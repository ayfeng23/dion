import math
import torch
import torch.distributed as dist
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union
from .muon import muon_update_pre_orthogonalize
# from .normuon_front import normuon_front_normalization

from .newton_schulz_triton import newton_schulz_triton, zeropower_via_newtonschulz5
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async

# Reuse Muon's helper functions
from .muon import (
    muon_update_newton_schulz,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)


def _full_dtype_and_shape(p: Tensor) -> Tuple[torch.Size, torch.dtype, torch.device]:
    if isinstance(p, DTensor):
        shape = p.size()  # global shape
        dev = p.to_local().device
        return shape, p.dtype, dev
    return p.size(), p.dtype, p.device


class FracNormuon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        interp: float = 1.0,
        ef_partial: bool = True,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
    ):
        # Chenk hyperparameter
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        if ef_decay < 0.0:
            raise ValueError(f"Invalid error-feedback decay (ef_decay): {ef_decay}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            mu=mu,
            ef_decay=ef_decay,
            muon_beta2=muon_beta2,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            nesterov=nesterov,
            interp=interp,
            ef_partial=ef_partial,
            adjust_lr=adjust_lr,
            algorithm="fracnormuon",
            step=0,
            fraction=fraction,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh is supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Group by optimizers
        fracnormuon_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "fracnormuon":
                fracnormuon_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        fracnormuon_tasks = self._create_fracnormuon_tasks(fracnormuon_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(fracnormuon_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    #ignoring this for now
    def _get_or_initialize_dion2_state_layer(self, param: Tensor) -> dict:
        """
        Layer-sharded momentum state for dion2:
        - 'momentum_full' lives only on the owner rank (owner is implicitly device_rank).
        """
        st = self.state[param]
        if "momentum_full" not in st:
            st["momentum_full"] = None
        return st

    def _get_or_initialize_dion2_state_local(self, param: Tensor) -> dict:
        """
        Local-shard momentum state for dion2:
        - Each rank keeps 'momentum_local' matching its local shard shape.
        """
        st = self.state[param]
        if "momentum_local" not in st:
            st["momentum_local"] = torch.zeros_like(param)
        if "variance_neuron" not in st:
            st["variance_neuron"] = torch.zeros_like(param[..., 0:1])
        return st

    #Only used by Adam and Lion
    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _pad_states(self, states: List[dict], n: int) -> List[dict]:
        """
        Pad states to length n. Real entries get is_pad=False; padded entries get is_pad=True.
        """
        out = list(states)
        # Mark existing entries explicitly as not padded
        for st in out:
            if "is_pad" not in st:
                st["is_pad"] = False
        # Append padded placeholders
        while len(out) < n:
            out.append({"momentum_full": None, "is_pad": True})
        return out

    def _create_fracnormuon_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "fracnormuon",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "fracmuon optimizer only supports matrix parameters."

            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            # Wrap hyperparameters as tensors for torch.compile
            fracnormuon_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=torch.tensor(group["fraction"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                nesterov=group["nesterov"],
                interp=group["interp"],
                ef_partial=group["ef_partial"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
            )

            # Create batches of parameters of size self._world_size
            for batch_params in create_param_batches(
                params, batch_size=self._world_size
            ):
                grads = [p.grad for p in batch_params]

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(batch_params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, pl)
                        for i, pl in enumerate(batch_params[0].placements)
                        if pl.is_shard() and batch_params[0].device_mesh.size(i) > 1
                    ]

                    # If we don't flatten 3D matrices, we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    if not group["flatten"]:
                        matrix_dims = {
                            batch_params[0].ndim - 1,
                            batch_params[0].ndim - 2,
                        }
                        is_batch_sharded = any(
                            pl.dim not in matrix_dims for _, pl in shard_placements
                        )
                        shard_placements = [
                            (i, pl)
                            for i, pl in shard_placements
                            if pl.dim in matrix_dims
                        ]
                        
                    # We currently do not support tensors sharded along the last dimension because NorMuon
                    # normalization later assumes a full trailing axis when computing means.
                    if any(pl.dim == params[0].ndim - 1 for _, pl in shard_placements):
                        raise NotImplementedError(
                            "FracNorMuon currently does not support parameters sharded along the last dimension. "
                            "Please avoid shards at dim -1."
                        )

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that non-flattened 3D tensors can have additional sharded batch dimensions
                    # Flattened 3D tensors are limited to one sharded dimension out of all dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "FracNormuon does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and batch_params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if not is_matrix_sharded: # modified so I don't have ot use multiple devices

                    # For this case, we use local momentum per shard
                    for x, g in zip(batch_params, grads):
                        st = self._get_or_initialize_dion2_state_local(x)

                        # Create task for non-communicating local update
                        yield AsyncTask(
                            fracnormuon_update_local_async(
                                X=[x],
                                G=[g],
                                STATE=st,
                                **fracnormuon_args,
                            )
                        )
                    continue

                # Otherwise we use layer-sharded momentum and owner mapping
                states = [
                    self._get_or_initialize_dion2_state_layer(p) for p in batch_params
                ]

                # Create task for communicating batch update
                yield AsyncTask(
                    dion2_update_batch_async(
                        X=pad_batch(batch_params, self._world_size),
                        G=pad_batch(grads, self._world_size),
                        STATES=self._pad_states(states, self._world_size),
                        shard_dim=sharded_tensor_dim,
                        **fracnormuon_args,
                    )
                )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        # Check whether algo_name matches "lion"
        if algo_name != "lion":
            raise RuntimeError(f"lion is applied to {algo_name} groups")

        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        # Check whether algo_name matches "adamw"
        if algo_name != "adamw":
            raise RuntimeError(f"adamw is applied to {algo_name} groups")

        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def fracnormuon_update_local_async(
    X: List[Tensor],
    G: List[Tensor],
    STATE: dict,  # Should put local momentum state here
    lr: Tensor,
    momentum: Tensor,
    muon_beta2: Tensor,
    ef_decay: Tensor,
    fraction: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    interp: float,
    ef_partial: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    newton_schulz_func: Optional[Callable] = None,
    **kwargs,
) -> Generator[None, None, None]:
    assert len(X) == len(G) == 1
    x = X[0]
    # g = to_local(G)[0]  # local shard grad
    M = [STATE["momentum_local"]]  # local shard momentum
    if ef_partial:
        M_new = fracnormuon_update_pre_orthogonalize(
            G=to_local(G),
            M=M,
            momentum=torch.ones_like(momentum),
            nesterov=nesterov,
        )
    else:
        M_new = fracnormuon_update_pre_orthogonalize(
            G=to_local(G),
            M=M,
            momentum=momentum,
            nesterov=nesterov,
        )
    # U = torch.sign(M_new[0])
    U = list(normuon_front_normalization(
        M_new,
        V=[STATE["variance_neuron"]],
        muon_beta2=muon_beta2,
    ))[0]

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, x.shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, x.shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    if not ef_partial:
        ef_decay = 1
    O_local = fractional_orthonormalize_update(
        M_full=U,
        ef_M=M[0],
        fraction=float(fraction),
        ef_decay=ef_decay,
        flatten=flatten,
        epsilon=epsilon,
        newton_schulz_func=newton_schulz_func,
        interp=interp,
    )

    # Apply update locally
    dion2_update_post_orthogonalize(
        X=to_local([x]),
        U=[O_local],
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )
    yield


def dion2_update_batch_async(
    X: List[Tensor],  # DTensors or local Tensors
    G: List[Tensor],  # local shards (regular Tensors)
    STATES: List[dict],  # layer-sharded optimizer states (each has 'momentum_full')
    lr: Tensor,
    ef_decay: Tensor,
    fraction: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    dion2 layer-sharded path:
      - Matrix-dim sharded: all_to_all shards <-> full, compute once on owner, all_to_all back.
      - Replicated/unsharded: compute once on owner (index=device_rank), all_gather dense updates.
      - Single-GPU (batch size=1): compute once on owner, apply locally.
    """

    # Ownership-by-index: owner of batch index j is rank j.
    assert 0 <= device_rank < world_size
    assert len(X) == len(STATES) == world_size  # Guaranteed by padding

    # convert gradients to local shards
    G_local = to_local(G)
    frac = float(fraction)

    # Compute adjusted lr from global shape (matrix dims last)
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Matrix-dimension sharded path
    if shard_dim is not None:
        assert len(X) == world_size, "Batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert X[0].size(shard_dim) % world_size == 0, (
            f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} "
            f"is not divisible by world size {world_size}."
        )

        # Shards -> single full gradient on owner (bf16 comm)
        G_bf16 = [g.to(dtype=torch.bfloat16) for g in G_local]
        recv_shards = [torch.empty_like(g) for g in G_bf16]
        work = dist.all_to_all(recv_shards, G_bf16, group=process_group, async_op=True)
        yield
        work.wait()

        full_grad_bf16 = torch.cat(recv_shards, dim=shard_dim)

        # Ownership-by-index contract:
        # For layer-sharded path, the "owner" of batch index j is rank j (device_rank).
        owner_state = STATES[device_rank]
        owner_is_pad = owner_state.get("is_pad", False)

        # Build the shards to send (bf16), but only do actual work if not "pad"
        if owner_is_pad:
            # For pads, do NOT allocate momentum_full or run NS.
            # Just return zero shards.
            send_shards = [torch.zeros_like(g) for g in G_bf16]  # bf16 payloads
        else:
            # Non-pads: allocate/accumulate, run NS, split to bf16 shards
            if owner_state["momentum_full"] is None:
                full_shape, param_dtype, param_device = _full_dtype_and_shape(
                    X[device_rank]
                )
                owner_state["momentum_full"] = torch.zeros(
                    full_shape, dtype=param_dtype, device=param_device
                )
            M_full = owner_state["momentum_full"]
            full_grad = full_grad_bf16.to(dtype=M_full.dtype)
            M_full.add_(full_grad)

            O_full = fractional_orthonormalize_update(
                M_full=M_full,
                fraction=frac,
                ef_decay=ef_decay,
                flatten=flatten,
                epsilon=epsilon,
                newton_schulz_func=newton_schulz_func,
            )
            # Split back to shards
            send_shards = [
                t.contiguous().to(torch.bfloat16)
                for t in torch.tensor_split(O_full, world_size, dim=shard_dim)
            ]

        # All-to-all back to shards
        U = [torch.empty_like(g) for g in G_bf16]
        work = dist.all_to_all(U, send_shards, group=process_group, async_op=True)
        yield
        work.wait()

    # Replicated / unsharded
    else:
        # Owner index is device_rank
        x_owner = X[device_rank]
        g_owner = G_local[device_rank]
        st_owner = STATES[device_rank]
        owner_is_pad = st_owner.get("is_pad", False)

        # Check whether we are in multi-GPU setting
        multi_gpu = process_group is not None and world_size > 1

        if multi_gpu:
            # For pads, do not allocate momentum_full or run NS.
            if owner_is_pad:
                payload_bf16 = torch.zeros_like(
                    g_owner, dtype=torch.bfloat16
                ).contiguous()
            # Non-pads: allocate/accumulate, run NS, prepare bf16 payload
            else:
                if st_owner["momentum_full"] is None:
                    full_shape, param_dtype, param_device = _full_dtype_and_shape(
                        x_owner
                    )
                    st_owner["momentum_full"] = torch.zeros(
                        full_shape, dtype=param_dtype, device=param_device
                    )
                M_full = st_owner["momentum_full"]
                M_full.add_(g_owner.to(dtype=M_full.dtype))

                O_full = fractional_orthonormalize_update(
                    M_full=M_full,
                    fraction=frac,
                    ef_decay=ef_decay,
                    flatten=flatten,
                    epsilon=epsilon,
                    newton_schulz_func=newton_schulz_func,
                )
                payload_bf16 = O_full.to(dtype=torch.bfloat16).contiguous()

            # All-gather the computed updates
            U = [torch.empty_like(payload_bf16) for _ in range(world_size)]
            work = dist.all_gather(U, payload_bf16, group=process_group, async_op=True)
            yield
            work.wait()

        else:
            # Single-GPU case: produce local update directly.
            # No padding case handling required.
            if st_owner["momentum_full"] is None:
                full_shape, param_dtype, param_device = _full_dtype_and_shape(x_owner)
                st_owner["momentum_full"] = torch.zeros(
                    full_shape, dtype=param_dtype, device=param_device
                )
            M_full = st_owner["momentum_full"]
            M_full.add_(g_owner.to(dtype=M_full.dtype))

            O_full = fractional_orthonormalize_update(
                M_full=M_full,
                fraction=frac,
                ef_decay=ef_decay,
                flatten=flatten,
                epsilon=epsilon,
                newton_schulz_func=newton_schulz_func,
            )
            U = [O_full]

    # Ensure foreach dtypes match parameter shards for the update
    X_local = to_local(X)
    U = [u.to(dtype=xi.dtype) for u, xi in zip(U, X_local)]

    dion2_update_post_orthogonalize(
        X=X_local,
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def make_work_view(M: Tensor) -> Tuple[Tensor, bool]:
    I, J = M.size(-2), M.size(-1)
    if I < J:
        return M.mT, True
    return M, False


@torch.compile(fullgraph=True)
def fracnormuon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    # torch._foreach_add_(M, G)
    torch._foreach_add_(M, torch._foreach_sub(G, torch._foreach_mul(G, momentum)))

    if nesterov: #which part should be nesterov updated? Dion doesn't use nesterov. Isn't only w.r.t K?
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U

def fractional_orthonormalize_update(
    M_full: Tensor,
    ef_M: Tensor,
    fraction: float,
    ef_decay: Tensor,
    flatten: bool,
    epsilon: Tensor,
    newton_schulz_func: Callable,
    interp: float,
) -> Tensor:
    M_work, transposed = make_work_view(M_full)
    ef_M_work, _ = make_work_view(ef_M)
    I, J = M_work.size(-2), M_work.size(-1)
    if fraction >= 1.0:
        # Full orthonormalization
        ortho_update = muon_update_newton_schulz(
            M_work, newton_schulz_func, flatten=flatten, epsilon=epsilon
        )
        ef_M_work.mul_(ef_decay)
    else:
        # Fractional orthonormalization
        k = int(math.ceil(fraction * J))
        ortho_update = topk_and_orthonormalize(
            M_work,
            ef_M=ef_M_work,
            ef_decay=ef_decay,
            k=k,
            flatten=flatten,
            epsilon=epsilon,
            newton_schulz_func=newton_schulz_func,
            interp=interp,
        )
    return ortho_update.mT.contiguous() if transposed else ortho_update


def topk_and_orthonormalize(
    M_work: Tensor,
    ef_M: Tensor,
    ef_decay: Tensor,
    k: int,
    flatten: bool,
    epsilon: Tensor,
    newton_schulz_func,
    interp: float,
) -> Tensor:
    """ """
    # Compute the top-k columns by L1 norm
    alpha = M_work.abs().sum(dim=-2)  # [J]
    K = torch.topk(alpha, k, sorted=False).indices  # [k]
    # Select and orthonormalize
    M_sel = torch.index_select(M_work, dim=-1, index=K)  # [I, k]
    O_sel = muon_update_newton_schulz(
        M_sel, newton_schulz_func, flatten=flatten, epsilon=epsilon
    )
    # In-place error-feedback decay only on selected columns:
    ef_M[..., K] *= ef_decay
    # Construct the full update matrix

    O_full = M_work.clone().to(O_sel.dtype) * interp
    O_full.index_copy_(dim=-1, index=K, source=O_sel)
    return O_full


def dion2_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


@torch.compile(fullgraph=True)
def normuon_front_normalization(
    M_new: List[Tensor],
    V: List[Tensor],
    muon_beta2: Tensor,
) -> List[Tensor]:
    """
    NorMuon normalization step after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    V_dtype = V[0].dtype
    M_new = [m_new.to(dtype=V_dtype) for m_new in M_new]

    M_new_sq = torch._foreach_mul(M_new, M_new)  # list of m*m, same shapes as M_new
    neuron_norms = [m_new_sq.mean(dim=-1, keepdim=True) for m_new_sq in M_new_sq]  # Shape: [*, 1]
    torch._foreach_lerp_(
        V, neuron_norms, 1 - muon_beta2
    )  # Update variance neuron buffer

    denom = torch._foreach_sqrt(V)  # list of sqrt(v)
    torch._foreach_add_(denom, 1e-8)  # denom[i] += 1e-8
    normalized_U = torch._foreach_div(M_new, denom)  # list of u / denom
    return normalized_U