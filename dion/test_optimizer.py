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
from .normuon_front import normuon_front_normalization
from .dion2 import dion2_update_post_orthogonalize

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


class TestOptimizer(Optimizer):
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
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
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
            adjust_lr=adjust_lr,
            algorithm="testoptimizer",
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
        testoptimizer_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "testoptimizer":
                testoptimizer_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        testoptimizer_tasks = self._create_testoptimizer_tasks(testoptimizer_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(testoptimizer_tasks, lion_tasks, adamw_tasks)
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
            st["variance_neuron"] = torch.zeros((param.size(-1), param.size(-1)), device=param.device, dtype=param.dtype)
            # st["variance_neuron"] = torch.zeros_like(param[..., 0:1])
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

    def _create_testoptimizer_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "testoptimizer",
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
            testoptimizer_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=torch.tensor(group["fraction"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                nesterov=group["nesterov"],
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
                            "TestOptimizer currently does not support parameters sharded along the last dimension. "
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
                            "TestOptimizer does not support parameters with multiple sharded dimensions."
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
                            update_local_async(
                                X=[x],
                                G=[g],
                                STATE=st,
                                **testoptimizer_args,
                            )
                        )
                    continue

                # Otherwise we use layer-sharded momentum and owner mapping
                states = [
                    self._get_or_initialize_dion2_state_layer(p) for p in batch_params
                ]

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


def update_local_async(
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
    flatten: bool,
    adjust_lr: Optional[str],
    newton_schulz_func: Optional[Callable] = None,
    **kwargs,
) -> Generator[None, None, None]:
    assert len(X) == len(G) == 1
    x = X[0]
    # g = to_local(G)[0]  # local shard grad
    M = [STATE["momentum_local"]]  # local shard momentum
    U = adam_update(
        G=to_local(G),
        M=M,
        V=[STATE["variance_neuron"]], 
        momentum=momentum,
        nesterov=nesterov,
        newton_schulz_func=newton_schulz_func, 
        flatten=flatten,
        epsilon=epsilon,
    )

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, x.shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, x.shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Apply update locally
    dion2_update_post_orthogonalize(
        X=to_local([x]),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )
    yield


def make_work_view(M: Tensor) -> Tuple[Tensor, bool]:
    I, J = M.size(-2), M.size(-1)
    if I < J:
        return M.mT, True
    return M, False

# def psd_inv_sqrt(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     A32 = A.float()
#     w, Q = torch.linalg.eigh(A32)
#     w = w.clamp_min(eps)
#     return ((Q * w.rsqrt().unsqueeze(-2)) @ Q.mT).to(A.dtype)

def psd_inv_sqrt(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    try:
        A32 = A.float()
        w, Q = torch.linalg.eigh(A32)
        w = w.clamp_min(eps)
        return ((Q * w.rsqrt().unsqueeze(0)) @ Q.mT).to(A.dtype)
    except RuntimeError:
        A32 = A.float()
        n = A32.shape[-1]
        I = torch.eye(n, device=A.device, dtype=torch.float32)
        scale = A32.diagonal().mean().clamp_min(1.0)
        jitter = 1e-6 * scale
        A32 = A32 + jitter * I
        w, Q = torch.linalg.eigh(A32)
        w = w.clamp_min(jitter)
        return (Q * w.rsqrt().unsqueeze(0)) @ Q.mT



def adam_update(
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
    newton_schulz_func, 
    flatten,
    epsilon,
) -> List[Tensor]:
    V_dtype = V[0].dtype
    G = [g.to(dtype=V_dtype) for g in G]
    M = [m.to(dtype=V_dtype) for m in M]

    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    U_new = []
    for g, m, v in zip(G, U, V):
        d = g - m
        gram = momentum * d.mT @ d              # (J, J)
        v.mul_(momentum).add_(gram, alpha=(1.0 - momentum))
        # r = torch.linalg.cholesky(v).mT
        # stack = torch.cat((m, r), dim=-1, out=None)
        # w = muon_update_newton_schulz(stack, newton_schulz_func, flatten, epsilon)
        # w_m, w_r = torch.split(w, [m.size(-1), r.size(-1)], dim=-1)
        u = m @ psd_inv_sqrt(m.mT @ m)
        U_new.append(u)

    return U_new