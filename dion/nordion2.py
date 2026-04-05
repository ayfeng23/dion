import math
import torch
import torch.distributed as dist
import wandb
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .newton_schulz_triton import newton_schulz_triton
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    create_named_batches,
    pad_batch,
    pad_names,
    to_local,
)
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async

# Reuse Muon's helper functions
from .muon import (
    muon_update_newton_schulz,
    zeropower_via_newtonschulz5,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)
from .dion2 import dion2_post_orthogonalize


class NorDion2(Optimizer):
    """
    Distributed NorDion2 optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For NorDion2, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        fraction: Fraction of submatrix to orthogonalize per update (0 < fraction <= 1).
        mu: Momentum factor for NorDion2 algorithm.
        muon_beta2: Second beta parameter for NorDion2 algorithm's adaptive updates.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        cautious_wd: Whether to apply weight decay only where update and parameter signs align.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    NorDion2 optimizer: https://arxiv.org/abs/2510.05491
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        fraction: float = 0.25,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        k_sel: str = "topk",
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "rms_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if muon_beta2 < 0.0:
            raise ValueError(f"Invalid muon_beta2: {muon_beta2}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            fraction=fraction,
            mu=mu,
            muon_beta2=muon_beta2,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="nordion2",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
            k_sel=k_sel,
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
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        nordion2_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "nordion2":
                nordion2_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        nordion2_tasks = self._create_nordion2_tasks(nordion2_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(nordion2_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

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
            if algo == "nordion2":
                state["variance_neuron"] = torch.zeros_like(param[..., 0:1])
        return state

    def _create_nordion2_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "nordion2",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of NorDion2 matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "NorDion2 optimizer only supports matrix parameters."

            if "param_names" in group:
                group_items = [
                    (p, n)
                    for p, n in zip(group["params"], group["param_names"])
                    if p.grad is not None
                ]
            else:
                group_items = [(p, "<unnamed>") for p in group["params"] if p.grad is not None]

            if not group_items:
                continue

            # Wrap hyperparameters in tensors for torch.compile
            nordion2_update_args = dict(
                lr=torch.tensor(group["lr"]),
                fraction=group["fraction"],
                momentum=torch.tensor(group["mu"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                k_sel=group["k_sel"],
                epsilon=torch.tensor(group["epsilon"]),
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
                cautious_wd=group["cautious_wd"],
            )

            # Create batches of parameters of size self._world_size
            for batch in create_named_batches(group_items, batch_size=self._world_size):
                params = [p for p, _ in batch]
                names  = [n for _, n in batch]
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, algo_name) for p in params]
                momentums = [s["momentum"] for s in states]
                variances_neuron = [s["variance_neuron"] for s in states]

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # If we don't flatten 3D matrices, we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    # We currently do not support tensors sharded along the last dimension because NorDion2
                    # normalization later assumes a full trailing axis when computing means.
                    if any(p.dim == params[0].ndim - 1 for _, p in shard_placements):
                        raise NotImplementedError(
                            "NorDion2 currently does not support parameters sharded along the last dimension. "
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
                            "NorDion2 does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m, v, n in zip(
                        params, gradients, momentums, variances_neuron, names
                    ):
                        yield AsyncTask(
                            nordion2_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                V=[v],
                                names=[n],
                                shard_dim=None,  # No sharded matrix dim
                                **nordion2_update_args,
                            )
                        )
                # Otherwise, we parallelize the NorDion2 update across devices
                else:
                    yield AsyncTask(
                        nordion2_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            V=pad_batch(variances_neuron, self._world_size),
                            names=pad_names(names, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **nordion2_update_args,
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
            cautious_wd = group["cautious_wd"]

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    cautious_wd=cautious_wd,
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
            cautious_wd = group["cautious_wd"]
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
                    cautious_wd=cautious_wd,
                )
            )


def nordion2_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance neuron buffer (modified in place)
    names: List[str],  # Names of the parameters
    lr: Tensor,  # Learning rate (scalar tensor)
    fraction: float,  # Fraction of submatrix to orthogonalize (0 < fraction <= 1)
    momentum: Tensor,  # Momentum factor (scalar tensor)
    muon_beta2: Tensor,  # Muon beta2 for normalization
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    k_sel: str,  # How to select submatrix ("topk" or "random")
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    """
    Batched version of Muon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)

    # Choose Neurons
    select_dim = -2

    # Update momentum and compute the inputs for orthogonalization
    U_selected, indices_list = nordion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        momentum=momentum,
        nesterov=nesterov,
        k_sel=k_sel,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert len(X) == world_size, "Batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U_selected[0], DTensor), "U should contain local shards"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        # Allocate buffers to receive shards of one whole matrix from other devices
        single_matrix_shards = [torch.empty_like(u) for u in U_selected]

        # Redistribute the shards to form one unique full tensor on each device
        work = dist.all_to_all(
            single_matrix_shards, U_selected, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        U_ortho = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(
            U_ortho, single_matrix_shards, group=process_group, async_op=True
        )
        #
        yield
        work.wait()

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    elif len(U_selected) > 1:
        assert len(U_selected) == world_size, "Batch size must equal world size"
        assert process_group is not None

        single_matrix = U_selected[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Allocate empty tensors to receive updates from other devices
        U_ortho = [torch.empty_like(u) for u in U_selected]

        # All gather orthogonalized results from other devices into buffer
        work = dist.all_gather(
            U_ortho, single_matrix.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # Single tensor with no sharded dimension. This happens in 2 cases:
    # - Running on a single GPU
    # - 3D+ tensors sharded along a batch dimension (different whole matrices per device)
    else:
        assert len(U_selected) == 1
        U_ortho[0] = muon_update_newton_schulz(
            U_selected[0],
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

    # NorDion2 normalization
    # Select V for each top-k row
    V_sel = []
    for v, indices in zip(V, indices_list):
        selected_v = v.index_select(dim=select_dim, index=indices)
        V_sel.append(selected_v)

    U_normed, V_sel = nordion2_normalization(
        U_ortho,
        V_sel=V_sel,
        muon_beta2=muon_beta2,
    )
    
    # Copy back update V_sel to V using indices
    for v, v_sel, indices in zip(V, V_sel, indices_list):
        v.index_copy_(dim=select_dim, index=indices, source=v_sel)

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Update model parameters with orthogonalized output
    if wandb.run is not None:
        for name, idx in zip(names, indices_list):
            wandb.log({f"ortho_sel_k/{name}": idx.tolist(),}, commit=False)

    dion2_post_orthogonalize(
        X=to_local(X),
        U=U_normed,
        indices=indices_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        select_dim=-2,
    )


@torch.compile(fullgraph=True)
def nordion2_normalization(
    U_ortho: List[Tensor],
    V_sel: List[Tensor],
    muon_beta2: Tensor,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    NorDion2 normalization step after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    V_dtype = V_sel[0].dtype
    U_ortho = [u.to(dtype=V_dtype) for u in U_ortho]
    
    norm_U = [
        u.norm(p=2, dim=(-2, -1), keepdim=True) for u in U_ortho
    ]  # list of ||u||_F, shape [*, 1, 1]

    U_sq = torch._foreach_mul(U_ortho, U_ortho)  # list of u*u, same shapes as U_ortho
    neuron_norms = [u_sq.mean(dim=-1, keepdim=True) for u_sq in U_sq]  # Shape: [*, rows, 1]
    
    torch._foreach_lerp_(
        V_sel, neuron_norms, 1 - muon_beta2
    )  # Update variance neuron buffer

    denom = torch._foreach_sqrt(V_sel)  # list of sqrt(v)
    torch._foreach_add_(denom, 1e-8)  # denom[i] += 1e-8
    normalized_U = torch._foreach_div(U_ortho, denom)  # list of u / denom

    norm_U_new = [
        nu.norm(p=2, dim=(-2, -1), keepdim=True) for nu in normalized_U
    ]  # list of ||normalized_u||_F, shape [*, 1, 1]
    
    # Protect against division by zero when norm_U_new is zero.
    # This can happen when U is all zeros (e.g., zero gradients from zero-initialized weights).
    # In this case, norm_U is also zero, so after clamping norm_U_new to ε the ratio becomes 0/ε ≈ 0,
    # and normalized_U * ratio correctly remains zero, preserving the zero state.
    norm_U_new_safe = [nu.clamp(min=1e-8) for nu in norm_U_new]
    
    ratio = torch._foreach_div(
        norm_U, norm_U_new_safe
    )  # list of ||u||_F / ||normalized_u||_F, shape [*, 1, 1]
    
    torch._foreach_mul_(normalized_U, ratio)  # normalized_u[i] *= ratio

    return normalized_U, V_sel

def nordion2_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    fraction: float,
    momentum: Tensor,
    nesterov: bool,
    k_sel: str,
) -> List[Tensor]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype

    num_select = M[0].size(-2)
    k = max(1, int(math.ceil(fraction * num_select)))

    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_add_(M, G)
    # print(nesterov)
    assert nesterov == False

    M_stacked = torch.stack(M, dim=0)

    if k_sel == "topk":
        slice_norms = M_stacked.norm(p=1, dim=-1)
        scores = slice_norms
    elif k_sel == "random":
        batch_size = M_stacked.size(0)
        num_rows = M_stacked.size(1)
        scores = torch.rand(batch_size, num_rows, device=M_stacked.device)
    else:
        raise ValueError(f"Unknown k_sel value: {k_sel}")

    _, indices = torch.topk(scores, k, dim=-1, sorted=False)
    
    # Selecting rows
    num_cols = M[0].size(-1)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_cols)
    selected_stacked = torch.gather(M_stacked, dim=-2, index=indices_expanded)
    U_selected = list(selected_stacked.to(dtype=torch.bfloat16).unbind(dim=0))

    indices_list = list(indices.unbind(dim=0))
    for m, idx in zip(M, indices_list):
        selected_slice = m.index_select(dim=-2, index=idx)
        m.index_copy_(dim=-2, index=idx, source=selected_slice * momentum)

    return U_selected, indices_list