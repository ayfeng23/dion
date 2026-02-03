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
from .normuon_front import normuon_front_second_moment
import wandb

from .newton_schulz_triton import newton_schulz_triton, zeropower_via_newtonschulz5
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
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)


class FracNormuon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        interp: float = 1.0,
        ef_partial: bool = True,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        verbose: bool = False,
        # Austin: partial_warmup determines whether to use warmup on all updates (False) or orthonormalized ones (True)
        partial_warmup: bool = True, 
        # Austin: warmup_cutoff is an auxiliary variable to ensure partial_warmup logic only affects warmup (and not warmdown crucially)
        warmup_cutoff: int = 0, 
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if ef_decay < 0.0:
            raise ValueError(f"Invalid ef_decay: {ef_decay}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            ef_decay=ef_decay,
            muon_beta2=muon_beta2,
            fraction=fraction,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            interp=interp,
            ef_partial=ef_partial,
            adjust_lr=adjust_lr,
            algorithm="fracnormuon",
            # Austin: step needed to see if we reach warmup_cutoff
            step=0,
            warmup_cutoff=warmup_cutoff,
            # Austin: store warmup-non-adjusted learning rate
            full_lr=torch.tensor(lr) if partial_warmup else None,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
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
        self.verbose = verbose

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        fracnormuon_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
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
        fracnormuon_tasks = self._create_fracnormuon_tasks(fracnormuon_groups, verbose=self.verbose)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(fracnormuon_tasks, lion_tasks, adamw_tasks)
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
            if algo == "fracnormuon":
                state["variance_neuron"] = torch.zeros_like(param[..., 0:1])
        return state

    def _create_fracnormuon_tasks(
        self,
        param_groups: List[dict],
        verbose: bool = False,
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of FracNormuon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
        """
        for group in param_groups:
            assert group["algorithm"] == "fracnormuon"
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Fracnormuon only supports matrix parameters."

            if "param_names" in group:
                group_items = [
                    (p, n)
                    for p, n in zip(group["params"], group["param_names"])
                    if p.grad is not None
                ]
            else:
                group_items = [(p, "<unnamed>") for p in group["params"] if p.grad is not None]

            # group_params = [p for p in group["params"] if p.grad is not None]
            if not group_items:
                continue

            # Most hyperparameters as tensors for torch.compile
            # Here "fraction" only determines the dimension of the submatrix
            # to be orthonormalized. Hence, it doesn't need to be a tensor
            fracnormuon_args = dict(
                lr=torch.tensor(group["lr"]),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=group["fraction"],
                muon_beta2=torch.tensor(group["muon_beta2"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                interp=group["interp"],
                ef_partial=group["ef_partial"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                full_lr=group["full_lr"],
                warmup_cutoff=group["warmup_cutoff"],
                step=group["step"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
            )

            # Create batches of parameters of size self._world_size
            for batch in create_named_batches(group_items, batch_size=self._world_size):
                params = [p for p, _ in batch]
                names  = [n for _, n in batch]
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "fracnormuon") for p in params]
                momentums = [s["momentum"] for s in states]
                variances = [s["variance_neuron"] for s in states]

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

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that non-flattened 3D tensors can have additional sharded batch dimensions
                    # Flattened 3D tensors are limited to one sharded dimension out of all dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Fracnormuon does not support parameters with multiple sharded dimensions."
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
                    for x, g, m, v in zip(params, gradients, momentums, variances):
                        yield AsyncTask(
                            fracnormuon_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                names=[n],
                                V=[v],
                                shard_dim=None,  # No sharded matrix dim
                                **fracnormuon_args,
                                verbose=verbose,
                            )
                        )
                # Otherwise, we parallelize the Muon update across devices
                else:
                    yield AsyncTask(
                        fracnormuon_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            names=pad_names(names, self._world_size),
                            V=pad_batch(variances, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **fracnormuon_args,
                            verbose=verbose,
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


def fracnormuon_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    names: List[str], # Name of Tensor parameters
    V: List[Tensor],
    lr: Tensor,  # Learning rate (scalar tensor)
    muon_beta2: Tensor,
    ef_decay: Tensor,  # Error-feedback factor (scalar tensor)
    fraction: float,  # Fraction of submatrix to orthogonalize (0 < fraction <= 1)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    interp: float,
    ef_partial: bool,
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    step: int,
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    verbose: bool = False,
    full_lr: Optional[Tensor] = None, 
    warmup_cutoff: int = 0,
) -> Generator[None, None, None]:
    """
    Batched version of FracNormuon update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """
    assert len(X) == len(G)
    assert len(X) == len(M)

    # Determine selection dimension based on sharding and tensor shape:
    # For sharded matrices, we align select_dim with shard_dim
    # For unsharded matrices (DDP or single-GPU), we select the shorter dimension
    ndim = X[0].ndim
    select_dim = -2
    # select_dim = None
    # if shard_dim is not None:
    #     # Normalize shard_dim to negative indexing for unified treatment
    #     shard_dim = shard_dim if shard_dim < 0 else shard_dim - ndim
    #     if shard_dim == -2:
    #         select_dim = -2  # Row-sharded
    #     elif shard_dim == -1:
    #         select_dim = -1  # Column-sharded
    #         # Austin: if shard_dim=-1, this affects neuron_variances
    #         raise NotImplementedError(
    #             "NorMuon currently does not support parameters sharded along the last dimension. "
    #             "Please avoid shards at dim -1."
    #         )
    # if select_dim is None:
    #     num_rows, num_cols = X[0].shape[-2:]
    #     select_dim = -2 if num_rows <= num_cols else -1

    # # Update momentum and select top-α fraction along select_dim
    # num_select = to_local(M)[0].size(select_dim)
    # print("M", to_local(M)[0].size(), "Select Dim", select_dim)
    # k = max(1, int(math.ceil(fraction * num_select)))
    # print("K",k)
    U, U_selected, indices_list = fracnormuon_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        V=to_local(V),
        fraction=fraction,
        select_dim=select_dim,
        muon_beta2=muon_beta2,
    )

    # Austin: error feedback only updated params
    # torch._foreach_mul_(M, ef_decay)
    if interp == 0:
        for m, idx in zip(M, indices_list):
            selected_slice = m.index_select(dim=select_dim, index=idx)
            m.index_copy_(dim=select_dim, index=idx, source=selected_slice * ef_decay)
    else:
        torch._foreach_mul_(M, ef_decay)

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert len(X) == world_size, "Batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        # Allocate buffers to receive shards of one whole submatrix from other devices
        recv_shards = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(
            recv_shards, U_selected, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        # Only submatrix is orthogonalized!
        full_submatrix = torch.cat(recv_shards, dim=select_dim)
        full_submatrix = muon_update_newton_schulz(
            full_submatrix, newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        send_shards = [
            t.contiguous()
            for t in torch.tensor_split(full_submatrix, world_size, dim=select_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        O = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(O, send_shards, group=process_group, async_op=True)
        yield
        work.wait()

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    elif len(U_selected) > 1:
        assert len(U_selected) == world_size, "Batch size must equal world size"
        assert process_group is not None

        single_matrix = U_selected[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_ortho = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Allocate empty tensors to receive updates from other devices
        O = [torch.empty_like(u) for u in U_selected]
        # All gather orthogonalized results from other devices into buffer
        work = dist.all_gather(
            O, single_ortho.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # Single tensor with no sharded dimension. This happens in 2 cases:
    # - Running on a single GPU
    # - 3D+ tensors sharded along a batch dimension (different whole matrices per device)
    else:
        assert len(U_selected) == 1
        O = [
            muon_update_newton_schulz(
                U_selected[0], newton_schulz_func, flatten=flatten, epsilon=epsilon
            )
        ]

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
        adjusted_full_lr = full_lr if full_lr is not None else None
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
        adjusted_full_lr = adjust_lr_spectral_norm(full_lr, X[0].shape, flatten=flatten) if full_lr is not None else None
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
        adjusted_full_lr = adjust_lr_rms_norm(full_lr, X[0].shape, flatten=flatten) if full_lr is not None else None
    else:
        raise ValueError(f"Unknown adjust_lr: {adjust_lr}")

    # print("LR", full_lr, lr) # Austin: checking if full_lr actually is the right lr and it is
    dion2_post_orthogonalize(
        X=to_local(X),
        O=O,
        U=U, 
        indices=indices_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        select_dim=select_dim,
        interp=interp,
        full_lr=full_lr,
        adjusted_full_lr=adjusted_full_lr,
        # Austin: full_lr is not None occurs only when we use partial_warmup for the run, step < warmup_cutoff so that orthonormalized parts can warm down
        partial_warmup=(full_lr is not None) and (step < warmup_cutoff),
        process_group=process_group,
        device_rank=device_rank,
        names=names,
    )


@torch.compile(fullgraph=True)
def fracnormuon_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    fraction: float,
    select_dim: int,
    muon_beta2: Tensor,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    More specifically, it does the following steps:
        - updates the momentum with gradient
        - computes the top-k indices (according to L1 norm) to determine submatrices
        - (other norms can be used such as L2 norm)
        - We move error feedback out for more flexibility
        - output submatrices and indices
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype

    # norm_dim is the dimension we compute norm over
    # select_dim is the dimension we select submatrix from
    num_select = M[0].size(select_dim)
    norm_dim = -1  
    k = max(1, int(math.ceil(fraction * num_select)))

    # Update momentum: M = M + G
    G = [g.to(dtype=dtype) for g in G]
    torch._foreach_add_(M, G)

    #Non-distributed / no-communication for row-sharding
    U = normuon_front_second_moment(G, M, V, muon_beta2)

    U_stacked = torch.stack(U, dim=0)

    # Compute L1 norm along norm_dim (sum of absolute values)
    slice_norms = U_stacked.norm(p=1, dim=norm_dim)

    # Batched topk: indices shape (batch_size, k)
    _, indices = torch.topk(slice_norms, k, dim=-1, sorted=False)

    # Always row-wise selection
    num_cols = M[0].size(-1)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_cols)
    selected_stacked = torch.gather(U_stacked, dim=-2, index=indices_expanded)
     

    indices_list = list(indices.unbind(dim=0))
    
    # Convert to bf16 and unstack for communication
    U_selected = list(selected_stacked.to(dtype=torch.bfloat16).unbind(dim=0))

    return [u.clone() for u in U], U_selected, indices_list

# @torch.compile(fullgraph=True)
def dion2_post_orthogonalize(
    X: List[Tensor],
    O: List[Tensor],
    U: List[Tensor],
    indices: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    select_dim: int,
    interp: float,
    full_lr: Optional[Tensor],
    adjusted_full_lr: Optional[Tensor],
    partial_warmup: bool,
    process_group: Optional[ProcessGroup],
    device_rank: int,
    names: List[str],
):
    dtype = X[0].dtype
    # assert len(X) == 1
    O = [o.to(dtype=dtype) for o in O]
    U = [u.to(dtype=dtype) for u in U]
    if not partial_warmup: #default to full updates (mostly for warmdown)
        full_lr = base_lr
        adjusted_full_lr = adjusted_lr

    for x, o, u, name, idx in zip(X, O, U, names, indices):
        # Austin: O is NS(U) and U=Adam-Normalized(M)
        # u_subset is the "replaced" part of U
        o_frob = o.norm(p="fro")
        o_frob_per_elem = o_frob / math.sqrt(o.numel())
        u_subset = u.index_select(select_dim, idx)
        u_subset_frob = u_subset.norm(p="fro")
        u_frob_per_elem = u.norm(p="fro") / math.sqrt(x.numel())

        # if process_group is None or device_rank == 0: 
            # Austin: under DDP, I assume this gets the "whole batch", because you accum gradient anyway?
            # And you don't need to shard params?
        wandb.log({
            f"o_frob/{name}": o_frob.item(),
            f"u_replace_frob/{name}": u_subset_frob.item(),
            f"o_frob_norm/{name}": o_frob_per_elem.item(),
            f"u_frob_norm/{name}": u_frob_per_elem.item()
        }, commit=False)
        interp = o_frob / u_subset_frob
        # Austin: first do weight decay and then updates
        # split weight_decay based on learning rates
        x_sel = x.index_select(select_dim, idx)
        x.mul_(1 - base_lr * weight_decay)
        x_sel.mul_(1 - full_lr * weight_decay) #full_lr to avoid warmup issues
        x.index_copy_(select_dim, idx, x_sel)

        x_sel = x.index_select(select_dim, idx) 
        x.sub_((u * interp) * adjusted_lr)
        x_sel.sub_(o * adjusted_full_lr)  
        x.index_copy_(select_dim, idx, x_sel)
 

# A helper function to print selection chocie for each matrix
# It only prints once `verbose` is set True
_printed_configs: set = set()


def _print_selection_choice(
    shape: torch.Size,
    shard_dim: Optional[int],
    select_dim: int,
    ndim: int,
):
    config_key = (tuple(shape), shard_dim, select_dim)
    if config_key not in _printed_configs:
        _printed_configs.add(config_key)

        num_rows, num_cols = shape[-2:]
        select_info = "rows" if select_dim == -2 else "columns"
        norm_info = "row norms" if select_dim == -2 else "col norms"

        if shard_dim is None:
            mode = "DDP/Single-GPU"
            shorter = "rows" if num_rows <= num_cols else "cols"
            reason = f"shorter dim = {shorter} ({min(num_rows, num_cols)})"
        else:
            # Normalize shard_dim for display
            normalized = shard_dim if shard_dim < 0 else shard_dim - ndim
            if normalized == -2:
                mode = "FSDP"
                reason = f"row-sharded (shard_dim={shard_dim}→-2)"
            elif normalized == -1:
                mode = "FSDP"
                reason = f"col-sharded (shard_dim={shard_dim}→-1)"
            else:
                mode = "FSDP batch-sharded"
                shorter = "rows" if num_rows <= num_cols else "cols"
                reason = f"shard_dim={shard_dim} (batch), shorter = {shorter}"

        print(
            f"[FracNormuon] Shape {tuple(shape)}: {mode}, {reason} → "
            f"select top-α {select_info} by {norm_info}"
        )
