import math
import torch
from collections import defaultdict
import torch.distributed as dist
import wandb
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import (
    DistributedOrthoBase,
    megabatch_orthogonalize_async,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)

from .opt_utils import (
    AsyncTask,
    create_named_batches,
    pad_names,
    to_local,
)
from .dion2 import dion2_post_orthogonalize


class NorDion2(DistributedOrthoBase):
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
        use_gram_newton_schulz: Whether to use Gram Newton-Schulz for orthogonalization.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is ``func(input: Tensor, epsilon: float) -> Tensor``.

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
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        newton_schulz_func: Optional[Callable] = None,
    ):
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
        super().__init__(
            params, distributed_mesh, "nordion2", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == self._algo_name and "variance_neuron" not in state:
            state["variance_neuron"] = torch.zeros_like(param[..., 0:1])
        return state

    def _get_shard_info(self, param: Tensor, group: dict):
        result = super()._get_shard_info(param, group)
        _, is_matrix_sharded, sharded_tensor_dim = result
        if is_matrix_sharded and sharded_tensor_dim == param.ndim - 1:
            raise NotImplementedError(
                "NorDion2 currently does not support parameters sharded along the last dimension. "
                "Please avoid shards at dim -1."
            )
        return result

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched NorDion2 task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
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

            update_args = dict(
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

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p, name in group_items:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append((p, name))

            for (_shape, _sharding, _dtype), items in shape_groups.items():
                params = [p for p, _ in items]
                names  = [n for _, n in items]
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, self._algo_name) for p in params]
                momentums = [s["momentum"] for s in states]
                variances_neuron = [s["variance_neuron"] for s in states]

                is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                    self._get_shard_info(params[0], group)
                )

                megabatch_args = update_args
                if is_batch_sharded and not is_matrix_sharded:
                    megabatch_args = {**update_args, "process_group": None}

                yield AsyncTask(
                    nordion2_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances_neuron,
                        names=names,
                        shard_dim=sharded_tensor_dim,
                        **megabatch_args,
                    )
                )


def nordion2_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    names: List[str], 
    lr: Tensor,
    fraction: float,
    momentum: Tensor,
    muon_beta2: Tensor,
    weight_decay: Tensor,
    k_sel: str,  # How to select submatrix ("topk" or "random")
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    """
    Mega-batched NorDion2 update: processes ALL same-shape parameters in one
    communication round instead of world_size-sized batches.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V)

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

    # Convert shard_dim to negative for comm_dim
    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None

    # Orthogonalize via shared megabatch communication
    U_ortho = yield from megabatch_orthogonalize_async(
        U_selected,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
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
    indices, _ = indices.sort(dim=-1) #Very necessary for some reason
    
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