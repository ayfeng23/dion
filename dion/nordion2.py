import math
import torch
from collections import defaultdict
import torch.distributed as dist
try:
    import wandb
except ImportError:
    wandb = None
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
    to_local,
)
from .dion2 import dion2_pre_orthogonalize, dion2_post_orthogonalize
from .normuon import normuon_normalization_stacked


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
        epsilon: Small value to avoid division by zero.
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
                group_items = [
                    (p, "<unnamed>")
                    for p in group["params"]
                    if p.grad is not None
                ]

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

            num_heads = self._resolve_num_heads(group)

            for (_shape, _sharding, _dtype), items in shape_groups.items():
                params = [p for p, _ in items]
                names  = [n for _, n in items]
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, self._algo_name) for p in params]
                momentums = [s["momentum"] for s in states]
                variances_neuron = [s["variance_neuron"] for s in states]

                if num_heads is not None:
                    params, gradients, momentums, variances_neuron = self._prepare_head_split(
                        num_heads, params, gradients, momentums, variances_neuron
                    )
                    megabatch_args = {**update_args, "process_group": None}
                    shard_dim = None
                else:
                    is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                        self._get_shard_info(params[0], group)
                    )
                    megabatch_args = update_args
                    if is_batch_sharded and not is_matrix_sharded:
                        megabatch_args = {**update_args, "process_group": None}
                    shard_dim = sharded_tensor_dim

                yield AsyncTask(
                    nordion2_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances_neuron,
                        names=names,
                        shard_dim=shard_dim,
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
    k_sel: str,
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
    is_sharded = shard_dim is not None

    # Update momentum and compute the inputs for orthogonalization
    U_selected, indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=momentum,
        select_dim=-2,
    )

    # comm_dim is just -2
    comm_dim = select_dim if is_sharded else None

    # On the sharded path X[0] must still be a DTensor, so .shape[comm_dim]
    # is the unsharded global size. The megabatch fn uses this to compute
    # the rank-consistent pad size for its alltoall. Catch the case where a
    # future refactor moves to_local(X) above this point and silently
    # collapses .shape to the local size.
    if comm_dim is not None:
        if not isinstance(X[0], DTensor):
            raise TypeError(
                "Sharded path requires X[0] to be a DTensor so .shape gives "
                f"the global size; got {type(X[0]).__name__}."
            )
        global_comm_dim_size = X[0].shape[comm_dim]
    else:
        global_comm_dim_size = None

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
        global_comm_dim_size=global_comm_dim_size,
    )

    # NorDion2 normalization
    # Select V for each top-k row
    V_local = to_local(V)
    V_sel = []
    for v, indices in zip(V_local, indices_list):
        selected_v = v.index_select(dim=select_dim, index=indices)
        V_sel.append(selected_v)
    U_stacked = torch.stack(U_ortho)
    V_sel_stacked = torch.stack(V_sel).float()

    U_stacked, V_stacked = normuon_normalization_stacked(U_stacked, V_sel_stacked, muon_beta2)
    U_normed = [U_stacked[i] for i in range(N)]

    # Copy back updated V_sel to V using indices (cast back from f32)
    for i, (v, indices) in enumerate(zip(V_local, indices_list)):
        v.index_copy_(dim=select_dim, index=indices, source=V_stacked[i].to(v.dtype))

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
    if wandb is not None and wandb.run is not None:
        for name, idx, v in zip(names, indices_list, V_local):
            wandb.log({
                f"ortho_sel_k/{name}": idx.tolist(),
                f"nordion2_V/{name}": v.flatten().tolist(),
            }, commit=False)

    # dion2_post_orthogonalize handles f32 upcast internally.
    X_local = to_local(X)

    dion2_post_orthogonalize(
        X=X_local,
        U=U_normed,
        indices=indices_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        select_dim=-2,
    )


_inductor_workaround = (
    torch._inductor.config.patch(loop_ordering_after_fusion=False)
    if torch.__version__ < "2.13"
    else lambda fn: fn
)
