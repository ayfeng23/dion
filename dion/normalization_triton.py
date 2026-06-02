import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    import types

    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.Config = dict
    triton.heuristics = lambda _: lambda fn: fn
    triton.cdiv = lambda a, b: (a + b - 1) // b
    tl = types.ModuleType("triton.language")
    tl.constexpr = int


@triton.heuristics(
    {"BLOCK_SIZE_N": lambda args: min(triton.next_power_of_2(args["N"]), 4096)}
)
@triton.jit
def _normuon_norm_stats_kernel(
    U_ptr,
    V_ptr,
    Row_ss_ptr,
    denom_ptr,
    muon_beta2,
    eps,
    M,
    N,
    u_stride_m,
    u_stride_n,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Updates V in-place and stores row-wise sum of U^2 and denom (V + eps)

    For each row i:
        Compute row_ss = sum_j U[i,j]^2
        Update V[i] = muon_beta2 * V[i] + (1 - muon_beta2) * row_ss
        Compute denom = V[i] + eps
        Store row_ss and denom for norm_U and norm_U_new computation

    Stats all computed in fp32 for higher precision for normalization;
    fp32 writes have minimal overhead for 1B model, even at fp4 (<1%)
    """
    pid = tl.program_id(0)

    row_sum_sq = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for col_start in range(0, N, BLOCK_SIZE_N):
        offs = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = offs < N
        u = tl.load(
            U_ptr + pid * u_stride_m + offs * u_stride_n, mask=mask, other=0.0
        ).to(tl.float32)
        row_sum_sq += u * u

    row_ss = tl.sum(row_sum_sq)

    v = tl.load(V_ptr + pid, mask=pid < M, other=0.0).to(tl.float32)
    v = muon_beta2 * v + (1 - muon_beta2) * (row_ss / N)
    tl.store(V_ptr + pid, v, mask=pid < M)

    denom = tl.sqrt(v) + eps
    tl.store(Row_ss_ptr + pid, row_ss, mask=pid < M)
    tl.store(denom_ptr + pid, denom, mask=pid < M)


def normuon_normalization_triton(
    U: Tensor,
    V: Tensor,
    muon_beta2: Tensor,
):
    """Triton-fused normalization step for NorMuon normalization

    Computes row-wise sum of squares and denom in kernel for normalization
    Applies Adam-style normalization and norm rescaling in one step for one U write.

    Args:
        U: update tensor (*leading, M, N)
        V: second-moment tensor (*leading, M)
        muon_beta2: second-moment decay factor
        eps: small constant for numerical stability in normalization
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for normuon_normalization_triton")
    
    if not U.is_contiguous() or not V.is_contiguous():
        raise ValueError("normuon_normalization_triton requires contiguous U and V")

    # Treats U as a batch of M row vectors, irrespective of leading dims
    M, N = U.numel() // U.shape[-1], U.shape[-1]
    muon_beta2 = muon_beta2.item() if isinstance(muon_beta2, Tensor) else muon_beta2

    row_ss = torch.empty(M, dtype=torch.float32, device=U.device)
    denom = torch.empty(M, dtype=torch.float32, device=U.device)

    grid = (M,)
    _normuon_norm_stats_kernel[grid](
        U,
        V,
        row_ss,
        denom,
        muon_beta2=muon_beta2,
        eps=1e-8,
        M=M,
        N=N,
        u_stride_m=U.stride(-2),
        u_stride_n=U.stride(-1),
    )

    # Reshape row_ss and denom to match U's leading dims
    row_ss_shaped = row_ss.reshape(*U.shape[:-1])
    denom_shaped = denom.reshape(*U.shape[:-1])

    norm_U = torch.sqrt(row_ss_shaped.sum(dim=-1, keepdim=True))
    norm_U_new = torch.sqrt((row_ss_shaped / (denom_shaped * denom_shaped)).sum(dim=-1, keepdim=True))

    # Adam-style row-normalization and norm rescaling combined for one U write
    multiplier = (norm_U / norm_U_new) / denom_shaped  # [M]
    U = (U * multiplier.unsqueeze(-1)).to(U.dtype)

    return U, V
