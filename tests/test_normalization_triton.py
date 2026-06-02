"""Tests for the Triton kernel ``normuon_normalization_triton``.

Two levels of testing:
1. **Function-level**: compare the Triton kernel output directly against the
    reference of ``normuon_normalization_stacked`` on identical inputs. Tested
    with various dtypes and entries should only differ at FP-rounding level.
2. **End-to-end: run NorMuon and NorDion2 optimizers with ``use_triton=True`` vs default
    and verify parameters are close.
"""

import pytest
import torch

from dion.normuon import normuon_normalization_stacked
from dion.dion.normalization_triton import TRITON_AVAILABLE, normuon_normalization_triton

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AND_CUDA = CUDA_AVAILABLE and TRITON_AVAILABLE

torch._dynamo.config.cache_size_limit = 64


def _make_test_data(shape, seed=42, dtype=torch.float32):
    """Create U and V tensors

    Returns:
        U: (*leading, M, N) in u_dtype
        V: (*leading, M) in v_dtype
    """
    torch.manual_seed(seed)
    M, _ = shape[-2], shape[-1]
    leading = shape[:-2]

    U = torch.randn(shape, device=DEVICE, dtype=dtype)
    V = torch.randn((*leading, M, 1), device=DEVICE, dtype=dtype)

    return U, V


# ---------------------------------------------------------------------------
# Function-level tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TRITON_AND_CUDA, reason="CUDA and Triton required")
class TestNorMuonNormalizationKernel:
    """Compare Triton kernel output against stacked implementation"""
    # Test N values to see performance with heuristic-chosen BLOCK_SIZE
    @pytest.mark.parametrize("M, N", [
        (32, 512),
        (32, 1024),
        (32, 1536),
        (32, 4096),
        (32, 13312),
    ])
    @pytest.mark.parametrize("leading", [
        (1,),
        (4,),
        (2, 4),
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    # Test for first-step performance when V is zero
    @pytest.mark.parametrize("v_init", ["zeros", "nonzero"])
    def test_single_tensor(self, M, N, leading, dtype, v_init):
        shape = (*leading, M, N)

        U, V = _make_test_data(shape, dtype=dtype)
        if v_init == "zeros":
            V = torch.zeros_like(V)
        else:
            # V is always positive in practice after first step
            V = V.abs() 

        muon_beta2 = torch.tensor(0.95, device=DEVICE)

        U_ref, V_ref = normuon_normalization_stacked(U, V, muon_beta2)
        U_tri, V_tri = normuon_normalization_triton(U.clone(), V.clone(), muon_beta2)

        # Dtype-aware tolerances: bf16 has coarser rounding (~1 ULP = 2^-7 * value)
        # so the one-vs-two-rounding difference is larger.
        if dtype == torch.bfloat16:
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 1e-6, 1e-5

        # Selected entries: fused vs two-step rounding
        assert torch.allclose(U_ref, U_tri, atol=atol, rtol=rtol), (
            f"U differs beyond tolerance: "
            f"max diff = {(U_ref - U_tri).abs().max().item():.2e}"
        )

        # Overall comparison
        assert torch.allclose(V_ref, V_tri, atol=atol, rtol=rtol), (
            f"V differs beyond tolerance: "
            f"max diff = {(V_ref - V_tri).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# End-to-end optimizer tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TRITON_AND_CUDA, reason="CUDA and Triton required")
class TestNormalizationTritonEndToEnd:
    """Run NorMuon / NorDion2 optimizer with use_triton=True vs default and compare."""
    
    def _make_params(self, shapes, dtype=torch.float32):
        torch.manual_seed(42)
        return [torch.nn.Parameter(torch.randn(s, device=DEVICE, dtype=dtype)) for s in shapes]

    def _run_steps(self, opt_class, params, opt_kwargs, n_steps=3):
        opt = opt_class(params, **opt_kwargs)
        for step in range(n_steps):
            torch.manual_seed(100 + step)
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()
        return opt

    from dion import NorMuon, NorDion2
    @pytest.mark.parametrize("shapes", [
        [(32, 128)],
        [(64, 128)] * 4,
    ])
    @pytest.mark.parametrize("opt_class, extra_kwargs", [
        (NorMuon, {}),
        (NorDion2, {"fraction": 0.5, "triton_post_ortho": True}),
        (NorDion2, {"fraction": 0.25, "triton_post_ortho": False}),
    ])
    @pytest.mark.parametrize("use_triton", [False, True])
    @pytest.mark.parametrize("use_gram_newton_schulz", [False, True])
    @pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32])
    def test_triton_vs_default(self, shapes, opt_class, extra_kwargs, use_triton, use_gram_newton_schulz, param_dtype):
        """Triton normalization should match default up to fused-rounding tolerance."""
        kwargs = dict(
            lr=0.01, use_triton=use_triton, use_gram_newton_schulz=use_gram_newton_schulz,
        )
        kwargs.update(extra_kwargs)

        p_default = self._make_params(shapes, dtype=param_dtype)
        opt_default = self._run_steps(opt_class, p_default, {**kwargs, "triton_normalization": False}, n_steps=3)

        p_triton = self._make_params(shapes, dtype=param_dtype)
        opt_triton = self._run_steps(opt_class, p_triton, {**kwargs, "triton_normalization": True}, n_steps=3)

        # Momentum should be bitwise identical (same pre-ortho path)
        for pd, pt in zip(p_default, p_triton):
            md = opt_default.state[pd]["momentum"]
            mt = opt_triton.state[pt]["momentum"]
            assert torch.equal(md, mt), "Momentum buffers differ"

        if param_dtype == torch.bfloat16:
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 1e-5, 1e-4

        # Parameters and V should all be close within fused rounding tolerance
        for pd, pt in zip(p_default, p_triton):
            assert torch.allclose(pd.data, pt.data, atol=atol, rtol=rtol), (
                f"Parameters differ beyond tolerance: "
                f"max diff = {(pd.data - pt.data).abs().max().item():.2e}"
            )
        for pd, pt in zip(p_default, p_triton):
            vd = opt_default.state[pd]["variance_neuron"]
            vt = opt_triton.state[pt]["variance_neuron"]
            assert torch.allclose(vd, vt, atol=atol, rtol=rtol), (
                f"Variance neuron buffers differ beyond tolerance: "
                f"max diff = {(vd - vt).abs().max().item():.2e}"
            )
