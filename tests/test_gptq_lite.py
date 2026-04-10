import torch
import torch.nn as nn
import pytest
from nanoptq.algorithms.gptq_lite import compute_hessian, quantize_linear_gptq
from nanoptq.model.quant_linear import QuantLinear


def make_linear(in_f=64, out_f=32):
    l = nn.Linear(in_f, out_f, bias=False)
    nn.init.normal_(l.weight)
    return l


def make_calibration_acts(in_f=64, n=32):
    return torch.randn(n, in_f)


def test_compute_hessian_shape():
    acts = make_calibration_acts(64, 32)
    H = compute_hessian(acts)
    assert H.shape == (64, 64)


def test_compute_hessian_symmetric():
    acts = make_calibration_acts(64, 32)
    H = compute_hessian(acts)
    assert torch.allclose(H, H.T, atol=1e-5)


def test_compute_hessian_psd():
    """Hessian should be positive semi-definite."""
    acts = make_calibration_acts(64, 32)
    H = compute_hessian(acts)
    eigvals = torch.linalg.eigvalsh(H)
    assert (eigvals >= -1e-4).all()


def test_quantize_linear_gptq_returns_quant_linear():
    linear = make_linear()
    acts = make_calibration_acts()
    ql = quantize_linear_gptq(linear, acts, bits=4, group_size=64)
    assert isinstance(ql, QuantLinear)


def test_gptq_fills_buffers():
    linear = make_linear()
    acts = make_calibration_acts()
    ql = quantize_linear_gptq(linear, acts, bits=4, group_size=64)
    assert ql.weight_q.abs().sum() > 0
    assert ql.scales.abs().sum() > 0


def test_gptq_output_shape():
    linear = make_linear(in_f=64, out_f=32)
    acts = make_calibration_acts(in_f=64)
    ql = quantize_linear_gptq(linear, acts, bits=4, group_size=64)
    assert ql.weight_q.shape == (32, 64)
    assert ql.scales.shape == (32, 1)


def test_gptq_not_worse_than_rtn():
    """GPTQ should have comparable or lower reconstruction error vs RTN."""
    torch.manual_seed(0)
    linear = make_linear(in_f=64, out_f=32)
    acts = make_calibration_acts(in_f=64, n=128)

    from nanoptq.algorithms.rtn import quantize_linear_rtn
    ql_rtn = quantize_linear_rtn(linear, bits=4, group_size=64)
    ql_gptq = quantize_linear_gptq(linear, acts, bits=4, group_size=64)

    x = make_calibration_acts(in_f=64, n=16)
    with torch.no_grad():
        err_rtn = (linear(x) - ql_rtn(x)).pow(2).mean()
        err_gptq = (linear(x) - ql_gptq(x)).pow(2).mean()

    # GPTQ should not be catastrophically worse (2x tolerance)
    assert err_gptq < err_rtn * 2.0
