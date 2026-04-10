import torch
import torch.nn as nn
import pytest
from nanoptq.algorithms.rtn import quantize_model_rtn, quantize_linear_rtn
from nanoptq.model.quant_linear import QuantLinear


def make_linear(in_f=128, out_f=64):
    l = nn.Linear(in_f, out_f)
    nn.init.normal_(l.weight)
    return l


def test_quantize_linear_rtn_returns_quant_linear():
    linear = make_linear()
    ql = quantize_linear_rtn(linear, bits=8, group_size=128)
    assert isinstance(ql, QuantLinear)


def test_quantize_linear_rtn_fills_weights():
    linear = make_linear()
    ql = quantize_linear_rtn(linear, bits=8, group_size=128)
    assert ql.weight_q.abs().sum() > 0
    assert ql.scales.abs().sum() > 0


def test_quantize_linear_rtn_int8_output_shape():
    linear = make_linear(128, 64)
    ql = quantize_linear_rtn(linear, bits=8, group_size=128)
    assert ql.weight_q.shape == (64, 128)
    assert ql.scales.shape == (64, 1)  # 128/128 = 1 group


def test_quantize_linear_rtn_int4_output_shape():
    linear = make_linear(128, 64)
    ql = quantize_linear_rtn(linear, bits=4, group_size=64)
    assert ql.weight_q.shape == (64, 128)
    assert ql.scales.shape == (64, 2)  # 128/64 = 2 groups


def test_rtn_forward_close_to_original():
    torch.manual_seed(0)
    linear = make_linear(128, 64)
    ql = quantize_linear_rtn(linear, bits=8, group_size=128)
    x = torch.randn(4, 128)
    with torch.no_grad():
        y_orig = linear(x)
        y_quant = ql(x)
    assert (y_orig - y_quant).abs().mean() < 0.1


def test_quantize_model_rtn_replaces_linears():
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
    quantize_model_rtn(model, bits=8, group_size=64)
    quant_layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    assert len(quant_layers) == 2


def test_quantize_model_rtn_no_linear_remains():
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32))
    quantize_model_rtn(model, bits=4, group_size=64)
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 0


def test_quantize_model_rtn_skip_modules():
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32))
    quantize_model_rtn(model, bits=4, group_size=64, skip_modules=["1"])
    # Module "1" skipped → still nn.Linear
    assert isinstance(model[1], nn.Linear), "skipped module should remain nn.Linear"
    # Module "0" not skipped → replaced
    assert isinstance(model[0], QuantLinear), "non-skipped module should be replaced"
