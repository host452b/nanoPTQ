import torch
import torch.nn as nn
import pytest
from nanoptq.model.quant_linear import QuantLinear

def make_linear(in_f=64, out_f=32):
    l = nn.Linear(in_f, out_f, bias=True)
    nn.init.normal_(l.weight)
    return l

def test_from_linear_preserves_bias():
    linear = make_linear()
    ql = QuantLinear.from_linear(linear, bits=8, group_size=64)
    assert ql.bias is not None
    assert torch.allclose(ql.bias, linear.bias.detach())

def test_from_linear_no_bias():
    linear = nn.Linear(64, 32, bias=False)
    ql = QuantLinear.from_linear(linear, bits=8, group_size=64)
    assert ql.bias is None

def test_forward_output_shape():
    linear = make_linear(in_f=64, out_f=32)
    ql = QuantLinear.from_linear(linear, bits=8, group_size=64)
    from nanoptq.core.group_quant import group_quantize
    q, scales = group_quantize(linear.weight.detach(), group_size=64, bits=8)
    ql.weight_q = q.to(torch.int8)
    ql.scales = scales.to(torch.float16)
    x = torch.randn(4, 64)
    out = ql(x)
    assert out.shape == (4, 32)

def test_forward_dtype_passthrough():
    linear = make_linear(in_f=64, out_f=32)
    ql = QuantLinear.from_linear(linear, bits=8, group_size=64)
    from nanoptq.core.group_quant import group_quantize
    q, scales = group_quantize(linear.weight.detach(), group_size=64, bits=8)
    ql.weight_q = q.to(torch.int8)
    ql.scales = scales.to(torch.float16)
    x = torch.randn(2, 64, dtype=torch.float16)
    out = ql(x)
    assert out.dtype == torch.float16

def test_dequantize_shape():
    linear = make_linear(in_f=128, out_f=64)
    ql = QuantLinear.from_linear(linear, bits=4, group_size=128)
    from nanoptq.core.group_quant import group_quantize
    q, scales = group_quantize(linear.weight.detach(), group_size=128, bits=4)
    ql.weight_q = q.to(torch.int8)
    ql.scales = scales.to(torch.float16)
    W = ql.dequantize()
    assert W.shape == (64, 128)

def test_config_roundtrip():
    linear = make_linear()
    ql = QuantLinear.from_linear(linear, bits=4, group_size=64)
    assert ql.bits == 4
    assert ql.group_size == 64
    assert ql.in_features == 64
    assert ql.out_features == 32
