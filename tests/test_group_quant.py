import torch
import pytest
from nanoptq.core.group_quant import group_quantize, group_dequantize

def test_output_shapes_symmetric():
    W = torch.randn(64, 128)
    group_size = 32
    q, scales = group_quantize(W, group_size=group_size, bits=4, symmetric=True)
    assert q.shape == (64, 128)
    assert scales.shape == (64, 4)  # 128 / 32 = 4 groups

def test_output_shapes_asymmetric():
    W = torch.randn(32, 64)
    q, scales, zps = group_quantize(W, group_size=64, bits=4, symmetric=False)
    assert q.shape == (32, 64)
    assert scales.shape == (32, 1)
    assert zps.shape == (32, 1)

def test_group_dequantize_roundtrip_int8():
    torch.manual_seed(0)
    W = torch.randn(64, 128)
    q, scales = group_quantize(W, group_size=128, bits=8, symmetric=True)
    W_hat = group_dequantize(q, scales, group_size=128, symmetric=True)
    assert W_hat.shape == W.shape
    assert (W - W_hat).abs().mean() < 0.02

def test_group_int4_worse_than_int8():
    torch.manual_seed(1)
    W = torch.randn(64, 128)
    q8, s8 = group_quantize(W, group_size=128, bits=8, symmetric=True)
    q4, s4 = group_quantize(W, group_size=128, bits=4, symmetric=True)
    W8 = group_dequantize(q8, s8, group_size=128, symmetric=True)
    W4 = group_dequantize(q4, s4, group_size=128, symmetric=True)
    assert (W - W4).abs().mean() > (W - W8).abs().mean()

def test_smaller_group_better_precision():
    torch.manual_seed(2)
    W = torch.randn(64, 256)
    q_g256, s_g256 = group_quantize(W, group_size=256, bits=4, symmetric=True)
    q_g32, s_g32 = group_quantize(W, group_size=32, bits=4, symmetric=True)
    W_g256 = group_dequantize(q_g256, s_g256, group_size=256, symmetric=True)
    W_g32 = group_dequantize(q_g32, s_g32, group_size=32, symmetric=True)
    assert (W - W_g32).abs().mean() < (W - W_g256).abs().mean()

def test_invalid_group_size_raises():
    W = torch.randn(64, 100)
    with pytest.raises(AssertionError):
        group_quantize(W, group_size=32, bits=4)
