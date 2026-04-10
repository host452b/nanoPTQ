import torch
import pytest
from nanoptq.core.quant_primitives import (
    quantize_symmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    dequantize_asymmetric,
    fake_quantize,
)

def test_symmetric_int8_range():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q, scale = quantize_symmetric(x, bits=8)
    assert q.dtype == torch.int8
    assert q.abs().max() <= 127

def test_symmetric_roundtrip_error_small():
    torch.manual_seed(0)
    x = torch.randn(256)
    q, scale = quantize_symmetric(x, bits=8)
    x_hat = dequantize_symmetric(q, scale)
    assert (x - x_hat).abs().mean() < 0.05

def test_asymmetric_int4_range():
    x = torch.tensor([0.1, 0.5, 1.2, 3.0, -0.3])
    q, scale, zp = quantize_asymmetric(x, bits=4)
    assert q.dtype == torch.uint8
    assert q.max() <= 15
    assert q.min() >= 0

def test_asymmetric_roundtrip():
    torch.manual_seed(42)
    x = torch.randn(128).abs()
    q, scale, zp = quantize_asymmetric(x, bits=8)
    x_hat = dequantize_asymmetric(q, scale, zp)
    assert (x - x_hat).abs().mean() < 0.05

def test_fake_quantize_preserves_dtype():
    x = torch.randn(64, dtype=torch.float16)
    x_hat = fake_quantize(x, bits=8, symmetric=True)
    assert x_hat.dtype == torch.float16

def test_fake_quantize_introduces_error():
    torch.manual_seed(0)
    x = torch.randn(1024)
    x_hat_8 = fake_quantize(x, bits=8)
    x_hat_4 = fake_quantize(x, bits=4)
    err_8 = (x - x_hat_8).abs().mean()
    err_4 = (x - x_hat_4).abs().mean()
    assert err_4 > err_8

def test_scale_positive():
    x = torch.randn(64)
    _, scale = quantize_symmetric(x, bits=8)
    assert scale > 0
