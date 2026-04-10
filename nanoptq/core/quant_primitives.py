# nanoptq/core/quant_primitives.py
"""
Fundamental quantization math.
读这里理解量化的数学本质。

Symmetric:  Q = round(x / S),        S = max|x| / (2^(b-1) - 1)
Asymmetric: Q = round(x / S + Z),    S = (max-min) / (2^b - 1)
"""
import torch


def quantize_symmetric(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-tensor quantization.
    Maps float tensor to signed integers in [-2^(b-1)+1, 2^(b-1)-1].
    """
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    q = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct float from symmetric quantized tensor."""
    return q.float() * scale


def quantize_asymmetric(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric per-tensor quantization.
    Maps float tensor to unsigned integers in [0, 2^b - 1].
    Zero-point shifts the range to cover the actual distribution.
    """
    qmin, qmax = 0, 2 ** bits - 1
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin) / (qmax - qmin)
    scale = scale.clamp(min=1e-8)
    zero_point = (qmin - xmin / scale).round().clamp(qmin, qmax).to(torch.uint8)
    q = ((x / scale) + zero_point.float()).round().clamp(qmin, qmax).to(torch.uint8)
    return q, scale, zero_point


def dequantize_asymmetric(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Reconstruct float from asymmetric quantized tensor."""
    return (q.float() - zero_point.float()) * scale


def fake_quantize(x: torch.Tensor, bits: int, symmetric: bool = True) -> torch.Tensor:
    """
    Quantize then immediately dequantize — stays in float but simulates quant error.
    "伪量化": 模拟量化误差，但不改变数据类型，便于研究精度损失。
    This is how all PTQ research starts: measure the damage before optimizing.
    """
    orig_dtype = x.dtype
    x_f = x.float()
    if symmetric:
        q, scale = quantize_symmetric(x_f, bits)
        x_hat = dequantize_symmetric(q, scale)
    else:
        q, scale, zp = quantize_asymmetric(x_f, bits)
        x_hat = dequantize_asymmetric(q, scale, zp)
    return x_hat.to(orig_dtype)
