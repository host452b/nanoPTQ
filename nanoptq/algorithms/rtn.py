# nanoptq/algorithms/rtn.py
"""
RTN: Round-to-Nearest quantization — the simplest possible PTQ baseline.
RTN：四舍五入量化 —— 最简单的 PTQ 基线。

No calibration. No optimization. Just round the weights.
没有校准数据，没有优化，直接四舍五入。

This is your baseline. Every fancier algorithm (AWQ, GPTQ) is measured
against how much better it does than RTN.
这是你的基线。每个更复杂的算法（AWQ、GPTQ）都在与 RTN 比较它好了多少。

Formula:
  scale = max|W_group| / (2^(b-1) - 1)
  W_q   = round(W / scale)
  W_hat = W_q * scale   ← dequantized for matmul
"""
import torch
import torch.nn as nn
from nanoptq.core.group_quant import group_quantize
from nanoptq.model.quant_linear import QuantLinear
from nanoptq.model.hf_loader import get_linear_layers, _set_module_by_name


def quantize_linear_rtn(
    linear: nn.Linear,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
) -> QuantLinear:
    """
    Apply RTN to a single nn.Linear and return a filled QuantLinear.
    One function, one idea.
    """
    ql = QuantLinear.from_linear(linear, bits=bits, group_size=group_size, symmetric=symmetric)
    W = linear.weight.detach().float()

    if symmetric:
        q, scales = group_quantize(W, group_size=group_size, bits=bits, symmetric=True)
        ql.weight_q = q.to(torch.int8)
        ql.scales = scales.to(torch.float16)
    else:
        q, scales, zps = group_quantize(W, group_size=group_size, bits=bits, symmetric=False)
        ql.weight_q = q.to(torch.int8)
        ql.scales = scales.to(torch.float16)
        ql.zero_points = zps.to(torch.int8)

    return ql


def quantize_model_rtn(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    skip_modules: list[str] | None = None,
) -> None:
    """
    Apply RTN to every nn.Linear in model in-place.
    Fastest quantization possible: no data needed, just one pass over the weights.
    """
    skip_modules = skip_modules or []
    for name, linear in get_linear_layers(model):
        if any(skip in name for skip in skip_modules):
            continue
        ql = quantize_linear_rtn(linear, bits=bits, group_size=group_size, symmetric=symmetric)
        _set_module_by_name(model, name, ql)
