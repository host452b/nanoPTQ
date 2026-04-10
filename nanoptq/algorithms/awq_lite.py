# nanoptq/algorithms/awq_lite.py
"""
AWQ-lite: Activation-aware Weight Quantization (simplified).
AWQ 精简版：激活感知权重量化。

Original paper: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression
and Acceleration" (2023).

Core insight:
  Not all weights matter equally. Weights connected to large activations matter more —
  quantizing them coarsely causes larger output errors.

Math:
  Original:  y = W x
  AWQ:       y = (W * s) * (x / s)     ← mathematically equivalent!
  Where:     s[i] = act_scale[i]^alpha  ← per input-channel scale

  Quantize (W * s) — important channels are amplified → get more quantization levels → lower error.

alpha=0:   no scaling (equivalent to RTN)
alpha=1:   full scaling by activation magnitude
alpha=0.5: geometric mean (Lin et al. default, what we use here)

The full AWQ paper uses a grid search over alpha and also modifies the previous layer
to absorb 1/s. This lite version:
  - Uses alpha=0.5 (paper default, no grid search needed for teaching)
  - Stores input_channel_scales so callers can optionally absorb 1/s into the prev layer
"""
import torch
import torch.nn as nn
from nanoptq.core.group_quant import group_quantize
from nanoptq.model.quant_linear import QuantLinear


def find_channel_scales(
    activations: torch.Tensor,   # [n_samples, in_features]
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute per-channel scaling factors from calibration activations.
    Returns scales of shape [in_features].

    act_scale[i] = mean(|act[:, i]|)          ← how "active" is channel i?
    channel_scale[i] = act_scale[i]^alpha     ← raise to alpha (0=RTN, 1=full AWQ)
    """
    act_scale = activations.float().abs().mean(dim=0)  # [in_features]
    act_scale = act_scale.clamp(min=1e-8)
    return act_scale.pow(alpha)


def quantize_linear_awq(
    linear: nn.Linear,
    calibration_acts: torch.Tensor,   # [n_samples, in_features]
    bits: int = 4,
    group_size: int = 128,
    alpha: float = 0.5,
    symmetric: bool = True,
) -> QuantLinear:
    """
    AWQ-lite: scale weights by channel importance, quantize, store in QuantLinear.

    Steps:
      1. Find channel_scale[i] = mean(|act[:, i]|)^alpha  from calibration data
      2. Scale weights up: W_scaled[:, i] = W[:, i] * channel_scale[i]
         (important channels are amplified → get more quantization levels)
      3. Quantize W_scaled with RTN group-wise
      4. Return QuantLinear — its dequantize() gives W_scaled_hat (≈ W_scaled)

    Note: to fully reconstruct y ≈ Wx you need (x / channel_scale) as input.
    We store channel_scales in `input_channel_scales` buffer for the caller's use.
    """
    channel_scales = find_channel_scales(calibration_acts.float(), alpha=alpha)  # [in_features]

    W = linear.weight.detach().float()  # [out_features, in_features]

    # Scale up the weight columns of important channels
    # W_scaled[:, i] = W[:, i] * channel_scale[i]
    W_scaled = W * channel_scales.unsqueeze(0)  # broadcast: [1, in] * [out, in]

    ql = QuantLinear.from_linear(linear, bits=bits, group_size=group_size, symmetric=symmetric)

    if symmetric:
        q, scales = group_quantize(W_scaled, group_size=group_size, bits=bits, symmetric=True)
        ql.weight_q = q.to(torch.int8)
        ql.scales = scales.to(torch.float16)
    else:
        q, scales, zps = group_quantize(W_scaled, group_size=group_size, bits=bits, symmetric=False)
        ql.weight_q = q.to(torch.int8)
        ql.scales = scales.to(torch.float16)
        ql.zero_points = zps.to(torch.int8)

    # Store so callers can absorb 1/channel_scales into the previous layer (optional)
    ql.register_buffer("input_channel_scales", channel_scales.to(torch.float16))

    return ql


def quantize_model_awq(
    model: nn.Module,
    calibration_data: dict[str, torch.Tensor],  # {layer_name: activations [n_samples, in_features]}
    bits: int = 4,
    group_size: int = 128,
    alpha: float = 0.5,
    skip_modules: list[str] | None = None,
) -> None:
    """
    Apply AWQ-lite to all Linear layers that have calibration data.
    Layers without calibration data fall back to RTN.
    """
    from nanoptq.model.hf_loader import get_linear_layers, _set_module_by_name
    from nanoptq.algorithms.rtn import quantize_linear_rtn

    skip_modules = skip_modules or []
    for name, linear in get_linear_layers(model):
        if any(skip in name for skip in skip_modules):
            continue
        if name in calibration_data:
            ql = quantize_linear_awq(linear, calibration_data[name],
                                     bits=bits, group_size=group_size, alpha=alpha)
        else:
            ql = quantize_linear_rtn(linear, bits=bits, group_size=group_size)
        _set_module_by_name(model, name, ql)
