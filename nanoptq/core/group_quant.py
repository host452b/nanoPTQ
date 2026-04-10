# nanoptq/core/group_quant.py
"""
Group-wise quantization — the core trick that makes 4-bit LLMs work.
分组量化：让 4-bit LLM 成为可能的关键技巧。

Why groups?
  Per-tensor: one scale for the whole matrix → huge error for int4
  Per-channel: one scale per row → better, but still misses intra-row variation
  Per-group:   one scale per G elements → fine-grained, standard in AWQ/GPTQ/torchao

Industrial standard: group_size = 128 (GPTQ, AWQ, torchao default)
"""
import torch


def group_quantize(
    W: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool = True,
) -> tuple:
    """
    Quantize weight matrix W [out_features, in_features] group-wise.

    Returns:
      symmetric=True:  (q, scales)               shapes: [O, I], [O, I//G]
      symmetric=False: (q, scales, zero_points)   same shapes
    """
    out_features, in_features = W.shape
    assert in_features % group_size == 0, (
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    )

    num_groups = in_features // group_size
    Wg = W.float().reshape(out_features, num_groups, group_size)

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scales = Wg.abs().amax(dim=-1, keepdim=True) / qmax
        scales = scales.clamp(min=1e-8)
        q = (Wg / scales).round().clamp(-qmax, qmax)
        return q.reshape(out_features, in_features), scales.squeeze(-1)
    else:
        qmax = 2 ** bits - 1
        wmin = Wg.amin(dim=-1, keepdim=True)
        wmax = Wg.amax(dim=-1, keepdim=True)
        scales = (wmax - wmin) / qmax
        scales = scales.clamp(min=1e-8)
        zero_points = (-(wmin / scales)).round().clamp(0, qmax)
        q = ((Wg / scales) + zero_points).round().clamp(0, qmax)
        return q.reshape(out_features, in_features), scales.squeeze(-1), zero_points.squeeze(-1)


def group_dequantize(
    q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    symmetric: bool = True,
    zero_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reconstruct float weights from group-quantized representation.
    scales shape: [out_features, num_groups]
    """
    out_features, in_features = q.shape
    num_groups = in_features // group_size

    Wg = q.float().reshape(out_features, num_groups, group_size)
    s = scales.float().unsqueeze(-1)

    if symmetric:
        W = Wg * s
    else:
        assert zero_points is not None
        zp = zero_points.float().unsqueeze(-1)
        W = (Wg - zp) * s

    return W.reshape(out_features, in_features)
