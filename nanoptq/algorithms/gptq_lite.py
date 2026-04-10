# nanoptq/algorithms/gptq_lite.py
"""
GPTQ-lite: Hessian-based layer-wise weight quantization (simplified).
GPTQ 精简版：基于 Hessian 矩阵的逐层权重量化。

Original paper: Frantar et al., "GPTQ: Accurate Post-Training Quantization
for Generative Pre-trained Transformers" (NeurIPS 2022).

Core insight:
  RTN quantizes columns independently, ignoring that error in column j
  shifts the optimal value for column j+1. GPTQ compensates:

  H = (2/n) X^T X   ← input Hessian (proxy for weight sensitivity)
  For column j (left to right):
    1. Quantize W[:, j]
    2. err = W[:, j] - W_hat[:, j]
    3. W[:, j+1:] -= err ⊗ (H_inv[j, j+1:] / H_inv[j,j])  ← OBQ update

  This is OBQ / GPTQ without column reordering (simpler, close in quality).

Full GPTQ also uses Cholesky decomposition for O(n^2) Hessian inverse update
and optional column reordering for better numerical stability.
This lite version uses direct Cholesky inversion for clarity.
"""
import torch
import torch.nn as nn
from nanoptq.core.group_quant import group_quantize
from nanoptq.model.quant_linear import QuantLinear


def compute_hessian(
    activations: torch.Tensor,   # [n_samples, in_features]
    damping: float = 0.01,
) -> torch.Tensor:
    """
    Compute the input Hessian H = (2/n) * X^T X + damping * I.

    The damping term (ridge) ensures H is invertible even when the calibration
    data is rank-deficient (common when n < in_features).

    Returns H of shape [in_features, in_features].
    """
    n, d = activations.shape
    X = activations.float()
    H = (2.0 / n) * (X.T @ X)
    H += damping * torch.eye(d, dtype=H.dtype, device=H.device)
    return H


def quantize_linear_gptq(
    linear: nn.Linear,
    calibration_acts: torch.Tensor,   # [n_samples, in_features]
    bits: int = 4,
    group_size: int = 128,
    damping: float = 0.01,
    symmetric: bool = True,
) -> QuantLinear:
    """
    GPTQ-lite: quantize one Linear layer using Hessian-guided error compensation.

    Algorithm:
      For each column j of W (each input channel):
        1. Quantize W[:, j] → q_col  (per-column scale, one scalar per output channel)
        2. Compute quant error: err = W[:, j] - q_col
        3. Propagate error to remaining columns via Hessian inverse row:
           W[:, j+1:] -= outer(err, H_inv[j, j+1:] / H_inv[j, j])

    After the column-wise update, re-apply group-wise quantization for consistent
    QuantLinear format (scales per group rather than per column).
    """
    W = linear.weight.detach().float().clone()  # [out_features, in_features]
    out_features, in_features = W.shape

    H = compute_hessian(calibration_acts.float(), damping=damping)

    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        H_inv = torch.linalg.pinv(H)

    # Column-wise quantization with Hessian compensation
    qmax = 2 ** (bits - 1) - 1
    for j in range(in_features):
        col = W[:, j]  # [out_features]

        # Per-column scale (scalar)
        scale = col.abs().max() / qmax
        scale = scale.clamp(min=1e-8)

        col_q = (col / scale).round().clamp(-qmax, qmax)
        col_hat = col_q * scale  # dequantized

        # Hessian compensation: update all remaining columns
        err = col - col_hat  # quantization error for this column
        if j < in_features - 1 and H_inv[j, j].abs() > 1e-8:
            # W[:, j+1:] -= err[:, None] * (H_inv[j, j+1:] / H_inv[j, j])[None, :]
            h_factor = H_inv[j, j + 1:] / H_inv[j, j]  # [remaining]
            W[:, j + 1:] -= err.unsqueeze(1) * h_factor.unsqueeze(0)

    # After compensation, W now contains Hessian-adjusted float values.
    # Apply group-wise quantization for the standard QuantLinear format.
    ql = QuantLinear.from_linear(linear, bits=bits, group_size=group_size, symmetric=symmetric)

    if symmetric:
        q, scales = group_quantize(W, group_size=group_size, bits=bits, symmetric=True)
        ql.weight_q = q.to(torch.int8)
        ql.scales = scales.to(torch.float16)
    else:
        q, scales, zps = group_quantize(W, group_size=group_size, bits=bits, symmetric=False)
        ql.weight_q = q.to(torch.int8)
        ql.scales = scales.to(torch.float16)
        ql.zero_points = zps.to(torch.uint8)

    return ql
