# nanoptq/algorithms/

Three quantization algorithms. Each lives in one file. Each file is self-contained.

## The Three Algorithms

| File | Algorithm | Calibration | Core idea |
|------|-----------|-------------|-----------|
| `rtn.py` | RTN | None | Round every weight to the nearest integer code |
| `awq_lite.py` | AWQ-lite | ~128 samples | Scale up weights of high-activation channels before quantizing |
| `gptq_lite.py` | GPTQ-lite | ~128 samples | Use Hessian to compensate downstream columns after each quantization step |

## The Mathematical Essence

```python
# RTN — rtn.py
W_q = round(W / scale) * scale

# AWQ — awq_lite.py
s   = mean(|activations|, dim=0) ** alpha   # channel importance
W_q = quantize(W * s)                       # scale up important channels
out = W_q @ (x / s)                         # divide back at inference

# GPTQ — gptq_lite.py
H     = X.T @ X                             # input Hessian
H_inv = cholesky_inverse(cholesky(H))
for j in range(in_features):
    err          = W[:,j] - quantize(W[:,j])
    W[:,j+1:] -= err * H_inv[j, j+1:] / H_inv[j,j]   # compensate future columns
```

## Public API

Each file exports a `quantize_linear_*` function and a `quantize_model_*` wrapper:

```python
# RTN
from nanoptq.algorithms.rtn import quantize_linear_rtn, quantize_model_rtn

# AWQ
from nanoptq.algorithms.awq_lite import quantize_linear_awq, quantize_model_awq

# GPTQ
from nanoptq.algorithms.gptq_lite import quantize_linear_gptq
```

## Expected Results (Qwen2-0.5B, int4 group_size=128)

| Method | PPL vs FP16 baseline | Notes |
|--------|---------------------|-------|
| RTN | +2–4 PPL | No calibration needed; fastest |
| AWQ | +0.5–2 PPL | Better on models with activation outliers |
| GPTQ | +0.5–2 PPL | Similar to AWQ; slower due to Hessian computation |

## What "lite" Means

Both `awq_lite` and `gptq_lite` implement the core mathematical idea without production optimizations:
- No blocked GPTQ (no 64-column Cholesky dampening blocks)
- No grid search for AWQ alpha
- No mixed-precision per-layer bit assignment
- No fused CUDA kernels

The math is faithful. The engineering shortcuts are left to production frameworks (AutoAWQ, GPTQModel).
