# nanoptq/core/

Quantization math primitives. These two files are the mathematical foundation.
Read these first before anything else.

## Files

### `quant_primitives.py`

The four functions every quantization library needs:

| Function | What it does |
|----------|-------------|
| `compute_scale_symmetric(W, bits)` | `S = max|W| / (2^(b-1)-1)` |
| `compute_scale_zero_asymmetric(W, bits)` | `S = (max-min)/(2^b-1)`, `Z = -round(min/S)` |
| `quantize_tensor(W, scale, zero_point, bits)` | `Q = clip(round(W/S + Z), qmin, qmax)` |
| `dequantize_tensor(Q, scale, zero_point)` | `W_approx = S × (Q - Z)` |
| `fake_quantize(W, bits, symmetric)` | Round-trip: quantize then dequantize in one call |

**Start here.** Five functions, ~50 lines of code, pure PyTorch. No hidden state.

### `group_quant.py`

Applies the primitives above across groups of 128 weights.

| Function | What it does |
|----------|-------------|
| `group_quantize(W, group_size, bits, symmetric)` | Returns `(W_q, scales, zero_points)` |
| `group_dequantize(W_q, scales, zero_points, group_size)` | Reconstructs float weights from stored tensors |

**Why group-wise?** One scale per 128 weights instead of one per layer.
Outlier columns get their own scale — neighboring columns aren't polluted.

## Key Numbers

```python
int4 symmetric:  values -8 to +7  (15 quantization levels)
int4 asymmetric: values  0 to 15  (16 quantization levels)
int8 symmetric:  values -127 to +127
```

## No Side Effects

These are pure functions. No model state, no hooks, no buffers. Safe to call on any tensor.
