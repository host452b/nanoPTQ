# nanoptq/model/

The quantized layer abstraction and HuggingFace model integration.

## Files

### `quant_linear.py`

`QuantLinear` is a drop-in replacement for `nn.Linear`.

**What it stores (on GPU):**
```python
self.weight_q        # int8 tensor, shape [out_features, in_features]
self.scales          # fp16 tensor, shape [out_features, num_groups]
self.zero_points     # uint8 tensor, shape [out_features, num_groups]  (asymmetric only)
self.input_channel_scales  # fp16 tensor, shape [in_features]  (AWQ only)
```

**What happens in `forward(x)`:**
1. AWQ only: `x = x / input_channel_scales`
2. Dequantize: `W_fp16 = weight_q * scales` (broadcast per group)
3. Standard matmul: `output = F.linear(x, W_fp16, bias)`

No custom CUDA kernel. Works on any PyTorch-supported GPU.

**Key methods:**
- `QuantLinear.from_linear(linear, bits, group_size)` — create from an existing `nn.Linear`
- `ql.dequantize()` — reconstruct float weight matrix (used in forward and in GPTQ)

### `hf_loader.py`

Utilities for loading HuggingFace models and swapping layers in-place.

| Function | What it does |
|----------|-------------|
| `load_hf_model(model_id, device)` | `from_pretrained` with fp16, single GPU |
| `get_linear_layers(model)` | Iterator over `(name, nn.Linear)` pairs (skips embedding/norm) |
| `_set_module_by_name(model, name, new_module)` | Replace a named module in-place using `setattr` |

## Design Note

`device_map="auto"` is not used. All models load to a single GPU via `device_map=device`.
This keeps the code simple and avoids silent multi-GPU behavior.
