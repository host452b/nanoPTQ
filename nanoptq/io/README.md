# nanoptq/io/

Save and load quantized model checkpoints using the safetensors format.

## File

### `safetensors_io.py`

Two functions:

**`save_quantized_model(model, output_dir, bits, group_size, method)`**

Walks every `QuantLinear` in the model and serializes:
- `{name}.weight_q` — int8 integer codes
- `{name}.scales` — fp16 scale factors
- `{name}.zero_points` — uint8 zero-points (only if asymmetric and non-zero)
- `{name}.input_channel_scales` — fp16 (AWQ only; the per-channel `s` vector)

Also writes `quant_config.json` with metadata: `bits`, `group_size`, `method`, `skipped_modules`.

**`load_quantized_model(model, checkpoint_dir)`**

Reads the `.safetensors` file and the `quant_config.json`, then:
1. Reads `quant_config.json` to get `bits`, `group_size`
2. Replaces each `nn.Linear` (that has saved tensors) with a new `QuantLinear`
3. Loads `weight_q`, `scales`, `zero_points`, and `input_channel_scales` into the layer

## Why Safetensors?

- Safe: no code execution on load (unlike pickle)
- Fast: memory-mapped loading for large files
- Compatible: loadable by vLLM, HF Transformers, llama.cpp

## File Layout in Output Directory

```
output_dir/
├── quantized_model.safetensors   # all quantized layer tensors
├── quant_config.json             # bits, group_size, method, etc.
├── config.json                   # original HF model config (from tokenizer.save_pretrained)
├── tokenizer.json                # tokenizer
└── tokenizer_config.json         # tokenizer config
```

## AWQ Note

`input_channel_scales` must survive save/load — without it, the AWQ scaling trick
`(W*s) @ (x/s) = W@x` breaks silently (outputs wrong, no error).
The loader explicitly checks for and restores this buffer.
