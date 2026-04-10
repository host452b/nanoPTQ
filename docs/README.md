# docs/

Project documentation beyond the root README.

## Files

### `flow.md` / `flow.zh.md`

Detailed flowcharts of two quantization processes:
1. **Offline quantization** — what happens when you run `nanoptq quantize` (step-by-step, with math)
2. **Runtime inference** — what happens in `QuantLinear.forward()` at serving time

Read these to understand the full end-to-end lifecycle of a quantized model.

### `Glossary.md` / `Glossary.zh.md`

Every quantization term explained with an analogy first, then a formal definition.
Covers: quantization, scale, zero-point, group-wise, perplexity, RTN, AWQ, GPTQ,
calibration data, int4/int8/fp8/bf16/nvfp4, Hessian, safetensors, outlier channels.

Start here if any term in the codebase is unfamiliar.

## Subdirectories

### `superpowers/`

Internal planning and development tooling (AI agent skill files and plan documents).
Not relevant to learning quantization.
