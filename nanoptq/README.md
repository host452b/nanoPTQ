# nanoptq/

The core library. Everything runnable lives here.

## Structure

```
nanoptq/
├── core/          # Quantization math primitives
├── model/         # QuantLinear layer + HF model loading
├── algorithms/    # RTN, AWQ, GPTQ implementations
├── io/            # Save/load quantized checkpoints
├── eval/          # Perplexity and latency benchmarks
├── data/          # Calibration/eval dataset loader
└── cli.py         # Entry point: nanoptq quantize / eval / compare
```

## Reading Order

If you are learning, read in this order:

| Step | File | What you learn |
|------|------|----------------|
| 1 | `core/quant_primitives.py` | Symmetric, asymmetric, fake_quant math |
| 2 | `core/group_quant.py` | Why group-wise dramatically improves int4 |
| 3 | `model/quant_linear.py` | Unified quantized layer; dequant-on-the-fly |
| 4 | `algorithms/rtn.py` | Baseline: round and done |
| 5 | `algorithms/awq_lite.py` | Activation-aware improvement |
| 6 | `algorithms/gptq_lite.py` | Hessian-based compensation |

## Entry Point

`cli.py` ties everything together. Each subcommand imports only what it needs.

```
nanoptq quantize  →  algorithms/ + io/
nanoptq eval      →  io/ + eval/
nanoptq compare   →  algorithms/ + eval/
```
