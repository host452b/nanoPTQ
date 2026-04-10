# nanoPTQ

> One file, one idea. One script, one experiment. One metric table, one truth.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[中文文档 README.zh.md](README.zh.md)**

---

## Philosophy

Modern quantization frameworks (vLLM, TensorRT-LLM, AutoAWQ) are extraordinary engineering —
CUDA kernel fusion, distributed sharding, multi-backend compatibility, hardware-specific precision
formats. That engineering is production-critical, but it buries the math.

**The 80/20 rule of quantization:** A small number of techniques — RTN, AWQ, GPTQ — cover the vast
majority of real-world weight quantization deployments. Everything else (QAT, sparsity, mixed-precision
search, distillation) exists but rarely moves the needle in practice. Learn the critical few, skip
the rest. `nanoPTQ` teaches exactly those 3 algorithms, the same way `nanoGPT` teaches the Transformer
by stripping everything non-essential. Each algorithm lives in **one file**. Every formula maps to one line of code.

**Clean demos of only:**
- `nn.Linear` layers only — the quantization target in every major LLM framework
- `int4 / int8` weight quantization — covers W4A16, W8A16 industrial workloads
- Group-wise quantization (group_size=128) — the universal precision/size tradeoff
- Safetensors I/O — real artifacts you can load into vLLM or HF Transformers
- Perplexity + tokens/s — the two metrics that actually matter
- Bundled calibration + eval data — one-stop eval, no internet needed after setup

**This project does not include:**
- QAT, pruning, distillation, sparsity
- CUDA/Triton kernels — dequant on the fly, backend handles matmul
- Multi-GPU, FSDP, pipeline parallelism
- VLMs, encoder-decoders, MoE routing

---

## Key Concepts

Three numbers explain 90% of the code:

| Concept | Formula | Intuition |
|---|---|---|
| Symmetric quant | `S = max\|W\| / (2^(b-1)-1)`, `Q = round(W/S)` | Scale by max absolute value, signed integers |
| Asymmetric quant | `S = (max-min)/(2^b-1)`, `Z = -round(min/S)` | Zero-point shift, unsigned integers, covers skewed distributions |
| Group-wise | Apply scale per 128-weight block, not per-tensor | One scale per 128 weights — precision up, storage cost tiny |

**Why int4 group_size=128?**

```
Bits per weight:
  fp16         = 16 bits
  int8          =  8 bits   (2× compression)
  int4 g=128   ≈  4.25 bits (nearly 4× compression, with scale overhead)

group_size=128 is the industry consensus (AWQ, GPTQ, torchao all default to it).
```

---

## Algorithms

| Algorithm | Core Idea | Calibration | Industrial Use |
|-----------|-----------|-------------|----------------|
| **RTN** | Round-to-Nearest. No data needed. Your baseline. | None | bitsandbytes baseline |
| **AWQ-lite** | Protect outlier channels by scaling weights up before quant. | ~128 samples | AutoAWQ |
| **GPTQ-lite** | Use activation Hessian to compensate downstream columns after each quant step. | ~128 samples | GPTQModel |

**The mathematical essence:**

```python
# RTN (nanoptq/algorithms/rtn.py)
W_q = round(W / scale) * scale            # that's it

# AWQ (nanoptq/algorithms/awq_lite.py)
s   = mean(|activations|, dim=0) ** alpha  # channel importance
W_q = quantize(W * s)                      # scale up important channels
out = W_q @ (x / s)                        # divide back at runtime

# GPTQ (nanoptq/algorithms/gptq_lite.py)
H     = X.T @ X                            # input Hessian
H_inv = cholesky_inverse(cholesky(H))
for j in range(in_features):
    err          = W[:,j] - quantize(W[:,j])
    W[:,j+1:] -= err ⊗ H_inv[j, j+1:] / H_inv[j,j]  # compensate
```

---

## Expected Results

On Qwen2-0.5B, int4 group_size=128 (your numbers will vary slightly):

| Method | PPL (wikitext-2) | ΔPPL | Notes |
|--------|-----------------|------|-------|
| fp16 baseline | ~14.5 | — | reference |
| RTN int4 | ~16–18 | +2–4 | no calibration needed |
| AWQ int4 | ~15–16 | +0.5–2 | better outlier handling |
| GPTQ int4 | ~15–16 | +0.5–2 | similar to AWQ |

> Lower perplexity = better. FP16 is the ceiling. RTN is the floor.

---

## Quickstart

**Prerequisites:**
```
python >= 3.10, pytorch >= 2.0, transformers, safetensors
```

**Install:**
```bash
git clone https://github.com/host452b/nanoPTQ
cd nanoPTQ
pip install -e ".[dev]"

# Prepare bundled eval data (one-time, ~30s, needs internet once)
python scripts/prepare_data.py
```

**Run:**
```bash
# Quantize with RTN (no calibration data needed)
nanoptq quantize --model Qwen/Qwen2-0.5B --method rtn --bits 4 --group-size 128 --output ./qwen-rtn-int4

# Evaluate perplexity (uses bundled wikitext-2, no internet needed)
nanoptq eval --model ./qwen-rtn-int4 --metric ppl

# Compare RTN vs FP16 baseline
nanoptq compare --model Qwen/Qwen2-0.5B --bits 4 --group-size 128

# End-to-end example with latency
python examples/quant_model.py --model Qwen/Qwen2-0.5B --bits 4

# Compare all three methods side by side
python examples/compare_methods.py --model Qwen/Qwen2-0.5B --bits 4
```

---

## Reading Order

If you are learning, read in this order:

| Step | File | What you learn | Time |
|------|------|----------------|------|
| 0 | [docs/Glossary.md](docs/Glossary.md) | Every term with an analogy — read before anything else | 10 min |
| 1 | [nanoptq/core/quant_primitives.py](nanoptq/core/quant_primitives.py) | The math: symmetric, asymmetric, fake_quant | 5 min |
| 2 | [nanoptq/core/group_quant.py](nanoptq/core/group_quant.py) | Why group-wise dramatically improves int4 | 5 min |
| 3 | [nanoptq/model/quant_linear.py](nanoptq/model/quant_linear.py) | Unified layer abstraction; dequant-on-the-fly | 10 min |
| 4 | [nanoptq/algorithms/rtn.py](nanoptq/algorithms/rtn.py) | Baseline: round and done | 5 min |
| 5 | [nanoptq/algorithms/awq_lite.py](nanoptq/algorithms/awq_lite.py) | Activation-aware improvement | 15 min |
| 6 | [nanoptq/algorithms/gptq_lite.py](nanoptq/algorithms/gptq_lite.py) | Hessian-based compensation | 20 min |
| 7 | [examples/compare_methods.py](examples/compare_methods.py) | See them all side by side | — |
| 8 | [docs/flow.md](docs/flow.md) | End-to-end lifecycle: offline quant → runtime inference | 10 min |

---

## Project Structure

| Directory | README | What's inside |
|-----------|--------|---------------|
| [nanoptq/](nanoptq/) | [→](nanoptq/README.md) | Core library |
| [nanoptq/core/](nanoptq/core/) | [→](nanoptq/core/README.md) | Quantization math primitives |
| [nanoptq/model/](nanoptq/model/) | [→](nanoptq/model/README.md) | QuantLinear + HF model loading |
| [nanoptq/algorithms/](nanoptq/algorithms/) | [→](nanoptq/algorithms/README.md) | RTN, AWQ, GPTQ implementations |
| [nanoptq/io/](nanoptq/io/) | [→](nanoptq/io/README.md) | Save/load safetensors checkpoints |
| [nanoptq/eval/](nanoptq/eval/) | [→](nanoptq/eval/README.md) | Perplexity + latency benchmarks |
| [nanoptq/data/](nanoptq/data/) | [→](nanoptq/data/README.md) | Dataset loader |
| [examples/](examples/) | [→](examples/README.md) | Runnable demos |
| [data/](data/) | [→](data/README.md) | Bundled calibration + eval datasets |
| [scripts/](scripts/) | [→](scripts/README.md) | One-time setup scripts |
| [tests/](tests/) | [→](tests/README.md) | Unit + integration tests |
| [docs/](docs/) | [→](docs/README.md) | Glossary, flow diagrams |

```
nanoptq/
├── core/
│   ├── quant_primitives.py   # symmetric/asymmetric/fake_quant math
│   └── group_quant.py        # group-wise quantization (the key trick)
├── model/
│   ├── quant_linear.py       # QuantLinear: drop-in for nn.Linear
│   └── hf_loader.py          # load HF model, replace Linear in-place
├── algorithms/
│   ├── rtn.py                # Round-to-Nearest (zero calibration)
│   ├── awq_lite.py           # AWQ-lite (activation-aware)
│   └── gptq_lite.py          # GPTQ-lite (Hessian compensation)
├── io/
│   └── safetensors_io.py     # save/load quantized checkpoints
├── eval/
│   ├── ppl.py                # sliding-window perplexity
│   └── latency.py            # prefill_ms, decode_tps, peak_mem_gb
└── data/
    └── loader.py             # load bundled calibration/eval data
data/
├── calibration/
│   └── {name}_128.jsonl      # 128 samples per dataset (7 datasets bundled)
└── eval/
    └── {name}_eval.jsonl     # full eval splits
examples/
├── quant_model.py            # end-to-end: load → quantize → eval → generate
├── compare_methods.py        # RTN vs AWQ vs GPTQ side-by-side table
├── precision_tour.py         # bf16 / fp8 / int4 / nvfp4 explained interactively
└── awq_explained.py          # AWQ step-by-step with live demos
docs/
├── Glossary.md               # every quantization term with an analogy
├── flow.md                   # flowcharts: offline quant + runtime inference
scripts/
└── prepare_data.py           # download datasets from HuggingFace (one-time)
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Only quantize `nn.Linear`, skip embeddings/norms | All major frameworks target Linear; norm layers have too few params to matter |
| Dequant-on-the-fly in `forward()` | No CUDA kernel needed; preserves `model.generate()` compat |
| `group_size=128` default | AWQ, GPTQ, torchao consensus; best precision/size balance for int4 |
| `symmetric=True` default for weights | Simpler hardware implementation; asymmetric is opt-in |
| Skip `lm_head` by default | Output projection is sensitive; quantizing it often hurts PPL disproportionately |
| Bundle datasets in `data/` | Reproducible eval without internet; one-stop eval for students |

---

## Additional Resources

| Resource | Description |
|----------|-------------|
| [docs/Glossary.md](docs/Glossary.md) | Every quantization term with an analogy |
| [docs/flow.md](docs/flow.md) | Flowcharts: offline quantization + runtime inference |
| [examples/precision_tour.py](examples/precision_tour.py) | Interactive tour of bf16, fp8, int4, nvfp4 |
| [examples/awq_explained.py](examples/awq_explained.py) | Step-by-step AWQ with live demos |

## Inspired by

- [nanoGPT](https://github.com/karpathy/nanoGPT) — the gold standard for educational ML repos
- [llm.c](https://github.com/karpathy/llm.c) — C clarity applied to deep learning
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) · [GPTQModel](https://github.com/modelcloud/gptqmodel) · [torchao](https://github.com/pytorch/ao) — industrial reference implementations
- [AngelSlim](https://github.com/tencent/AngelSlim) — bundled dataset eval design

---

## License

MIT
