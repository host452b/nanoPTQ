# nanoPTQ

> One file, one idea. One script, one experiment. One metric table, one truth.
> 一个文件，一个思想。一个脚本，一个实验。一张指标表，一个真相。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 项目哲学 / Philosophy

Modern quantization frameworks (vLLM, TensorRT-LLM, AutoAWQ) contain millions of lines of
engineering — CUDA kernel fusion, distributed sharding, multi-backend compatibility.
**80% of that code solves 20% of the conceptual problem.**

现代量化框架包含数百万行工程代码 —— CUDA 算子融合、分布式分片、多后端适配。
**其中 80% 的代码只解决了 20% 的概念问题。**

`nanoPTQ` strips quantization down to its mathematical soul, the same way `nanoGPT` strips
the Transformer. Each algorithm lives in **one file**. Every intermediate result can be printed.
Every formula maps directly to a line of code.

`nanoPTQ` 将量化剥离到数学本质，就像 `nanoGPT` 剥离 Transformer 一样。
每个算法住在**一个文件**里。每个中间结果都可以打印。每个公式都直接对应一行代码。

**What we keep / 保留什么:**
- `nn.Linear` layers only — the quantization target in every major LLM framework
- `int4 / int8` weight quantization — covers W4A16, W8A16 industrial workloads
- Group-wise quantization (group_size=128) — the universal precision/size tradeoff
- Safetensors I/O — real artifacts you can load into vLLM or HF Transformers
- Perplexity + tokens/s — the two metrics that actually matter
- Bundled calibration + eval data — one-stop eval, no internet needed after setup

**What we cut / 剔除什么:**
- QAT, pruning, distillation, sparsity
- CUDA/Triton kernels — dequant on the fly, backend handles matmul
- Multi-GPU, FSDP, pipeline parallelism
- VLMs, encoder-decoders, MoE routing

---

## 核心概念速查 / Key Concepts

理解三个数字就能读懂 90% 的代码：

| Concept / 概念 | Formula / 公式 | Intuition / 直觉 |
|---|---|---|
| Symmetric quant | `S = max\|W\| / (2^(b-1)-1)`, `Q = round(W/S)` | 以最大绝对值定刻度，有符号整数 |
| Asymmetric quant | `S = (max-min)/(2^b-1)`, `Z = -round(min/S)` | 零点偏移，无符号整数，覆盖非对称分布 |
| Group-wise | Apply scale per 128-weight block, not per-tensor | 每 128 个权重一个 scale，精度↑ 存储↑ 微小 |

**Why int4 group_size=128? / 为什么 int4 group_size=128?**

```
Bits per weight:
  fp16         = 16 bits
  int8          = 8 bits   (2× compression)
  int4 g=128   ≈ 4.25 bits (nearly 4× compression, with scale overhead)
  int4 g=32    ≈ 4.5 bits  (better precision, more overhead)

group_size=128 is the industry consensus (AWQ, GPTQ, torchao all default to it).
```

---

## Algorithms / 算法

| Algorithm | Core Idea / 核心思想 | Calibration | Industrial Use |
|-----------|---------------------|-------------|----------------|
| **RTN** | Round-to-Nearest. No data needed. Your baseline. <br>直接四舍五入。无需数据。这是你的基线。 | None / 无 | bitsandbytes baseline |
| **AWQ-lite** | Protect outlier channels by scaling weights up before quant. <br>量化前放大异常通道权重，保护精度。 | ~128 samples | AutoAWQ |
| **GPTQ-lite** | Use activation Hessian to compensate downstream columns after each quant step. <br>用 Hessian 矩阵逐列补偿量化误差。 | ~128 samples | GPTQModel |

**The mathematical essence / 数学本质:**

```python
# RTN (nanoptq/algorithms/rtn.py)
W_q = round(W / scale) * scale            # that's it / 就这一行

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

## Expected Results / 典型结果

On Qwen2-0.5B, int4 group_size=128 (your numbers will vary slightly):

| Method | PPL (wikitext-2) | ΔPPL | Notes |
|--------|-----------------|------|-------|
| fp16 baseline | ~14.5 | — | reference |
| RTN int4 | ~16–18 | +2–4 | no calibration needed |
| AWQ int4 | ~15–16 | +0.5–2 | better outlier handling |
| GPTQ int4 | ~15–16 | +0.5–2 | similar to AWQ |

> Lower perplexity = better. FP16 is the ceiling. RTN is the floor.
> 困惑度越低越好。FP16 是天花板，RTN 是地板。

---

## Quickstart / 快速开始

**Prerequisites / 前置要求:**
```bash
python >= 3.10
pytorch >= 2.0
transformers, safetensors
```

**Install / 安装:**
```bash
git clone https://github.com/host452b/nanoPTQ
cd nanoPTQ
pip install -e ".[dev]"

# Prepare bundled eval data (one-time, ~30s, needs internet once)
# 准备内置评测数据（仅需运行一次，约 30 秒，需要网络）
python scripts/prepare_data.py
```

**Run / 运行:**
```bash
# Quantize with RTN (no calibration data needed)
# RTN 量化（无需校准数据）
nanoptq quantize --model Qwen/Qwen2-0.5B --method rtn --bits 4 --group-size 128 --output ./qwen-rtn-int4

# Evaluate perplexity (uses bundled wikitext-2, no internet needed)
# 评测困惑度（使用内置数据，无需联网）
nanoptq eval --model ./qwen-rtn-int4 --metric ppl

# Compare RTN vs FP16 baseline
nanoptq compare --model Qwen/Qwen2-0.5B --bits 4 --group-size 128

# End-to-end example with latency
python examples/quant_model.py --model Qwen/Qwen2-0.5B --bits 4

# Compare all three methods side by side
# 三种方法横向对比
python examples/compare_methods.py --model Qwen/Qwen2-0.5B --bits 4
```

---

## Reading Order / 阅读顺序

If you are learning, read in this order / 如果你是来学习的，按此顺序阅读:

| Step | File | What you learn / 学到什么 | Time |
|------|------|--------------------------|------|
| 1 | `nanoptq/core/quant_primitives.py` | The math: symmetric, asymmetric, fake_quant <br>量化数学本质 | 5 min |
| 2 | `nanoptq/core/group_quant.py` | Why group-wise dramatically improves int4 <br>为什么逐组量化大幅改善 int4 精度 | 5 min |
| 3 | `nanoptq/model/quant_linear.py` | Unified layer abstraction; dequant-on-the-fly <br>统一量化层抽象；动态反量化 | 10 min |
| 4 | `nanoptq/algorithms/rtn.py` | Baseline: round and done <br>基线：四舍五入搞定 | 5 min |
| 5 | `nanoptq/algorithms/awq_lite.py` | Activation-aware improvement <br>激活感知改进 | 15 min |
| 6 | `nanoptq/algorithms/gptq_lite.py` | Hessian-based compensation <br>基于 Hessian 的误差补偿 | 20 min |
| 7 | `examples/compare_methods.py` | See them all side by side <br>三种方法横向对比 | — |

---

## Project Structure / 项目结构

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
│   └── wikitext2_train_128.jsonl   # 128 samples, committed to repo
└── eval/
    └── wikitext2_test.jsonl        # full wikitext-2 test split
examples/
├── quant_model.py            # end-to-end: load → quantize → eval → generate
└── compare_methods.py        # RTN vs AWQ vs GPTQ side-by-side table
scripts/
└── prepare_data.py           # regenerate data/ from HuggingFace (optional)
```

---

## Design Decisions / 设计决策

| Decision | Rationale |
|----------|-----------|
| Only quantize `nn.Linear`, skip embeddings/norms | All major frameworks target Linear; norm layers have too few params to matter |
| Dequant-on-the-fly in `forward()` | No CUDA kernel needed; preserves `model.generate()` compat |
| `group_size=128` default | AWQ, GPTQ, torchao consensus; best precision/size balance for int4 |
| `symmetric=True` default for weights | Simpler hardware implementation; asymmetric is opt-in |
| Skip `lm_head` by default | Output projection has different sensitivity; quantizing it often hurts PPL disproportionately |
| Bundle datasets in `data/` | Reproducible eval without internet; one-stop eval for students |

---

## Inspired by / 致敬

- [nanoGPT](https://github.com/karpathy/nanoGPT) — the gold standard for educational ML repos
- [llm.c](https://github.com/karpathy/llm.c) — C clarity applied to deep learning
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) · [GPTQModel](https://github.com/modelcloud/gptqmodel) · [torchao](https://github.com/pytorch/ao) — industrial reference implementations
- [AngelSlim](https://github.com/tencent/AngelSlim) — bundled dataset eval design

---

## License

MIT
