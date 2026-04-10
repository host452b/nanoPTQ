# nanoPTQ

> One file, one idea. One script, one experiment. One metric table, one truth.
> 一个文件，一个思想。一个脚本，一个实验。一张指标表，一个真相。

---

## 项目哲学 / Philosophy

Modern quantization frameworks (vLLM, TensorRT-LLM, AutoAWQ) contain millions of lines of
engineering — CUDA kernel fusion, distributed sharding, multi-backend compatibility.
**80% of that code solves 20% of the conceptual problem.**

现代量化框架包含数百万行工程代码 —— CUDA 算子融合、分布式分片、多后端适配。
**其中 80% 的代码只解决了 20% 的概念问题。**

`nanoPTQ` strips quantization down to its mathematical soul, the same way `nanoGPT` strips
the Transformer. Each algorithm lives in one file. Every intermediate result can be printed.
Every formula maps directly to a line of code.

`nanoPTQ` 将量化剥离到数学本质，就像 `nanoGPT` 剥离 Transformer 一样。
每个算法住在一个文件里。每个中间结果都可以打印。每个公式都直接对应一行代码。

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

## Algorithms / 算法

| Algorithm | Method | Bits | Calibration | Industrial equivalent |
|-----------|--------|------|-------------|----------------------|
| RTN | Round-to-Nearest | int4/int8 | None | bitsandbytes NF4 baseline |
| AWQ-lite | Activation-aware channel scaling | int4 | ~128 samples | AutoAWQ |
| GPTQ-lite | Hessian-based layer-wise update | int4 | ~128 samples | GPTQModel |

---

## Quickstart / 快速开始

```bash
pip install -e ".[dev]"

# Prepare bundled eval data (one-time, ~30s)
python scripts/prepare_data.py

# Quantize a model with RTN int4
nanoptq quantize --model Qwen/Qwen2-0.5B --method rtn --bits 4 --group-size 128 --output ./qwen-rtn-int4

# Evaluate perplexity (uses bundled wikitext-2, no internet needed)
nanoptq eval --model ./qwen-rtn-int4 --metric ppl

# Compare all methods
nanoptq compare --model Qwen/Qwen2-0.5B --bits 4 --group-size 128
```

---

## Reading Order / 阅读顺序

If you are learning, read in this order / 如果你是来学习的，按此顺序阅读:

1. `nanoptq/core/quant_primitives.py` — the math (5 min)
2. `nanoptq/core/group_quant.py` — why group-wise matters (5 min)
3. `nanoptq/model/quant_linear.py` — unified layer abstraction (10 min)
4. `nanoptq/algorithms/rtn.py` — baseline, no calibration needed (5 min)
5. `nanoptq/algorithms/awq_lite.py` — activation-aware improvement (15 min)
6. `nanoptq/algorithms/gptq_lite.py` — Hessian-based refinement (20 min)
7. `examples/compare_methods.py` — see them all side by side

---

## Project Structure / 项目结构

```
nanoptq/core/          # quantization math primitives
nanoptq/model/         # QuantLinear + HF model loading
nanoptq/algorithms/    # RTN, AWQ-lite, GPTQ-lite
nanoptq/io/            # safetensors save/load
nanoptq/eval/          # perplexity + latency
nanoptq/data/          # bundled dataset loader
data/calibration/      # 128 wikitext-2 train samples (committed)
data/eval/             # wikitext-2 test split (committed)
scripts/               # data preparation utilities
examples/              # end-to-end runnable scripts
```

---

## Inspired by / 致敬

- [nanoGPT](https://github.com/karpathy/nanoGPT) — the gold standard for educational ML repos
- [llm.c](https://github.com/karpathy/llm.c) — C clarity applied to deep learning
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [GPTQModel](https://github.com/modelcloud/gptqmodel), [torchao](https://github.com/pytorch/ao) — industrial reference
- [AngelSlim](https://github.com/tencent/AngelSlim) — bundled dataset eval design
