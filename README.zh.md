# nanoPTQ

> 一个文件，一个思想。一个脚本，一个实验。一张指标表，一个真相。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[English README](README.md)**

---

## 项目哲学

现代量化框架（vLLM、TensorRT-LLM、AutoAWQ）包含数百万行工程代码 —— CUDA 算子融合、分布式分片、多后端适配。**其中 80% 的代码只解决了 20% 的概念问题。**

`nanoPTQ` 将量化剥离到数学本质，就像 `nanoGPT` 剥离 Transformer 一样。每个算法住在**一个文件**里。每个中间结果都可以打印。每个公式都直接对应一行代码。

**保留什么：**
- 只量化 `nn.Linear` —— 所有主流 LLM 框架的量化目标
- `int4 / int8` 权重量化 —— 覆盖 W4A16、W8A16 工业场景
- 逐组量化（group_size=128）—— 精度与压缩比的通用折中方案
- Safetensors 存储 —— 生成的权重文件可直接加载到 vLLM 或 HF Transformers
- 困惑度 + tokens/s —— 两个真正有意义的评测指标
- 内置校准与评测数据 —— 一站式评测，setup 后无需联网

**剔除什么：**
- QAT、剪枝、蒸馏、稀疏化
- CUDA/Triton 算子 —— 动态反量化，matmul 交给后端处理
- 多 GPU、FSDP、流水线并行
- VLM、编码器-解码器、MoE 路由

---

## 核心概念速查

理解三个数字就能读懂 90% 的代码：

| 概念 | 公式 | 直觉理解 |
|---|---|---|
| 对称量化 | `S = max\|W\| / (2^(b-1)-1)`，`Q = round(W/S)` | 以最大绝对值定刻度，有符号整数 |
| 非对称量化 | `S = (max-min)/(2^b-1)`，`Z = -round(min/S)` | 零点偏移，无符号整数，覆盖非对称分布 |
| 逐组量化 | 每 128 个权重一个 scale，而非整层一个 | 每组一个刻度 —— 精度显著提升，存储开销极小 |

**为什么 int4 group_size=128？**

```
每个权重占用的比特数：
  fp16          = 16 bits
  int8           =  8 bits   （压缩 2×）
  int4 g=128    ≈  4.25 bits （压缩近 4×，含 scale 开销）

group_size=128 是业界共识（AWQ、GPTQ、torchao 均以此为默认值）。
```

---

## 算法

| 算法 | 核心思想 | 校准数据 | 工业对应 |
|------|---------|---------|---------|
| **RTN** | 直接四舍五入，无需任何数据。这是你的基线。 | 无 | bitsandbytes 基线 |
| **AWQ-lite** | 量化前放大异常通道的权重，保护关键精度。 | ~128 条 | AutoAWQ |
| **GPTQ-lite** | 用激活 Hessian 矩阵对后续列逐步补偿量化误差。 | ~128 条 | GPTQModel |

**数学本质（各 3~4 行）：**

```python
# RTN（nanoptq/algorithms/rtn.py）
W_q = round(W / scale) * scale            # 就这一行

# AWQ（nanoptq/algorithms/awq_lite.py）
s   = mean(|activations|, dim=0) ** alpha  # 通道重要性
W_q = quantize(W * s)                      # 放大重要通道后量化
out = W_q @ (x / s)                        # 推理时除回去

# GPTQ（nanoptq/algorithms/gptq_lite.py）
H     = X.T @ X                            # 输入 Hessian
H_inv = cholesky_inverse(cholesky(H))
for j in range(in_features):
    err          = W[:,j] - quantize(W[:,j])
    W[:,j+1:] -= err ⊗ H_inv[j, j+1:] / H_inv[j,j]  # 补偿后续列
```

---

## 典型结果

以 Qwen2-0.5B，int4 group_size=128 为例（实际结果会有微小波动）：

| 方法 | 困惑度（wikitext-2） | ΔPPL | 备注 |
|------|-------------------|------|------|
| fp16 基线 | ~14.5 | — | 参考上限 |
| RTN int4 | ~16–18 | +2–4 | 无需校准数据 |
| AWQ int4 | ~15–16 | +0.5–2 | 异常通道处理更好 |
| GPTQ int4 | ~15–16 | +0.5–2 | 与 AWQ 相近 |

> 困惑度越低越好。fp16 是天花板，RTN 是地板。

---

## 快速开始

**前置要求：**
```
python >= 3.10，pytorch >= 2.0，transformers，safetensors
```

**安装：**
```bash
git clone https://github.com/host452b/nanoPTQ
cd nanoPTQ
pip install -e ".[dev]"

# 准备内置评测数据（仅需运行一次，约 30 秒，需要网络）
python scripts/prepare_data.py
```

**运行：**
```bash
# RTN 量化（无需校准数据）
nanoptq quantize --model Qwen/Qwen2-0.5B --method rtn --bits 4 --group-size 128 --output ./qwen-rtn-int4

# 评测困惑度（使用内置数据，无需联网）
nanoptq eval --model ./qwen-rtn-int4 --metric ppl

# 与 FP16 基线对比
nanoptq compare --model Qwen/Qwen2-0.5B --bits 4 --group-size 128

# 端到端示例（含延迟测试）
python examples/quant_model.py --model Qwen/Qwen2-0.5B --bits 4

# 三种方法横向对比
python examples/compare_methods.py --model Qwen/Qwen2-0.5B --bits 4
```

---

## 阅读顺序

如果你是来学习的，建议按此顺序阅读：

| 步骤 | 文件 | 学到什么 | 时间 |
|------|------|---------|------|
| 1 | `nanoptq/core/quant_primitives.py` | 量化数学本质：对称、非对称、伪量化 | 5 分钟 |
| 2 | `nanoptq/core/group_quant.py` | 为什么逐组量化大幅改善 int4 精度 | 5 分钟 |
| 3 | `nanoptq/model/quant_linear.py` | 统一量化层抽象；动态反量化 | 10 分钟 |
| 4 | `nanoptq/algorithms/rtn.py` | 基线：四舍五入搞定 | 5 分钟 |
| 5 | `nanoptq/algorithms/awq_lite.py` | 激活感知改进 | 15 分钟 |
| 6 | `nanoptq/algorithms/gptq_lite.py` | 基于 Hessian 的误差补偿 | 20 分钟 |
| 7 | `examples/compare_methods.py` | 三种方法横向对比 | — |

---

## 项目结构

```
nanoptq/
├── core/
│   ├── quant_primitives.py   # 对称/非对称/伪量化数学
│   └── group_quant.py        # 逐组量化（核心技巧）
├── model/
│   ├── quant_linear.py       # QuantLinear：nn.Linear 的量化替代
│   └── hf_loader.py          # 加载 HF 模型，原位替换 Linear 层
├── algorithms/
│   ├── rtn.py                # RTN（零校准）
│   ├── awq_lite.py           # AWQ-lite（激活感知）
│   └── gptq_lite.py          # GPTQ-lite（Hessian 补偿）
├── io/
│   └── safetensors_io.py     # 量化检查点的保存与加载
├── eval/
│   ├── ppl.py                # 滑动窗口困惑度
│   └── latency.py            # prefill_ms、decode_tps、peak_mem_gb
└── data/
    └── loader.py             # 加载内置校准/评测数据
data/
├── calibration/
│   └── wikitext2_train_128.jsonl   # 128 条训练样本（已提交至仓库）
└── eval/
    └── wikitext2_test.jsonl        # wikitext-2 完整测试集
examples/
├── quant_model.py            # 端到端：加载 → 量化 → 评测 → 生成
└── compare_methods.py        # RTN vs AWQ vs GPTQ 横向对比表
scripts/
└── prepare_data.py           # 从 HuggingFace 重新生成数据（可选）
```

---

## 设计决策

| 决策 | 原因 |
|------|------|
| 只量化 `nn.Linear`，跳过 embedding 和 norm | 所有主流框架均针对 Linear；norm 层参数量太少，量化收益微乎其微 |
| `forward()` 中动态反量化 | 无需 CUDA 算子，保持 `model.generate()` 兼容性 |
| 默认 `group_size=128` | AWQ/GPTQ/torchao 的共识默认值；int4 精度与压缩比的最优平衡点 |
| 权重默认 `symmetric=True` | 硬件实现更简单；非对称量化作为可选项 |
| 默认跳过 `lm_head` | 输出投影层对量化更敏感，量化后 PPL 损失往往不成比例 |
| 将数据集内置到 `data/` | 可复现的评测，无需联网；方便学生一站式复现实验 |

---

## 致敬

- [nanoGPT](https://github.com/karpathy/nanoGPT) —— 教学类 ML 仓库的黄金标准
- [llm.c](https://github.com/karpathy/llm.c) —— C 语言的简洁性应用于深度学习
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) · [GPTQModel](https://github.com/modelcloud/gptqmodel) · [torchao](https://github.com/pytorch/ao) —— 工业级参考实现
- [AngelSlim](https://github.com/tencent/AngelSlim) —— 内置数据集的评测设计灵感来源

---

## License

MIT
