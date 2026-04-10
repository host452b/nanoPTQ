# 量化流程图解

两个独立的过程，两张独立的流程图。

1. **离线量化** — 运行 `nanoptq quantize` 时发生了什么
2. **运行时推理** — 量化模型在后端框架中运行时发生了什么

---

## 第一部分 — 离线量化

```
┌─────────────────────────────────────────────────────────────────┐
│  输入：HuggingFace 模型（FP16 权重）+ 量化方法 + 位宽            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  加载模型                                                        │
│  AutoModelForCausalLM.from_pretrained()                         │
│  → 权重以 FP16 格式加载到 GPU/CPU                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
          ┌─────────────────┐    ┌───────────────────────────────┐
          │  RTN（无需校准） │    │  AWQ / GPTQ（需要校准数据）   │
          │                 │    │                               │
          │  → 跳过步骤2    │    │  收集校准数据                 │
          └────────┬────────┘    │  加载约 128 条文本样本        │
                   │             │  前向传播，收集逐层激活值      │
                   │             │  （每个 nn.Linear 挂 hook）   │
                   │             └──────────────┬────────────────┘
                   │                            │
                   └────────────┬───────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  对每个 nn.Linear（跳过 lm_head）：                              │
│                                                                  │
│  RTN 路径：                                                      │
│    1. 每组计算 scale：S = max|W| / (2^(b-1) - 1)                │
│    2. Q = round(W / S)        [int4 或 int8]                   │
│    3. 将 (Q, S) 存入 QuantLinear                               │
│                                                                  │
│  AWQ 路径：                                                      │
│    1. 计算通道重要性：s[i] = mean(|X[:,i]|)^alpha               │
│    2. 缩放权重：W_scaled = W * s  （逐元素）                    │
│    3. 对 W_scaled 逐组量化 → (Q, S_group)                      │
│    4. 将 (Q, S_group, s) 存入 QuantLinear                      │
│       [s = input_channel_scales，推理时用于除以 x]              │
│                                                                  │
│  GPTQ 路径：                                                     │
│    1. 计算 H = X^T X（该层的输入 Hessian）                      │
│    2. H_inv = Cholesky 分解求逆                                 │
│    3. 对 j = 0, 1, ..., in_features-1：                        │
│         Q[:,j] = quantize(W[:,j])                               │
│         err    = W[:,j] - Q[:,j]                                │
│         W[:,j+1:] -= err × H_inv[j, j+1:] / H_inv[j,j]        │
│    4. 将 (Q, S_group) 存入 QuantLinear                         │
│                                                                  │
│  → 原位替换：nn.Linear → QuantLinear                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  保存量化检查点                                                   │
│  safetensors 文件：weights_q（int4/int8）、scales、zero_points  │
│                    input_channel_scales（仅 AWQ）                 │
│  quant_config.json：bits、group_size、method、skipped_modules   │
│  tokenizer 文件：从原始模型复制                                   │
└─────────────────────────────────────────────────────────────────┘

输出：包含量化模型的目录，可直接加载到 vLLM / HF Transformers
```

### 每个步骤实际计算了什么

```
逐组 SCALE 计算
────────────────────────────────────────
W 形状：[out_features, in_features]
重塑为：[out_features × num_groups, group_size]   (group_size=128)

对每一组：
  S = max|W_group| / (2^(b-1) - 1)    ← 对称量化
  Q = clip(round(W_group / S), -2^(b-1), 2^(b-1)-1)

scales 形状：[out_features, num_groups]            ← 以 fp16 存储
weights_q 形状：[out_features, in_features]        ← 以 int8 存储（int4 实际上打包存储）


AWQ 通道缩放
────────────────────────────────────────
激活值 X：[num_tokens, in_features]   ← 从 128 条校准样本收集

act_scale = mean(|X|, dim=0)          ← [in_features]  逐通道均值绝对值
s = act_scale ^ alpha                 ← [in_features]  alpha=0.5 为默认值

W_scaled = W * s.unsqueeze(0)         ← 在 out_features 维度上广播
Q, S_group = group_quantize(W_scaled) ← 标准逐组量化


GPTQ 误差补偿
────────────────────────────────────────
H = X.T @ X                           ← [in_features, in_features]  Hessian 矩阵
H_inv = cholesky_inverse(cholesky(H)) ← 数值稳定的求逆方式

for j = 0, 1, 2, ..., in_features-1:
  q_j       = quantize(W[:, j])       ← 量化第 j 列
  err_j     = W[:, j] - q_j          ← 量化误差
  W[:, j:]  -= err_j × H_inv[j, j:] / H_inv[j, j]   ← 误差传播到后续列
  W[:, j]   = q_j                     ← 锁定当前列的量化结果
```

---

## 第二部分 — 运行时推理

```
┌─────────────────────────────────────────────────────────────────┐
│  输入：量化模型目录 + 提示文本（prompt）                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  加载量化模型                                                    │
│  1. AutoModelForCausalLM 加载原始架构（FP16）                   │
│  2. load_quantized_model() 读取 .safetensors：                  │
│       - 将 nn.Linear 替换为 QuantLinear                         │
│       - 加载 weights_q（int4/int8）、scales、zero_points        │
│       - 加载 input_channel_scales（仅 AWQ）                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  分词（Tokenize Prompt）                                         │
│  tokenizer(prompt) → input_ids  [1, seq_len]                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  预填充（Prefill）— 一次性处理整个 prompt                        │
│                                                                  │
│  对每个 Transformer 层：                                         │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │  QuantLinear.forward(x)：                                 │ │
│    │                                                           │ │
│    │  ① 仅 AWQ：x = x / input_channel_scales  （FP16 操作）  │ │
│    │                                                           │ │
│    │  ② 反量化权重：                                          │ │
│    │     scales  形状：[out, num_groups]                       │ │
│    │     weights 形状：[out, in]                               │ │
│    │     W_fp16 = weights_q * scales.repeat_interleave(128, 1) │ │
│    │     [非对称时加上 zero_point]                             │ │
│    │                                                           │ │
│    │  ③ 矩阵乘法：output = F.linear(x, W_fp16, bias)         │ │
│    │     [标准 FP16 matmul，无需特殊 CUDA 算子]               │ │
│    └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│  → KV Cache 填充完毕（保存所有 prompt token 的 K/V）            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  解码循环（Decode Loop）— 每次生成一个 token                     │
│                                                                  │
│  while 未结束：                                                  │
│    x = embedding(上一个token)         [1, 1, hidden]            │
│    for 每层：                                                    │
│      q, k, v = QKV_linear(x)         ← QuantLinear 动态反量化  │
│      attn = attend(q, k, v, kv_cache) ← FP16 注意力机制        │
│      x = proj_linear(attn)            ← QuantLinear 动态反量化  │
│      x = mlp_linear(x)                ← QuantLinear 动态反量化  │
│    logits = lm_head(x)                ← 未量化（已跳过）        │
│    next_token = sample(logits)                                   │
│    把 next_token 追加到 kv_cache                                 │
│    if next_token == EOS: break                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出：生成的 token ID → 解码为文本                              │
└─────────────────────────────────────────────────────────────────┘
```

### 推理时的显存布局

```
GPU 显存占用
──────────────────────────────────────────────────
FP16 模型 (Qwen2-0.5B)：    约 1.0 GB
INT4 模型 (Qwen2-0.5B)：    约 0.3 GB（缩小约 3.7×）

前向传播时，逐层处理：
  存储：int4 权重    [小]
  创建：FP16 W_fp16  [暂时占用显存]
  使用：FP16 matmul  [标准 CUBLAS]
  释放：W_fp16       [matmul 后即释放]

KV Cache 随序列长度增长：
  每个 token 的 kv_cache ≈ 2 × 层数 × hidden × sizeof(fp16)
  对于 Qwen2-0.5B：约 0.2 MB/token（可控）
```

### 为什么跳过 lm_head？

```
lm_head 形状：[hidden, vocab_size]   以 Qwen2-0.5B 为例：[896, 151936]

量化这一层的问题：
  ✗ vocab_size 很大 → 异常行多 → scale 大 → PPL 损失显著
  ✗ 这是最终的 logit 投影层 → 误差直接影响 token 采样
  ✗ 参数量占比：约 0.27 GB（FP16），仅占约 7% 的参数 — 压缩收益不值得冒险

决策：始终跳过 lm_head。压缩收益有限，PPL 代价可能很大。
```

---

## 小结

```
离线（一次性操作，由开发者或 MLOps 完成）：
  FP16 模型  →  [量化算法]  →  INT4 检查点（.safetensors）

在线（每次请求，由推理服务器完成）：
  INT4 检查点  →  [动态反量化]  →  FP16 矩阵乘法  →  生成文本

核心洞察：存储廉价，算力宝贵。
  存 4 位，算 16 位。
  无需自定义算子，PyTorch 在任何 GPU 上即可运行。
```
