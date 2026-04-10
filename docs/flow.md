# Quantization Process — Flow Diagrams

Two separate processes. Two separate diagrams.

1. **Offline quantization** — what happens when you run `nanoptq quantize`
2. **Runtime inference** — what happens inside the backend when the quantized model runs

---

## Part 1 — Offline Quantization

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: HuggingFace model (FP16 weights) + method + bits        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOAD MODEL                                                      │
│  AutoModelForCausalLM.from_pretrained()                         │
│  → weights in FP16 on GPU/CPU                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
          ┌─────────────────┐    ┌───────────────────────────────┐
          │  RTN (no calib) │    │  AWQ / GPTQ (needs calib)     │
          │                 │    │                               │
          │  → skip step 2  │    │  COLLECT CALIBRATION DATA     │
          └────────┬────────┘    │  load ~128 text samples        │
                   │             │  forward pass through model    │
                   │             │  collect per-layer activations │
                   │             │  (hooks on each nn.Linear)     │
                   │             └──────────────┬────────────────┘
                   │                            │
                   └────────────┬───────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  FOR EACH nn.Linear (except lm_head):                           │
│                                                                  │
│  RTN path:                                                       │
│    1. Compute scale S = max|W| / (2^(b-1) - 1) per group        │
│    2. Q = round(W / S)        [int4 or int8]                    │
│    3. Store (Q, S) in QuantLinear                               │
│                                                                  │
│  AWQ path:                                                       │
│    1. Compute channel importance: s[i] = mean(|X[:,i]|)^alpha   │
│    2. Scale weights: W_scaled = W * s   (element-wise)          │
│    3. Group-quantize W_scaled → (Q, S_group)                    │
│    4. Store (Q, S_group, s) in QuantLinear                      │
│       [s = input_channel_scales, used at inference to divide x] │
│                                                                  │
│  GPTQ path:                                                      │
│    1. Compute H = X^T X  (input Hessian for this layer)         │
│    2. H_inv = inverse via Cholesky decomposition                │
│    3. FOR j in [0 .. in_features):                              │
│         Q[:,j] = quantize(W[:,j])                               │
│         err    = W[:,j] - Q[:,j]                                │
│         W[:,j+1:] -= err × H_inv[j, j+1:] / H_inv[j,j]        │
│    4. Store (Q, S_group) in QuantLinear                         │
│                                                                  │
│  → nn.Linear replaced by QuantLinear in-place                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  SAVE QUANTIZED CHECKPOINT                                       │
│  safetensors file:  weights_q (int4/int8), scales, zero_points  │
│                     input_channel_scales (AWQ only)              │
│  quant_config.json: bits, group_size, method, skipped_modules   │
│  tokenizer files:   copied from original model                  │
└─────────────────────────────────────────────────────────────────┘

OUTPUT: directory with quantized model, loadable by vLLM / HF Transformers
```

### What each step actually computes

```
GROUP-WISE SCALE COMPUTATION
─────────────────────────────
W shape: [out_features, in_features]
Reshape: [out_features × num_groups, group_size]   (group_size=128)

For each group:
  S = max|W_group| / (2^(b-1) - 1)    ← symmetric
  Q = clip(round(W_group / S), -2^(b-1), 2^(b-1)-1)

scales shape: [out_features, num_groups]            ← stored as fp16
weights_q shape: [out_features, in_features]        ← stored as int8 (int4 packed in practice)


AWQ CHANNEL SCALING
───────────────────
activations X: [num_tokens, in_features]   ← collected over 128 calibration samples

act_scale = mean(|X|, dim=0)              ← [in_features]  per-channel mean magnitude
s = act_scale ^ alpha                     ← [in_features]  alpha=0.5 default

W_scaled = W * s.unsqueeze(0)             ← broadcast over out_features
Q, S_group = group_quantize(W_scaled)    ← standard group-wise quantization


GPTQ COMPENSATION
─────────────────
H = X.T @ X                              ← [in_features, in_features]  Hessian
H_inv = cholesky_inverse(cholesky(H))   ← stable inverse

for j = 0, 1, 2, ..., in_features-1:
  q_j       = quantize(W[:, j])          ← quantize column j
  err_j     = W[:, j] - q_j             ← quantization error
  W[:, j:]  -= err_j × H_inv[j, j:] / H_inv[j, j]   ← propagate error to future columns
  W[:, j]   = q_j                        ← lock in quantized column
```

---

## Part 2 — Runtime Inference

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: quantized model dir + prompt text                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOAD QUANTIZED MODEL                                            │
│  1. AutoModelForCausalLM loads original architecture (FP16)     │
│  2. load_quantized_model() reads .safetensors:                  │
│       - replaces nn.Linear with QuantLinear                      │
│       - loads weights_q (int4/int8), scales, zero_points        │
│       - loads input_channel_scales (AWQ only)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  TOKENIZE PROMPT                                                 │
│  tokenizer(prompt) → input_ids  [1, seq_len]                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PREFILL (process the whole prompt at once)                      │
│                                                                  │
│  For each transformer layer:                                     │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │  QuantLinear.forward(x):                                  │ │
│    │                                                           │ │
│    │  ① AWQ only: x = x / input_channel_scales  (FP16 op)    │ │
│    │                                                           │ │
│    │  ② DEQUANTIZE weights:                                   │ │
│    │     scales  shape: [out, num_groups]                      │ │
│    │     weights shape: [out, in]                              │ │
│    │     W_fp16 = weights_q * scales.repeat_interleave(128, 1) │ │
│    │     [optionally add zero_point if asymmetric]             │ │
│    │                                                           │ │
│    │  ③ MATMUL: output = F.linear(x, W_fp16, bias)           │ │
│    │     [standard FP16 matmul — no special kernel needed]     │ │
│    └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│  → KV cache populated for all prompt tokens                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  DECODE LOOP (generate one token at a time)                      │
│                                                                  │
│  while not done:                                                 │
│    x = embedding(last_token)           [1, 1, hidden]           │
│    for each layer:                                               │
│      q, k, v = QKV_linear(x)          ← QuantLinear dequant    │
│      attn = attend(q, k, v, kv_cache) ← FP16 attention         │
│      x = proj_linear(attn)            ← QuantLinear dequant    │
│      x = mlp_linear(x)                ← QuantLinear dequant    │
│    logits = lm_head(x)                ← NOT quantized (skipped) │
│    next_token = sample(logits)                                   │
│    append next_token to kv_cache                                 │
│    if next_token == EOS: break                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: generated token IDs → decoded text                      │
└─────────────────────────────────────────────────────────────────┘
```

### Memory layout during inference

```
GPU VRAM USAGE
──────────────────────────────────────────────────
FP16 model (Qwen2-0.5B):    ~1.0 GB
INT4 model (Qwen2-0.5B):    ~0.3 GB  (3.7× smaller)

During forward pass, one layer at a time:
  Stored:  int4 weights    [small]
  Created: FP16 W_fp16     [temporarily in VRAM]
  Used:    FP16 matmul     [standard CUBLAS]
  Freed:   W_fp16          [after matmul]

KV cache grows with sequence length:
  kv_cache per token ≈ 2 × num_layers × hidden × sizeof(fp16)
  For Qwen2-0.5B: ~0.2 MB per token (manageable)
```

### Why skip lm_head?

```
lm_head shape: [hidden, vocab_size]   e.g., [896, 151936] for Qwen2-0.5B

Quantizing this layer:
  ✗ vocab_size is huge → many outlier rows → large scale → big PPL loss
  ✗ This is the final logit projection → errors here directly affect token sampling
  ✗ Relative size: ~0.27 GB in FP16, just ~7% of model params — not worth the risk

Decision: always skip lm_head. The compression gain is small; the PPL cost can be large.
```

---

## Summary

```
OFFLINE (one-time, done by developer / MLOps):
  FP16 model  →  [quantize algorithm]  →  INT4 checkpoint (.safetensors)

ONLINE (every request, done by inference server):
  INT4 checkpoint  →  [dequant on the fly]  →  FP16 matmul  →  generated text

The key insight: storage is cheap, compute is precious.
  Store 4 bits, compute in 16 bits.
  No custom kernel needed. Works on any GPU with PyTorch.
```
