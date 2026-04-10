# Quantization Glossary — Analogy Edition

> Every term explained with a real-world analogy first, then the math.
> Read the analogy. Then the definition will click immediately.

---

## Quantization (量化)

**Analogy:** You're packing books into a moving truck. The books (weights) are currently in a warehouse
where each shelf position is labeled with a real number (float32). Your truck only has 16 labeled slots
(int4: values -8 to +7). You re-label the shelves using fewer slot labels — some books now share a slot.
You lose some positional precision, but everything fits.

**Definition:** Mapping a continuous floating-point weight to a discrete integer.
`Q = round(W / S)` where S is the scale factor and Q is the integer code.

---

## Scale Factor / Scale (缩放因子 / Scale)

**Analogy:** A ruler's unit markings. If you use a ruler marked in millimeters, you can measure 12.3 mm.
If you only have a ruler marked in centimeters, you round to 12 cm. The centimeter is the scale factor —
it determines how fine your measurements can be.

**Definition:** `S = max|W| / (2^(b-1) - 1)` for symmetric int-b quantization.
One scale per group (group_size=128). Stored as float16 alongside the integer weights.

---

## Zero-Point (零点, ZP)

**Analogy:** Celsius vs Kelvin. Kelvin has zero at absolute zero (-273°C). Celsius has zero at water's
freezing point. To convert between them you shift by 273. The zero-point is that shift — it moves
the integer range to cover asymmetric real-value distributions.

**Definition:** `Z = -round(min(W) / S)`. Used in asymmetric quantization:
`W_approx = S × (Q - Z)`. Symmetric quantization always has Z=0.

---

## Symmetric vs Asymmetric Quantization (对称 vs 非对称量化)

**Analogy (Symmetric):** A balance scale. Zero is in the exact middle. Both sides are equal.
Works well when the data is centered near zero (like most neural network weights after training).

**Analogy (Asymmetric):** A thermometer that starts from a non-zero baseline.
Works better for activations after ReLU (which are always ≥ 0) — no need to waste half the range on
negative numbers that can't exist.

**In nanoPTQ:** Weights use symmetric (default). Asymmetric is opt-in.

---

## Group-wise Quantization (逐组量化)

**Analogy:** Grading students on a curve — but separately for each exam room.
If you grade the whole school on one curve, a room full of geniuses drags up the "normal" scale so that
average students all get F. Grading each room separately gives a fair curve for each group.

**Definition:** Instead of one scale for the whole weight matrix, use one scale per 128 weights (one group).
Result: outlier columns get their own scale and don't corrupt neighboring columns' precision.

```
Per-tensor:  [---w1---w2---w3---outlier---w4---w5---]   ← one scale for all
Group-wise:  [---w1--][--w2--][--outlier--][--w3--]   ← one scale per group
```

---

## Perplexity (PPL, 困惑度)

**Analogy:** Imagine reading a mystery novel one word at a time. After each word, you guess the next.
If you're rarely surprised, your perplexity is low. If every word shocks you, it's high.
PPL measures how "surprised" a language model is by real text — lower is better.

**Definition:** `PPL = exp(-1/N × Σ log P(token_i))`. Evaluated on a fixed test set (wikitext-2).
FP16 baseline is the ceiling. Quantized models should be as close to FP16 as possible.

---

## RTN — Round-to-Nearest (直接四舍五入)

**Analogy:** You're converting a price list from dollars to integer cents by rounding.
`$3.14159` → `314` cents. No information about neighboring items. Just round each number independently.

**Definition:** `Q = round(W / S)`. No calibration data needed. Fastest. Most loss of precision.
The floor — every other method should beat this.

---

## AWQ — Activation-Aware Weight Quantization (激活感知权重量化)

**Analogy:** A teacher grading with partial credit. Some questions (channels) matter more to the final
grade than others. The teacher allocates more careful grading time to the high-weight questions.
AWQ identifies which weight columns receive large activations at runtime, and gives those columns
more precision budget before quantization.

**Key insight:** Two types of outliers exist:
- Weight outliers: a column of W has large magnitude → group-wise already handles this
- Activation outliers: a column of X has large magnitude → large activations amplify any weight error

**Math:** Multiply W[:,i] by s[i] = mean(|X[:,i]|)^alpha before quantizing.
Divide x by s at inference time. Net effect: (W·s) @ (x/s) = W @ x. Math is exact, precision better.

---

## GPTQ — Generative Pre-trained Transformer Quantization (Hessian 补偿量化)

**Analogy:** A sculptor fixing a statue. You chip away at column 1 (quantize it), then notice the
balance is off. You compensate by adjusting columns 2, 3, 4 based on how they were correlated with
column 1. Then chip column 2, compensate the rest. Repeat left-to-right across the statue.

**Math:** After quantizing column j, the error `err = W[:,j] - Q[:,j]` propagates to remaining columns:
`W[:,j+1:] -= err × H_inv[j, j+1:] / H_inv[j,j]`
where H = X^T X is the input Hessian matrix (measures how correlated inputs are).

---

## Calibration Data (校准数据)

**Analogy:** A doctor taking a blood sample before prescribing medication.
The sample (calibration data) is small (~128 sentences) but representative. The doctor (quantizer)
uses it to learn which channels are "sensitive" — without running the full clinical trial (training).

**In nanoPTQ:** 128 wikitext2 sentences stored in `data/calibration/wikitext2_128.jsonl`.
Never touches the eval set — eval data stays held-out for honest PPL measurement.

---

## Weight vs Activation (权重 vs 激活值)

**Analogy:**
- **Weights:** The recipe. Fixed once the model is trained. Quantizing weights reduces model file size.
- **Activations:** The ingredients at serving time. They change with every input. 
  W4A16 means 4-bit weights + 16-bit activations. W8A8 means both are 8-bit.

**nanoPTQ quantizes weights only (W4A16 / W8A16).** Activation quantization (W8A8) requires special
hardware (H100 FP8 units) and is not included.

---

## int4 / int8 (整数格式)

**Analogy:**
- int4: A sticky note with 16 possible values (-8 to +7). Great compression (4× vs float16), some loss.
- int8: A business card with 256 possible values (-128 to +127). 2× compression, minimal loss.

**In practice:** int4 with group_size=128 ≈ 4.25 bits/weight (scale overhead). Nearly 4× vs float16.

---

## bf16 / fp16 (浮点格式)

**Analogy:**
- fp16: A short ruler with 5 exponent bits (max ≈ 65504). Very precise near zero, but overflows
  for large numbers. Common in older GPUs (V100).
- bf16: A wide-range ruler with 8 exponent bits (same as float32). Same range as float32, but
  only 7 bits of decimal precision. H100/A100 native. Never overflows in practice.

**Rule of thumb:** Use bf16 for training and inference on modern GPUs. fp16 for older hardware.

---

## fp8 (8位浮点)

**Analogy:** A compact camera with two lenses:
- e4m3 (4 exp + 3 mantissa): Sharp detail near zero. Max value = 448. For activations in forward pass.
- e5m2 (5 exp + 2 mantissa): Wide range. Max value = 57344. For gradients in backward pass.

**Hardware:** H100 has native fp8 tensor cores. On older GPUs, fp8 emulates in software (slow).
W8A8 FP8 inference is the next frontier after W4A16.

---

## nvfp4 / FP4 (4位浮点，e2m1)

**Analogy:** A pocket ruler with only 8 tick marks on each side (16 total).
The marks aren't evenly spaced — they're closer together near zero and farther apart for large values.
This matches how neural network weights are distributed (bell curve near zero).

**Values (positive side only):** 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
**Hardware:** NVIDIA Blackwell (B200, GB200). Not available on H100/A100.
**Format:** 1 sign bit + 2 exponent bits + 1 mantissa bit = 4 bits total.

---

## Dequant-on-the-fly (动态反量化)

**Analogy:** A compressed music file (MP3) being decoded every time you play it.
The file stays compressed on disk. The player decodes it to full quality in real time.
Similarly, nanoPTQ stores int4 weights on GPU memory, then reconstructs float16 every forward pass.

**Trade-off:** Saves VRAM (4× compression), costs a tiny bit of compute per matmul.
No special CUDA kernel needed — works with standard PyTorch `F.linear`.

---

## Hessian Matrix (Hessian 矩阵, H = X^T X)

**Analogy:** A sensitivity map of a soundboard. If you push fader A, how much does it affect the
output? And how correlated is fader A with fader B? The Hessian captures these correlations.
GPTQ uses it to know: when I introduce error in column j, how much do I need to nudge column j+1?

**In nanoPTQ:** `H = X.T @ X` where X is the calibration activations for this layer.
Shape: [in_features, in_features]. Inverted via Cholesky for numerical stability.

---

## Safetensors (.safetensors)

**Analogy:** A standardized shipping container format. Any port (vLLM, HuggingFace Transformers,
llama.cpp) can unload it. More secure than pickle (no code execution on load). Faster for large files.

**In nanoPTQ:** Quantized weights, scales, zero_points, and input_channel_scales are all stored in
`.safetensors`. A `quant_config.json` alongside records bits, group_size, and method.

---

## Outlier Channel (异常通道)

**Analogy:** One extremely loud instrument in an orchestra. If you record the whole orchestra with
one volume knob, you turn it down for the loud instrument — and everyone else becomes inaudible.
Outlier channels (columns with unusually large weight or activation magnitude) force the global scale
to be large, crushing precision for all normal channels.

**Solution:** group-wise quantization (separate scale per group) or AWQ (per-channel importance scaling).
