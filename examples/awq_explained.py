#!/usr/bin/env python3
# examples/awq_explained.py
"""
AWQ Deep Dive — from "why naive int4 fails" to "how AWQ fixes it".
AWQ 原理精讲：从"为什么 int4 失败"到"AWQ 如何修复"。

AWQ = Activation-aware Weight Quantization (Lin et al., 2023)
Paper: https://arxiv.org/abs/2306.00978

Run:
  python examples/awq_explained.py
  python examples/awq_explained.py --step 3   # only one step

No GPU, no model download. Pure PyTorch.
Read alongside: nanoptq/algorithms/awq_lite.py
"""
import argparse
import math
import torch
import torch.nn as nn

SEP = "─" * 64


def header(n, title_en, title_zh=""):
    print(f"\n{'═'*64}")
    print(f"  STEP {n} — {title_en}")
    print(f"{'═'*64}\n")


# ─────────────────────────────────────────────────────────────
# Synthetic model setup
# ─────────────────────────────────────────────────────────────

def make_weight_with_outlier_at_column(out_features=16, in_features=128,
                                        outlier_col=5, seed=42):
    """
    Weight matrix where column `outlier_col` has large magnitude (weight outlier).
    This outlier dominates the per-tensor scale, hurting all other columns.

    In real LLMs, ~1% of weight columns are "outliers" (10-100× larger than avg).
    This is the problem AWQ was designed to solve.
    """
    torch.manual_seed(seed)
    W = torch.randn(out_features, in_features) * 0.3   # normal columns
    W[:, outlier_col] = torch.randn(out_features) * 8.0  # weight outlier
    return W


def make_activations_with_high_channel(n_tokens=64, in_features=128,
                                        high_channel=20, seed=1):
    """
    Activations where channel `high_channel` is frequently large (activation outlier).
    Note: activation outlier channel != weight outlier channel (different locations).

    AWQ insight: if x[:,i] is consistently large, then quantization error of W[:,i]
    has outsized impact on output: output_err ≈ delta_W @ x, so large x[:,i] amplifies
    any error in W[:,i] — even if W[:,i] itself looks small.
    """
    torch.manual_seed(seed)
    X = torch.randn(n_tokens, in_features) * 0.5
    X[:, high_channel] = torch.randn(n_tokens).abs() * 5.0  # activation outlier
    return X


# ═══════════════════════════════════════════════════════════════
# STEP 1: Why naive per-tensor int4 fails
# ═══════════════════════════════════════════════════════════════

def step1_why_naive_int4_fails():
    header(1, "Why naive per-tensor int4 fails",
              "为什么朴素 per-tensor int4 量化会失败")

    W = make_weight_with_outlier_at_column()
    X = make_activations_with_high_channel()
    qmax = 7

    print("Weight matrix W [16 out × 128 in] — column 5 is the weight outlier:")
    print(f"  Normal columns (0-4, 6-127): std ≈ {W[:, :5].std():.2f}")
    print(f"  Outlier column 5:            std ≈ {W[:, 5].std():.2f}  ← dominates scale\n")

    # Per-tensor naive int4
    scale_naive = W.abs().max() / qmax      # dominated by column 5
    W_q = (W / scale_naive).round().clamp(-qmax, qmax) * scale_naive
    err = (W - W_q).abs()

    print("Naive int4 — one global scale for the whole matrix:")
    print(f"  Global scale = max|W| / 7 = {W.abs().max():.2f} / 7 = {scale_naive:.3f}")
    print(f"  Quantization step size = {scale_naive:.3f}  "
          f"(that's HUGE relative to normal column std={W[:, :5].std():.2f}!)\n")

    # Show column-level error
    print("Per-column mean |error|  (first 8 columns):")
    for i in range(8):
        bar = "█" * int(err[:, i].mean() / scale_naive * 40)
        note = "  ← WEIGHT OUTLIER (dominates scale)" if i == 5 else ""
        print(f"  col {i:>3}: {err[:, i].mean():.4f}  {bar}{note}")

    # Output error
    Y_fp = X @ W.T
    Y_q  = X @ W_q.T
    output_err = (Y_fp - Y_q).abs().mean().item()
    print(f"\nOutput error (Y = X @ W^T):  {output_err:.4f}")
    print(f"Relative error:              {output_err / Y_fp.abs().mean() * 100:.1f}%")
    print("""
Conclusion:
  One large-weight column forces the global scale to be large.
  All normal columns lose precision because their values barely use any integer slots.
  This is the core limitation of per-tensor quantization.
""")
    return W, X, Y_fp


# ═══════════════════════════════════════════════════════════════
# STEP 2: Group-wise quantization already helps weight outliers
# ═══════════════════════════════════════════════════════════════

def step2_groupwise_fixes_weight_outliers():
    header(2, "Group-wise quantization: fix weight outliers",
              "逐组量化：已经能解决权重异常")

    W = make_weight_with_outlier_at_column()
    X = make_activations_with_high_channel()
    qmax = 7

    # Per-tensor naive
    scale_pt = W.abs().max() / qmax
    W_q_pt = (W / scale_pt).round().clamp(-qmax, qmax) * scale_pt
    err_pt = (W - W_q_pt).abs().mean().item()

    # Group-wise (group_size=32, so 4 groups across 128 cols)
    group_size = 32
    W_g = W.reshape(W.shape[0], -1, group_size)          # [16, 4, 32]
    scales = W_g.abs().amax(dim=-1, keepdim=True) / qmax  # [16, 4, 1]
    W_q_g  = (W_g / scales).round().clamp(-qmax, qmax) * scales
    W_q_gw = W_q_g.reshape(W.shape)
    err_gw = (W - W_q_gw).abs().mean().item()

    print(f"Per-tensor int4:  global scale = {scale_pt:.3f}  mean|err| = {err_pt:.4f}")
    print(f"Group-wise int4:  separate scale per {group_size} cols  mean|err| = {err_gw:.4f}")
    print(f"Group-wise improvement: {err_pt/err_gw:.1f}× more accurate on weight reconstruction\n")

    print("Output error:")
    Y_fp  = X @ W.T
    Y_pt  = X @ W_q_pt.T
    Y_gw  = X @ W_q_gw.T
    e_pt  = (Y_fp - Y_pt).abs().mean().item()
    e_gw  = (Y_fp - Y_gw).abs().mean().item()
    print(f"  Per-tensor int4:  {e_pt:.4f}  ({e_pt/Y_fp.abs().mean()*100:.1f}%)")
    print(f"  Group-wise int4:  {e_gw:.4f}  ({e_gw/Y_fp.abs().mean()*100:.1f}%)")
    print(f"  Group-wise improvement: {e_pt/e_gw:.1f}×\n")

    print("""But group-wise still has a gap! Look at activation channel 20:
  X[:, 20] has mean|x| ≈ large values. Even if W[:, 20] is normal (small weights),
  the output error from W[:, 20] is AMPLIFIED by the large activations.

  output_err_channel_20 ≈ δW[:,20] × x[:,20]  ← small_error × LARGE_activation

Group-wise fixes per-column scale, but doesn't know which columns MATTER MORE
due to large activations. That's exactly what AWQ adds.
""")
    return W, X, Y_fp


# ═══════════════════════════════════════════════════════════════
# STEP 3: The AWQ insight — activation-aware importance weighting
# ═══════════════════════════════════════════════════════════════

def step3_awq_insight():
    header(3, "The AWQ insight — activation-aware weight scaling",
              "AWQ 核心洞察：激活感知权重缩放")

    X = make_activations_with_high_channel()
    act_scale = X.abs().mean(dim=0)  # mean activation magnitude per channel

    print("Activation magnitude per input channel (mean |x[:,i]| across all tokens):")
    print("  Channel 20 has consistently large activations — it MATTERS MORE for output.\n")
    print(f"  Channels 0-4:  {[f'{act_scale[i]:.3f}' for i in range(5)]}")
    print(f"  Channels 18-22:{[f'{act_scale[i]:.3f}' for i in range(18, 23)]}")
    print(f"  → Channel 20 mean|x|: {act_scale[20]:.3f}  (10-15× larger than avg)\n")

    print("""Why does this matter for quantization?

  Output error from quantizing column i:
    δY[:,i] ≈ δW[:,i] × x[:,i]

  If x[:,i] is large (high activation), even a SMALL error δW[:,i] causes
  a LARGE output error. Column 20 is 10× more sensitive than column 0.

  Naive int4 + group-wise treats all columns equally within each group.
  AWQ gives more precision to high-activation columns.

AWQ solution — mathematical equivalence trick:
  Y = W @ x  =  (W·s) @ (x/s)     for any per-channel scale s

  Step 1: s[i] = mean(|x[:,i]|)^alpha   (activation-based importance)
  Step 2: W_scaled[:,i] = W[:,i] × s[i]  (amplify salient columns)
  Step 3: quantize(W_scaled)             (salient columns get more integer range)
  Step 4: at inference: x' = x / s, then Y = W_q_scaled @ x'

  The math cancels out: (W·s) @ (x/s) = W @ x  ✓
  The quantization is more accurate because salient columns are amplified.
""")

    alpha = 0.5
    s = act_scale.pow(alpha)
    print(f"Channel scales s = mean(|x|)^{alpha}:")
    print(f"  Normal channels (avg): {s[:5].mean():.3f}")
    print(f"  Channel 20 scale:      {s[20]:.3f}  ← amplified {s[20]/s[:5].mean():.1f}×")
    print("\n  After scaling W[:,20] by s[20], it occupies more of the integer range")
    print("  → finer quantization for this important column ✓")


# ═══════════════════════════════════════════════════════════════
# STEP 4: Side-by-side comparison — naive vs group-wise vs AWQ
# ═══════════════════════════════════════════════════════════════

def step4_compare_all_three():
    header(4, "Side-by-side: naive int4 vs group-wise vs AWQ",
              "三种方案对比：朴素 int4 vs 逐组 vs AWQ")

    from nanoptq.algorithms.rtn import quantize_linear_rtn
    from nanoptq.algorithms.awq_lite import quantize_linear_awq

    # Setup: weight outlier at col 5, activation outlier at col 20 (different columns!)
    W_mat = make_weight_with_outlier_at_column(out_features=16, in_features=128,
                                               outlier_col=5)
    X     = make_activations_with_high_channel(n_tokens=64, in_features=128,
                                               high_channel=20)

    linear = nn.Linear(128, 16, bias=False)
    linear.weight.data = W_mat.clone()

    x_test = torch.randn(8, 128)
    y_fp   = linear(x_test)

    # 1. Naive per-tensor int4 (group_size = whole row = 128, 1 group)
    ql_rtn_g128 = quantize_linear_rtn(linear, bits=4, group_size=128)
    y_rtn_g128  = ql_rtn_g128(x_test)

    # 2. Group-wise int4 (group_size=32, 4 groups per row)
    ql_rtn_g32 = quantize_linear_rtn(linear, bits=4, group_size=32)
    y_rtn_g32  = ql_rtn_g32(x_test)

    # 3. AWQ int4 (group_size=32, activation-aware scaling)
    ql_awq = quantize_linear_awq(linear, X, bits=4, group_size=32, alpha=0.5)
    y_awq  = ql_awq(x_test)

    e_naive = (y_fp - y_rtn_g128).abs().mean().item()
    e_gw    = (y_fp - y_rtn_g32).abs().mean().item()
    e_awq   = (y_fp - y_awq).abs().mean().item()
    ref     = y_fp.abs().mean().item()

    print(f"  {'Method':<25} {'Output error':>14} {'Relative %':>12} {'vs naive':>10}")
    print(f"  {SEP[:63]}")
    print(f"  {'RTN int4 (g=128, naive)':<25} {e_naive:>14.5f} {e_naive/ref*100:>11.1f}%  {'1.0×':>10}")
    print(f"  {'RTN int4 (g=32)':<25} {e_gw:>14.5f} {e_gw/ref*100:>11.1f}%  {e_naive/e_gw:>9.1f}×")
    print(f"  {'AWQ int4 (g=32)':<25} {e_awq:>14.5f} {e_awq/ref*100:>11.1f}%  {e_naive/e_awq:>9.1f}×")

    print(f"""
Key insight from this comparison:
  • Group-wise helps by isolating the weight outlier (col 5) in its own group.
  • AWQ further helps by protecting the high-activation channel (col 20)
    even though its weight magnitude is normal.

  The activation-aware scaling adds precision exactly where the output
  is most sensitive — proportional to mean activation magnitude.
""")


# ═══════════════════════════════════════════════════════════════
# STEP 5: AWQ mapped to nanoPTQ code
# ═══════════════════════════════════════════════════════════════

def step5_awq_in_code():
    header(5, "AWQ implementation — concept to code mapping",
              "AWQ 实现 — 概念到代码的一一对应")

    print("""nanoptq/algorithms/awq_lite.py  (read alongside this file)
─────────────────────────────────────────────────────────────

# Concept: s[i] = mean(|X[:,i]|)^alpha
def compute_activation_channel_scales(activations, alpha=0.5):
    act_scale = activations.float().abs().mean(dim=0).clamp(min=1e-8)
    return act_scale.pow(alpha)          # shape: [in_features]

# Concept: store (W * s) quantized, store s for inference
def quantize_linear_awq(linear, calibration_acts, bits=4, ...):
    s = compute_activation_channel_scales(calibration_acts)  # [in_features]

    W_scaled = linear.weight.float() * s.unsqueeze(0)        # W * s
    q, scales = group_quantize(W_scaled, group_size, bits)   # quantize(W*s)

    ql = QuantLinear.from_linear(linear, bits, group_size)
    ql.weight_q = q
    ql.scales   = scales
    ql.register_buffer("input_channel_scales", s.to(torch.float16))
    return ql

# Concept: at inference x' = x/s, then (W*s) @ x' = W @ x
class QuantLinear(nn.Module):
    def forward(self, x):
        if self.input_channel_scales is not None:
            x = x / self.input_channel_scales   # ← divide x by s
        W = self.dequantize()                    # reconstruct (W*s) as float
        return F.linear(x, W, self.bias)         # (W*s) @ (x/s) = W@x  ✓

Mathematical equivalence verification:
""")

    torch.manual_seed(0)
    W = torch.randn(4, 8)
    x = torch.randn(8)
    s = torch.tensor([1.0, 1.0, 3.5, 1.0, 1.0, 2.1, 1.0, 4.0])  # channel scales

    Y_original = W @ x
    Y_awq      = (W * s) @ (x / s)

    print(f"  Y_original  (W @ x):         {[f'{v:.4f}' for v in Y_original.tolist()]}")
    print(f"  Y_awq ((W*s) @ (x/s)):       {[f'{v:.4f}' for v in Y_awq.tolist()]}")
    print(f"  Max difference:              {(Y_original - Y_awq).abs().max().item():.2e}")
    print(f"  → Mathematically identical: (W*s)@(x/s) = W@x  ✓\n")

    print("""Key design decisions in AWQ:
  1. alpha=0.5: square root of activation scale — balanced amplification.
     alpha=1.0 over-protects salient channels; alpha=0.0 = no AWQ.
  2. ~128 calibration samples is enough — the activation statistics stabilize quickly.
  3. input_channel_scales is stored per-layer and survives save/load
     (this was a bug we fixed: the save_quantized_model now serializes it).
  4. The forward() division is fused into the existing matmul path — no extra kernel.
""")


# ═══════════════════════════════════════════════════════════════
# STEP 6: Full end-to-end with save/load verification
# ═══════════════════════════════════════════════════════════════

def step6_end_to_end():
    header(6, "End-to-end AWQ: quantize → save → load → verify",
              "端到端验证：量化 → 保存 → 加载 → 推理一致性检查")

    from nanoptq.algorithms.awq_lite import quantize_linear_awq
    from nanoptq.io.safetensors_io import save_quantized_model, load_quantized_model
    import tempfile

    torch.manual_seed(42)
    linear = nn.Linear(128, 64, bias=False)
    nn.init.normal_(linear.weight, std=0.3)
    linear.weight.data[:, 20] = torch.randn(64) * 6.0  # large weight column

    calibration_acts = torch.randn(64, 128).abs() * 0.5
    calibration_acts[:, 20] = torch.randn(64).abs() * 4.0  # large activation channel

    x_test = torch.randn(4, 128)
    y_fp   = linear(x_test)

    ql = quantize_linear_awq(linear, calibration_acts, bits=4, group_size=32)
    y_awq = ql(x_test)

    print(f"  input_channel_scales: shape={ql.input_channel_scales.shape}")
    print(f"  Channel 20 scale: {ql.input_channel_scales[20].item():.4f}  (should be > 1.0)")
    print(f"  Output error before save: {(y_fp - y_awq).abs().mean().item():.5f}\n")

    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Sequential(ql)
        # wrap bare QuantLinear in Sequential so save API works
        model_orig = nn.Sequential(linear)
        model_orig[0] = ql
        save_quantized_model(model_orig, tmp, bits=4, group_size=32, method='awq')

        model_loaded = nn.Sequential(nn.Linear(128, 64, bias=False))
        load_quantized_model(model_loaded, tmp)
        y_loaded = model_loaded(x_test)

        roundtrip_diff = (y_awq - y_loaded).abs().max().item()
        print(f"  After save/load — max diff: {roundtrip_diff:.2e}  (must be < 1e-3)")
        assert roundtrip_diff < 1e-3, "Save/load roundtrip failed!"

    print("  input_channel_scales preserved through save/load ✓")
    print("  Full AWQ pipeline verified ✓")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

STEPS = {
    1: step1_why_naive_int4_fails,
    2: step2_groupwise_fixes_weight_outliers,
    3: step3_awq_insight,
    4: step4_compare_all_three,
    5: step5_awq_in_code,
    6: step6_end_to_end,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWQ deep dive — step by step")
    parser.add_argument("--step", type=int, choices=list(STEPS),
                        help="Run only this step (default: all)")
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        for fn in STEPS.values():
            fn()

    print(f"\n{SEP}")
    print("Next steps:")
    print("  python examples/precision_tour.py    — bf16 / fp8 / nvfp4 explained")
    print("  python examples/compare_methods.py   — RTN vs AWQ vs GPTQ on a real model")
    print("  nanoptq/algorithms/awq_lite.py        — nanoPTQ AWQ source code")
    print(SEP)
