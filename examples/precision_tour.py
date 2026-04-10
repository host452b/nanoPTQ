#!/usr/bin/env python3
# examples/precision_tour.py
"""
精度格式全景导览 — 从 bf16 到 nvfp4，量化领域基础设施认知普及。
A tour of precision formats used in LLM inference — from bf16 to nvfp4.

目标受众：对量化感兴趣、但还没有深入了解硬件精度格式的学生和研究者。
Audience: students and researchers curious about quantization, no hardware background needed.

运行方式 / Run:
  python examples/precision_tour.py
  python examples/precision_tour.py --section 3   # 只跑某一节

无需 GPU，无需下载模型。
No GPU, no model download needed.
"""
import argparse
import math
import torch

SEP = "─" * 64


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def header(n, title_en, title_zh):
    print(f"\n{'═'*64}")
    print(f"  SECTION {n} — {title_en}")
    print(f"  第{n}节   — {title_zh}")
    print(f"{'═'*64}\n")


def show_number_on_ruler(value: float, slots: list, label: str):
    """Visualise where a number falls among discrete quantization slots."""
    nearest = min(slots, key=lambda s: abs(s - value))
    error = value - nearest
    print(f"  Real value: {value:+.3f}")
    print(f"  Slots: {[f'{s:.1f}' for s in slots]}")
    print(f"  Nearest slot ({label}): {nearest:+.3f}  (error: {error:+.4f})")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — The core idea: what IS a number format?
# ═══════════════════════════════════════════════════════════════

def section1_what_is_a_number_format():
    header(1, "What IS a number format?", "数字格式到底是什么？")

    print("""类比 / Analogy — 把数字写在纸上
─────────────────────────────────────────────────────────────
想象你要把数字写在一张纸上，纸的大小决定了你能写多少位数字。

  32位纸 (fp32) → "3.14159265"  ← 很精确
  16位纸 (fp16) → "3.14159"     ← 稍差一点
   8位纸 (int8) → "3"           ← 只能是整数！
   4位纸 (int4) → 只能是 -8 到 7 之间的整数

Imagine writing a number on paper. The paper size = how many digits you can write.
  32-bit paper (fp32) → "3.14159265"  precise
  16-bit paper (fp16) → "3.14159"     slightly less
   8-bit paper (int8) → "3"           integers only!
   4-bit paper (int4) → only -8 to 7

关键问题：LLM 有几十亿个权重，每个数字少用几个比特，就能少用几 GB 显存。
Key point: LLMs have billions of weights. Fewer bits per number = fewer GB of VRAM.
""")

    print("实际比较 / Concrete comparison — 存储 1B 参数模型需要多少显存？")
    print(f"  {'Format':<12} {'Bits/param':>12} {'1B model (GB)':>14} {'4B model (GB)':>14}")
    print(f"  {SEP[:54]}")
    formats = [
        ("fp32",       32, "training baseline"),
        ("bf16/fp16",  16, "inference standard"),
        ("fp8",         8, "H100 inference"),
        ("int8",        8, "T4/A100 serving"),
        ("int4",        4, "AWQ/GPTQ target"),
        ("nvfp4",       4, "B200 Blackwell"),
    ]
    for name, bits, note in formats:
        gb_1b = 1e9 * bits / 8 / 1e9
        gb_4b = 4e9 * bits / 8 / 1e9
        print(f"  {name:<12} {bits:>12} {gb_1b:>13.1f} {gb_4b:>13.1f}  ← {note}")

    print("""
→ int4 比 fp32 节省 8 倍显存。一张 80GB A100 可以放下 fp16 的 40B 模型，
  或者放下 int4 的 160B 模型。这就是量化的核心价值。

→ int4 is 8× smaller than fp32. An 80GB A100 fits a 40B fp16 model,
  OR a 160B int4 model. That's why quantization matters.
""")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — bf16 vs fp16: the sibling formats
# ═══════════════════════════════════════════════════════════════

def section2_bf16_vs_fp16():
    header(2, "bf16 vs fp16 — The Sibling Formats", "bf16 与 fp16 — 亲兄弟有啥不同？")

    print("""比特布局 / Bit layout:
  fp32: [S·1][Exponent·8][Mantissa·23]  → range: ±3.4×10³⁸,  precision: 1/8M
  bf16: [S·1][Exponent·8][Mantissa·7]   → range: ±3.4×10³⁸,  precision: 1/128
  fp16: [S·1][Exponent·5][Mantissa·10]  → range: ±65504,      precision: 1/1024

  bf16 = "brain float 16" — Google Brain team, 2018
  bf16 把 fp32 直接截断后 16 位，保留了完整的 8 位指数。
  bf16 simply truncates fp32 to 16 bits, keeping the full 8-bit exponent.

类比 / Analogy:
  fp32: "3.14159265"   (10 decimal digits)
  bf16: "3.1"          (2 decimal digits, same magnitude range)
  fp16: "3.141"        (4 decimal digits, but max ~65000 not billions!)
""")

    print("实际演示 / Live demo — 把同一个数存进不同格式，看看损失了多少：")
    test_values = [3.14159, 65000.0, 0.0001, -1234.56, 70000.0]
    print(f"\n  {'Value':>12} {'fp32 (ref)':>14} {'bf16':>10} {'fp16':>10} {'fp16 overflow':>14}")
    print(f"  {SEP[:60]}")
    for v in test_values:
        t = torch.tensor(v)
        bf16_v = t.to(torch.bfloat16).float().item()
        try:
            fp16_v = t.to(torch.float16).float().item()
            overflow = "⚠️ inf!" if math.isinf(fp16_v) else ""
        except Exception:
            fp16_v = float('inf')
            overflow = "⚠️ inf!"
        print(f"  {v:>12.4f} {v:>14.4f} {bf16_v:>10.4f} {fp16_v:>10.4f}  {overflow}")

    print("""
→ 70000 超出 fp16 范围（max=65504），变成 inf，导致训练崩溃！
  这就是为什么现代 LLM 训练从 fp16 换到了 bf16。
→ 70000 exceeds fp16 range (max=65504) → inf → training explodes!
  This is why modern LLM training switched from fp16 to bf16.

→ bf16 精度比 fp16 低（尾数少 3 位），但对权重量化来说够用。
  bf16 is less precise than fp16 (3 fewer mantissa bits), but fine for weights.
""")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — fp8: the new inference format
# ═══════════════════════════════════════════════════════════════

def section3_fp8():
    header(3, "fp8 — The H100 Inference Format", "fp8 — H100 的推理利器")

    print("""背景 / Background:
  NVIDIA H100 (Hopper 架构, 2023) 引入了原生 fp8 张量核心。
  NVIDIA H100 (Hopper arch, 2023) introduced native fp8 tensor cores.
  这意味着 H100 能直接做 fp8 × fp8 矩阵乘法，不需要先转换格式。
  This means H100 can do fp8 × fp8 matmul natively — no format conversion.

两种 fp8 / Two fp8 variants:
  fp8 e4m3: [S·1][E·4][M·3] — max=448   — for ACTIVATIONS (推理时激活值)
  fp8 e5m2: [S·1][E·5][M·2] — max=57344 — for GRADIENTS  (训练时梯度)

  为什么要两种？/ Why two variants?
  激活值（如 attention scores）分布集中，小范围就够 → e4m3
  梯度分布很宽，需要更大的动态范围 → e5m2

  Activations (e.g., attention scores) are concentrated → e4m3 (smaller range)
  Gradients can be huge → e5m2 (wider range)
""")

    print("fp8 e4m3 的所有正数值 / All positive fp8 e4m3 values (there are only 2^7=128):")
    print("  用 4 位指数 + 3 位尾数构造出的正数 (前 32 个) /")
    print("  Numbers constructable with 4-bit exponent + 3-bit mantissa (first 32):\n")

    # Generate fp8 e4m3 values (simplified: exponent bias=7, special: e=1111 → NaN)
    vals = []
    for e in range(1, 15):   # normal values (e=0 → subnormal, e=15 → NaN)
        for m in range(8):
            v = (1 + m / 8) * (2 ** (e - 7))
            vals.append(round(v, 5))
    vals = sorted(set(vals))[:32]

    # Print in rows of 8
    for i in range(0, len(vals), 8):
        row = vals[i:i+8]
        print("  " + "  ".join(f"{v:7.4f}" for v in row))

    print(f"\n  fp8 e4m3 最大值 = {max(vals):.0f}  (对比 fp16 最大值 65504)")
    print(f"  fp8 e4m3 max = {max(vals):.0f}  (vs fp16 max 65504)\n")

    print("量化误差演示 / Quantization error demo:")
    torch.manual_seed(0)
    W = torch.randn(32, 32)  # typical weight block

    # Simulate fp8 e4m3 quantization
    scale = 448.0 / W.abs().max().clamp(min=1e-8)
    W_scaled = (W * scale).clamp(-448, 448)
    W_fp8 = (W_scaled * 8).round() / 8 / scale  # precision = 1/8

    err = (W - W_fp8).abs()
    print(f"  Weight stats:   mean={W.mean():.3f}  std={W.std():.3f}  max={W.abs().max():.3f}")
    print(f"  fp8 e4m3 error: mean={err.mean():.5f}  max={err.max():.5f}")
    print(f"  Signal-to-noise: {10*math.log10((W.pow(2).mean()/(err.pow(2).mean()+1e-12)).item()):.1f} dB\n")

    print("""部署路径 / Deployment:
  H100 上用 fp8:
    TensorRT-LLM → W8A8-fp8 (自动校准 activation scale)
    Transformer Engine (te.Linear) → fp8 training
    vLLM 0.3+ → 支持 fp8 模型加载

  H100 fp8 deployment:
    TensorRT-LLM → W8A8-fp8 (auto-calibrates activation scales)
    Transformer Engine → fp8 training
    vLLM 0.3+ → supports fp8 model loading
""")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — int4 group-wise: the LLM compression workhorse
# ═══════════════════════════════════════════════════════════════

def section4_int4_groupwise():
    header(4, "int4 Group-wise — The LLM Compression Workhorse",
              "int4 逐组量化 — LLM 压缩的主力军")

    print("""int4 有什么问题？/ What's the problem with naive int4?
  int4 只有 16 个整数值: -8, -7, ..., 0, ..., 6, 7
  Int4 has only 16 integer values: -8, -7, ..., 0, ..., 6, 7

类比 / Analogy — 把一张有大有小数字的列表量化到 16 个值：
  原始数据: [0.001, 0.002, ..., 100.0, 200.0]
  问题: 如果 scale 由最大值决定 (200/7 ≈ 28.5)
        小数 (0.001) 全都四舍五入成 0，精度全损！

  Original: [0.001, 0.002, ..., 100.0, 200.0]
  Problem: if scale = max/7 = 200/7 ≈ 28.5,
           small values (0.001) all round to 0 — precision destroyed!

解决方案：逐组量化 / Solution: group-wise quantization
  把权重矩阵切成小块（每块 128 个数字），每块单独计算 scale。
  Split the weight matrix into blocks (128 numbers each), compute separate scale per block.
  这样每个小块内部的动态范围都被完整保留。
  This preserves full dynamic range within each block.
""")

    # Visual demo: per-tensor vs group-wise
    torch.manual_seed(1)
    # Simulate a weight row with one outlier channel
    W_row = torch.randn(256)
    W_row[128] = 15.0  # outlier — one large value

    print("可视演示 / Visual demo — 一行权重，第128个元素是异常值 15.0")
    print(f"  Weight row: 256 values, mostly N(0,1), but W[128] = 15.0 (outlier)\n")

    # Per-tensor int4
    qmax = 7
    scale_pt = W_row.abs().max() / qmax  # dominated by outlier
    W_pt = (W_row / scale_pt).round().clamp(-qmax, qmax) * scale_pt
    err_pt = (W_row - W_pt).abs()

    # Group-wise int4 (group_size=128)
    W_grouped = W_row.reshape(2, 128)
    scale_gw = W_grouped.abs().amax(dim=1, keepdim=True) / qmax
    W_gw = ((W_grouped / scale_gw).round().clamp(-qmax, qmax) * scale_gw).reshape(256)
    err_gw = (W_row - W_gw).abs()

    print(f"  Per-tensor int4:  scale={scale_pt:.3f}  mean_err={err_pt.mean():.4f}  "
          f"max_err={err_pt.max():.4f}")
    print(f"  Group-wise int4:  scales=[{scale_gw[0,0]:.3f}, {scale_gw[1,0]:.3f}]  "
          f"mean_err={err_gw.mean():.4f}  max_err={err_gw.max():.4f}")

    improvement = err_pt.mean() / err_gw.mean()
    print(f"\n  Group-wise is {improvement:.1f}× more accurate on this example!")
    print(f"  逐组量化在这个例子中精度提升了 {improvement:.1f} 倍！")

    print(f"""
Group-size 的选择 / Choosing group_size:
  group_size=32  → 更精确，但 scale 开销更大 (约 4.5 bits/weight)
  group_size=128 → 工业共识，AWQ/GPTQ/torchao 默认值 (约 4.125 bits/weight)
  group_size=256 → 更低开销，精度稍差

  Smaller group = more accurate, more scale overhead
  group_size=128 is the industry consensus (AWQ, GPTQ, torchao all default to it)
""")


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — nvfp4: NVIDIA Blackwell's extreme format
# ═══════════════════════════════════════════════════════════════

def section5_nvfp4():
    header(5, "nvfp4 (e2m1) — NVIDIA Blackwell's Extreme Format",
              "nvfp4 — NVIDIA Blackwell 的极限格式（4位浮点）")

    print("""背景 / Background:
  NVIDIA Blackwell GPU（B100、B200，2025年）引入了原生 nvfp4 张量核心。
  NVIDIA Blackwell GPUs (B100, B200, 2025) introduced native nvfp4 tensor cores.
  nvfp4 格式：e2m1，即 1位符号 + 2位指数 + 1位尾数。
  nvfp4 format: e2m1 — 1 sign bit + 2 exponent bits + 1 mantissa bit.

所有可能的值 / ALL possible nvfp4 values (there are only 16 total!):
""")

    # e2m1 values: exponent bias = 1
    # e=00: subnormal → m/2 * 2^(1-1) = m * 0.5 → values: 0.0, 0.5
    # e=01: (1 + m/2) * 2^(1-1) = 1.0 or 1.5
    # e=10: (1 + m/2) * 2^(2-1) = 2.0 or 3.0
    # e=11: (1 + m/2) * 2^(3-1) = 4.0 or 6.0
    pos_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    all_vals = sorted([-v for v in pos_vals if v > 0] + pos_vals)

    print(f"  Positive values (8 total): {pos_vals}")
    print(f"  All values (16 total):     {all_vals}\n")

    print("类比 / Analogy — 一把只有 8 个刻度的尺子：")
    print("  普通尺子 (fp16): 尺子上有 65536 个刻度，几乎连续")
    print("  nvfp4 尺子:      只有 8 个正数刻度: 0, 0.5, 1, 1.5, 2, 3, 4, 6")
    print()
    print("  Normal ruler (fp16): 65536 marks, nearly continuous")
    print("  nvfp4 ruler:         only 8 positive marks: 0, 0.5, 1, 1.5, 2, 3, 4, 6")
    print()
    print("  注意：刻度不是均匀的！靠近0的地方密，大数字之间间隔大。")
    print("  Note: marks are NOT uniform! Dense near 0, sparse for large values.")
    print("  这就是'浮点'的本质：相对精度恒定，绝对精度随数值变化。")
    print("  This is 'floating point': constant relative precision, varying absolute.")
    print()

    # Visual ruler
    print("  可视化刻度 / Visualize the ruler:")
    ruler = " " * 60
    ruler = list(ruler)
    max_val = 6.0
    width = 58
    for v in pos_vals:
        pos = int(v / max_val * width)
        ruler[pos] = "|"
    print("  0" + "".join(ruler) + "6")
    print("  " + "↑".join(["    "] * 9))
    print("  └─ nvfp4 slots: " + " ".join(f"{v}" for v in pos_vals))

    print()
    print("实际演示：把一组随机权重映射到 nvfp4 / Live demo: map random weights to nvfp4:")
    torch.manual_seed(42)
    sample = torch.randn(8)
    fp4_tensor = torch.tensor(pos_vals)
    scale = sample.abs().max() / 6.0

    print(f"\n  Scale factor (max_abs / 6.0): {scale:.4f}")
    print(f"\n  {'Original':>10} {'Scaled':>10} {'Nearest nvfp4':>14} {'Error':>10}")
    print(f"  {SEP[:46]}")
    for v in sample:
        scaled = v.item() / scale.item()
        # Find nearest fp4 value (with sign)
        all_fp4 = torch.tensor(all_vals)
        nearest_fp4 = all_fp4[(all_fp4 - scaled).abs().argmin()].item()
        reconstructed = nearest_fp4 * scale.item()
        error = v.item() - reconstructed
        print(f"  {v.item():>10.4f} {scaled:>10.4f} {nearest_fp4:>14.4f} {error:>10.4f}")

    print(f"""
为什么 nvfp4 比 int4 好？/ Why nvfp4 over int4?
  int4: 值均匀分布 -8,-7,...,0,...,7 (需要 scale 把它映射到浮点范围)
  nvfp4: 值本身就是浮点分布，天然适配权重的统计分布（接近正态分布）

  int4: uniform values -8..7 (needs scale to map to float range)
  nvfp4: values themselves are float-distributed, naturally fits weight statistics

  nvfp4 在 Blackwell 上有原生硬件支持：W4A8 矩阵乘法无需先反量化。
  nvfp4 has native hardware support on Blackwell: W4A8 matmul without dequant.

局限 / Limitation:
  只有 8 个正值，必须配合非常小的 group_size（通常 16）或 MX（Microscaling）才能有足够精度。
  Only 8 positive values — must use very small group_size (typically 16) or MX (Microscaling).
  TensorRT-LLM 用 MX 格式：每 16 个权重共享一个 fp8 scale。
  TensorRT-LLM uses MX format: 16 weights share one fp8 scale.
""")


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — Hardware map
# ═══════════════════════════════════════════════════════════════

def section6_hardware_map():
    header(6, "Hardware Map — Which Format Runs Where",
              "硬件支持矩阵 — 哪个格式在哪跑")

    print(f"""
  {'Format':<14} {'GPU 支持':<28} {'用途':<30}
  {SEP}
  {'fp32':<14} {'所有 GPU':<28} {'训练基线':<30}
  {'bf16':<14} {'Ampere+ (A100, 3090...)':<28} {'训练/推理主流':<30}
  {'fp16':<14} {'Pascal+ (V100, 2080...)':<28} {'推理 (2023年前主流)':<30}
  {'fp8 e4m3':<14} {'Hopper+ (H100)':<28} {'W8A8 推理':<30}
  {'fp8 e5m2':<14} {'Hopper+ (H100)':<28} {'fp8 训练 (梯度)':<30}
  {'int8':<14} {'Turing+ (T4, 2080Ti...)':<28} {'W8A8 serving':<30}
  {'int4':<14} {'软件模拟 (任意 GPU)':<28} {'W4A16 压缩 (AWQ/GPTQ)':<30}
  {'nvfp4':<14} {'Blackwell (B100/B200)':<28} {'W4A8 极限压缩':<30}
  {'GGUF Q4_K_M':<14} {'CPU (任意 x86/ARM)':<28} {'llama.cpp 本地推理':<30}

注：int4 在 GPU 上没有原生矩阵乘法核，必须先反量化到 fp16 再做 matmul。
Note: int4 has no native GPU matmul kernel — must dequant to fp16 first.
nanoPTQ 就是这样做的：存 int4，forward() 里动态反量化为 fp16。
nanoPTQ does exactly this: store int4, dequant to fp16 in forward().

速度排名 (单位时间吞吐量，同型号 GPU) / Speed ranking (throughput, same GPU):
  nvfp4 W4A8 (B200) > fp8 W8A8 (H100) > int4 W4A16 > int8 W8A8 > bf16

显存排名 (越小越好) / Memory ranking (smaller = better):
  nvfp4 ≈ int4 (4bit) < int8 ≈ fp8 (8bit) < bf16 ≈ fp16 (16bit) < fp32 (32bit)
""")


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — When to use what (decision tree)
# ═══════════════════════════════════════════════════════════════

def section7_decision_tree():
    header(7, "When to Use What — Decision Tree",
              "我该用哪种精度？决策树")

    print("""
  你有什么 GPU？
  What GPU do you have?
  │
  ├── B100 / B200 (Blackwell)
  │     → nvfp4 W4A8 (TensorRT-LLM 2.x)        极限压缩，最高吞吐
  │
  ├── H100 / H800 (Hopper)
  │     → fp8 W8A8 (TensorRT-LLM / TE)           质量↑ 速度↑
  │       如果不想碰 fp8：int4 AWQ/GPTQ 也行
  │
  ├── A100 / A10G (Ampere)
  │     → bf16 serving (如果显存够)
  │       int4 AWQ/GPTQ (如果显存紧张)
  │
  ├── T4 / 3090 / 4090 (Turing/Ampere consumer)
  │     → int4 AWQ (vLLM 原生支持，最实用)
  │       int8 bitsandbytes (简单但稍慢)
  │
  └── 只有 CPU
        → GGUF Q4_K_M (llama.cpp，最成熟的 CPU 量化)
          int8 bitsandbytes (transformers 直接支持)

质量排名 (越左越好) / Quality ranking (left = better):
  fp32 > bf16 ≈ fp16 > fp8 e4m3 > int8 > int4+AWQ ≈ int4+GPTQ >> int4+RTN

实用建议 2025 / Practical picks 2025:
  消费级显卡      → int4 AWQ    (vLLM + AutoAWQ)
  A100 数据中心   → bf16 or int8
  H100 数据中心   → fp8 W8A8   (TensorRT-LLM)
  B200 前沿部署   → nvfp4 W4A8 (TensorRT-LLM 2.x)
  离线 / 本地 CPU → Q4_K_M     (llama.cpp / Ollama)
""")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

SECTIONS = {
    1: section1_what_is_a_number_format,
    2: section2_bf16_vs_fp16,
    3: section3_fp8,
    4: section4_int4_groupwise,
    5: section5_nvfp4,
    6: section6_hardware_map,
    7: section7_decision_tree,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, choices=list(SECTIONS),
                        help="Run only this section (default: all)")
    args = parser.parse_args()

    if args.section:
        SECTIONS[args.section]()
    else:
        for fn in SECTIONS.values():
            fn()

    print(f"\n{SEP}")
    print("下一步 / Next steps:")
    print("  python examples/awq_explained.py     — AWQ 原理精讲")
    print("  python examples/compare_methods.py   — RTN vs AWQ vs GPTQ 实测对比")
    print(SEP)
