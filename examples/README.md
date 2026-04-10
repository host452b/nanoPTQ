# examples/

Runnable demos. Each file is standalone — no imports from other example files.

## Files

### `quant_model.py` — End-to-End Pipeline

Demonstrates the full workflow in one script:
1. Load FP16 model
2. Benchmark FP16 latency and PPL
3. Quantize with RTN (or AWQ/GPTQ via `--method`)
4. Benchmark quantized latency and PPL
5. Generate sample text to verify coherence

```bash
python examples/quant_model.py --model Qwen/Qwen2-0.5B --bits 4 --method rtn
```

### `compare_methods.py` — Side-by-Side Table

Runs RTN, AWQ, and GPTQ on the same model, prints a comparison table.
Shows PPL and memory for all three methods.

```bash
python examples/compare_methods.py --model Qwen/Qwen2-0.5B --bits 4
```

### `precision_tour.py` — Number Format Tutorial

7-section interactive tour of floating-point and integer number formats:
- Section 1: What is a number format?
- Section 2: bf16 vs fp16 (live overflow demo)
- Section 3: fp8 (e4m3, e5m2)
- Section 4: int4 group-wise (outlier visual)
- Section 5: nvfp4 e2m1 (Blackwell)
- Section 6: Hardware support map
- Section 7: Decision tree

```bash
python examples/precision_tour.py            # all sections
python examples/precision_tour.py --section 3  # just fp8
```

### `awq_explained.py` — AWQ Step-by-Step

6-step explanation of why AWQ works, with live demos:
- Step 1: Why naive per-tensor int4 fails (weight outlier dominates scale)
- Step 2: Group-wise already helps — but is incomplete
- Step 3: The AWQ insight — activation-aware importance
- Step 4: Side-by-side comparison (RTN vs group-wise vs AWQ)
- Step 5: Code mapping to `nanoptq/algorithms/awq_lite.py`
- Step 6: End-to-end save/load verification

```bash
python examples/awq_explained.py
```

## Design Philosophy

These examples are for learning, not production. They use concrete numbers, ASCII art,
and step-by-step print output to make the math tangible. Target audience: someone who
knows PyTorch but has never implemented quantization.
