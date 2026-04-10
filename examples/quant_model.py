#!/usr/bin/env python3
# examples/quant_model.py
"""
End-to-end: load → quantize → eval → generate.
端到端示例：加载 → 量化 → 评测 → 生成。

Run:
  python examples/quant_model.py --model Qwen/Qwen2-0.5B --method rtn --bits 4

This script shows the complete nanoPTQ workflow in ~50 lines.
"""
import argparse
import torch
from nanoptq.model.hf_loader import load_hf_model
from nanoptq.algorithms.rtn import quantize_model_rtn
from nanoptq.io.safetensors_io import save_quantized_model
from nanoptq.eval.ppl import evaluate_ppl_wikitext
from nanoptq.eval.latency import benchmark_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--method", default="rtn", choices=["rtn", "awq", "gptq"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128, dest="group_size")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[1/4] Loading {args.model} ...")
    model, tokenizer = load_hf_model(args.model, device=args.device)

    print(f"[2/4] FP16 baseline evaluation ...")
    ppl_fp16 = evaluate_ppl_wikitext(model, tokenizer, device=args.device)
    lat_fp16 = benchmark_latency(model, tokenizer, device=args.device)
    print(f"  FP16 PPL: {ppl_fp16:.2f} | {lat_fp16.decode_tps:.1f} tok/s | {lat_fp16.peak_mem_gb:.2f} GB")

    print(f"[3/4] RTN int{args.bits} quantization (group={args.group_size}) ...")
    quantize_model_rtn(model, bits=args.bits, group_size=args.group_size,
                       skip_modules=["lm_head"])

    print(f"[4/4] Quantized evaluation ...")
    ppl_q = evaluate_ppl_wikitext(model, tokenizer, device=args.device)
    lat_q = benchmark_latency(model, tokenizer, device=args.device)
    print(f"  Quant PPL: {ppl_q:.2f} | {lat_q.decode_tps:.1f} tok/s | {lat_q.peak_mem_gb:.2f} GB")

    save_quantized_model(model, args.output, bits=args.bits,
                         group_size=args.group_size, method=args.method)
    tokenizer.save_pretrained(args.output)
    print(f"  Saved to {args.output}")

    print("\nGeneration sample:")
    enc = tokenizer("The meaning of quantization is", return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=50, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
