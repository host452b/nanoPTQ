#!/usr/bin/env python3
# examples/compare_methods.py
"""
Compare RTN vs AWQ-lite vs GPTQ-lite side by side.
并排比较三种量化方法的困惑度。

Run:
  python examples/compare_methods.py --model Qwen/Qwen2-0.5B --bits 4

Reads bundled wikitext-2 data — no internet needed after prepare_data.py.
"""
import argparse
import copy
import torch
import torch.nn as nn
import sys

from nanoptq.model.hf_loader import load_hf_model, get_linear_layers, _set_module_by_name
from nanoptq.algorithms.rtn import quantize_model_rtn
from nanoptq.eval.ppl import evaluate_ppl_bundled


def collect_calibration_data(model, tokenizer, n_texts=32, max_length=512, device="cuda"):
    """Collect per-layer activation stats for AWQ/GPTQ."""
    from nanoptq.data.loader import load_calibration_texts
    texts = load_calibration_texts()[:n_texts]
    activations = {}

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach().cpu()
            activations.setdefault(name, []).append(x.reshape(-1, x.shape[-1]))
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            handles.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)

    for h in handles:
        h.remove()

    return {name: torch.cat(acts, dim=0) for name, acts in activations.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128, dest="group_size")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model_fp16, tokenizer = load_hf_model(args.model, device=args.device)
    ppl_fp16 = evaluate_ppl_bundled(model_fp16, tokenizer, device=args.device)

    results = [("fp16 (baseline)", ppl_fp16)]

    print("Collecting calibration data ...")
    calib = collect_calibration_data(model_fp16, tokenizer, device=args.device)

    print("RTN ...")
    m_rtn = copy.deepcopy(model_fp16)
    quantize_model_rtn(m_rtn, bits=args.bits, group_size=args.group_size,
                       skip_modules=["lm_head"])
    results.append((f"RTN int{args.bits}", evaluate_ppl_bundled(m_rtn, tokenizer, device=args.device)))

    print("AWQ-lite ...")
    from nanoptq.algorithms.awq_lite import quantize_model_awq
    m_awq = copy.deepcopy(model_fp16)
    quantize_model_awq(m_awq, calib, bits=args.bits, group_size=args.group_size,
                       skip_modules=["lm_head"])
    results.append((f"AWQ-lite int{args.bits}", evaluate_ppl_bundled(m_awq, tokenizer, device=args.device)))

    print("GPTQ-lite ...")
    from nanoptq.algorithms.gptq_lite import quantize_linear_gptq
    from nanoptq.algorithms.rtn import quantize_linear_rtn
    m_gptq = copy.deepcopy(model_fp16)
    for name, linear in get_linear_layers(m_gptq):
        if "lm_head" in name:
            continue
        if name in calib:
            ql = quantize_linear_gptq(linear, calib[name], bits=args.bits, group_size=args.group_size)
        else:
            ql = quantize_linear_rtn(linear, bits=args.bits, group_size=args.group_size)
        _set_module_by_name(m_gptq, name, ql)
    results.append((f"GPTQ-lite int{args.bits}", evaluate_ppl_bundled(m_gptq, tokenizer, device=args.device)))

    print(f"\n{'Method':<25} {'PPL':>8} {'ΔPPL':>8}")
    print("-" * 45)
    for name, ppl in results:
        delta = ppl - ppl_fp16
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        print(f"{name:<25} {ppl:>8.2f} {delta_str:>8}")


if __name__ == "__main__":
    main()
