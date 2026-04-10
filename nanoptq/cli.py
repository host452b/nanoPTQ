# nanoptq/cli.py
"""
nanoPTQ CLI: quantize, eval, compare.
三个命令，仅此而已。

nanoptq quantize --model <hf_id> --method rtn --bits 4 --group-size 128 --output ./out
nanoptq eval     --model ./out --metric ppl
nanoptq compare  --model <hf_id> --bits 4 --group-size 128
"""
import argparse
import sys
import torch


def cmd_quantize(args):
    from nanoptq.model.hf_loader import load_hf_model
    from nanoptq.io.safetensors_io import save_quantized_model

    print(f"Loading {args.model} ...")
    model, tokenizer = load_hf_model(args.model, device=args.device)

    if args.method == "rtn":
        from nanoptq.algorithms.rtn import quantize_model_rtn
        quantize_model_rtn(model, bits=args.bits, group_size=args.group_size,
                           skip_modules=["lm_head"])
    elif args.method == "awq":
        print("Collecting calibration activations for AWQ ...")
        calib_data = _collect_calibration_data(model, tokenizer, args)
        from nanoptq.algorithms.awq_lite import quantize_model_awq
        quantize_model_awq(model, calib_data, bits=args.bits, group_size=args.group_size,
                           skip_modules=["lm_head"])
    elif args.method == "gptq":
        print("GPTQ layer-wise quantization ...")
        _apply_gptq(model, tokenizer, args)
    else:
        print(f"Unknown method: {args.method}. Choose: rtn, awq, gptq")
        sys.exit(1)

    save_quantized_model(model, args.output, bits=args.bits,
                         group_size=args.group_size, method=args.method)
    print(f"Saved to {args.output}")
    tokenizer.save_pretrained(args.output)


def cmd_eval(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nanoptq.io.safetensors_io import load_quantized_model

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                  device_map=args.device)
    load_quantized_model(model, args.model)

    if args.metric == "ppl":
        from nanoptq.eval.ppl import evaluate_ppl_bundled
        ppl = evaluate_ppl_bundled(model, tokenizer, dataset=args.dataset, device=args.device)
        print(f"Perplexity ({args.dataset}): {ppl:.2f}")
    elif args.metric == "latency":
        from nanoptq.eval.latency import benchmark_latency
        result = benchmark_latency(model, tokenizer, device=args.device)
        print(f"Prefill: {result.prefill_ms:.1f} ms | "
              f"Decode: {result.decode_tps:.1f} tok/s | "
              f"Mem: {result.peak_mem_gb:.2f} GB")
    else:
        print(f"Unknown metric: {args.metric}. Choose from: ppl, latency")
        sys.exit(1)


def cmd_compare(args):
    import copy
    from nanoptq.model.hf_loader import load_hf_model
    from nanoptq.algorithms.rtn import quantize_model_rtn
    from nanoptq.eval.ppl import evaluate_ppl_bundled

    print(f"Loading {args.model} ...")
    model_fp16, tokenizer = load_hf_model(args.model, device=args.device)

    print(f"FP16 baseline PPL ({args.dataset}) ...")
    ppl_fp16 = evaluate_ppl_bundled(model_fp16, tokenizer, dataset=args.dataset, device=args.device)

    print("RTN quantization ...")
    model_rtn = copy.deepcopy(model_fp16)
    quantize_model_rtn(model_rtn, bits=args.bits, group_size=args.group_size,
                       skip_modules=["lm_head"])
    ppl_rtn = evaluate_ppl_bundled(model_rtn, tokenizer, dataset=args.dataset, device=args.device)

    print(f"\n{'Method':<20} {'PPL':>8} {'ΔPPL':>8}")
    print("-" * 40)
    print(f"{'fp16':<20} {ppl_fp16:>8.2f} {'—':>8}")
    delta = ppl_rtn - ppl_fp16
    delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
    print(f"{'rtn-int' + str(args.bits):<20} {ppl_rtn:>8.2f} {delta_str:>8}")


def _collect_calibration_data(model, tokenizer, args):
    """Collect per-layer activation statistics for AWQ/GPTQ calibration."""
    import torch.nn as nn
    from nanoptq.data.loader import load_calibration_texts

    dataset = getattr(args, "dataset", "wikitext2")
    texts = load_calibration_texts(dataset=dataset)[:32]
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
    device = next(model.parameters()).device
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)

    for h in handles:
        h.remove()

    return {name: torch.cat(acts, dim=0) for name, acts in activations.items()}


def _apply_gptq(model, tokenizer, args):
    """Apply GPTQ layer by layer using bundled calibration data."""
    import torch.nn as nn
    from nanoptq.algorithms.gptq_lite import quantize_linear_gptq
    from nanoptq.algorithms.rtn import quantize_linear_rtn
    from nanoptq.model.hf_loader import get_linear_layers, _set_module_by_name

    calib_data = _collect_calibration_data(model, tokenizer, args)

    for name, linear in get_linear_layers(model):
        if "lm_head" in name:
            continue
        if name in calib_data:
            ql = quantize_linear_gptq(linear, calib_data[name],
                                       bits=args.bits, group_size=args.group_size)
        else:
            ql = quantize_linear_rtn(linear, bits=args.bits, group_size=args.group_size)
        _set_module_by_name(model, name, ql)
        print(f"  quantized {name}")


def main():
    parser = argparse.ArgumentParser(
        prog="nanoptq",
        description="nanoPTQ — minimal PTQ toolkit / 极简后训练量化工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _DATASETS = ["wikitext2", "alpaca", "gsm8k", "humaneval", "qa", "sharegpt", "sum"]

    p_quant = sub.add_parser("quantize", help="Quantize a HuggingFace model")
    p_quant.add_argument("--model", required=True, help="HF model ID or local path")
    p_quant.add_argument("--method", default="rtn", choices=["rtn", "awq", "gptq"])
    p_quant.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p_quant.add_argument("--group-size", type=int, default=128, dest="group_size")
    p_quant.add_argument("--output", required=True, help="Output directory")
    p_quant.add_argument("--calib-dataset", default="wikitext2", choices=_DATASETS,
                         dest="dataset", help="Calibration dataset for AWQ/GPTQ")
    p_quant.add_argument("--device", default="cuda", help="cuda or cpu")

    p_eval = sub.add_parser("eval", help="Evaluate a quantized model")
    p_eval.add_argument("--model", required=True, help="Path to quantized model dir")
    p_eval.add_argument("--metric", default="ppl", choices=["ppl", "latency"])
    p_eval.add_argument("--dataset", default="wikitext2", choices=_DATASETS,
                        help="Eval dataset for PPL (default: wikitext2)")
    p_eval.add_argument("--device", default="cuda", help="cuda or cpu")

    p_cmp = sub.add_parser("compare", help="Compare FP16 vs RTN quantized PPL")
    p_cmp.add_argument("--model", required=True)
    p_cmp.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p_cmp.add_argument("--group-size", type=int, default=128, dest="group_size")
    p_cmp.add_argument("--dataset", default="wikitext2", choices=_DATASETS,
                       help="Eval dataset for PPL (default: wikitext2)")
    p_cmp.add_argument("--device", default="cuda", help="cuda or cpu")

    args = parser.parse_args()
    dispatch = {"quantize": cmd_quantize, "eval": cmd_eval, "compare": cmd_compare}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
