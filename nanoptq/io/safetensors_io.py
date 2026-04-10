# nanoptq/io/safetensors_io.py
"""
Save and load quantized models using safetensors format.
使用 safetensors 格式保存和加载量化模型。

Why safetensors?
  - Safe: no arbitrary code execution unlike pickle/pt
  - Fast: zero-copy mmap loading
  - Standard: vLLM, HF Transformers, llama.cpp all speak it
"""
import json
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from nanoptq.model.quant_linear import QuantLinear
from nanoptq.model.hf_loader import _set_module_by_name


def save_quantized_model(
    model: nn.Module,
    output_dir: str | Path,
    bits: int,
    group_size: int,
    method: str = "rtn",
    symmetric: bool = True,
) -> None:
    """
    Serialize all QuantLinear buffers to a safetensors file.
    Also writes quant_config.json so the loader knows how to reconstruct.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            prefix = name.replace(".", "_")
            tensors[f"{prefix}.weight_q"] = mod.weight_q.cpu()
            tensors[f"{prefix}.scales"] = mod.scales.cpu()
            if mod.zero_points is not None:
                tensors[f"{prefix}.zero_points"] = mod.zero_points.cpu()
            if mod.bias is not None:
                tensors[f"{prefix}.bias"] = mod.bias.cpu()
            ics = getattr(mod, "input_channel_scales", None)
            if ics is not None:
                tensors[f"{prefix}.input_channel_scales"] = ics.cpu()

    save_file(tensors, output_dir / "model.safetensors")

    config = {
        "bits": bits,
        "group_size": group_size,
        "method": method,
        "symmetric": symmetric,
    }
    with open(output_dir / "quant_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_quantized_model(
    model: nn.Module,
    checkpoint_dir: str | Path,
) -> None:
    """
    Load a quantized checkpoint into a model (in-place).
    Replaces nn.Linear layers with QuantLinear and fills them from the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "quant_config.json", encoding="utf-8") as f:
        config = json.load(f)

    bits = config["bits"]
    group_size = config["group_size"]
    symmetric = config.get("symmetric", True)

    tensors = load_file(checkpoint_dir / "model.safetensors")

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        prefix = name.replace(".", "_")
        wq_key = f"{prefix}.weight_q"
        if wq_key not in tensors:
            continue
        ql = QuantLinear.from_linear(mod, bits=bits, group_size=group_size, symmetric=symmetric)
        ql.weight_q = tensors[wq_key]
        ql.scales = tensors[f"{prefix}.scales"]
        if f"{prefix}.zero_points" in tensors:
            ql.zero_points = tensors[f"{prefix}.zero_points"]
        if f"{prefix}.bias" in tensors:
            ql.bias = tensors[f"{prefix}.bias"]
        ics_key = f"{prefix}.input_channel_scales"
        if ics_key in tensors:
            ql.register_buffer("input_channel_scales", tensors[ics_key])
        _set_module_by_name(model, name, ql)
