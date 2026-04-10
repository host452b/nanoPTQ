# nanoptq/model/hf_loader.py
"""
HF model loading and Linear layer surgery.
HF 模型加载和 Linear 层替换工具。

This module answers: "how do we get a model, find all its Linear layers,
and swap them for QuantLinear shells ready to be filled by a PTQ algorithm?"
"""
import torch
import torch.nn as nn
from nanoptq.model.quant_linear import QuantLinear


def load_hf_model(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """
    Load a HuggingFace causal LM + tokenizer.
    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def get_linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """
    Walk the model and return all (name, nn.Linear) pairs.
    Only targets nn.Linear — embeddings, LayerNorm, etc. are intentionally skipped.
    """
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Navigate dotted name path and replace the leaf module."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def replace_linear_with_quant(
    model: nn.Module,
    bits: int,
    group_size: int,
    symmetric: bool = True,
    skip_modules: list[str] | None = None,
) -> None:
    """
    Replace all nn.Linear layers with QuantLinear shells (in-place).
    skip_modules: list of name substrings to leave untouched (e.g. ["lm_head"]).

    After this call, weight_q and scales are UNINITIALIZED ZEROS.
    A quantization algorithm must fill them.
    """
    skip_modules = skip_modules or []
    for name, linear in get_linear_layers(model):
        if any(skip in name for skip in skip_modules):
            continue
        ql = QuantLinear.from_linear(linear, bits=bits, group_size=group_size, symmetric=symmetric)
        _set_module_by_name(model, name, ql)


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())
