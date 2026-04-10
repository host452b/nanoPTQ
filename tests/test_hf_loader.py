import torch
import torch.nn as nn
import pytest
from nanoptq.model.hf_loader import (
    get_linear_layers,
    replace_linear_with_quant,
    count_parameters,
)
from nanoptq.model.quant_linear import QuantLinear


def make_toy_model():
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )


def test_get_linear_layers_finds_all():
    model = make_toy_model()
    linears = get_linear_layers(model)
    assert len(linears) == 2
    for name, mod in linears:
        assert isinstance(mod, nn.Linear)


def test_get_linear_layers_returns_names():
    model = make_toy_model()
    linears = get_linear_layers(model)
    names = [name for name, _ in linears]
    assert "0" in names
    assert "2" in names


def test_replace_linear_with_quant_swaps_layers():
    model = make_toy_model()
    replace_linear_with_quant(model, bits=8, group_size=64)
    for mod in model.modules():
        assert not isinstance(mod, nn.Linear)


def test_replace_linear_with_quant_installs_quant_linear():
    model = make_toy_model()
    replace_linear_with_quant(model, bits=4, group_size=64)
    quant_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, QuantLinear)]
    assert len(quant_layers) == 2


def test_replace_preserves_bias():
    model = make_toy_model()
    replace_linear_with_quant(model, bits=8, group_size=64)
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            assert mod.bias is not None


def test_count_parameters():
    model = make_toy_model()
    total = count_parameters(model)
    assert total == 64 * 128 + 128 + 128 * 64 + 64


def test_replace_does_not_replace_skipped_modules():
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    # Module "0" is skipped by name substring "0"
    replace_linear_with_quant(model, bits=8, group_size=64, skip_modules=["0"])
    # "0" should still be nn.Linear, "2" should be QuantLinear
    assert isinstance(model[0], nn.Linear), "skipped module should remain nn.Linear"
    assert isinstance(model[2], QuantLinear), "non-skipped module should be replaced"


def test_set_module_by_name_dotted_path():
    """_set_module_by_name must handle multi-level dotted paths."""
    from nanoptq.model.hf_loader import _set_module_by_name
    import torch.nn as nn

    class TwoLevel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(16, 16))

    model = TwoLevel()
    new_linear = nn.Linear(16, 8)
    _set_module_by_name(model, "encoder.0", new_linear)
    assert model.encoder[0] is new_linear
