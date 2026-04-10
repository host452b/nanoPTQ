import torch
import torch.nn as nn
import json
import pytest
from nanoptq.model.quant_linear import QuantLinear
from nanoptq.algorithms.rtn import quantize_model_rtn
from nanoptq.io.safetensors_io import save_quantized_model, load_quantized_model


def make_quantized_model():
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )
    quantize_model_rtn(model, bits=4, group_size=64)
    return model


def test_save_creates_files(tmp_path):
    model = make_quantized_model()
    save_quantized_model(model, tmp_path, bits=4, group_size=64)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "quant_config.json").exists()


def test_save_load_weight_q_matches(tmp_path):
    model = make_quantized_model()
    orig_wq = None
    for m in model.modules():
        if isinstance(m, QuantLinear):
            orig_wq = m.weight_q.clone()
            break

    save_quantized_model(model, tmp_path, bits=4, group_size=64)

    model2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
    load_quantized_model(model2, tmp_path)

    loaded_wq = None
    for m in model2.modules():
        if isinstance(m, QuantLinear):
            loaded_wq = m.weight_q.clone()
            break

    assert torch.equal(orig_wq, loaded_wq)


def test_quant_config_json_has_required_keys(tmp_path):
    model = make_quantized_model()
    save_quantized_model(model, tmp_path, bits=4, group_size=64)
    with open(tmp_path / "quant_config.json") as f:
        cfg = json.load(f)
    assert "bits" in cfg
    assert "group_size" in cfg
    assert "method" in cfg


def test_save_load_inference_close(tmp_path):
    model = make_quantized_model()
    x = torch.randn(2, 128)
    with torch.no_grad():
        y_before = model(x)

    save_quantized_model(model, tmp_path, bits=4, group_size=64)
    model2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
    load_quantized_model(model2, tmp_path)

    with torch.no_grad():
        y_after = model2(x)

    assert torch.allclose(y_before, y_after, atol=1e-4)
