import torch
import torch.nn as nn
import json
import pytest
from nanoptq.model.quant_linear import QuantLinear
from nanoptq.algorithms.rtn import quantize_model_rtn
from nanoptq.algorithms.awq_lite import quantize_linear_awq
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


def test_awq_save_load_roundtrip_identical_output(tmp_path):
    """AWQ QuantLinear must produce identical output before and after save+load.

    Regression test for the bug where input_channel_scales was not persisted:
    after loading the checkpoint, forward() skipped the x/s pre-scaling step,
    causing wrong results (max error ~0.76 for layers with outlier channels).
    """
    torch.manual_seed(0)
    in_features, out_features, group_size = 128, 64, 128

    # Build a small model with one AWQ-quantized linear layer
    linear = nn.Linear(in_features, out_features)
    nn.init.normal_(linear.weight)

    # Calibration activations with a strong outlier channel to make scales non-trivial
    acts = torch.randn(32, in_features)
    acts[:, 7] *= 50.0   # outlier channel

    ql_before = quantize_linear_awq(linear, acts, bits=4, group_size=group_size, alpha=0.5)

    # Verify input_channel_scales was actually set (pre-condition)
    assert hasattr(ql_before, "input_channel_scales")
    assert ql_before.input_channel_scales is not None

    # Wrap in a Sequential so save/load helpers can find it by name
    model_before = nn.Sequential(ql_before)

    x = torch.randn(4, in_features)
    with torch.no_grad():
        y_before = model_before(x)

    # Save
    save_quantized_model(model_before, tmp_path, bits=4, group_size=group_size, method="awq")

    # Load into a fresh model with plain nn.Linear
    model_after = nn.Sequential(nn.Linear(in_features, out_features))
    load_quantized_model(model_after, tmp_path)

    # The loaded layer must have input_channel_scales restored
    loaded_ql = model_after[0]
    assert isinstance(loaded_ql, QuantLinear)
    assert hasattr(loaded_ql, "input_channel_scales")
    assert loaded_ql.input_channel_scales is not None

    # Outputs must be bit-exact (same weights, same scales, same pre-scaling)
    with torch.no_grad():
        y_after = model_after(x)

    assert torch.allclose(y_before, y_after, atol=1e-4), (
        f"Max error after AWQ save/load: {(y_before - y_after).abs().max().item():.6f}"
    )
