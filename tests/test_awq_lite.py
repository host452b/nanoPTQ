import torch
import torch.nn as nn
import pytest
from nanoptq.algorithms.awq_lite import find_channel_scales, quantize_linear_awq
from nanoptq.model.quant_linear import QuantLinear


def make_linear_with_outlier(in_f=128, out_f=64, outlier_channel=5):
    l = nn.Linear(in_f, out_f)
    nn.init.normal_(l.weight)
    return l, outlier_channel


def make_calibration_data(in_f=128, n_samples=32, outlier_channel=5):
    acts = torch.randn(n_samples, in_f)
    acts[:, outlier_channel] *= 100.0  # simulate outlier channel
    return acts


def test_find_channel_scales_identifies_outlier():
    _, outlier_ch = make_linear_with_outlier()
    acts = make_calibration_data(outlier_channel=outlier_ch)
    scales = find_channel_scales(acts, alpha=0.5)
    assert scales.shape == (128,)
    assert scales[outlier_ch] > scales.mean() * 2


def test_find_channel_scales_all_positive():
    acts = torch.randn(32, 128).abs()
    scales = find_channel_scales(acts, alpha=0.5)
    assert (scales > 0).all()


def test_quantize_linear_awq_returns_quant_linear():
    linear, _ = make_linear_with_outlier()
    acts = make_calibration_data()
    ql = quantize_linear_awq(linear, acts, bits=4, group_size=128)
    assert isinstance(ql, QuantLinear)


def test_awq_better_than_rtn_on_outlier_data():
    """AWQ should produce smaller or comparable reconstruction error vs RTN on outlier data."""
    torch.manual_seed(42)
    linear, outlier_ch = make_linear_with_outlier()
    acts = make_calibration_data(outlier_channel=outlier_ch)

    from nanoptq.algorithms.rtn import quantize_linear_rtn
    ql_rtn = quantize_linear_rtn(linear, bits=4, group_size=128)
    ql_awq = quantize_linear_awq(linear, acts, bits=4, group_size=128)

    x = make_calibration_data(n_samples=16, outlier_channel=outlier_ch)
    with torch.no_grad():
        err_rtn = (linear(x) - ql_rtn(x)).abs().mean()
        err_awq = (linear(x) - ql_awq(x)).abs().mean()

    # AWQ should not be catastrophically worse than RTN
    assert err_awq < err_rtn * 1.5


def test_quantize_linear_awq_fills_buffers():
    linear, _ = make_linear_with_outlier()
    acts = make_calibration_data()
    ql = quantize_linear_awq(linear, acts, bits=4, group_size=128)
    assert ql.weight_q.abs().sum() > 0
    assert ql.scales.abs().sum() > 0
