# tests/test_data_loader.py
import pytest
from nanoptq.data.loader import load_calibration_texts, load_eval_texts

def test_load_calibration_texts_count():
    texts = load_calibration_texts()
    assert len(texts) == 128

def test_load_calibration_texts_nonempty():
    texts = load_calibration_texts()
    assert all(len(t) > 0 for t in texts)

def test_load_eval_texts_nonempty():
    texts = load_eval_texts()
    assert len(texts) > 0

def test_load_calibration_texts_returns_strings():
    texts = load_calibration_texts()
    assert all(isinstance(t, str) for t in texts)

def test_load_eval_texts_returns_strings():
    texts = load_eval_texts()
    assert all(isinstance(t, str) for t in texts)
