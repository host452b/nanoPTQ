import torch
import torch.nn as nn
import pytest
from nanoptq.eval.ppl import compute_perplexity_tokens


def make_lm_stub(vocab_size=100, seq_len=32):
    class FakeLM(nn.Module):
        def forward(self, input_ids, **kwargs):
            B, T = input_ids.shape
            logits = torch.zeros(B, T, vocab_size)
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(logits=logits)
    return FakeLM()


def test_compute_ppl_returns_positive_float():
    model = make_lm_stub()
    tokens = torch.randint(0, 100, (1, 64))
    ppl = compute_perplexity_tokens(model, tokens, stride=32, device="cpu")
    assert isinstance(ppl, float)
    assert ppl > 0


def test_uniform_logits_ppl_near_vocab_size():
    """Uniform distribution over V tokens → perplexity ≈ V."""
    vocab_size = 100
    model = make_lm_stub(vocab_size=vocab_size)
    tokens = torch.randint(0, vocab_size, (1, 128))
    ppl = compute_perplexity_tokens(model, tokens, stride=64, device="cpu")
    assert abs(ppl - vocab_size) < 5


def test_stride_less_than_seq_len():
    model = make_lm_stub()
    tokens = torch.randint(0, 100, (1, 256))
    ppl = compute_perplexity_tokens(model, tokens, stride=64, device="cpu")
    assert ppl > 0
