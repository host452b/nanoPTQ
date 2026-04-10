# nanoptq/eval/ppl.py
"""
Perplexity evaluation — the canonical PTQ quality metric.
困惑度评估：PTQ 质量的标准衡量指标。

Perplexity = exp(average negative log-likelihood over tokens).
Lower is better. FP16 baseline → quantized: how much did PPL rise?

We use a sliding window (stride) to handle sequences longer than max_length.
This matches how llama.cpp, lm-evaluation-harness, and vLLM compute PPL.
"""
import math
import torch
import torch.nn as nn


def compute_perplexity_tokens(
    model: nn.Module,
    token_ids: torch.Tensor,       # shape [1, total_tokens]
    stride: int = 512,
    max_length: int = 2048,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity over a pre-tokenized token sequence using sliding window.
    Returns a float PPL value.
    """
    model.eval()
    seq_len = token_ids.size(1)
    nlls = []
    total_scored = 0
    prev_end = 0

    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - max(begin, prev_end)
            if target_len <= 0:
                prev_end = end
                continue

            chunk = token_ids[:, begin:end].to(device)
            outputs = model(chunk)
            logits = outputs.logits  # [1, chunk_len, vocab_size]

            # Shift: predict token i+1 from token i
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()

            # Score only the new stride portion (avoid double-counting overlap)
            score_logits = shift_logits[:, -target_len:, :]
            score_labels = shift_labels[:, -target_len:]

            loss = nn.CrossEntropyLoss()(
                score_logits.view(-1, score_logits.size(-1)),
                score_labels.view(-1),
            )
            nlls.append(loss.item() * target_len)
            total_scored += target_len
            prev_end = end
            if end == seq_len:
                break

    return math.exp(sum(nlls) / total_scored)


def evaluate_ppl_bundled(
    model: nn.Module,
    tokenizer,
    stride: int = 512,
    max_length: int = 2048,
    device: str = "cuda",
) -> float:
    """
    Evaluate PPL using bundled wikitext-2 data — no internet required.
    This is the default eval path for nanoPTQ.
    使用内置数据集评估困惑度，无需联网。
    """
    from nanoptq.data.loader import load_eval_texts
    texts = load_eval_texts()
    text = "\n\n".join(texts)
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    return compute_perplexity_tokens(model, tokens, stride=stride, max_length=max_length, device=device)


def evaluate_ppl_wikitext(
    model: nn.Module,
    tokenizer,
    n_samples: int = 40,
    stride: int = 512,
    max_length: int = 2048,
    device: str = "cuda",
) -> float:
    """
    Evaluate PPL downloading wikitext-2 from HuggingFace.
    Use evaluate_ppl_bundled() for offline evaluation.
    """
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"][:n_samples])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"]
    return compute_perplexity_tokens(model, tokens, stride=stride, max_length=max_length, device=device)
