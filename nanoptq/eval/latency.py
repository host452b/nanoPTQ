# nanoptq/eval/latency.py
"""
Latency benchmarking: tokens/s, prefill time, peak memory.
延迟基准测试：tokens/s、预填充时间、峰值显存。

Two metrics that matter for production:
  prefill_ms: time to process the prompt (batch of input tokens)
  decode_tps: tokens per second during autoregressive generation
"""
import time
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class LatencyResult:
    prefill_ms: float       # time to process prompt
    decode_tps: float        # tokens/s during model.generate() (includes prefill overhead)
    peak_mem_gb: float      # peak GPU memory in GB
    n_new_tokens: int


def benchmark_latency(
    model: nn.Module,
    tokenizer,
    prompt: str = "The quick brown fox",
    n_new_tokens: int = 128,
    n_warmup: int = 2,
    n_runs: int = 5,
    device: str = "cuda",
) -> LatencyResult:
    """
    Benchmark prefill and decode latency.
    Returns average over n_runs after n_warmup warm-up passes.

    Note: decode_tps measures total wall time for model.generate() (prefill + decode).
    It is generate throughput, not pure decode-only speed.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    prefill_times = []
    decode_times = []

    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        prefill_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=n_new_tokens, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        decode_times.append(time.perf_counter() - t0)

    n_gen = gen.shape[1] - input_len
    avg_decode_s = sum(decode_times) / n_runs
    decode_tps = n_gen / avg_decode_s if avg_decode_s > 0 else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    return LatencyResult(
        prefill_ms=sum(prefill_times) / n_runs,
        decode_tps=decode_tps,
        peak_mem_gb=peak_mem,
        n_new_tokens=n_gen,
    )
