# nanoptq/eval/

Evaluation: perplexity and latency. The two metrics that matter.

## Files

### `ppl.py`

Sliding-window perplexity evaluation on wikitext-2 (or any bundled dataset).

**`evaluate_ppl_bundled(model, tokenizer, dataset, device, stride, max_length)`**

- Loads eval texts from `data/eval/{dataset}_eval.jsonl` (no internet needed)
- Concatenates all texts into one long token sequence
- Slides a window of `max_length=2048` tokens with `stride=512`
- Returns scalar PPL: `exp(mean cross-entropy loss over all windows)`

Lower is better. FP16 baseline is the ceiling. A good quantization should stay within +2 PPL.

### `latency.py`

Two benchmarks in a named tuple `LatencyResult`:

| Field | Meaning |
|-------|---------|
| `prefill_ms` | Time to process a 512-token prompt, in milliseconds |
| `decode_tps` | Tokens per second during `model.generate()` over 100 new tokens |
| `peak_mem_gb` | Peak GPU VRAM allocated during the benchmark |

**`benchmark_latency(model, tokenizer, device, prompt_len, decode_steps)`**

Runs a warmup pass, then times prefill and decode separately.
Single GPU only. Results vary by GPU; compare FP16 vs quantized on the same machine.

## Usage

```python
from nanoptq.eval.ppl import evaluate_ppl_bundled
from nanoptq.eval.latency import benchmark_latency

ppl = evaluate_ppl_bundled(model, tokenizer, dataset="wikitext2", device="cuda")
lat = benchmark_latency(model, tokenizer, device="cuda")

print(f"PPL: {ppl:.2f}")
print(f"Prefill: {lat.prefill_ms:.1f} ms | Decode: {lat.decode_tps:.1f} tok/s")
```

## Note on Eval Data

The full wikitext-2 test set is bundled in `data/eval/wikitext2_eval.jsonl`.
No internet connection needed after `python scripts/prepare_data.py` is run once.
