# nanoptq/data/

Dataset loader for calibration and evaluation. Reads from the `data/` directory at the repo root.

## File

### `loader.py`

**`load_calibration_texts(dataset, path) → list[str]`**

Returns a list of text strings for AWQ/GPTQ calibration.
Default: 128 samples from `data/calibration/wikitext2_128.jsonl`.

**`load_eval_texts(dataset, path) → list[str]`**

Returns the full eval split for perplexity evaluation.
Default: `data/eval/wikitext2_eval.jsonl`.

**`list_available_datasets() → dict`**

Returns the DATASETS registry — all dataset names and their file paths.

## Available Datasets

| Name | Calibration file | Eval file | Domain |
|------|-----------------|-----------|--------|
| `wikitext2` | `wikitext2_128.jsonl` | `wikitext2_eval.jsonl` | Wikipedia prose |
| `alpaca` | `alpaca_128.jsonl` | `alpaca_eval.jsonl` | Instruction following |
| `gsm8k` | `gsm8k_128.jsonl` | `gsm8k_eval.jsonl` | Math word problems |
| `humaneval` | `humaneval_128.jsonl` | `humaneval_eval.jsonl` | Python code |
| `qa` | `qa_128.jsonl` | `qa_eval.jsonl` | Question answering (SQuAD) |
| `sharegpt` | `sharegpt_128.jsonl` | `sharegpt_eval.jsonl` | Multi-turn conversations |
| `sum` | `sum_128.jsonl` | `sum_eval.jsonl` | News summarization (XSum) |

## Note

This module only reads. It never downloads. Use `scripts/prepare_data.py` to download
datasets from HuggingFace and generate the JSONL files.
