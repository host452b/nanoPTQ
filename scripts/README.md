# scripts/

Utility scripts. Run once to set up the environment.

## Files

### `prepare_data.py`

Downloads calibration and evaluation data from HuggingFace and saves them as JSONL files
in `data/calibration/` and `data/eval/`.

**Run once after cloning:**
```bash
python scripts/prepare_data.py
```

**Download specific datasets only:**
```bash
python scripts/prepare_data.py --dataset gsm8k alpaca humaneval
```

**Available dataset names:**
`wikitext2`, `alpaca`, `gsm8k`, `humaneval`, `qa`, `sharegpt`, `sum`

**Output files (per dataset):**
- `data/calibration/{name}_128.jsonl` — 128 samples for AWQ/GPTQ calibration
- `data/eval/{name}_eval.jsonl` — full eval split for PPL measurement

**Requirements:** Internet connection (one-time). The downloaded files are committed to the
repository, so collaborators and CI don't need to run this again.
