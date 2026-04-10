#!/usr/bin/env python3
# scripts/prepare_data.py
"""
Download wikitext-2 from HuggingFace and produce small, curated JSONL files
for bundling directly in the repo.

Run once:
  python scripts/prepare_data.py

Output:
  data/calibration/wikitext2_train_128.jsonl   — 128 samples, calibration
  data/eval/wikitext2_test.jsonl               — test split for PPL
"""
import json
import pathlib
from datasets import load_dataset

ROOT = pathlib.Path(__file__).parent.parent
CALIB_OUT = ROOT / "data" / "calibration" / "wikitext2_train_128.jsonl"
EVAL_OUT   = ROOT / "data" / "eval"        / "wikitext2_test.jsonl"

def main():
    CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading wikitext-2 train (calibration) ...")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Take non-empty articles; limit to 128 samples
    calib = [{"text": t} for t in train["text"] if len(t.strip()) > 100][:128]
    with open(CALIB_OUT, "w") as f:
        for rec in calib:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(calib)} samples → {CALIB_OUT}")

    print("Downloading wikitext-2 test (eval) ...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_records = [{"text": t} for t in test["text"] if len(t.strip()) > 50]
    with open(EVAL_OUT, "w") as f:
        for rec in eval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(eval_records)} samples → {EVAL_OUT}")

if __name__ == "__main__":
    main()
