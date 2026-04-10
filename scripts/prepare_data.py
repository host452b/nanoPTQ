#!/usr/bin/env python3
# scripts/prepare_data.py
"""
Download and curate bundled calibration + eval datasets.
下载并整理内置校准/评测数据集（参考 AngelSlim /dataset/ 体系）。

Datasets (each ~128 calibration + eval samples):
  wikitext2   — general text, standard PPL benchmark
  alpaca      — instruction-following (tatsu-lab/alpaca)
  gsm8k       — math word problems (openai/gsm8k)
  humaneval   — code generation (openai/openai_humaneval)
  qa          — reading comprehension (rajpurkar/squad)
  sharegpt    — multi-turn chat (lmsys/chatbot_arena_conversations)
  sum         — summarization (EdinburghNLP/xsum)

Run once:
  python scripts/prepare_data.py              # all datasets
  python scripts/prepare_data.py --dataset gsm8k alpaca  # specific

Output layout (mirrors AngelSlim /dataset/):
  data/calibration/<dataset>_128.jsonl
  data/eval/<dataset>_eval.jsonl
"""
import argparse
import json
import pathlib
from datasets import load_dataset

ROOT      = pathlib.Path(__file__).parent.parent
CALIB_DIR = ROOT / "data" / "calibration"
EVAL_DIR  = ROOT / "data" / "eval"


def write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → {path}  ({len(records)} samples)")


# ---------------------------------------------------------------------------
# Dataset preparers — each returns list[{"text": str}]
# ---------------------------------------------------------------------------

def prep_wikitext2(n_calib=128):
    print("wikitext-2 ...")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calib = [{"text": t} for t in train["text"] if len(t.strip()) > 100][:n_calib]
    eval_ = [{"text": t} for t in test["text"]  if len(t.strip()) > 50]
    return calib, eval_


def prep_alpaca(n_calib=128):
    print("alpaca ...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    records = []
    for row in ds:
        parts = []
        if row.get("instruction"):
            parts.append(row["instruction"])
        if row.get("input"):
            parts.append(row["input"])
        if row.get("output"):
            parts.append(row["output"])
        text = "\n".join(parts).strip()
        if len(text) > 80:
            records.append({"text": text})
    calib = records[:n_calib]
    eval_ = records[n_calib:n_calib + 500]
    return calib, eval_


def prep_gsm8k(n_calib=128):
    print("gsm8k ...")
    train = load_dataset("openai/gsm8k", "main", split="train")
    test  = load_dataset("openai/gsm8k", "main", split="test")
    def fmt(row):
        return {"text": f"Question: {row['question']}\nAnswer: {row['answer']}"}
    calib = [fmt(r) for r in train][:n_calib]
    eval_ = [fmt(r) for r in test]
    return calib, eval_


def prep_humaneval(n_calib=128):
    print("humaneval ...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    def fmt(row):
        text = row["prompt"]
        if row.get("canonical_solution"):
            text += row["canonical_solution"]
        return {"text": text.strip()}
    records = [fmt(r) for r in ds if len(r["prompt"].strip()) > 50]
    # humaneval only has 164 problems; use all for both calib + eval
    calib = records[:n_calib]
    eval_ = records
    return calib, eval_


def prep_qa(n_calib=128):
    """SQuAD reading comprehension contexts as text."""
    print("qa (squad) ...")
    train = load_dataset("rajpurkar/squad", split="train")
    val   = load_dataset("rajpurkar/squad", split="validation")
    seen  = set()
    calib, eval_ = [], []
    for row in train:
        ctx = row["context"].strip()
        if ctx not in seen and len(ctx) > 100:
            calib.append({"text": f"{ctx}\nQ: {row['question']}\nA: {row['answers']['text'][0]}"})
            seen.add(ctx)
        if len(calib) >= n_calib:
            break
    for row in val:
        ctx = row["context"].strip()
        if ctx not in seen and len(ctx) > 100:
            eval_.append({"text": f"{ctx}\nQ: {row['question']}\nA: {row['answers']['text'][0]}"})
            seen.add(ctx)
        if len(eval_) >= 500:
            break
    return calib, eval_


def prep_sharegpt(n_calib=128):
    """Multi-turn conversations from chatbot arena."""
    print("sharegpt (chatbot_arena_conversations) ...")
    ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    records = []
    for row in ds:
        turns = row.get("conversation_a") or []
        text = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in turns if t.get("content")
        ).strip()
        if len(text) > 100:
            records.append({"text": text})
        if len(records) >= n_calib + 500:
            break
    calib = records[:n_calib]
    eval_ = records[n_calib:n_calib + 500]
    return calib, eval_


def prep_sum(n_calib=128):
    """Summarization: document + summary (XSum)."""
    print("sum (xsum) ...")
    train = load_dataset("EdinburghNLP/xsum", split="train")
    test  = load_dataset("EdinburghNLP/xsum", split="test")
    def fmt(row):
        return {"text": f"{row['document'].strip()}\nSummary: {row['summary'].strip()}"}
    calib = [fmt(r) for r in train if len(r["document"]) > 100][:n_calib]
    eval_ = [fmt(r) for r in test  if len(r["document"]) > 100][:500]
    return calib, eval_


# ---------------------------------------------------------------------------

PREPARERS = {
    "wikitext2": prep_wikitext2,
    "alpaca":    prep_alpaca,
    "gsm8k":     prep_gsm8k,
    "humaneval": prep_humaneval,
    "qa":        prep_qa,
    "sharegpt":  prep_sharegpt,
    "sum":       prep_sum,
}


def main():
    parser = argparse.ArgumentParser(description="Prepare bundled datasets for nanoPTQ")
    parser.add_argument(
        "--dataset", nargs="*",
        choices=list(PREPARERS), default=list(PREPARERS),
        help="Which datasets to prepare (default: all)",
    )
    args = parser.parse_args()

    for name in args.dataset:
        print(f"\n[{name}]")
        calib, eval_ = PREPARERS[name]()
        write_jsonl(CALIB_DIR / f"{name}_128.jsonl",  calib)
        write_jsonl(EVAL_DIR  / f"{name}_eval.jsonl", eval_)

    print("\nDone. All datasets written to data/calibration/ and data/eval/")


if __name__ == "__main__":
    main()
