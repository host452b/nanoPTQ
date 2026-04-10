# nanoptq/data/loader.py
"""
Load bundled calibration and evaluation datasets.
加载内置的校准和评估数据集。

Bundled data lives in data/ at the repo root so researchers can eval
immediately after `git clone` — no internet access needed for basic experiments.
内置数据在 data/ 目录中，克隆仓库后无需联网即可立即评估。

If you need a different dataset, pass custom text lists to ppl.py / awq_lite.py directly.
"""
import json
import pathlib

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_CALIB_PATH = _REPO_ROOT / "data" / "calibration" / "wikitext2_train_128.jsonl"
_EVAL_PATH  = _REPO_ROOT / "data" / "eval"        / "wikitext2_test.jsonl"


def load_calibration_texts(path: str | pathlib.Path | None = None) -> list[str]:
    """
    Load calibration text samples.
    Returns list of raw strings — tokenize with your model's tokenizer.
    Default: bundled wikitext-2 train subset (128 samples).
    """
    p = pathlib.Path(path) if path else _CALIB_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Bundled calibration data not found at {p}.\n"
            "Run `python scripts/prepare_data.py` to generate it."
        )
    with open(p) as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def load_eval_texts(path: str | pathlib.Path | None = None) -> list[str]:
    """
    Load evaluation text samples for perplexity computation.
    Default: bundled wikitext-2 test split.
    """
    p = pathlib.Path(path) if path else _EVAL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Bundled eval data not found at {p}.\n"
            "Run `python scripts/prepare_data.py` to generate it."
        )
    with open(p) as f:
        return [json.loads(line)["text"] for line in f if line.strip()]
