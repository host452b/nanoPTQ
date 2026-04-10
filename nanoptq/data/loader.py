# nanoptq/data/loader.py
"""
Load bundled calibration and evaluation datasets.
加载内置校准与评测数据集。

Bundled data lives in data/ so researchers can eval immediately after
`git clone` — no internet access needed for basic experiments.
内置数据在 data/ 目录，克隆后无需联网即可立即评测。

Supported datasets / 支持的数据集:
  wikitext2  — general text, standard PPL benchmark (default)
  alpaca     — instruction-following
  gsm8k      — math word problems
  humaneval  — code generation
  qa         — reading comprehension (SQuAD)
  sharegpt   — multi-turn conversation
  sum        — summarization (XSum)

Usage:
  from nanoptq.data import load_calibration_texts, load_eval_texts
  texts = load_calibration_texts()                  # wikitext2 default
  texts = load_calibration_texts(dataset="alpaca")  # alpaca calibration
  texts = load_eval_texts(dataset="gsm8k")          # gsm8k eval
"""
import json
import pathlib

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_CALIB_DIR = _REPO_ROOT / "data" / "calibration"
_EVAL_DIR  = _REPO_ROOT / "data" / "eval"

# Canonical dataset names and their file stems
DATASETS = {
    "wikitext2": "wikitext2",
    "alpaca":    "alpaca",
    "gsm8k":     "gsm8k",
    "humaneval": "humaneval",
    "qa":        "qa",
    "sharegpt":  "sharegpt",
    "sum":       "sum",
}

_DEFAULT_DATASET = "wikitext2"


def _calib_path(dataset: str) -> pathlib.Path:
    return _CALIB_DIR / f"{DATASETS[dataset]}_128.jsonl"


def _eval_path(dataset: str) -> pathlib.Path:
    return _EVAL_DIR / f"{DATASETS[dataset]}_eval.jsonl"


def _load_jsonl(path: pathlib.Path, dataset: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found at {path}.\n"
            f"Run: python scripts/prepare_data.py --dataset {dataset}"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def load_calibration_texts(
    dataset: str = _DEFAULT_DATASET,
    path: str | pathlib.Path | None = None,
) -> list[str]:
    """
    Load calibration text samples for AWQ/GPTQ.
    加载 AWQ/GPTQ 校准文本样本。

    Args:
        dataset: one of wikitext2, alpaca, gsm8k, humaneval, qa, sharegpt, sum
        path: override with a custom JSONL file path

    Returns:
        list of raw text strings — tokenize with your model's tokenizer.
    """
    if path:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Custom calibration data not found: {p}")
        with open(p, encoding="utf-8") as f:
            return [json.loads(line)["text"] for line in f if line.strip()]
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASETS)}")
    return _load_jsonl(_calib_path(dataset), dataset)


def load_eval_texts(
    dataset: str = _DEFAULT_DATASET,
    path: str | pathlib.Path | None = None,
) -> list[str]:
    """
    Load evaluation text samples for perplexity computation.
    加载困惑度评测文本样本。

    Args:
        dataset: one of wikitext2, alpaca, gsm8k, humaneval, qa, sharegpt, sum
        path: override with a custom JSONL file path

    Returns:
        list of raw text strings.
    """
    if path:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Custom eval data not found: {p}")
        with open(p, encoding="utf-8") as f:
            return [json.loads(line)["text"] for line in f if line.strip()]
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASETS)}")
    return _load_jsonl(_eval_path(dataset), dataset)


def list_available_datasets() -> dict[str, dict]:
    """
    Return which datasets are available (files exist on disk).
    返回哪些数据集已就绪（文件存在）。
    """
    status = {}
    for name in DATASETS:
        status[name] = {
            "calibration": _calib_path(name).exists(),
            "eval":        _eval_path(name).exists(),
        }
    return status
