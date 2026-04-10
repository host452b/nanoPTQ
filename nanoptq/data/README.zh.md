# nanoptq/data/

校准和评测数据集加载器。从仓库根目录的 `data/` 读取文件。

## 文件

### `loader.py`

**`load_calibration_texts(dataset, path) → list[str]`**

返回用于 AWQ/GPTQ 校准的文本字符串列表。
默认：从 `data/calibration/wikitext2_128.jsonl` 读取 128 条样本。

**`load_eval_texts(dataset, path) → list[str]`**

返回用于困惑度评测的完整评测集。
默认：`data/eval/wikitext2_eval.jsonl`。

**`list_available_datasets() → dict`**

返回 DATASETS 注册表——所有数据集名称及其文件路径。

## 可用数据集

| 名称 | 校准文件 | 评测文件 | 领域 |
|------|---------|---------|------|
| `wikitext2` | `wikitext2_128.jsonl` | `wikitext2_eval.jsonl` | 维基百科散文 |
| `alpaca` | `alpaca_128.jsonl` | `alpaca_eval.jsonl` | 指令跟随 |
| `gsm8k` | `gsm8k_128.jsonl` | `gsm8k_eval.jsonl` | 数学应用题 |
| `humaneval` | `humaneval_128.jsonl` | `humaneval_eval.jsonl` | Python 代码 |
| `qa` | `qa_128.jsonl` | `qa_eval.jsonl` | 问答（SQuAD） |
| `sharegpt` | `sharegpt_128.jsonl` | `sharegpt_eval.jsonl` | 多轮对话 |
| `sum` | `sum_128.jsonl` | `sum_eval.jsonl` | 新闻摘要（XSum） |

## 注意

此模块只读取数据，不下载。使用 `scripts/prepare_data.py` 从 HuggingFace 下载数据集并生成 JSONL 文件。
