# data/

内置校准和评测数据集。已提交到仓库。
运行一次 `scripts/prepare_data.py` 后，无需联网。

## 目录结构

```
data/
├── calibration/          # 每个数据集 128 条样本，用于 AWQ/GPTQ 校准
│   ├── wikitext2_128.jsonl
│   ├── alpaca_128.jsonl
│   ├── gsm8k_128.jsonl
│   ├── humaneval_128.jsonl
│   ├── qa_128.jsonl
│   ├── sharegpt_128.jsonl
│   └── sum_128.jsonl
└── eval/                 # 完整评测集，用于 PPL 测量
    ├── wikitext2_eval.jsonl
    ├── alpaca_eval.jsonl
    ├── gsm8k_eval.jsonl
    ├── humaneval_eval.jsonl
    ├── qa_eval.jsonl
    ├── sharegpt_eval.jsonl
    └── sum_eval.jsonl
```

## 文件格式

每个 `.jsonl` 文件每行一个 JSON 对象：`{"text": "..."}`。

## 数据集来源

| 名称 | HuggingFace 来源 | 领域 |
|------|----------------|------|
| `wikitext2` | `Salesforce/wikitext`，`wikitext-2-raw-v1` | 维基百科散文 |
| `alpaca` | `tatsu-lab/alpaca` | 指令跟随 |
| `gsm8k` | `openai/gsm8k`，main | 数学应用题 |
| `humaneval` | `openai/openai_humaneval` | Python 编程 |
| `qa` | `rajpurkar/squad` | 阅读理解 |
| `sharegpt` | `lmsys/chatbot_arena_conversations` | 多轮对话 |
| `sum` | `EdinburghNLP/xsum` | 新闻摘要 |

## 为什么内置数据集？

1. **可复现** — 每次使用相同的校准数据，不受 HF 数据集更新影响
2. **离线评测** — 初始设置后运行 `nanoptq eval` 无需联网
3. **学生友好** — 克隆即用，无需账号或 API Key

## 重新生成

```bash
python scripts/prepare_data.py                           # 所有数据集
python scripts/prepare_data.py --dataset gsm8k alpaca   # 指定数据集
```
