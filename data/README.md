# data/

Bundled calibration and evaluation datasets. Committed to the repository.
No internet needed after `scripts/prepare_data.py` is run once.

## Structure

```
data/
├── calibration/          # 128 samples per dataset, for AWQ/GPTQ calibration
│   ├── wikitext2_128.jsonl
│   ├── alpaca_128.jsonl
│   ├── gsm8k_128.jsonl
│   ├── humaneval_128.jsonl
│   ├── qa_128.jsonl
│   ├── sharegpt_128.jsonl
│   └── sum_128.jsonl
└── eval/                 # Full eval splits, for PPL measurement
    ├── wikitext2_eval.jsonl
    ├── alpaca_eval.jsonl
    ├── gsm8k_eval.jsonl
    ├── humaneval_eval.jsonl
    ├── qa_eval.jsonl
    ├── sharegpt_eval.jsonl
    └── sum_eval.jsonl
```

## File Format

Each `.jsonl` file has one JSON object per line: `{"text": "..."}`.

## Dataset Sources

| Name | HuggingFace Source | Domain |
|------|--------------------|--------|
| `wikitext2` | `Salesforce/wikitext`, `wikitext-2-raw-v1` | Wikipedia prose |
| `alpaca` | `tatsu-lab/alpaca` | Instruction following |
| `gsm8k` | `openai/gsm8k`, main split | Math word problems |
| `humaneval` | `openai/openai_humaneval` | Python programming |
| `qa` | `rajpurkar/squad` | Reading comprehension |
| `sharegpt` | `lmsys/chatbot_arena_conversations` | Multi-turn chat |
| `sum` | `EdinburghNLP/xsum` | News summarization |

## Why Bundle Datasets?

1. **Reproducibility** — same calibration data every time, regardless of HF dataset updates
2. **Offline eval** — run `nanoptq eval` without internet after initial setup
3. **Student-friendly** — clone and run, no accounts or API keys needed

## Regenerating

```bash
python scripts/prepare_data.py                           # all datasets
python scripts/prepare_data.py --dataset gsm8k alpaca   # specific datasets
```
