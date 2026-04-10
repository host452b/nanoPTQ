# scripts/

工具脚本。环境准备时运行一次即可。

## 文件

### `prepare_data.py`

从 HuggingFace 下载校准和评测数据，保存为 JSONL 文件，
分别存入 `data/calibration/` 和 `data/eval/`。

**克隆仓库后运行一次：**
```bash
python scripts/prepare_data.py
```

**仅下载指定数据集：**
```bash
python scripts/prepare_data.py --dataset gsm8k alpaca humaneval
```

**可用数据集名称：**
`wikitext2`、`alpaca`、`gsm8k`、`humaneval`、`qa`、`sharegpt`、`sum`

**每个数据集生成的文件：**
- `data/calibration/{name}_128.jsonl` — 128 条校准样本
- `data/eval/{name}_eval.jsonl` — 完整评测集

**要求：** 首次运行需要网络。生成的文件已提交到仓库，
协作者和 CI 无需重复运行。
