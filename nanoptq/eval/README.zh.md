# nanoptq/eval/

评测：困惑度和延迟。两个真正有意义的指标。

## 文件

### `ppl.py`

在 wikitext-2（或任何内置数据集）上进行滑动窗口困惑度评测。

**`evaluate_ppl_bundled(model, tokenizer, dataset, device, stride, max_length)`**

- 从 `data/eval/{dataset}_eval.jsonl` 加载评测文本（无需联网）
- 将所有文本拼接成一个长 token 序列
- 以 `max_length=2048` 为窗口、`stride=512` 步长滑动
- 返回标量 PPL：`exp(所有窗口交叉熵损失的均值)`

越低越好。FP16 基线是上限。好的量化结果应在 +2 PPL 以内。

### `latency.py`

两个基准测试，返回命名元组 `LatencyResult`：

| 字段 | 含义 |
|------|------|
| `prefill_ms` | 处理 512 token prompt 的耗时（毫秒） |
| `decode_tps` | `model.generate()` 生成 100 新 token 的速度（tokens/秒） |
| `peak_mem_gb` | 基准测试期间峰值 GPU 显存占用（GB） |

**`benchmark_latency(model, tokenizer, device, prompt_len, decode_steps)`**

先运行一次热身，再分别计时 prefill 和 decode。
单卡运行。结果因 GPU 而异；比较 FP16 和量化模型时需在同一台机器上进行。

## 使用示例

```python
from nanoptq.eval.ppl import evaluate_ppl_bundled
from nanoptq.eval.latency import benchmark_latency

ppl = evaluate_ppl_bundled(model, tokenizer, dataset="wikitext2", device="cuda")
lat = benchmark_latency(model, tokenizer, device="cuda")

print(f"PPL: {ppl:.2f}")
print(f"Prefill: {lat.prefill_ms:.1f} ms | Decode: {lat.decode_tps:.1f} tok/s")
```

## 关于评测数据

完整的 wikitext-2 测试集已内置在 `data/eval/wikitext2_eval.jsonl`。
运行一次 `python scripts/prepare_data.py` 后，无需联网即可完成评测。
