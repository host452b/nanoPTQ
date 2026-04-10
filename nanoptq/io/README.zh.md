# nanoptq/io/

用 safetensors 格式保存和加载量化模型检查点。

## 文件

### `safetensors_io.py`

两个函数：

**`save_quantized_model(model, output_dir, bits, group_size, method)`**

遍历模型中所有 `QuantLinear` 并序列化：
- `{name}.weight_q` — int8 整数编码
- `{name}.scales` — fp16 缩放因子
- `{name}.zero_points` — uint8 零点（仅非对称且非零时保存）
- `{name}.input_channel_scales` — fp16（仅 AWQ；逐通道缩放向量 s）

同时写入 `quant_config.json`，记录元数据：`bits`、`group_size`、`method`、`skipped_modules`。

**`load_quantized_model(model, checkpoint_dir)`**

读取 `.safetensors` 文件和 `quant_config.json`，然后：
1. 读取 `quant_config.json` 获取 `bits`、`group_size`
2. 将有存储张量的每个 `nn.Linear` 替换为新的 `QuantLinear`
3. 加载 `weight_q`、`scales`、`zero_points` 和 `input_channel_scales` 到对应层

## 为什么用 Safetensors？

- 安全：加载时不执行代码（不同于 pickle）
- 快速：大文件内存映射加载
- 兼容：可被 vLLM、HF Transformers、llama.cpp 直接加载

## 输出目录文件布局

```
output_dir/
├── quantized_model.safetensors   # 所有量化层张量
├── quant_config.json             # bits、group_size、method 等
├── config.json                   # 原始 HF 模型配置
├── tokenizer.json                # 分词器
└── tokenizer_config.json         # 分词器配置
```

## AWQ 注意事项

`input_channel_scales` 必须在保存/加载中完整保留——
否则 AWQ 的缩放技巧 `(W*s) @ (x/s) = W@x` 会静默失效（输出错误，无报错）。
加载器会显式检查并恢复这个 buffer。
