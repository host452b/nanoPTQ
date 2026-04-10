# nanoptq/model/

量化层抽象与 HuggingFace 模型集成。

## 文件

### `quant_linear.py`

`QuantLinear` 是 `nn.Linear` 的即插即用替代品。

**存储在 GPU 上的内容：**
```python
self.weight_q        # int8 张量，形状 [out_features, in_features]
self.scales          # fp16 张量，形状 [out_features, num_groups]
self.zero_points     # uint8 张量，形状 [out_features, num_groups]  （仅非对称量化）
self.input_channel_scales  # fp16 张量，形状 [in_features]  （仅 AWQ）
```

**`forward(x)` 中发生了什么：**
1. 仅 AWQ：`x = x / input_channel_scales`
2. 反量化：`W_fp16 = weight_q * scales`（按组广播）
3. 标准矩阵乘法：`output = F.linear(x, W_fp16, bias)`

无需自定义 CUDA 算子，在任何支持 PyTorch 的 GPU 上均可运行。

**主要方法：**
- `QuantLinear.from_linear(linear, bits, group_size)` — 从已有 `nn.Linear` 创建
- `ql.dequantize()` — 重建浮点权重矩阵（用于 forward 和 GPTQ）

### `hf_loader.py`

加载 HuggingFace 模型并原位替换层的工具函数。

| 函数 | 功能 |
|------|------|
| `load_hf_model(model_id, device)` | 以 fp16 加载到单卡 |
| `get_linear_layers(model)` | 遍历 `(名称, nn.Linear)` 对（跳过 embedding/norm） |
| `_set_module_by_name(model, name, new_module)` | 用 `setattr` 原位替换命名模块 |

## 设计说明

不使用 `device_map="auto"`。所有模型通过 `device_map=device` 加载到单卡。
保持代码简洁，避免隐式多卡行为。
