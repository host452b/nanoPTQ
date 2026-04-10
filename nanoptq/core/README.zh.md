# nanoptq/core/

量化数学基元。这两个文件是整个项目的数学基础。
**请先读这里，再看其他任何文件。**

## 文件

### `quant_primitives.py`

每个量化库都需要的四个函数：

| 函数 | 功能 |
|------|------|
| `compute_scale_symmetric(W, bits)` | `S = max|W| / (2^(b-1)-1)` |
| `compute_scale_zero_asymmetric(W, bits)` | `S = (max-min)/(2^b-1)`，`Z = -round(min/S)` |
| `quantize_tensor(W, scale, zero_point, bits)` | `Q = clip(round(W/S + Z), qmin, qmax)` |
| `dequantize_tensor(Q, scale, zero_point)` | `W_approx = S × (Q - Z)` |
| `fake_quantize(W, bits, symmetric)` | 一次调用完成量化再反量化 |

**从这里开始。** 五个函数，约 50 行代码，纯 PyTorch，无隐藏状态。

### `group_quant.py`

将上述基元应用于每 128 个权重的分组。

| 函数 | 功能 |
|------|------|
| `group_quantize(W, group_size, bits, symmetric)` | 返回 `(W_q, scales, zero_points)` |
| `group_dequantize(W_q, scales, zero_points, group_size)` | 从存储张量重建浮点权重 |

**为什么要逐组量化？** 每 128 个权重一个 scale，而不是整层一个 scale。
异常列有自己的 scale，不会污染相邻列的精度。

## 关键数值

```python
int4 对称：  取值范围 -8 到 +7  （15 个量化级别）
int4 非对称：取值范围  0 到 15  （16 个量化级别）
int8 对称：  取值范围 -127 到 +127
```

## 无副作用

这些都是纯函数。没有模型状态、没有 hook、没有 buffer。可以安全地在任意张量上调用。
