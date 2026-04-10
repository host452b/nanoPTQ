# nanoptq/algorithms/

三种量化算法。每种算法住在一个文件里，每个文件自成体系。

## 三种算法

| 文件 | 算法 | 校准数据 | 核心思想 |
|------|------|---------|---------|
| `rtn.py` | RTN | 无 | 将每个权重直接四舍五入到最近的整数编码 |
| `awq_lite.py` | AWQ-lite | 约 128 条 | 量化前放大高激活通道的权重，保护关键精度 |
| `gptq_lite.py` | GPTQ-lite | 约 128 条 | 用 Hessian 对后续列逐步补偿量化误差 |

## 数学本质

```python
# RTN — rtn.py
W_q = round(W / scale) * scale

# AWQ — awq_lite.py
s   = mean(|activations|, dim=0) ** alpha   # 通道重要性
W_q = quantize(W * s)                       # 放大重要通道后量化
out = W_q @ (x / s)                         # 推理时除回去

# GPTQ — gptq_lite.py
H     = X.T @ X                             # 输入 Hessian
H_inv = cholesky_inverse(cholesky(H))
for j in range(in_features):
    err          = W[:,j] - quantize(W[:,j])
    W[:,j+1:] -= err * H_inv[j, j+1:] / H_inv[j,j]   # 补偿后续列
```

## 公开 API

每个文件导出 `quantize_linear_*` 函数和 `quantize_model_*` 封装器：

```python
# RTN
from nanoptq.algorithms.rtn import quantize_linear_rtn, quantize_model_rtn

# AWQ
from nanoptq.algorithms.awq_lite import quantize_linear_awq, quantize_model_awq

# GPTQ
from nanoptq.algorithms.gptq_lite import quantize_linear_gptq
```

## 典型结果（Qwen2-0.5B，int4 group_size=128）

| 方法 | 困惑度（vs FP16 基线） | 备注 |
|------|---------------------|------|
| RTN | +2–4 PPL | 无需校准数据，最快 |
| AWQ | +0.5–2 PPL | 激活异常通道处理更好 |
| GPTQ | +0.5–2 PPL | 与 AWQ 相近；Hessian 计算略慢 |

## "lite" 的含义

`awq_lite` 和 `gptq_lite` 都实现了核心数学思想，但没有生产级工程优化：
- 无分块 GPTQ（无 64 列 Cholesky 阻尼块）
- 无 AWQ alpha 网格搜索
- 无逐层混合精度位宽分配
- 无融合 CUDA 算子

数学上是忠实的，工程捷径留给生产框架（AutoAWQ、GPTQModel）。
