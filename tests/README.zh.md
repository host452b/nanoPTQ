# tests/

单元测试和集成测试。所有测试均离线运行——无需下载 HuggingFace 模型。

## 运行测试

```bash
pytest                      # 运行全部测试
pytest -v                   # 详细输出
pytest tests/test_ppl.py    # 运行指定文件
```

## 测试文件

| 文件 | 测试内容 |
|------|---------|
| `test_quant_primitives.py` | 对称/非对称 scale 计算、量化、反量化、往返误差 |
| `test_group_quant.py` | 逐组量化和反量化的形状与数值 |
| `test_quant_linear.py` | QuantLinear 前向传播、from_linear 工厂函数、偏置处理 |
| `test_rtn.py` | 对合成 nn.Linear 模块的 RTN 量化 |
| `test_awq_lite.py` | AWQ 通道缩放、权重缩放、输出等价性 |
| `test_gptq_lite.py` | GPTQ 逐列补偿收敛性 |
| `test_safetensors_io.py` | RTN、非对称、AWQ 模型的保存/加载往返测试 |
| `test_ppl.py` | PPL 函数返回有限正浮点数 |
| `test_data_loader.py` | 校准和评测文本加载 |

## 设计原则

- 无网络：所有测试使用合成张量或 `data/` 中的内置数据
- 无真实 HF 模型：若测试需要真实模型，用 `@pytest.mark.skip` 标记
- 快速：全套测试 60 秒内完成
- `nanoptq/` 中的每个公开函数都有至少一个测试

## 预期输出

```
60 passed in N.Ns
```
