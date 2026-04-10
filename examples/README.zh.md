# examples/

可直接运行的演示脚本。每个文件独立运行，不依赖其他演示文件。

## 文件

### `quant_model.py` — 端到端流程

一个脚本演示完整工作流程：
1. 加载 FP16 模型
2. 测量 FP16 延迟和困惑度
3. 用 RTN（或通过 `--method` 选 AWQ/GPTQ）量化
4. 测量量化后的延迟和困惑度
5. 生成示例文本验证连贯性

```bash
python examples/quant_model.py --model Qwen/Qwen2-0.5B --bits 4 --method rtn
```

### `compare_methods.py` — 横向对比表

在同一模型上运行 RTN、AWQ 和 GPTQ，打印对比表。

```bash
python examples/compare_methods.py --model Qwen/Qwen2-0.5B --bits 4
```

### `precision_tour.py` — 数值格式教程

7 节互动式数值格式讲解：
- 第1节：什么是数值格式？
- 第2节：bf16 vs fp16（溢出现场演示）
- 第3节：fp8（e4m3、e5m2）
- 第4节：int4 逐组量化（异常列可视化）
- 第5节：nvfp4 e2m1（Blackwell）
- 第6节：硬件支持一览表
- 第7节：如何选择格式的决策树

```bash
python examples/precision_tour.py            # 全部章节
python examples/precision_tour.py --section 3  # 仅看 fp8
```

### `awq_explained.py` — AWQ 逐步讲解

6 步解析 AWQ 为什么有效，配有现场演示：
- 第1步：朴素 per-tensor int4 为什么失败
- 第2步：逐组量化已经有帮助，但还不够
- 第3步：AWQ 核心洞察——激活感知重要性
- 第4步：三种方案横向对比
- 第5步：代码与 `nanoptq/algorithms/awq_lite.py` 的一一对应
- 第6步：端到端保存/加载验证

```bash
python examples/awq_explained.py
```

## 设计理念

这些示例用于学习，而非生产。使用具体数值、ASCII 图表和逐步打印输出，让数学变得直观。
目标读者：懂 PyTorch、但从未实现过量化的人。
