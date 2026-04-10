# nanoptq/

核心代码库。所有可运行的功能都在这里。

## 目录结构

```
nanoptq/
├── core/          # 量化数学基元
├── model/         # QuantLinear 层 + HF 模型加载
├── algorithms/    # RTN、AWQ、GPTQ 实现
├── io/            # 量化检查点的保存与加载
├── eval/          # 困惑度和延迟评测
├── data/          # 校准/评测数据集加载器
└── cli.py         # 入口：nanoptq quantize / eval / compare
```

## 阅读顺序

如果你是来学习的，建议按此顺序阅读：

| 步骤 | 文件 | 学到什么 |
|------|------|---------|
| 1 | `core/quant_primitives.py` | 对称/非对称量化数学，伪量化 |
| 2 | `core/group_quant.py` | 为什么逐组量化大幅改善 int4 精度 |
| 3 | `model/quant_linear.py` | 统一量化层抽象；动态反量化 |
| 4 | `algorithms/rtn.py` | 基线：四舍五入搞定 |
| 5 | `algorithms/awq_lite.py` | 激活感知改进 |
| 6 | `algorithms/gptq_lite.py` | 基于 Hessian 的误差补偿 |

## 入口

`cli.py` 把所有模块串联起来。每个子命令只导入自己需要的部分：

```
nanoptq quantize  →  algorithms/ + io/
nanoptq eval      →  io/ + eval/
nanoptq compare   →  algorithms/ + eval/
```
