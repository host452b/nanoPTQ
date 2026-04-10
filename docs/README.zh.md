# docs/

根目录 README 之外的项目文档。

## 文件

### `flow.md` / `flow.zh.md`

两个量化过程的详细流程图：
1. **离线量化** — 运行 `nanoptq quantize` 时的步骤（含数学细节）
2. **运行时推理** — 推理服务器中 `QuantLinear.forward()` 的执行过程

阅读这两个文档，了解量化模型的完整生命周期。

### `Glossary.md` / `Glossary.zh.md`

每个量化术语先给类比，再给正式定义。
涵盖：量化、缩放因子、零点、逐组量化、困惑度、RTN、AWQ、GPTQ、
校准数据、int4/int8/fp8/bf16/nvfp4、Hessian 矩阵、safetensors、异常通道。

代码中遇到陌生术语时，先来这里查。

## 子目录

### `superpowers/`

内部规划和开发工具（AI agent 技能文件和计划文档）。
与学习量化无关。
