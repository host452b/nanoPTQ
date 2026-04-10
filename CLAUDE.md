# nanoPTQ — Project Rules for AI Agents

This file governs how AI agents (Claude Code, Copilot, etc.) should behave when working in this repository.

---

## Philosophy

nanoPTQ is an **educational** repository. Every design decision prioritizes clarity over cleverness.
The target reader is a graduate student who understands PyTorch but has never implemented quantization.

**28 Rule:** Focus on the small critical portion of quantization techniques that cover most industrial
deployment scenarios. Skip quantization variants with limited real-world impact (QAT, sparsity, mixed
precision search, etc.). RTN + AWQ + GPTQ already covers ~80% of production weight quantization.

---

## Language Rules

- **All `print()` statements and user-facing output must be in English.** Chinese comments in code are OK,
  but console output must be English — readable by any global contributor and legible in CI logs.
- **Docstrings:** English primary. Chinese translation welcome as an additional line below, not instead of.
- **README files:** Maintain both `README.md` (English) and `README.zh.md` (Chinese) as sibling files.
  Whenever one changes, the other must be kept in sync.

---

## Naming Rules

Function names, variable names, file names, and CLI flags must be **maximally expressive**.
A reader should understand intent without reading the body. Examples:

- Good: `quantize_linear_awq`, `evaluate_ppl_bundled`, `input_channel_scales`
- Bad: `quant`, `proc`, `data`, `x2`, `tmp`

Reduce naming ambiguity — both human engineers and AI agents read names in isolation. Make them count.

---

## Device / Hardware Rules

- **Single GPU only.** Default device is `cuda` (single card). Never add multi-GPU, FSDP, or pipeline
  parallel code. If a function receives `device: str`, it routes the whole model there.
- `device_map="auto"` is banned — it silently enables multi-GPU. Use explicit `device_map=device`.
- All scripts must run on a single GPU with ≤24 GB VRAM (Qwen2-0.5B target model).

---

## Code Style

- **No speculative abstractions.** Don't add helpers for hypothetical future use. Three similar lines
  is better than one premature helper.
- **No error handling for impossible cases.** Only validate at system boundaries (CLI args, file I/O).
  Internal functions can assert or let PyTorch raise naturally.
- **Skip embeddings and norms.** Only quantize `nn.Linear`. Never touch `embed_tokens`, `norm`, `lm_head`
  by default (lm_head is skip-listed in all algorithms).
- **group_size=128 is the default.** Do not change without explicit reason.
- **symmetric=True is the default** for weights. Asymmetric is opt-in.

---

## Testing Rules

- Tests live in `tests/`. Run with `pytest`.
- Every new public function in `nanoptq/` must have a test.
- Tests must not require internet access. Use bundled data in `data/` or synthetic tensors.
- If a test requires a real HF model, skip it with `pytest.mark.skip` and document why.

---

## Examples Rules

- Examples in `examples/` are **educational demos**, not production code.
- Each example should be runnable standalone: `python examples/foo.py`
- Target audience: someone who has never seen quantization code. Assume PyTorch familiarity.
- Use concrete numbers and visual output (ASCII art, bars, tables) to make math tangible.
- `--section N` CLI pattern is preferred for multi-step educational scripts.

---

## Documentation Rules

- **flow.md / flow.zh.md**: High-level flowcharts of the PTQ process. Keep in `docs/`.
- **Glossary.md / Glossary.zh.md**: Analogy-based explanations of quantization terms. Keep in `docs/`.
- **Per-directory READMEs**: Each subdirectory has `README.md` + `README.zh.md` explaining its role.
- Update READMEs whenever a file is added, removed, or significantly restructured.

---

## What NOT to Add

Do not add any of the following, even if they seem helpful:

- QAT (quantization-aware training)
- Pruning / sparsity
- Knowledge distillation
- Multi-GPU / FSDP / pipeline parallel
- CUDA / Triton kernels
- VLMs, encoder-decoders, MoE routing
- Mixed-precision search (e.g., sensitivity-based bit assignment beyond group_size)
- Hardware-specific backends (TensorRT, ONNX export)

If a user asks for these, explain why they're out of scope and point to the appropriate industrial
framework (AutoAWQ, GPTQModel, torchao, TensorRT-LLM).
