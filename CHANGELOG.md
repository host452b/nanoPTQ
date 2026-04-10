# Changelog

All notable changes to nanoPTQ are documented here.
每次重要改动的原因和需求来源都记录在此。

Format: `[version] YYYY-MM-DD — <type>: <what changed> | <why / demand source>`

Types: `feat` · `fix` · `refactor` · `docs` · `data` · `eval` · `break`

---

## [Unreleased] — In Development

### Planned
- `feat: RTN baseline quantizer` — P0 foundation; establishes the measurement floor all other algorithms beat
- `feat: AWQ-lite` — most practical calibration-based method; short calibration, good int4 quality
- `feat: GPTQ-lite` — Hessian compensation; shows students *why* calibration data improves quantization
- `feat: bundled wikitext-2 calibration + eval data` — one-stop eval without internet; inspired by AngelSlim /dataset/ design
- `feat: safetensors save/load` — real artifacts usable by vLLM/HF; not just in-memory experiments
- `feat: CLI (quantize / eval / compare)` — single entry point; mirrors nanoGPT's `train.py` philosophy

---

## [0.1.0] — 2026-04-10 — Initial Architecture

### Added
- Project specification finalized with user
- Implementation plan written: `docs/superpowers/plans/2026-04-10-nanoptq.md`
- `PLAN.md` progress tracker created
- `CHANGELOG.md` created

### Design source
- Requirement: "28定律 量化中最重要最普世最广泛工业使用的量化技术" (Pareto-optimal PTQ)
- Requirement: Karpathy-style — one file, one idea, readable by students
- Requirement: HF safetensors in/out + `model.generate()` compat + vLLM-ready artifacts
- Requirement: Pure PyTorch + transformers, no bitsandbytes, no CUDA kernels
- Requirement: Bundled eval datasets in `data/` for one-stop reproducible evaluation (AngelSlim-inspired)
- Scope cut: No QAT, no pruning, no distillation, no CUDA/Triton kernels, no multi-GPU, no VLMs

### Algorithms selected (28定律 core set)
| Algorithm | Why included |
|-----------|-------------|
| RTN | Universal baseline; zero calibration cost; explains what "rounding" means |
| AWQ-lite | Short calibration, activation-aware, most practical int4 path; huge HF model hub coverage |
| GPTQ-lite | Hessian-based; teaches *why* second-order information helps; industrial standard for offline PTQ |

### Algorithms excluded from v0.1
| Algorithm | Why excluded |
|-----------|-------------|
| SmoothQuant | W8A8 path; valuable but adds activation quantization complexity; v0.2 candidate |
| FP8 | Hardware-specific (H100+); kernel-dependent; out of scope for pure PyTorch target |
| KV Cache quant | Requires runtime hook into attention; separate concern from weight quantization |
| QAT | Requires training; out of scope for PTQ-focused educational repo |

---

<!-- Template for future entries:

## [x.y.z] — YYYY-MM-DD — <title>

### Added / Changed / Fixed / Removed
- `type: description` — demand source / motivation

### Breaking changes (if any)
- ...

-->
