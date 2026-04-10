# nanoPTQ — Project Plan & Progress

> Progress tracking for the nanoPTQ implementation.
> 进度追踪文件：记录计划、当前状态和已完成任务。

---

## Milestones

| Phase | Description | Status |
|-------|-------------|--------|
| P0 | Core math + RTN baseline + safetensors I/O + PPL eval | 🔲 Not started |
| P1 | Group-wise int4 (group_size=128) + bundled datasets | 🔲 Not started |
| P2 | AWQ-lite (activation-aware channel scaling) | 🔲 Not started |
| P3 | GPTQ-lite (Hessian-based layer-wise update) | 🔲 Not started |
| P4 | CLI + end-to-end examples + compare script | 🔲 Not started |

---

## Task Checklist

### P0 — Foundation

- [x] Task 0.5: Bundled datasets (data/calibration/ + data/eval/) + dataset loader
- [ ] Task 1: `nanoptq/core/quant_primitives.py` — symmetric, asymmetric, fake_quant
- [ ] Task 2: `nanoptq/core/group_quant.py` — group-wise quantization
- [ ] Task 3: `nanoptq/model/quant_linear.py` — QuantLinear unified abstraction
- [ ] Task 4: `nanoptq/model/hf_loader.py` — HF model loading + Linear surgery
- [ ] Task 5: `nanoptq/algorithms/rtn.py` — RTN baseline (no calibration)
- [ ] Task 6: `nanoptq/io/safetensors_io.py` — save/load quantized checkpoints
- [ ] Task 7: `nanoptq/eval/ppl.py` — perplexity (wikitext-2, bundled)

### P2 — AWQ

- [ ] Task 8: `nanoptq/algorithms/awq_lite.py` — activation-aware weight scaling
- [ ] Task 8.5: Wire AWQ to use bundled calibration data by default

### P3 — GPTQ

- [ ] Task 9: `nanoptq/algorithms/gptq_lite.py` — Hessian-based layer-wise update
- [ ] Task 9.5: Wire GPTQ to use bundled calibration data by default

### P4 — Usability

- [ ] Task 10: `nanoptq/eval/latency.py` — tokens/s, prefill, peak memory
- [ ] Task 11: `nanoptq/cli.py` — quantize / eval / compare
- [ ] Task 12: `examples/` — end-to-end scripts

---

## Design Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-10 | Only quantize `nn.Linear`, skip embeddings/norms | All major frameworks (AWQ, GPTQ, torchao) target Linear; norm layers have too few params to matter |
| 2026-04-10 | Dequant-on-the-fly in `QuantLinear.forward()` | No CUDA kernel needed; backend handles matmul; preserves model.generate() compat |
| 2026-04-10 | group_size=128 default | AWQ, GPTQ, torchao consensus; good precision/size balance for int4 |
| 2026-04-10 | Bundle small datasets in `data/` | Reproducible eval without internet; mirrors AngelSlim approach; one-stop eval for students |
| 2026-04-10 | Skip `lm_head` by default | Output projection has different sensitivity profile; quantizing it often hurts PPL disproportionately |
| 2026-04-10 | `symmetric=True` default for weights | Most industrial deployments use symmetric for weights (simpler hardware); asymmetric is opt-in |

---

## Open Questions

- [ ] Should `data/calibration/` use raw text (`.jsonl`) or pre-tokenized tensors? (raw text is model-agnostic; pre-tokenized saves runtime)
- [ ] Add a `data/eval/c4_eval_200.jsonl` for a second eval domain?
- [ ] P4+: Triton dequant kernel to show what production fusion looks like?

---

## Completed Tasks

_(moved here as tasks finish)_

- [x] Task 0: Project scaffold (pyproject.toml, README bilingual, __init__ stubs)

---

*Last updated: 2026-04-10*
