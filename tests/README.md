# tests/

Unit and integration tests. All tests run offline — no HuggingFace model downloads.

## Running Tests

```bash
pytest                      # all tests
pytest -v                   # verbose output
pytest tests/test_ppl.py    # specific file
```

## Test Files

| File | What it tests |
|------|--------------|
| `test_quant_primitives.py` | Symmetric/asymmetric scale, quantize, dequantize, round-trip error |
| `test_group_quant.py` | Group-wise quantization and dequantization shapes and values |
| `test_quant_linear.py` | QuantLinear forward pass, from_linear factory, bias handling |
| `test_rtn.py` | RTN quantization on synthetic nn.Linear modules |
| `test_awq_lite.py` | AWQ channel scales, weight scaling, output equivalence |
| `test_gptq_lite.py` | GPTQ column-by-column compensation convergence |
| `test_safetensors_io.py` | Save/load roundtrip for RTN, asymmetric, and AWQ models |
| `test_ppl.py` | PPL function returns finite positive float |
| `test_data_loader.py` | Calibration and eval text loading |

## Design Rules

- No internet: all tests use synthetic tensors or bundled data in `data/`
- No real HF models: if a test needs a real model, mark with `@pytest.mark.skip`
- Fast: total suite runs in under 60 seconds
- Every public function in `nanoptq/` has at least one test

## Expected Output

```
60 passed in N.Ns
```
