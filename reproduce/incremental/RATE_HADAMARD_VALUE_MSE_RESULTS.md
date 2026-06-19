# Rate-Hadamard Value MSE Results

## Purpose

Explore a method-level increment closer to TurboQuant's core rotation/preconditioning mechanism, instead of a prompt gate or bit-width search.

Earlier value-side Hadamard probes showed complementary behavior:

- Conservative per-vector value outlier-Hadamard improves 2.5-bit MultiQA but fails 3.5-bit MultiQA.
- Late-layer value outlier-Hadamard improves 3.5-bit full 2WikiMQA but fails 2.5-bit full 2WikiMQA.

This experiment combines those observations into one fixed quantizer rule.

## Method

Quantizer: `rate_hadamard_value_mse`

- K path: always use reproduced TurboQuant MSE.
- V path at low rate, `bits < 3.0`: use conservative per-vector value outlier-Hadamard, activated only when the Hadamard candidate improves local reconstruction MSE by at least 5%.
- V path at higher rate, `bits >= 3.0`: use reproduced TurboQuant MSE for layers 0-15 and value outlier-Hadamard for layers 16-31.
- Block-Hadamard size: 16.

This is a rate-aware value preconditioning rule inside the KV cache quantizer. It uses the same method name and code path for 2.5-bit and 3.5-bit runs.

## Validation

```bash
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
```

Result:

```text
91 passed
```

## Full MultiQA Results

| Method | KV bits | HotpotQA | 2WikiMQA | MuSiQue | MultiQA Avg | Delta vs TQ MultiQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant | 2.5 | 44.83 | 37.96 | 25.24 | 36.01 | +0.00 |
| Rate-Hadamard Value MSE | 2.5 | 49.05 | 38.88 | 23.54 | 37.16 | +1.15 |
| TurboQuant | 3.5 | 54.65 | 44.47 | 30.01 | 43.04 | +0.00 |
| Rate-Hadamard Value MSE | 3.5 | 56.00 | 46.09 | 29.11 | 43.73 | +0.69 |

## Artifacts

- `reproduce/runs/incremental/rate_hadamard_value_mse_{hotpotqa,2wikimqa,musique}_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/rate_hadamard_value_mse_multiqa_progress.md`
- `reproduce/incremental/rate_hadamard_value_mse_multiqa_progress.json`
- `reproduce/incremental/rate_hadamard_value_mse_multiqa_progress.csv`

## Analysis

- The method passes the current MultiQA gate at both bit budgets: `+1.15` at 2.5-bit and `+0.69` at 3.5-bit.
- The gains come from a Hadamard/value-preconditioning rule, not a prompt/task gate and not a new intermediate bit-width setting.
- MuSiQue remains negative at both bit budgets, so the method is not yet proven as a full Table 1 contribution.
- The next validation step is to extend the same quantizer to all 16 Table 1 tasks and compare the macro average against reproduced TurboQuant at both 2.5-bit and 3.5-bit.
