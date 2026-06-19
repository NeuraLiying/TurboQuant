# Incremental Result Summary

This document summarizes the current post-reproduction state for `meta-llama/Llama-3.1-8B-Instruct`.
The full experiment log is in `reproduce/INCREMENTAL_EXPERIMENTS.md`.

## Reproduction Baseline

The reproduction baseline is fixed in:

- `reproduce/TABLE1_OFFICIAL_COMPARISON.md`
- `reproduce/TABLE1_OFFICIAL_COMPARISON.csv`
- `reproduce/TABLE1_OFFICIAL_COMPARISON.json`

The reproduced Table 1 settings are:

| Method | KV bits | Complete Tasks | Average |
| --- | ---: | ---: | ---: |
| Full Cache | 16.0 | 16 / 16 | 50.38 |
| TurboQuant | 2.5 | 16 / 16 | 45.42 |
| TurboQuant | 3.5 | 16 / 16 | 49.38 |

## Complete Incremental Result

Unified Regular-Gain Gate is the current complete reportable increment. It uses one shared gate rule for both 2.5-bit and 3.5-bit runs and is evaluated on all 16 LongBench Table 1 tasks.

| Method | KV bits | Complete Tasks | TurboQuant Avg | Method Avg | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unified Regular-Gain Gate | 2.5 | 16 / 16 | 45.42 | 45.95 | +0.53 |
| Unified Regular-Gain Gate | 3.5 | 16 / 16 | 49.38 | 49.53 | +0.15 |

Detailed category and task-level results are in:

- `reproduce/incremental/UNIFIED_REGULAR_GAIN_GATE_FINAL.md`
- `reproduce/incremental/unified_regular_gain_gate_table1_runner.md`

## Additional Method-Level Evidence

Rate-Regime MSE is a newer reconstruction/preconditioning branch. Its 3.5-bit Table 1 run is complete; the 2.5-bit full run is still in progress, so the 2.5-bit average is not fixed yet.

| Method | KV bits | Complete Tasks | Evidence |
| --- | ---: | ---: | --- |
| Rate-Regime MSE | 2.5 | 10 / 16 | MultiQA is complete: 36.01 -> 36.55, delta +0.54. |
| Rate-Regime MSE | 3.5 | 16 / 16 | Full Table 1 average: 49.38 -> 49.86, delta +0.48. |

Detailed progress is in `reproduce/incremental/rate_regime_mse_table1.md`.

## Validation

Last core validation command:

```bash
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
```

Expected result: `89 passed`.
