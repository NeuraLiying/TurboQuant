# Guarded Quantized-Prefix Multi-Task Gate

This aggregates three 200-row repeated-context gates. It is stronger than a smoke test but still not a full 2400-row decision result.

## Aggregate

- Records: `600`.
- Groups: `150`.
- Candidate score: `0.570496`.
- fp16 Shared-KV reference score: `0.570105`.
- Score delta: `0.000392`.
- Shared-total saving vs fp16 Shared-KV: `61.01%`.
- Score mismatches: `21`.
- Prediction mismatches: `42`.

## By Task

| Task | Records | Candidate | Reference | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | 200 | 0.434845 | 0.438445 | -0.003600 | 53.75% | 3.26x | 12 | 23 |
| `musique` | 200 | 0.276644 | 0.271869 | 0.004775 | 65.96% | 5.55x | 9 | 19 |
| `passage_retrieval_en` | 200 | 1.000000 | 1.000000 | 0.000000 | 59.10% | 3.77x | 0 | 0 |

## Interpretation

Use this gate to decide whether a guarded policy is worth a full run. Passing evidence requires aggregate quality near the fp16 Shared-KV reference, no large task-level regression, and enough storage saving to justify the stronger full validation. Runtime remains a separate blocker because the current prototype decodes/materializes quantized prefix K/V in Python.
