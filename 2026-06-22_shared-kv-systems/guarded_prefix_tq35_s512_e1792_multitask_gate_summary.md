# Guarded Quantized-Prefix Multi-Task Gate

This aggregates three 200-row repeated-context gates. It is stronger than a smoke test but still not a full 2400-row decision result.

## Aggregate

- Records: `600`.
- Groups: `150`.
- Candidate score: `0.562825`.
- fp16 Shared-KV reference score: `0.570105`.
- Score delta: `-0.007280`.
- Shared-total saving vs fp16 Shared-KV: `59.44%`.
- Score mismatches: `24`.
- Prediction mismatches: `47`.

## By Task

| Task | Records | Candidate | Reference | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | 200 | 0.419467 | 0.438445 | -0.018978 | 51.20% | 3.30x | 13 | 22 |
| `musique` | 200 | 0.269007 | 0.271869 | -0.002862 | 64.72% | 5.54x | 11 | 25 |
| `passage_retrieval_en` | 200 | 1.000000 | 1.000000 | 0.000000 | 57.65% | 3.68x | 0 | 0 |

## Interpretation

Use this gate to decide whether a guarded policy is worth a full run. Passing evidence requires aggregate quality near the fp16 Shared-KV reference, no large task-level regression, and enough storage saving to justify the stronger full validation. Runtime remains a separate blocker because the current prototype decodes/materializes quantized prefix K/V in Python.
