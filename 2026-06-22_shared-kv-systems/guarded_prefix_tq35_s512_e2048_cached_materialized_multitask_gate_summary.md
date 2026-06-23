# Guarded Quantized-Prefix Multi-Task Gate

This aggregates three 200-row repeated-context gates. It is stronger than a smoke test but still not a full 2400-row decision result.

## Aggregate

- Records: `600`.
- Groups: `150`.
- Candidate score: `0.572944`.
- fp16 Shared-KV reference score: `0.570105`.
- Score delta: `0.002839`.
- Shared-total saving vs fp16 Shared-KV: `57.88%`.
- Transient materialized prefix-cache bytes: `181366423552`.
- Score mismatches: `25`.
- Prediction mismatches: `48`.

## By Task

| Task | Records | Candidate | Reference | Delta | Shared saving | Latency ratio | Transient prefix cache | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | 200 | 0.446931 | 0.438445 | 0.008486 | 48.71% | 1.00x | 30236868608 | 14 | 22 |
| `musique` | 200 | 0.271901 | 0.271869 | 0.000032 | 63.49% | 1.15x | 86328999936 | 11 | 26 |
| `passage_retrieval_en` | 200 | 1.000000 | 1.000000 | 0.000000 | 56.19% | 1.11x | 64800555008 | 0 | 0 |

## Interpretation

Use this gate to decide whether a guarded policy is worth a full run. Passing evidence requires aggregate quality near the fp16 Shared-KV reference, no large task-level regression, and enough storage saving to justify the stronger full validation. Runtime remains a separate blocker because the current prototype decodes/materializes quantized prefix K/V in Python.
