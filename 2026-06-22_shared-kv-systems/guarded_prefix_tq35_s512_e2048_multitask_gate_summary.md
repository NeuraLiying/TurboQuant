# Guarded Quantized-Prefix Multi-Task Gate

This aggregates three 200-row repeated-context gates. It is stronger than a smoke test but still not a full 2400-row decision result.

## Aggregate

- Records: `600`.
- Groups: `150`.
- Candidate score: `0.572944`.
- fp16 Shared-KV reference score: `0.570105`.
- Score delta: `0.002839`.
- Shared-total saving vs fp16 Shared-KV: `57.88%`.
- Score mismatches: `25`.
- Prediction mismatches: `48`.

## By Task

| Task | Records | Candidate | Reference | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | 200 | 0.446931 | 0.438445 | 0.008486 | 48.71% | 3.13x | 14 | 22 |
| `musique` | 200 | 0.271901 | 0.271869 | 0.000032 | 63.49% | 5.33x | 11 | 26 |
| `passage_retrieval_en` | 200 | 1.000000 | 1.000000 | 0.000000 | 56.19% | 3.59x | 0 | 0 |

## Interpretation

The guarded policy preserves aggregate quality across the three sampled tasks while cutting shared persistent storage by more than half. The remaining blocker is runtime: the current prototype decodes/materializes quantized prefix K/V in Python, so latency is worse than fp16 Shared-KV reference.
