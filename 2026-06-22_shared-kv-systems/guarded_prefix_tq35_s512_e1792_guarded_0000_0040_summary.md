# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `40`.
- Groups: `10`.
- Candidate score: `0.320714`.
- fp16 Shared-KV reference score: `0.308214`.
- Score delta: `0.012500`.
- Candidate avg latency: `0.816072s`.
- Reference avg latency: `0.286471s`.
- Candidate/reference latency ratio: `2.85x`.
- Score mismatches: `2`.
- Prediction mismatches: `3`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `53.98%`.
- Shared-total saving vs fp16 Shared-KV: `53.08%`.
- Candidate shared total bytes: `4890478464`.
- Reference shared total bytes: `10422060928`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 40 | 0.320714 | 0.308214 | 0.012500 | 53.08% | 2.85x | 2 | 3 |

## Candidate Metadata

```json
{
  "prefix_storage_mode": "turboquant",
  "prefix_kv_bits": 3.5,
  "prefix_key_quantizer": "mse",
  "prefix_value_quantizer": "mse",
  "prefix_raw_start_tokens": 512,
  "prefix_raw_end_tokens": 1792
}
```

## Score Mismatch Preview

- index `20`: candidate `0.25` vs reference `0.0`
- index `22`: candidate `0.25` vs reference `0.0`
