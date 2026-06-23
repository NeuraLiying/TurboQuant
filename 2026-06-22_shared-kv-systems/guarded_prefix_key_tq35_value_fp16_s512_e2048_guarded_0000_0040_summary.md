# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `40`.
- Groups: `10`.
- Candidate score: `0.339464`.
- fp16 Shared-KV reference score: `0.308214`.
- Score delta: `0.031250`.
- Candidate avg latency: `0.538863s`.
- Reference avg latency: `0.286471s`.
- Candidate/reference latency ratio: `1.88x`.
- Score mismatches: `2`.
- Prediction mismatches: `3`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `25.74%`.
- Shared-total saving vs fp16 Shared-KV: `25.33%`.
- Candidate shared total bytes: `7782295424`.
- Reference shared total bytes: `10422060928`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 40 | 0.339464 | 0.308214 | 0.031250 | 25.33% | 1.88x | 2 | 3 |

## Candidate Metadata

```json
{
  "prefix_storage_mode": "turboquant",
  "prefix_kv_bits": 3.5,
  "prefix_key_bits": 3.5,
  "prefix_value_bits": 16.0,
  "prefix_key_quantizer": "mse",
  "prefix_value_quantizer": "mse",
  "prefix_raw_start_tokens": 512,
  "prefix_raw_end_tokens": 2048
}
```

## Score Mismatch Preview

- index `1`: candidate `1.0` vs reference `0.0`
- index `22`: candidate `0.25` vs reference `0.0`
