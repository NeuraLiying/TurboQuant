# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `40`.
- Groups: `10`.
- Candidate score: `0.326964`.
- fp16 Shared-KV reference score: `0.308214`.
- Score delta: `0.018750`.
- Candidate avg latency: `0.545233s`.
- Reference avg latency: `0.286471s`.
- Candidate/reference latency ratio: `1.90x`.
- Score mismatches: `2`.
- Prediction mismatches: `3`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `25.74%`.
- Shared-total saving vs fp16 Shared-KV: `25.32%`.
- Candidate shared total bytes: `7783081856`.
- Reference shared total bytes: `10422060928`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 40 | 0.326964 | 0.308214 | 0.018750 | 25.32% | 1.90x | 2 | 3 |

## Candidate Metadata

```json
{
  "prefix_storage_mode": "turboquant",
  "prefix_kv_bits": 3.5,
  "prefix_key_bits": 16.0,
  "prefix_value_bits": 3.5,
  "prefix_key_quantizer": "mse",
  "prefix_value_quantizer": "mse",
  "prefix_raw_start_tokens": 512,
  "prefix_raw_end_tokens": 2048
}
```

## Score Mismatch Preview

- index `1`: candidate `1.0` vs reference `0.0`
- index `21`: candidate `0.0` vs reference `0.25`
