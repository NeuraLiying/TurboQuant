# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `40`.
- Groups: `10`.
- Candidate score: `0.351964`.
- fp16 Shared-KV reference score: `0.308214`.
- Score delta: `0.043750`.
- Candidate avg latency: `0.843567s`.
- Reference avg latency: `0.286471s`.
- Candidate/reference latency ratio: `2.94x`.
- Score mismatches: `3`.
- Prediction mismatches: `4`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `56.49%`.
- Shared-total saving vs fp16 Shared-KV: `55.54%`.
- Candidate shared total bytes: `4633708416`.
- Reference shared total bytes: `10422060928`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 40 | 0.351964 | 0.308214 | 0.043750 | 55.54% | 2.94x | 3 | 4 |

## Candidate Metadata

```json
{
  "prefix_storage_mode": "turboquant",
  "prefix_kv_bits": 3.5,
  "prefix_key_quantizer": "mse",
  "prefix_value_quantizer": "mse",
  "prefix_raw_start_tokens": 512,
  "prefix_raw_end_tokens": 1536
}
```

## Score Mismatch Preview

- index `0`: candidate `1.0` vs reference `0.0`
- index `1`: candidate `1.0` vs reference `0.0`
- index `21`: candidate `0.0` vs reference `0.25`
