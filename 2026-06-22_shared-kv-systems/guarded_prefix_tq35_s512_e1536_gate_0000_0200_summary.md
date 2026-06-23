# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `200`.
- Groups: `50`.
- Candidate score: `0.434845`.
- fp16 Shared-KV reference score: `0.438445`.
- Score delta: `-0.003600`.
- Candidate avg latency: `0.898829s`.
- Reference avg latency: `0.276031s`.
- Candidate/reference latency ratio: `3.26x`.
- Score mismatches: `12`.
- Prediction mismatches: `23`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `54.82%`.
- Shared-total saving vs fp16 Shared-KV: `53.75%`.
- Candidate shared total bytes: `21993260416`.
- Reference shared total bytes: `47551620480`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 200 | 0.434845 | 0.438445 | -0.003600 | 53.75% | 3.26x | 12 | 23 |

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
- index `42`: candidate `0.0` vs reference `1.0`
- index `44`: candidate `0.26666666666666666` vs reference `1.0`
- index `67`: candidate `1.0` vs reference `0.0`
- index `90`: candidate `1.0` vs reference `0.32`
- index `148`: candidate `0.5` vs reference `0.3333333333333333`
- index `151`: candidate `0.5` vs reference `0.3333333333333333`
- index `152`: candidate `0.25` vs reference `1.0`
