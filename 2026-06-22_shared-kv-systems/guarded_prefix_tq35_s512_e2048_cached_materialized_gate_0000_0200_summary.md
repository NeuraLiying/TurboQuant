# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `200`.
- Groups: `50`.
- Candidate score: `0.446931`.
- fp16 Shared-KV reference score: `0.438445`.
- Score delta: `0.008486`.
- Candidate avg latency: `0.276909s`.
- Reference avg latency: `0.276031s`.
- Candidate/reference latency ratio: `1.00x`.
- Score mismatches: `14`.
- Prediction mismatches: `22`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `49.67%`.
- Shared-total saving vs fp16 Shared-KV: `48.71%`.
- Candidate shared total bytes: `24387198336`.
- Reference shared total bytes: `47551620480`.
- Candidate transient materialized prefix-cache bytes: `30236868608`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Transient prefix cache | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 200 | 0.446931 | 0.438445 | 0.008486 | 48.71% | 1.00x | 30236868608 | 14 | 22 |

## Candidate Metadata

```json
{
  "prefix_storage_mode": "turboquant",
  "prefix_kv_bits": 3.5,
  "prefix_key_bits": 3.5,
  "prefix_value_bits": 3.5,
  "prefix_key_quantizer": "mse",
  "prefix_value_quantizer": "mse",
  "prefix_raw_start_tokens": 512,
  "prefix_raw_end_tokens": 2048
}
```

## Score Mismatch Preview

- index `1`: candidate `1.0` vs reference `0.0`
- index `14`: candidate `1.0` vs reference `0.0`
- index `21`: candidate `0.0` vs reference `0.25`
- index `45`: candidate `1.0` vs reference `0.6666666666666666`
- index `46`: candidate `1.0` vs reference `0.26666666666666666`
- index `47`: candidate `1.0` vs reference `0.6666666666666666`
- index `90`: candidate `0.3076923076923077` vs reference `0.32`
- index `106`: candidate `0.0` vs reference `1.0`
- index `129`: candidate `1.0` vs reference `0.6666666666666666`
- index `130`: candidate `1.0` vs reference `0.6666666666666666`
