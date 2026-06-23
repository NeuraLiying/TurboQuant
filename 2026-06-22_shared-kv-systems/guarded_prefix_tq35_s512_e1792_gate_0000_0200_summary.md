# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `200`.
- Groups: `50`.
- Candidate score: `0.419467`.
- fp16 Shared-KV reference score: `0.438445`.
- Score delta: `-0.018978`.
- Candidate avg latency: `0.909702s`.
- Reference avg latency: `0.276031s`.
- Candidate/reference latency ratio: `3.30x`.
- Score mismatches: `13`.
- Prediction mismatches: `22`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `52.23%`.
- Shared-total saving vs fp16 Shared-KV: `51.20%`.
- Candidate shared total bytes: `23204889984`.
- Reference shared total bytes: `47551620480`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 200 | 0.419467 | 0.438445 | -0.018978 | 51.20% | 3.30x | 13 | 22 |

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
- index `45`: candidate `1.0` vs reference `0.6666666666666666`
- index `47`: candidate `1.0` vs reference `0.6666666666666666`
- index `90`: candidate `0.11764705882352941` vs reference `0.32`
- index `106`: candidate `0.0` vs reference `1.0`
- index `130`: candidate `1.0` vs reference `0.6666666666666666`
- index `150`: candidate `0.5` vs reference `1.0`
- index `151`: candidate `0.5` vs reference `0.3333333333333333`
- index `166`: candidate `0.24000000000000002` vs reference `1.0`
