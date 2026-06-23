# Guarded Quantized-Prefix Shard Summary

This is a guarded shard result, not full-run decision evidence.

## Aggregate

- Records: `200`.
- Groups: `50`.
- Candidate score: `0.446931`.
- fp16 Shared-KV reference score: `0.438445`.
- Score delta: `0.008486`.
- Candidate avg latency: `0.864841s`.
- Reference avg latency: `0.276031s`.
- Candidate/reference latency ratio: `3.13x`.
- Score mismatches: `14`.
- Prediction mismatches: `22`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `49.67%`.
- Shared-total saving vs fp16 Shared-KV: `48.71%`.
- Candidate shared total bytes: `24387198336`.
- Reference shared total bytes: `47551620480`.

## Candidate Metadata

```json
{
  "prefix_storage_mode": "turboquant",
  "prefix_kv_bits": 3.5,
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
