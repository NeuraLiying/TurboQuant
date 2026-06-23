# Guarded Quantized-Prefix Shard Summary

This is a guarded shard result, not full-run decision evidence.

## Aggregate

- Records: `200`.
- Groups: `50`.
- Candidate score: `1.000000`.
- fp16 Shared-KV reference score: `1.000000`.
- Score delta: `0.000000`.
- Candidate avg latency: `1.072789s`.
- Reference avg latency: `0.298951s`.
- Candidate/reference latency ratio: `3.59x`.
- Score mismatches: `0`.
- Prediction mismatches: `0`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `60.82%`.
- Shared-total saving vs fp16 Shared-KV: `56.19%`.
- Candidate shared total bytes: `38680755584`.
- Reference shared total bytes: `88292861312`.

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

- None.
