# Guarded Quantized-Prefix Shard Summary

This is a guarded shard result, not full-run decision evidence.

## Aggregate

- Records: `40`.
- Groups: `10`.
- Candidate score: `0.351964`.
- fp16 Shared-KV reference score: `0.308214`.
- Score delta: `0.043750`.
- Candidate avg latency: `0.808822s`.
- Reference avg latency: `0.286471s`.
- Candidate/reference latency ratio: `2.82x`.
- Score mismatches: `3`.
- Prediction mismatches: `4`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `51.48%`.
- Shared-total saving vs fp16 Shared-KV: `50.61%`.
- Candidate shared total bytes: `5147248512`.
- Reference shared total bytes: `10422060928`.

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
