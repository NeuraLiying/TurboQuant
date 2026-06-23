# Guarded Quantized-Prefix Shard Summary

This is a guarded shard result, not full-run decision evidence.

## Aggregate

- Records: `200`.
- Groups: `50`.
- Candidate score: `0.271901`.
- fp16 Shared-KV reference score: `0.271869`.
- Score delta: `0.000032`.
- Candidate avg latency: `2.413447s`.
- Reference avg latency: `0.452763s`.
- Candidate/reference latency ratio: `5.33x`.
- Score mismatches: `11`.
- Prediction mismatches: `26`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `64.10%`.
- Shared-total saving vs fp16 Shared-KV: `63.49%`.
- Candidate shared total bytes: `38011221376`.
- Reference shared total bytes: `104107353472`.

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

- index `842`: candidate `0.0` vs reference `1.0`
- index `845`: candidate `0.28571428571428575` vs reference `0.125`
- index `870`: candidate `0.6666666666666666` vs reference `0.8`
- index `918`: candidate `1.0` vs reference `0.0`
- index `919`: candidate `1.0` vs reference `0.0`
- index `924`: candidate `0.18181818181818182` vs reference `0.7499999999999999`
- index `927`: candidate `0.18181818181818182` vs reference `0.7499999999999999`
- index `988`: candidate `0.12903225806451613` vs reference `0.06666666666666667`
- index `989`: candidate `0.12121212121212122` vs reference `0.06666666666666667`
- index `990`: candidate `0.06896551724137931` vs reference `0.125`
