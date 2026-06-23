# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `200`.
- Groups: `50`.
- Candidate score: `1.000000`.
- fp16 Shared-KV reference score: `1.000000`.
- Score delta: `0.000000`.
- Candidate avg latency: `1.101180s`.
- Reference avg latency: `0.298951s`.
- Candidate/reference latency ratio: `3.68x`.
- Score mismatches: `0`.
- Prediction mismatches: `0`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `62.39%`.
- Shared-total saving vs fp16 Shared-KV: `57.65%`.
- Candidate shared total bytes: `37396249984`.
- Reference shared total bytes: `88292861312`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| passage_retrieval_en | 200 | 1.000000 | 1.000000 | 0.000000 | 57.65% | 3.68x | 0 | 0 |

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

- None.
