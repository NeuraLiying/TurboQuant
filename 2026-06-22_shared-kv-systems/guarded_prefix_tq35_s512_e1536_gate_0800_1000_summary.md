# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `200`.
- Groups: `50`.
- Candidate score: `0.276644`.
- fp16 Shared-KV reference score: `0.271869`.
- Score delta: `0.004775`.
- Candidate avg latency: `2.512541s`.
- Reference avg latency: `0.452763s`.
- Candidate/reference latency ratio: `5.55x`.
- Score mismatches: `9`.
- Prediction mismatches: `19`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `66.60%`.
- Shared-total saving vs fp16 Shared-KV: `65.96%`.
- Candidate shared total bytes: `35441030528`.
- Reference shared total bytes: `104107353472`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| musique | 200 | 0.276644 | 0.271869 | 0.004775 | 65.96% | 5.55x | 9 | 19 |

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

- index `833`: candidate `1.0` vs reference `0.0`
- index `842`: candidate `0.0` vs reference `1.0`
- index `845`: candidate `0.28571428571428575` vs reference `0.125`
- index `888`: candidate `0.0` vs reference `0.20000000000000004`
- index `918`: candidate `1.0` vs reference `0.0`
- index `988`: candidate `0.12903225806451613` vs reference `0.06666666666666667`
- index `989`: candidate `0.0` vs reference `0.06666666666666667`
- index `990`: candidate `0.06896551724137931` vs reference `0.125`
- index `991`: candidate `0.12121212121212122` vs reference `0.06666666666666667`
