# Guarded Quantized-Prefix Summary

This summary compares guarded quantized-prefix output against the fp16 Shared-KV reference.

## Aggregate

- Records: `2400`.
- Groups: `600`.
- Candidate score: `0.590682`.
- fp16 Shared-KV reference score: `0.587374`.
- Score delta: `0.003308`.
- Candidate avg latency: `0.377205s`.
- Reference avg latency: `0.336493s`.
- Candidate/reference latency ratio: `1.12x`.
- Score mismatches: `105`.
- Prediction mismatches: `233`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `59.99%`.
- Shared-total saving vs fp16 Shared-KV: `57.82%`.
- Candidate shared total bytes: `404597824000`.
- Reference shared total bytes: `959330370048`.
- Candidate transient materialized prefix-cache bytes: `724544651264`.

## Task Breakdown

| Task | Records | Score | Ref score | Delta | Shared saving | Latency ratio | Transient prefix cache | Score mismatches | Prediction mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 800 | 0.464145 | 0.457359 | 0.006786 | 48.94% | 1.08x | 123549515776 | 45 | 89 |
| musique | 800 | 0.314151 | 0.309762 | 0.004389 | 63.44% | 1.16x | 343826235392 | 53 | 135 |
| passage_retrieval_en | 800 | 0.993750 | 0.995000 | -0.001250 | 56.08% | 1.11x | 257168900096 | 7 | 9 |

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
