# Group Admission Policy Analysis

This offline analysis simulates a policy that quantizes a shared-prefix group only
when an admission rule accepts it; rejected groups fall back to fp16 Shared-KV.

## Aggregate

- Policy: `all`.
- Probe variants: `1`.
- Rows/groups: `2400` / `600`.
- Accepted groups: `600`; rejected groups: `0`.
- Acceptance ratio: `100.00%`.
- Simulated score: `0.590682`.
- fp16 Shared-KV reference score: `0.587374`.
- Score delta vs reference: `0.003308`.
- Score mismatches vs reference: `105`.
- Prediction mismatches vs reference: `233`.
- Latency ratio vs reference: `4.32x`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `59.99%`.
- Shared-total saving vs fp16 Shared-KV: `57.82%`.
- Simulated shared-total bytes: `404597824000`.
- Reference shared-total bytes: `959330370048`.

## Task Breakdown

| Task | Rows | Accepted rows | Acceptance | Sim score | Ref score | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 800 | 800 | 100.00% | 0.464145 | 0.457359 | 0.006786 |
| musique | 800 | 800 | 100.00% | 0.314151 | 0.309762 | 0.004389 |
| passage_retrieval_en | 800 | 800 | 100.00% | 0.993750 | 0.995000 | -0.001250 |

## Decision Summary

```json
{
  "accepted_mean_delta": 0.003308177477083511,
  "rejected_mean_delta": null,
  "accepted_probe_delta": 0.0003609355640709446,
  "rejected_probe_delta": null
}
```
