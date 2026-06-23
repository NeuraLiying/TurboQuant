# Group Admission Policy Analysis

This offline analysis simulates a policy that quantizes a shared-prefix group only
when an admission rule accepts it; rejected groups fall back to fp16 Shared-KV.

## Aggregate

- Policy: `none`.
- Probe variants: `1`.
- Rows/groups: `2400` / `600`.
- Accepted groups: `0`; rejected groups: `600`.
- Acceptance ratio: `0.00%`.
- Simulated score: `0.587374`.
- fp16 Shared-KV reference score: `0.587374`.
- Score delta vs reference: `0.000000`.
- Score mismatches vs reference: `0`.
- Prediction mismatches vs reference: `0`.
- Latency ratio vs reference: `1.00x`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `0.00%`.
- Shared-total saving vs fp16 Shared-KV: `0.00%`.
- Simulated shared-total bytes: `959330370048`.
- Reference shared-total bytes: `959330370048`.

## Task Breakdown

| Task | Rows | Accepted rows | Acceptance | Sim score | Ref score | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 800 | 0 | 0.00% | 0.457359 | 0.457359 | 0.000000 |
| musique | 800 | 0 | 0.00% | 0.309762 | 0.309762 | 0.000000 |
| passage_retrieval_en | 800 | 0 | 0.00% | 0.995000 | 0.995000 | 0.000000 |

## Decision Summary

```json
{
  "accepted_mean_delta": null,
  "rejected_mean_delta": 0.003308177477083511,
  "accepted_probe_delta": null,
  "rejected_probe_delta": 0.0003609355640709446
}
```
