# Group Admission Policy Analysis

This offline analysis simulates a policy that quantizes a shared-prefix group only
when an admission rule accepts it; rejected groups fall back to fp16 Shared-KV.

## Aggregate

- Policy: `oracle_mean`.
- Probe variants: `1`.
- Rows/groups: `2400` / `600`.
- Accepted groups: `572`; rejected groups: `28`.
- Acceptance ratio: `95.33%`.
- Simulated score: `0.600308`.
- fp16 Shared-KV reference score: `0.587374`.
- Score delta vs reference: `0.012935`.
- Score mismatches vs reference: `63`.
- Prediction mismatches vs reference: `185`.
- Latency ratio vs reference: `4.06x`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `57.08%`.
- Shared-total saving vs fp16 Shared-KV: `55.02%`.
- Simulated shared-total bytes: `431473365504`.
- Reference shared-total bytes: `959330370048`.

## Task Breakdown

| Task | Rows | Accepted rows | Acceptance | Sim score | Ref score | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 800 | 744 | 93.00% | 0.475954 | 0.457359 | 0.018595 |
| musique | 800 | 748 | 93.50% | 0.326221 | 0.309762 | 0.016459 |
| passage_retrieval_en | 800 | 796 | 99.50% | 0.998750 | 0.995000 | 0.003750 |

## Decision Summary

```json
{
  "accepted_mean_delta": 0.013567688592925876,
  "rejected_mean_delta": -0.2062789781751248,
  "accepted_probe_delta": 0.0112972177994025,
  "rejected_probe_delta": -0.22305168724341656
}
```
