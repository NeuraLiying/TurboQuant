# Group Admission Policy Analysis

This offline analysis simulates a policy that quantizes a shared-prefix group only
when an admission rule accepts it; rejected groups fall back to fp16 Shared-KV.

## Aggregate

- Policy: `probe_mean`.
- Probe variants: `1`.
- Rows/groups: `2400` / `600`.
- Accepted groups: `590`; rejected groups: `10`.
- Acceptance ratio: `98.33%`.
- Simulated score: `0.595980`.
- fp16 Shared-KV reference score: `0.587374`.
- Score delta vs reference: `0.008606`.
- Score mismatches vs reference: `81`.
- Prediction mismatches vs reference: `208`.
- Latency ratio vs reference: `4.17x`.

## Storage

- Prefix-table saving vs fp16 Shared-KV: `58.83%`.
- Shared-total saving vs fp16 Shared-KV: `56.71%`.
- Simulated shared-total bytes: `415296342528`.
- Reference shared-total bytes: `959330370048`.

## Task Breakdown

| Task | Rows | Accepted rows | Acceptance | Sim score | Ref score | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 800 | 780 | 97.50% | 0.468644 | 0.457359 | 0.011285 |
| musique | 800 | 784 | 98.00% | 0.320546 | 0.309762 | 0.010784 |
| passage_retrieval_en | 800 | 796 | 99.50% | 0.998750 | 0.995000 | 0.003750 |

## Decision Summary

```json
{
  "accepted_mean_delta": 0.008752088709763,
  "rejected_mean_delta": -0.3178825852510063,
  "accepted_probe_delta": 0.011365420889572838,
  "rejected_probe_delta": -0.6489036986405408
}
```
