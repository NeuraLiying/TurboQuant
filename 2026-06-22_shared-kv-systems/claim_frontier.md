# Shared-KV Claim Frontier

This report consolidates complete generated repeated-context artifacts for the new Shared-KV systems direction.
Storage-only estimates are explicitly marked and are not quality claims.

## Claim-Ready Result

- fp16 Shared-KV score delta vs full fp16: `0.0097` points.
- fp16 Shared-KV latency speedup vs full fp16: `5.14x`.
- fp16 Shared-KV persistent storage saving vs materialized full fp16: `74.30%`.
- Clone parity: `2400` rows compared, `0` mismatches.

## Pareto Observation

- TurboQuant 3.5 independent persistent KV / fp16 Shared-KV persistent KV: `0.912x`.
- fp16 Shared-KV is `1.87` points higher than TurboQuant 3.5 and `28.78x` faster.
- fp16 Shared-KV is `8.58` points higher than TurboQuant 2.5 and `11.58x` faster.

Interpretation: in this repeated-context workload, exact cross-request sharing is a stronger first move than independent per-request low-bit compression. At roughly the same persistent KV size as TurboQuant 3.5, fp16 Shared-KV preserves quality and is much faster.

## Guarded TurboQuant Shared-Prefix Result

- Full guarded-prefix run: `2400` rows, `600` groups.
- Score delta vs fp16 Shared-KV reference: `0.3308` points.
- Score mismatches vs fp16 Shared-KV: `105`; prediction mismatches: `233`.
- Shared-total storage saving vs fp16 Shared-KV: `57.82%`.
- Persistent storage saving vs independent full fp16: `89.16%`.
- Latency is `4.32x` slower than fp16 Shared-KV reference in the current Python materialized prototype.

Interpretation: this is a full storage-quality result for a TurboQuant-composed Shared-KV table. It should not be reported as a speedup until the quantized prefix table has a non-materializing or fused decode path.

## Cached-Materialized Prefix Runtime Diagnostic

- Full run: `2400` rows, `600` groups.
- Score delta vs fp16 Shared-KV reference: `0.3308` points.
- Shared-total storage saving vs fp16 Shared-KV: `57.82%`.
- Latency ratios vs fp16 Shared-KV: `1.12x` aggregate, by task `2wikimqa` `1.08x`, `musique` `1.16x`, `passage_retrieval_en` `1.11x`.
- Additional transient dense prefix-cache bytes: `674.78 GB`.

Interpretation: caching one dense materialization of the quantized shared prefix removes most of the Python dequantization/materialization latency, but it is a diagnostic tradeoff rather than a final kernel result because it uses extra transient dense memory.

## Frontier Table

| Method | Status | Score | Delta vs fp16 | Latency | Persistent KV | Saving vs full | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full fp16 independent | claim-ready baseline | 58.73 | 0.00 | 1.730s | 3.40 TB | 0.00% | measured full run |
| fp16 Shared-KV reference | claim-ready systems result | 58.74 | 0.01 | 0.336s | 893.45 GB | 74.30% | measured full run, 2400/2400 clone parity |
| TurboQuant 2.5 independent | measured negative codec baseline | 50.15 | -8.57 | 3.896s | 597.77 GB | 82.81% | measured full run |
| TurboQuant 3.5 independent | measured Pareto comparator | 56.87 | -1.86 | 9.683s | 815.13 GB | 76.56% | measured full run |
| Guarded TQ3.5 shared prefix, fp16 local | full storage-quality result; no speed claim | 59.07 | 0.34 | 1.453s | 376.81 GB | 89.16% | measured full run, 2400 rows; 105 score mismatches vs fp16 Shared-KV |
| TurboQuant 2.5 plus shared-prefix estimate | storage-only estimate | 50.15 | -8.57 | n/a | 151.40 GB | 95.65% | not validated in shared-cache runtime |
| TurboQuant 3.5 plus shared-prefix estimate | storage-only estimate | 56.87 | -1.86 | n/a | 206.46 GB | 94.06% | not validated in shared-cache runtime |
| Quantized shared prefix, fp16 local estimate (2.5) | next-experiment storage target | n/a | n/a | n/a | 180.59 GB | 94.81% | unknown; must run full protocol |
| Quantized shared prefix, fp16 local estimate (3.5) | next-experiment storage target | n/a | n/a | n/a | 234.43 GB | 93.26% | unknown; must run full protocol |
| fp16 shared prefix, quantized local estimate (2.5) | low-value storage estimate | n/a | n/a | n/a | 866.81 GB | 75.07% | unknown; likely small upside |
| fp16 shared prefix, quantized local estimate (3.5) | low-value storage estimate | n/a | n/a | n/a | 868.84 GB | 75.01% | unknown; likely small upside |
| Guarded TQ3.5 shared prefix + dense prefix cache | runtime diagnostic; no final speed claim | 59.07 | 0.34 | 0.377s | 376.81 GB | 89.16% | full run, 2400 rows; extra transient prefix cache 674.78 GB |

## Storage Decomposition

- Shared prefix table: `861.22 GB`.
- Request-local suffix/decode KV: `32.51 GB`.
- Metadata: `112.50 KB`.
- Shared prefix fraction of non-metadata shared storage: `96.36%`.

This makes suffix-only quantization a weak next contribution. The dominant experiment is prefix-table quantization with fp16 request-local KV.

## Current Limitation

- Python split attention failed full row-level parity: `38` score mismatches and `77` prediction mismatches.
- Materialized eager vs clone also has backend drift: `32` score mismatches and `77` prediction mismatches.

Do not claim non-materializing split attention yet. The next implementation target is SDPA-equivalent prefix-aware attention or a predeclared relaxed score-parity protocol.
