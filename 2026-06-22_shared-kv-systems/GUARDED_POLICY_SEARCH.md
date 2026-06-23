# Guarded Prefix Policy Search

This note records the follow-up policy search after the first full guarded-prefix
result. The goal is not to invent another codec variant, but to find whether the
guarded shared-prefix storage policy can cross the original `90%` persistent-saving
target versus independent fp16 while preserving full quality.

## Baseline Full Policy

Policy:

```text
raw_start = 512
raw_end = 2048
middle = TurboQuant 3.5
local suffix/decode KV = fp16
```

Full result:

- Records: `2400`.
- Score: `0.590682` vs fp16 Shared-KV `0.587374`.
- Shared-total saving vs fp16 Shared-KV: `57.82%`.
- Persistent saving vs independent fp16 materialized KV: `89.16%`.
- Latency: `4.32x` slower than fp16 Shared-KV reference.
- Score mismatches: `105`; prediction mismatches: `233`.

Decision: reportable as storage-quality evidence, but it misses the `90%` saving target
and does not support speedup.

## Follow-Up Hypothesis

The simplest way to improve storage without changing the codec is to reduce the exact
tail guard. The protected tail is quality-sensitive, so this must be gated rather than
estimated.

Candidates:

| Policy | Rationale |
| --- | --- |
| `raw_start=512, raw_end=1536` | More aggressive; likely crosses `90%` full saving if quality holds. |
| `raw_start=512, raw_end=1792` | Conservative midpoint; should save more than `2048` while protecting more tail context than `1536`. |

## Gates

1. Rows `0:40` smoke/gate.
2. Balanced 600-row gate: `2wikimqa 0:200`, `musique 800:1000`,
   `passage_retrieval_en 1600:1800`.
3. Full `2400 / 2400` only if the 600-row gate has:
   - aggregate score delta near zero or positive vs fp16 Shared-KV;
   - no large task-level regression;
   - shared-total saving materially higher than the `2048` baseline.

## Current Results

### `raw_end=1536`

Rows `0:40`:

- Score: `0.351964` vs reference `0.308214`.
- Shared-total saving: `55.54%`.
- Score mismatches: `3`; prediction mismatches: `4`.

Balanced 600-row gate:

- Score: `0.570496` vs reference `0.570105`, delta `+0.000392`.
- Shared-total saving: `61.01%`.
- Score mismatches: `21`; prediction mismatches: `42`.
- Task deltas:
  - `2wikimqa`: `-0.003600`.
  - `musique`: `+0.004775`.
  - `passage_retrieval_en`: `+0.000000`.

Decision: useful but riskier than `2048` because 2Wiki regresses in the balanced gate.
Do not full-run it before checking the midpoint policy.

### `raw_end=1792`

Rows `0:40`:

- Score: `0.320714` vs reference `0.308214`.
- Shared-total saving: `53.08%`.
- Score mismatches: `2`; prediction mismatches: `3`.

Balanced 600-row gate:

- Score: `0.562825` vs reference `0.570105`, delta `-0.007280`.
- Shared-total saving: `59.44%`.
- Score mismatches: `24`; prediction mismatches: `47`.
- Task deltas:
  - `2wikimqa`: `-0.018978`.
  - `musique`: `-0.002862`.
  - `passage_retrieval_en`: `+0.000000`.

Decision: do not full-run. The midpoint protects more tail context than `1536`, but the
balanced gate is worse than both `1536` and `2048`.

## K/V-Side Separation Diagnostic

Added runner support for separate prefix key/value bit widths:

```text
--prefix-key-bits
--prefix-value-bits
```

If omitted, both default to `--prefix-kv-bits`, preserving old behavior.

Rows `0:40`, `raw_start=512`, `raw_end=2048`:

| Policy | Candidate | Reference | Delta | Shared saving | Latency ratio | Score mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K/V TQ3.5 | `0.351964` | `0.308214` | `+0.043750` | `50.61%` | `2.82x` slower | `3` |
| K fp16, V TQ3.5 | `0.326964` | `0.308214` | `+0.018750` | `25.32%` | `1.90x` slower | `2` |
| K TQ3.5, V fp16 | `0.339464` | `0.308214` | `+0.031250` | `25.33%` | `1.88x` slower | `2` |

Interpretation:

- Single-side quantization is viable on the small gate, so neither key-only nor
  value-only compression immediately collapses quality.
- Single-side quantization is not a strong storage contribution because it saves only
  about half as much as K/V quantization.
- The latency ratio is lower for single-side variants because only one side is
  dequantized, but it is still slower than fp16 Shared-KV reference.
- This is useful diagnostic evidence, not a full result.

## Updated Decision

The current strongest reportable result remains the full `raw_end=2048` policy:

- It is the only full `2400 / 2400` guarded-prefix run.
- It preserves aggregate quality and improves persistent storage by `57.82%` relative to
  fp16 Shared-KV.
- It reaches `89.16%` saving versus independent fp16, just below the original `90%`
  target.

The `raw_end=1536` policy is the only follow-up with a stronger 600-row storage gate, but
its estimated full saving is only about `89.97%` and it introduces a small `2wikimqa`
regression in the balanced gate. That is not enough upside to justify a full run before a
better mechanism is available.

Next useful mechanisms:

1. Fused/non-materializing quantized-prefix decode so the storage result can become a
   speed result.
2. More selective tail protection based on delimiter/question boundary or cheap group
   probe, instead of a fixed shorter tail guard.
3. Group-size scaling protocol, because the storage value of shared-prefix compression
   should improve with larger repeat groups.
