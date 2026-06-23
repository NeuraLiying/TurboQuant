# Quantized Prefix Table Design

This note documents the first implementation hook for a Shared-KV experiment:
quantize the shared prefix table while keeping request-local suffix/decode KV in fp16.

## API Added

```python
SharedKVPrefixTable.add_quantized_legacy_cache(
    entry_id: str,
    repeat_group_id: str,
    past_key_values: Any,
    config: KVQuantConfig,
) -> SharedKVQuantizedPrefixEntry
```

The guarded policy uses the segmented variant:

```python
SharedKVPrefixTable.add_guarded_quantized_legacy_cache(
    entry_id: str,
    repeat_group_id: str,
    past_key_values: Any,
    config: KVQuantConfig,
    raw_prefix_tokens: int,
    raw_suffix_tokens: int,
    cache_materialized_layers: bool = False,
) -> SharedKVSegmentedPrefixEntry
```

`SharedKVQuantizedPrefixEntry` wraps a `TurboQuantDynamicCache`:

- `storage_nbytes()` counts packed TurboQuant prefix segments plus group metadata.
- `materialized_nbytes()` reports the dense fp16 equivalent.
- `layer(layer_idx)` materializes the quantized prefix layer for compatibility with the
  current HuggingFace attention path.
- `compression_summary()` delegates to `TurboQuantDynamicCache`.

`SharedKVSegmentedPrefixEntry` keeps protected fp16 prefix boundary segments and stores
only the middle shared-prefix segment in TurboQuant form. It optionally caches one dense
materialization of the quantized middle segment per repeat group for runtime diagnosis;
that transient dense cache is counted separately from persistent storage.

## Why This Is The Right First Hook

The current fp16 Shared-KV reference result shows:

- shared prefix table: `861.22 GB`;
- request-local suffix/decode KV: `32.51 GB`;
- prefix fraction of non-metadata shared storage: `96.36%`.

Therefore, a TurboQuant integration that only compresses local suffix/decode KV has low
upside. Quantizing the prefix table is the storage-relevant path.

## Current Scope

This started as a storage-contract prototype and now has a full guarded-policy
validation.

It now does:

- share one guarded quantized prefix entry across requests;
- keep local suffix/decode K/V stored once per request in fp16;
- preserve the existing `SharedKVReferenceCache` materialization interface;
- expose storage accounting for persistent packed prefix bytes, dense equivalent bytes,
  and optional transient materialized-prefix-cache bytes;
- run the full `2400 / 2400` workload for `start=512/end=2048`, TurboQuant 3.5 middle
  prefix, and fp16 local KV.

It does not yet:

- avoid materializing decoded prefix K/V for stock attention;
- prove strict output equivalence to fp16 Shared-KV;
- provide a final speedup claim for the guarded TurboQuant prefix table.

## Runner Extension

The runner now exposes:

```text
--prefix-storage-mode fp16|turboquant
--prefix-kv-bits 3.5
--prefix-key-bits 3.5
--prefix-value-bits 3.5
--prefix-key-quantizer mse
--prefix-value-quantizer mse
--prefix-raw-start-tokens 512
--prefix-raw-end-tokens 2048
--prefix-cache-materialized-layers
```

When `--branch-cache-mode reference --prefix-storage-mode turboquant`, the runner should
create whole-prefix entries via `add_quantized_legacy_cache(...)` or guarded entries via
`add_guarded_quantized_legacy_cache(...)`, then keep request branches as
`SharedKVReferenceCache`.

## Smoke Result

Rows `0:4` were run with whole-prefix TurboQuant storage:

| Variant | Score | fp16 reference score | Prefix saving |
| --- | ---: | ---: | ---: |
| Whole-prefix TurboQuant 3.5 | `0.00` | `0.35` | `76.56%` |
| Whole-prefix TurboQuant 4.0 | `0.00` | `0.35` | `74.22%` |

This is a negative smoke. It validates that the implementation runs and saves prefix
storage, but it also shows direct whole-prefix quantization is too risky for a full run.

## Guarded Full-Run Result

The first reportable storage-quality result uses:

```text
--prefix-storage-mode turboquant
--prefix-kv-bits 3.5
--prefix-raw-start-tokens 512
--prefix-raw-end-tokens 2048
```

Full run:

| Rows | Candidate score | fp16 Shared-KV score | Shared-total saving | Saving vs independent fp16 | Latency ratio | Score mismatches | Prediction mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2400` | `0.590682` | `0.587374` | `57.82%` | `89.16%` | `4.32x` slower | `105` | `233` |

This passes the quality gate and gives a defensible storage-quality point, but it is not
a final speed result because the current compatibility path repeatedly
dequantizes/materializes the shared prefix.

The materialized-prefix-cache diagnostic uses the same persistent storage but caches one
dense materialization of each quantized middle-prefix segment:

| Rows | Candidate score | fp16 Shared-KV score | Shared-total saving | Latency ratio | Transient dense prefix cache |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `2400` | `0.590682` | `0.587374` | `57.82%` | `1.12x` vs fp16 Shared-KV | `724,544,651,264` bytes |

This diagnostic shows that the latency blocker is mostly repeated
dequantization/materialization, but it is not a final memory-speed point because the
extra dense cache is transient state.

## Full-Run Gate

- full rows: `2400 / 2400`;
- score delta vs fp16 Shared-KV reference: `>= -0.5`;
- average latency no worse than independent fp16 full cache;
- persistent KV saving vs independent fp16: `>= 90%`;
- report prediction and score mismatch counts vs fp16 Shared-KV reference.

Gate status for `start=512/end=2048`: full rows passed, quality passed, latency versus
independent fp16 passed, mismatch reporting passed, persistent saving narrowly missed
the original `90%` target at `89.16%`. The result is therefore reportable as a
storage-quality method, not as a strict `90%+` memory point or speedup.
