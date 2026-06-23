# Experiment Log

## 2026-06-22

### Direction Setup

Created this new direction folder:

```text
2026-06-22_shared-kv-systems/
```

Reason: prior work was reproduction and incremental TurboQuant codec exploration. This
direction changes the problem to cross-request Shared-KV systems behavior, so records
should be separated from `reproduce/incremental/` and `survey/`.

### Consolidated Existing Full-Run Evidence

Added:

```text
2026-06-22_shared-kv-systems/build_claim_frontier.py
2026-06-22_shared-kv-systems/claim_frontier.json
2026-06-22_shared-kv-systems/claim_frontier.md
```

Command:

```bash
python 2026-06-22_shared-kv-systems/build_claim_frontier.py
```

Inputs:

```text
reproduce/incremental/repeated_context_quality_summary_current.json
reproduce/incremental/repeated_context_shared_kv_plan_current.json
reproduce/incremental/repeated_context_reference_runtime_full_summary.json
reproduce/incremental/repeated_context_reference_materialization_overhead_full.json
reproduce/incremental/repeated_context_split_full_scoreparity_summary.json
```

Main generated findings:

- fp16 Shared-KV reference: score delta `+0.0097`, latency speedup `5.14x`, persistent KV
  saving `74.30%`, clone parity `2400 / 2400` with `0` mismatches.
- TurboQuant 3.5 independent persistent KV is close to fp16 Shared-KV (`0.912x`), but fp16
  Shared-KV is `+1.87` points higher quality and `28.78x` faster on this workload.
- Shared prefix table is `96.36%` of non-metadata shared storage, so suffix-only quantization
  is not a strong next contribution.
- Current Python split attention is not claim-ready: `38` score mismatches and `77`
  prediction mismatches on the full row-level parity protocol.

### Verification

Ran relevant existing tests:

```bash
python -m pytest \
  tests/test_repeated_context_shared_kv_plan.py \
  tests/test_repeated_context_shared_kv_memory.py \
  tests/test_repeated_context_quality_plan.py \
  tests/test_summarize_repeated_context_quality.py \
  tests/test_summarize_repeated_context_reference_runtime.py \
  tests/test_analyze_repeated_context_materialization_overhead.py
```

Result:

```text
22 passed
```

### Quantized Prefix Table Prototype

Added a first implementation hook for E1:

```text
turboquant/shared_kv_cache.py
tests/test_shared_kv_cache.py
```

New API:

```python
SharedKVPrefixTable.add_quantized_legacy_cache(
    entry_id=...,
    repeat_group_id=...,
    past_key_values=...,
    config=KVQuantConfig(...),
)
```

This stores shared prefix K/V once using `TurboQuantDynamicCache`, while request-local
suffix/decode K/V remains fp16 in `SharedKVReferenceCache`. The stock attention path can
still materialize prefix+local tensors, so this is a storage-contract prototype rather
than a final non-materializing kernel.

Verification:

```bash
python -m pytest \
  tests/test_shared_kv_cache.py \
  tests/test_repeated_context_prefix_sharing_runner.py
```

Result:

```text
14 passed
```

Smoke CLI:

```bash
python experiments/longbench/run_repeated_context_prefix_sharing_eval.py \
  --dry-run --start-index 0 --end-index 8 \
  --branch-cache-mode reference \
  --prefix-storage-mode turboquant \
  --prefix-kv-bits 3.5 \
  --output 2026-06-22_shared-kv-systems/dry_run_quantized_prefix_smoke.jsonl
```

Result: dry-run summary contains `prefix_storage_mode=turboquant` and expected token
sharing fields.

### Quantized Prefix Table Model-Loaded Smoke

Ran one repeat group, rows `0:4`, with quantized shared prefix table and fp16 local
suffix/decode KV.

Commands:

```bash
python experiments/longbench/run_repeated_context_prefix_sharing_eval.py \
  --start-index 0 --end-index 4 --device cuda:0 \
  --branch-cache-mode reference \
  --prefix-storage-mode turboquant \
  --prefix-kv-bits 3.5 \
  --output 2026-06-22_shared-kv-systems/quantized_prefix_reference_smoke_0000_0004.jsonl

python experiments/longbench/run_repeated_context_prefix_sharing_eval.py \
  --start-index 0 --end-index 4 --device cuda:0 \
  --branch-cache-mode reference \
  --prefix-storage-mode turboquant \
  --prefix-kv-bits 4 \
  --output 2026-06-22_shared-kv-systems/quantized_prefix_4bit_reference_smoke_0000_0004.jsonl
```

Comparison against existing fp16 Shared-KV reference smoke
`reproduce/runs/repeated_context/repeated_context_answer_preserving_qa_g4_fp16_prefix_shared_reference_smoke_0000_0004.jsonl`:

| Variant | Rows | Score | fp16 reference score | Avg latency | Prefix-table bytes | Prefix saving vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quantized prefix 3.5 | 4 | `0.00` | `0.35` | `2.270s` | `219,234,368` | `76.56%` |
| Quantized prefix 4.0 | 4 | `0.00` | `0.35` | `2.324s` | `241,139,776` | `74.22%` |

Prediction parity:

- Rows `0` and `1` match fp16 reference but both have zero score.
- Rows `2` and `3` collapse to the same wrong answer as rows `0/1`, while fp16 reference
  scores `0.4` and `1.0`.

Decision:

- Do not launch a full 3.5/4-bit quantized-prefix run as currently implemented.
- Directly quantizing the entire long shared prefix is too quality-sensitive, at least on
  the first group.
- The next TurboQuant integration needs an exact-prefix guard, selective prefix
  quantization, or a higher-level cache policy that only quantizes reused prefix segments
  after proving low quality sensitivity.

### Guarded Prefix Quantization Smoke

Implemented guarded prefix-table storage:

```text
SharedKVPrefixTable.add_guarded_quantized_legacy_cache(...)
--prefix-raw-start-tokens
--prefix-raw-end-tokens
```

This stores the beginning and end of the shared prefix as exact fp16 K/V and quantizes
only the middle prefix segment.

Verification:

```bash
python -m pytest tests/test_shared_kv_cache.py tests/test_repeated_context_prefix_sharing_runner.py
```

Result:

```text
17 passed
```

Smoke results on rows `0:4`:

| Variant | Rows | Score | fp16 reference score | Avg latency | Prefix-table bytes | Prefix saving vs fp16 | Prediction matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Guarded TQ3.5, start 256/end 512 | 4 | `0.00` | `0.35` | `2.209s` | `296,304,704` | `68.32%` | `2 / 4` |
| Guarded TQ3.5, start 512/end 2048 | 4 | `0.60` | `0.35` | `1.292s` | `476,135,488` | `49.09%` | `3 / 4` |

Decision:

- `start=512, end=2048` is the first positive TurboQuant-composed Shared-KV smoke.
- It is not decision evidence. The next gate is a `0:40` guarded shard against the fp16
  Shared-KV reference.

### Guarded Prefix Quantization 40-Row Gate

Ran rows `0:40` with `start=512`, `end=2048`, TurboQuant 3.5 middle prefix.

Command:

```bash
python experiments/longbench/run_repeated_context_prefix_sharing_eval.py \
  --start-index 0 --end-index 40 --device cuda:0 \
  --branch-cache-mode reference \
  --prefix-storage-mode turboquant \
  --prefix-kv-bits 3.5 \
  --prefix-raw-start-tokens 512 \
  --prefix-raw-end-tokens 2048 \
  --output 2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_guarded_0000_0040.jsonl
```

Summary command:

```bash
python 2026-06-22_shared-kv-systems/summarize_guarded_prefix_shard.py \
  --candidate 2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_guarded_0000_0040.jsonl \
  --reference reproduce/runs/repeated_context/repeated_context_answer_preserving_qa_g4_fp16_prefix_shared_reference_guarded_0000_0040.jsonl \
  --output 2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_guarded_0000_0040_summary.json
```

Result:

| Metric | Candidate | fp16 Shared-KV reference | Delta / ratio |
| --- | ---: | ---: | ---: |
| Rows | `40` | `40` | complete |
| Groups | `10` | `10` | complete |
| Score | `0.351964` | `0.308214` | `+0.043750` |
| Avg latency | `0.808822s` | `0.286471s` | `2.82x` slower |
| Prefix-table storage | `4,969,513,600` | `10,241,442,432` | `51.48%` saving |
| Shared total storage | `5,147,248,512` | `10,422,060,928` | `50.61%` saving |
| Score mismatches | `3` | - | report |
| Prediction mismatches | `4` | - | report |

Decision:

- This is a real positive candidate, not just an accounting estimate.
- It is not full decision evidence and should not be reported as a completed method.
- The latency regression means a full systems claim needs either cached decoded prefix,
  a faster materialization path, or non-materializing attention.
- Next gate should be either rows `0:200` or all `2wikimqa` repeated groups before
  spending a full `2400 / 2400` run.

### Guarded Prefix Quantization 200-Row Gate

Ran rows `0:200` with the same policy: fp16 start `512`, fp16 end `2048`, TurboQuant 3.5
middle prefix.

Command:

```bash
python experiments/longbench/run_repeated_context_prefix_sharing_eval.py \
  --start-index 0 --end-index 200 --device cuda:0 \
  --branch-cache-mode reference \
  --prefix-storage-mode turboquant \
  --prefix-kv-bits 3.5 \
  --prefix-raw-start-tokens 512 \
  --prefix-raw-end-tokens 2048 \
  --output 2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_gate_0000_0200.jsonl
```

Summary:

| Metric | Candidate | fp16 Shared-KV reference | Delta / ratio |
| --- | ---: | ---: | ---: |
| Rows | `200` | first 200 from full reference | complete |
| Groups | `50` | `50` | complete |
| Score | `0.446931` | `0.438445` | `+0.008486` |
| Avg latency | `0.864841s` | `0.276031s` | `3.13x` slower |
| Prefix-table saving | - | - | `49.67%` |
| Shared-total saving | - | - | `48.71%` |
| Score mismatches | `14` | - | report |
| Prediction mismatches | `22` | - | report |

Decision:

- This passes the 200-row quality/storage gate and is a credible candidate for a new
  TurboQuant-composed Shared-KV contribution.
- It still cannot be reported as a full method because the evidence is limited to 2Wiki
  rows `0:200`, and latency is worse under Python materialization.
- Next best experiment is either complete 2Wiki repeated-context (`0:800`) or a balanced
  cross-task gate (`2wikimqa`, `musique`, `passage_retrieval_en`) before full `2400`.

### Guarded Prefix Quantization Multi-Task 600-Row Gate

Ran the same policy on three 200-row gates:

- `2wikimqa`: rows `0:200`.
- `musique`: rows `800:1000`.
- `passage_retrieval_en`: rows `1600:1800`.

Aggregate artifact:

```text
2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_multitask_gate_summary.md
```

Aggregate result:

| Metric | Candidate | fp16 Shared-KV reference | Delta / ratio |
| --- | ---: | ---: | ---: |
| Rows | `600` | `600` | complete |
| Groups | `150` | `150` | complete |
| Score | `0.572944` | `0.570105` | `+0.002839` |
| Shared total storage | `101,079,175,296` | `239,951,835,264` | `57.88%` saving |
| Score mismatches | `25` | - | report |
| Prediction mismatches | `48` | - | report |

By task:

| Task | Candidate | Reference | Delta | Shared saving | Latency ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | `0.446931` | `0.438445` | `+0.008486` | `48.71%` | `3.13x` slower |
| `musique` | `0.271901` | `0.271869` | `+0.000032` | `63.49%` | `5.33x` slower |
| `passage_retrieval_en` | `1.000000` | `1.000000` | `+0.000000` | `56.19%` | `3.59x` slower |

Decision:

- This is the strongest candidate result so far: quality is preserved across sampled
  tasks and storage saving is large.
- It is still not a full `2400 / 2400` result, and latency is currently a blocker for a
  speed claim.
- Safe current framing: "guarded TurboQuant-compressed shared-prefix table is a promising
  storage extension to exact Shared-KV, with 600-row multi-task evidence."

### Current Research Decision

The direction is worth continuing, but the method claim must be scoped:

- Claim-ready now: fp16 Shared-KV reference representation and full-workload systems
  evidence.
- Claimable with narrow scope: guarded TurboQuant-compressed shared-prefix storage as a
  full storage-quality result, not a speedup or strict-equivalence result.
- Not claim-ready: Python split attention, non-materializing shared-prefix decode, or a
  general "better TurboQuant" codec claim.
- Highest-value next experiment: exact/non-materializing shared-prefix attention, or a
  more aggressive guarded/selective prefix-table quantization policy. Plain whole-prefix
  TurboQuant should not be expanded without a new guard.

### Guarded Prefix Quantization Full 2400-Row Run

Completed the full generated repeated-context workload for the guarded policy:

- fp16 prefix guard at the first `512` shared-prefix tokens.
- fp16 prefix guard at the last `2048` shared-prefix tokens.
- TurboQuant 3.5 for the middle shared-prefix segment.
- fp16 request-local suffix/decode KV.

Merged artifact:

```text
2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_full_0000_2400.jsonl
```

Summary artifact:

```text
2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_full_0000_2400_summary.md
```

Aggregate result:

| Metric | Candidate | fp16 Shared-KV reference | Delta / ratio |
| --- | ---: | ---: | ---: |
| Rows | `2400` | `2400` | complete |
| Groups | `600` | `600` | complete |
| Score | `0.590682` | `0.587374` | `+0.003308` raw / `+0.3308` points |
| Avg latency | `1.453345s` | `0.336493s` | `4.32x` slower |
| Shared total storage | `404,597,824,000` | `959,330,370,048` | `57.82%` saving |
| Saving vs independent fp16 materialized KV | - | - | `89.16%` |
| Score mismatches | `105` | - | report |
| Prediction mismatches | `233` | - | report |

Task-level result:

| Task | Candidate | Reference | Delta | Shared saving | Latency ratio | Score mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | `0.464145` | `0.457359` | `+0.006786` | `48.94%` | `3.44x` slower | `45` |
| `musique` | `0.314151` | `0.309762` | `+0.004389` | `63.44%` | `5.34x` slower | `53` |
| `passage_retrieval_en` | `0.993750` | `0.995000` | `-0.001250` | `56.08%` | `3.55x` slower | `7` |

Decision:

- This is a full storage-quality result for a TurboQuant-composed Shared-KV table.
- It supports a narrow novelty claim: guarded quantization of the dominant shared-prefix
  table preserves aggregate quality while reducing persistent storage.
- It does not support a speedup claim. Current latency is worse than fp16 Shared-KV
  because the prototype dequantizes/materializes prefix K/V in Python.
- It is not strict-equivalence evidence: report score and prediction mismatch counts.
- It narrowly misses the earlier `90%` persistent-saving target vs independent fp16
  (`89.16%` measured), but it cuts fp16 Shared-KV persistent storage by `57.82%`.

### Guarded Prefix Policy Search After Full Run

Added:

```text
2026-06-22_shared-kv-systems/GUARDED_POLICY_SEARCH.md
```

Purpose: check whether a stronger fixed guard can cross the original `90%` persistent
saving target without losing quality.

Balanced 600-row gate results:

| Policy | Score | fp16 Shared-KV | Delta | Shared saving | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `start=512/end=1536`, K/V TQ3.5 | `0.570496` | `0.570105` | `+0.000392` | `61.01%` | Not full-run yet; `2wikimqa` regresses `-0.003600`, estimated full saving only about `89.97%`. |
| `start=512/end=1792`, K/V TQ3.5 | `0.562825` | `0.570105` | `-0.007280` | `59.44%` | Stop; `2wikimqa` regresses `-0.018978`. |
| `start=512/end=2048`, K/V TQ3.5 | `0.572944` | `0.570105` | `+0.002839` | `57.88%` | Still the strongest quality/stability point. |

Added runner support for side-specific prefix bits:

```text
--prefix-key-bits
--prefix-value-bits
```

Rows `0:40`, `start=512/end=2048`:

| Policy | Score | fp16 Shared-KV | Delta | Shared saving | Latency ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| K/V TQ3.5 | `0.351964` | `0.308214` | `+0.043750` | `50.61%` | `2.82x` slower |
| K fp16, V TQ3.5 | `0.326964` | `0.308214` | `+0.018750` | `25.32%` | `1.90x` slower |
| K TQ3.5, V fp16 | `0.339464` | `0.308214` | `+0.031250` | `25.33%` | `1.88x` slower |

Decision:

- No stronger fixed-tail full-run candidate was found.
- Single-side K/V quantization is useful diagnostic evidence but not a strong storage
  contribution.
- Keep the full `start=512/end=2048` result as the reportable storage-quality point.
- The next novelty step should change mechanism: fused/non-materializing decode,
  boundary-aware protection, cheap group admission, or group-size scaling.

### Shared Dense Prefix Materialization Cache Runtime Diagnostic

Implemented a diagnostic runtime switch:

```text
--prefix-cache-materialized-layers
```

Code paths:

- `SharedKVQuantizedPrefixEntry` and `SharedKVSegmentedPrefixEntry` can keep dense
  materialized quantized prefix layers.
- `SharedKVPrefixTable.materialized_cache_nbytes()` reports transient dense cache bytes.
- Persistent storage accounting is unchanged.

Motivation:

The uncached guarded TQ3.5 prefix table repeatedly dequantizes/materializes the same
shared prefix for each request. This diagnostic caches that dense materialization once
per repeat group to estimate how much of the latency gap is repeated dequantization.

Rows `0:40`:

| Variant | Score | Reference | Delta | Latency ratio | Shared saving | Transient dense prefix cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uncached guarded TQ3.5 | `0.351964` | `0.308214` | `+0.043750` | `2.82x` | `50.61%` | `0` |
| cached-materialized guarded TQ3.5 | `0.351964` | `0.308214` | `+0.043750` | `1.00x` | `50.61%` | first group about `599,785,472` bytes |

Balanced 600-row gate:

| Variant | Score | Reference | Delta | Shared saving | 2Wiki latency | MuSiQue latency | Passage latency | Transient dense prefix cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uncached guarded TQ3.5 | `0.572944` | `0.570105` | `+0.002839` | `57.88%` | `3.13x` | `5.33x` | `3.59x` | `0` |
| cached-materialized guarded TQ3.5 | `0.572944` | `0.570105` | `+0.002839` | `57.88%` | `1.00x` | `1.15x` | `1.11x` | `181,366,423,552` bytes |

Full `2400 / 2400` cached-materialized run:

```text
2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_cached_materialized_full_0000_2400_summary.md
```

| Variant | Score | Reference | Delta | Shared saving | Avg latency | Latency ratio | Transient dense prefix cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uncached guarded TQ3.5 | `0.590682` | `0.587374` | `+0.003308` | `57.82%` | `1.453345s` | `4.32x` | `0` |
| cached-materialized guarded TQ3.5 | `0.590682` | `0.587374` | `+0.003308` | `57.82%` | `0.377205s` | `1.12x` | `724,544,651,264` bytes |

Task-level cached-materialized full result:

| Task | Score delta | Shared saving | Latency ratio | Transient dense prefix cache |
| --- | ---: | ---: | ---: | ---: |
| `2wikimqa` | `+0.006786` | `48.94%` | `1.08x` | `123,549,515,776` bytes |
| `musique` | `+0.004389` | `63.44%` | `1.16x` | `343,826,235,392` bytes |
| `passage_retrieval_en` | `-0.001250` | `56.08%` | `1.11x` | `257,168,900,096` bytes |

Artifact:

```text
2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_cached_materialized_multitask_gate_summary.md
2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_cached_materialized_full_0000_2400_summary.md
```

Decision:

- This is a useful runtime diagnostic: the latency blocker is mostly repeated
  dequantization/materialization of the shared prefix.
- It does not change the storage-quality claim because persistent bytes are unchanged.
- It is not a final speed claim because the speedup is bought with transient dense
  prefix memory rather than a fused kernel.
- It strengthens the next implementation target: fused or non-materializing quantized
  prefix decode should try to recover cached-materialized latency without paying the
  full transient dense cache cost.

### Official Baseline Repository Setup

Adopted a stricter baseline rule: if a baseline has official GitHub code, reportable
numbers must come from cloning and running the upstream project. Local reimplementations
are sanity checks only.

Cloned:

| Method | Repo | Commit |
| --- | --- | --- |
| CacheGen | `https://github.com/UChi-JCL/CacheGen` | `6bed34ca9d495289955ed754cf2ad0c43346ee48` |
| SKVQ | `https://github.com/cat538/SKVQ` | `fdfc7ec315c16293f68dd771850b47f5199787fc` |
| SpectrumKV | `https://github.com/YangSteve1223/kvcache-lab` | `e108438df1d903c448a123a43276da88542559d1` |

Artifacts:

```text
2026-06-22_shared-kv-systems/BASELINE_REPRODUCTION_PLAN.md
2026-06-22_shared-kv-systems/BASELINE_REPRODUCTION_STATUS.md
2026-06-22_shared-kv-systems/baseline_registry.json
```

Initial entry-point findings:

- SKVQ has official `eval_longbench.py` and supports LongBench tasks including
  `2wikimqa`, `musique`, and `passage_retrieval_en`, but needs a documented
  non-algorithm adapter for local model/dataset paths.
- CacheGen targets KV streaming/loading and paper artifact datasets; first comparison
  should focus on size/loading behavior unless a repeated-context adapter is documented.
- SpectrumKV targets PD-disaggregated transfer, WikiText-2 PPL, and NIAH; it is a strong
  mixed-precision policy baseline but not directly the same workload.
