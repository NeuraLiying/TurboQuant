# 2026-06-22 Shared-KV Systems Direction

This folder starts a new research direction, separate from the earlier reproduction and
incremental TurboQuant codec exploration.

## Current Thesis

For repeated long-context QA workloads, exact cross-request KV sharing is a stronger
first systems move than independently quantizing every request's KV cache.

The first claim-ready result is fp16 Shared-KV reference serving:

- Full generated workload: `2400 / 2400` rows.
- Quality delta versus independent fp16: `+0.0097` LongBench points.
- Clone parity: `2400` rows compared, `0` mismatches.
- Average latency: `1.730s -> 0.336s`, `5.14x` faster.
- Persistent KV storage: `3.40 TB -> 893.45 GB`, `74.30%` lower.

The current new-method result is a guarded TurboQuant-compressed shared-prefix table:

- Policy: fp16 guard for the first `512` and last `2048` shared-prefix tokens,
  TurboQuant 3.5 for the middle shared-prefix segment, fp16 request-local KV.
- Full generated workload: `2400 / 2400` rows.
- Score delta versus fp16 Shared-KV reference: `+0.3308` LongBench points.
- Shared-total storage saving versus fp16 Shared-KV: `57.82%`.
- Persistent KV saving versus independent fp16 materialized KV: `89.16%`.
- Runtime status: storage-quality result only; current uncached Python prototype is
  slower than fp16 Shared-KV, and the dense-prefix-cache variant is a diagnostic.

Generated artifact: [claim_frontier.md](claim_frontier.md).

Strict top-conference positioning: this is currently a storage-quality empirical result
and method seed, not a complete standalone top-conference paper. The main missing pieces
are a principled shared-prefix-specific policy or kernel module, stronger reusable-KV
and mixed-precision baselines, and end-to-end serving metrics. See
[TOP_CONFERENCE_ROADMAP.md](TOP_CONFERENCE_ROADMAP.md).

## Contribution Boundary

This is not a broad "shared KV cache" claim and not a new generic TurboQuant codec
claim.

The method contribution is a shared-KV systems representation, evaluation protocol, and
guarded shared-prefix storage policy:

1. Store shared prefix KV once in a prefix table.
2. Store only request-local suffix/decode KV per request.
3. Account for explicit reference metadata.
4. Validate quality and clone parity on a full repeated-context workload.
5. Quantify the remaining materialization overhead that a kernel path must remove.
6. Compress only the middle of the dominant shared-prefix table with TurboQuant while
   protecting prefix boundary regions and leaving request-local KV exact.

Independent per-request TurboQuant remains a negative comparator on this workload. The
validated TurboQuant result is specifically the guarded shared-prefix-table policy, not
blind whole-prefix quantization or suffix-only quantization.

## Novelty Assessment

Prefix caching itself is not novel. vLLM/PagedAttention, SGLang/RadixAttention,
Hydragen, ChunkAttention, and FlashInfer Cascade already cover prefix-aware KV reuse or
shared-prefix attention.

The viable novelty must be narrower:

- A LongBench-style answer-preserving repeated-context protocol with full quality gates.
- Explicit storage accounting that separates independent KV, shared prefix table,
  request-local KV, metadata, and duplicate prefix materialization.
- A systems-vs-codec Pareto result showing that fp16 Shared-KV dominates independent
  TurboQuant 2.5/3.5 on this repeated-context workload for quality and latency.
- A TurboQuant integration path focused on the dominant shared prefix table, not the small
  request-local suffix/decode tail.

## Key Observation

In the current shared representation, the shared prefix table is `861.22 GB` and
request-local suffix/decode KV is only `32.51 GB`. The prefix table is `96.36%` of
non-metadata shared storage.

Therefore, suffix-only quantization is a weak contribution. The next high-value
experiment is:

> quantize the shared prefix table while keeping request-local suffix/decode KV in fp16.

Storage target from current accounting:

- Quantized shared prefix at TurboQuant 2.5 ratio + fp16 local: `180.59 GB`, `94.81%`
  saving vs independent fp16 materialized KV.
- Quantized shared prefix at TurboQuant 3.5 ratio + fp16 local: `234.43 GB`, `93.26%`
  saving vs independent fp16 materialized KV.

These are storage targets only. They require full quality and latency validation before
being reported as method results.

## New Candidate Result: Guarded Prefix Quantization

Direct whole-prefix TurboQuant failed the first 4-row smoke, but guarded prefix
quantization produced positive gates and a complete full run:

- Policy: keep first `512` and last `2048` shared-prefix tokens in fp16; quantize the
  middle prefix segment with TurboQuant 3.5; keep suffix/decode KV in fp16.
- Full 2400-row score: `0.590682` vs fp16 Shared-KV reference `0.587374`, delta
  `+0.003308` raw score, or `+0.3308` LongBench points.
- Full shared-total storage saving vs fp16 Shared-KV: `57.82%`.
- Full persistent storage saving vs independent fp16 materialized KV: `89.16%`.
- Full score mismatches vs fp16 Shared-KV: `105`; prediction mismatches: `233`.
- Latency is worse than fp16 Shared-KV reference: `1.453s` vs `0.336s`, because this
  prototype decodes/materializes the quantized prefix in Python.

Task-level full result:

| Task | Score | fp16 Shared-KV | Delta | Shared saving | Latency ratio | Score mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | `0.464145` | `0.457359` | `+0.006786` | `48.94%` | `3.44x` slower | `45` |
| musique | `0.314151` | `0.309762` | `+0.004389` | `63.44%` | `5.34x` slower | `53` |
| passage_retrieval_en | `0.993750` | `0.995000` | `-0.001250` | `56.08%` | `3.55x` slower | `7` |

Artifact:

- [guarded_prefix_tq35_s512_e2048_full_0000_2400_summary.md](guarded_prefix_tq35_s512_e2048_full_0000_2400_summary.md)
- [guarded_prefix_tq35_s512_e2048_multitask_gate_summary.md](guarded_prefix_tq35_s512_e2048_multitask_gate_summary.md)
- [guarded_prefix_tq35_s512_e2048_gate_0000_0200_summary.md](guarded_prefix_tq35_s512_e2048_gate_0000_0200_summary.md)
- [guarded_prefix_tq35_s512_e2048_guarded_0000_0040_summary.md](guarded_prefix_tq35_s512_e2048_guarded_0000_0040_summary.md)

Interpretation: this is now a full storage-quality result for a TurboQuant-composed
Shared-KV table. It has a defensible novelty angle as a guarded shared-prefix storage
policy with full quality accounting. It is not a speed result, not a strict-equivalence
result, and it narrowly misses the earlier `90%` saving target vs independent fp16
(`89.16%` measured).

## Follow-Up Policy Search

After the full `start=512/end=2048` result, a stronger-guard search tested smaller exact
tail guards and K/V-side separated prefix quantization:

- `start=512/end=1536`: balanced 600-row score `0.570496` vs reference `0.570105`,
  shared saving `61.01%`, but `2wikimqa` regresses by `-0.003600`; estimated full saving
  is only about `89.97%`, still not a clean `90%+` result.
- `start=512/end=1792`: balanced 600-row score regresses to `0.562825` vs `0.570105`,
  with `2wikimqa` delta `-0.018978`; stopped.
- K-only and V-only prefix quantization both pass the 40-row gate, but each saves only
  about `25%` shared storage, so they are diagnostic rather than a main contribution.

Artifact:

- [GUARDED_POLICY_SEARCH.md](GUARDED_POLICY_SEARCH.md)

Decision: keep the full `start=512/end=2048` result as the current reportable storage
point. Do not spend a full run on `end=1536` or `end=1792` without a stronger mechanism.

## Runtime Diagnostic: Shared Dense Prefix Materialization Cache

The guarded TQ3.5 prefix table was slow because every request repeatedly materialized
the quantized shared prefix. A diagnostic cache now optionally keeps one dense
materialization of each quantized prefix layer per repeat group:

```text
--prefix-cache-materialized-layers
```

This does not change persistent storage accounting. It adds transient dense prefix-cache
memory and measures the runtime ceiling before a fused/non-materializing kernel exists.

Full `2400 / 2400` run, `start=512/end=2048`:

| Variant | Score delta | Shared saving | Latency ratio vs fp16 Shared-KV | Transient dense prefix cache |
| --- | ---: | ---: | ---: | ---: |
| uncached guarded TQ3.5 | `+0.003308` | `57.82%` | `4.32x` aggregate | `0` |
| cached-materialized guarded TQ3.5 | `+0.003308` | `57.82%` | `1.12x` aggregate | `724,544,651,264` bytes |

Task-level cached-materialized latency ratios:

| Task | Latency ratio | Transient dense prefix cache |
| --- | ---: | ---: |
| 2wikimqa | `1.08x` | `123,549,515,776` bytes |
| musique | `1.16x` | `343,826,235,392` bytes |
| passage_retrieval_en | `1.11x` | `257,168,900,096` bytes |

Artifact:

- [guarded_prefix_tq35_s512_e2048_cached_materialized_full_0000_2400_summary.md](guarded_prefix_tq35_s512_e2048_cached_materialized_full_0000_2400_summary.md)
- [guarded_prefix_tq35_s512_e2048_cached_materialized_multitask_gate_summary.md](guarded_prefix_tq35_s512_e2048_cached_materialized_multitask_gate_summary.md)

Interpretation: the latency problem is largely repeated dequantization/materialization,
not the shared-prefix storage policy itself. This is still a diagnostic, not a final
speed claim, because it trades transient dense memory for speed and is only validated on
the full run with an extra dense prefix cache rather than a fused kernel.

## Current Blocking Limitation

The current Python split attention path is not claim-ready:

- Full split attention score parity failed with `38` score mismatches and `77` prediction
  mismatches.
- Materialized eager vs clone also shows backend drift with `32` score mismatches and
  `77` prediction mismatches.

The next implementation target is SDPA-equivalent prefix-aware attention or a
predeclared relaxed score-parity protocol. Do not run more full validations of the
current Python split path without changing the implementation target.

## Files

- [claim_frontier.md](claim_frontier.md): consolidated evidence table and frontier.
- [claim_frontier.json](claim_frontier.json): machine-readable consolidated evidence.
- [build_claim_frontier.py](build_claim_frontier.py): reproducible artifact generator.
- [RELATED_WORK.md](RELATED_WORK.md): novelty and overlap matrix.
- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): chronological work log.
- [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md): required experiments and pass/fail gates.
- [COMPLETION_AUDIT.md](COMPLETION_AUDIT.md): objective-to-evidence completion and
  claim audit.
- [TOP_CONFERENCE_ROADMAP.md](TOP_CONFERENCE_ROADMAP.md): strict top-conference gap
  analysis, baseline matrix, and method-upgrade plan.
- [BASELINE_REPRODUCTION_PLAN.md](BASELINE_REPRODUCTION_PLAN.md): official-code policy
  for publishable baseline runs.
- [BASELINE_REPRODUCTION_STATUS.md](BASELINE_REPRODUCTION_STATUS.md): cloned baseline
  repos, commits, entry points, and current blockers.
- [baseline_registry.json](baseline_registry.json): machine-readable baseline metadata.
