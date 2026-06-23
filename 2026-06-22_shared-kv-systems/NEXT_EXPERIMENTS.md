# Next Experiments

The goal is to turn the current systems observation into a stronger, novel, reportable
method result.

Strict update: after reviewing reusable-KV storage and mixed-precision KV-cache work, the
current fixed guarded TurboQuant shared-prefix policy should be treated as an empirical
storage result, not a top-conference-ready method. The next phase must add baseline
coverage and a new method module. See [TOP_CONFERENCE_ROADMAP.md](TOP_CONFERENCE_ROADMAP.md).

## E0: Baseline Frontier And Claim-Risk Reduction

Status: required before any strong paper claim.

Hard rule: if the method has an official GitHub repository, use that upstream code for
reportable baseline numbers. Local reimplementations are not acceptable paper baselines;
they can only be sanity checks or format adapters. See
[BASELINE_REPRODUCTION_PLAN.md](BASELINE_REPRODUCTION_PLAN.md).

### Baselines To Add Or Align

| Family | Targets | Why needed |
| --- | --- | --- |
| Reusable KV storage | KVTC, CacheGen-style compression/offload | Tests whether shared-prefix table compression is already covered. |
| Sink/recent protection | SKVQ, InnerQ | Tests whether fixed fp16 boundary guards are merely known protection windows. |
| Mixed precision allocation | KVmix, SpectrumKV, TriAxialKV if feasible | Tests whether adaptive token/layer precision beats fixed guards. |
| Systems prefix cache | fp16 Shared-KV, vLLM/SGLang-style prefix cache if practical | Gives serving metrics beyond logical storage. |

### Execution Steps

1. Clone confirmed official repos into `external_baselines/`.
2. Record URL, commit, environment, command, raw output path, and any adapter patches.
3. Smoke-test on a tiny slice before running aligned LongBench/repeated-context slices.
4. Convert official raw outputs into this repo's common summary format without changing
   the method implementation.
5. Mark baselines without official code as `paper-only` rather than reporting local
   reimplementations as official numbers.

### Required Metrics

- LongBench-style score and paired mismatch counts.
- `M_independent_fp16`, `M_shared_fp16`, `M_shared_candidate`.
- Incremental saving over fp16 Shared-KV:
  `(M_shared_fp16 - M_shared_candidate) / M_shared_fp16`.
- Actual peak HBM or allocator footprint where available.
- Encode/decode/materialization overhead.
- TTFT, inter-token latency, throughput, cache hit/load latency, and resident prefix
  count for systems baselines.

## E0.5: New Method Module Candidate

Status: highest-priority novelty work.

The fixed `512/2048` guard should be replaced or justified by a shared-prefix-specific
policy.

Candidate module:

```text
Cross-suffix sensitivity protection:
  For each shared-prefix block, estimate how quantization error transfers across
  multiple suffix branches, then protect blocks whose mean/variance/worst-case risk is
  high under a memory budget.
```

Pass gates:

- Beats or matches fixed `512/2048` guard on 40-row gate at lower or equal memory.
- Beats whole-prefix TurboQuant and local-only quantization.
- Shows sensitivity scores are stable across at least two task slices.
- Provides a mathematical objective or risk bound that uses reuse count and cross-suffix
  variance, not only position.

## E1: Guarded Quantized Shared Prefix Table, fp16 Local KV

Status: full `2400 / 2400` validation completed for the first guarded policy.

### Motivation

The shared prefix table is `96.36%` of non-metadata shared storage. Compressing only
request-local suffix/decode KV barely changes total storage. The first TurboQuant
integration should therefore focus on the shared prefix table while keeping request-local
suffix/decode KV in fp16.

However, a direct whole-prefix TurboQuant smoke failed rows `0:4` at both 3.5 and 4.0
bits. The next version must be guarded or selective, not a blind full-prefix codec.

### Method

For each repeat group:

1. Prefill the exact shared prefix once.
2. Store selected prefix K/V segments in a quantized prefix table.
3. For each request, keep suffix/decode K/V in fp16.
4. Keep exact fp16 storage for protected prefix regions or groups.
5. Keep reference metadata accounting identical to the fp16 Shared-KV plan.

### Storage Targets

Storage-only estimates from current full-run accounting:

| Variant | Target persistent KV | Saving vs independent fp16 |
| --- | ---: | ---: |
| Prefix TurboQuant 2.5 + fp16 local | `180.59 GB` | `94.81%` |
| Prefix TurboQuant 3.5 + fp16 local | `234.43 GB` | `93.26%` |

### Required Guard Before Full Run

A candidate policy must pass rows `0:40` before a full run:

- Score delta vs fp16 Shared-KV reference: no worse than `-0.25` points.
- Prediction mismatch count reported.
- Prefix-table storage saving at least `40%` vs fp16 Shared-KV reference.
- Average latency no worse than independent fp16 full cache.

The failed direct whole-prefix smoke should remain the negative baseline. A stronger
guard gave the first positive smoke:

| Variant | Rows | Score | fp16 reference score | Prefix saving |
| --- | ---: | ---: | ---: | ---: |
| Whole-prefix TurboQuant 3.5 | 4 | `0.00` | `0.35` | `76.56%` |
| Whole-prefix TurboQuant 4.0 | 4 | `0.00` | `0.35` | `74.22%` |
| Guarded TQ3.5, start 512/end 2048 | 4 | `0.60` | `0.35` | `49.09%` |

Rows `0:40` with `start=512/end=2048` passed the first guarded shard as a candidate:

| Rows | Candidate score | fp16 reference score | Prefix saving | Shared-total saving | Latency ratio | Score mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `40` | `0.351964` | `0.308214` | `51.48%` | `50.61%` | `2.82x` slower | `3` |

Rows `0:200` passed the next gate:

| Rows | Candidate score | fp16 reference score | Prefix saving | Shared-total saving | Latency ratio | Score mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `200` | `0.446931` | `0.438445` | `49.67%` | `48.71%` | `3.13x` slower | `14` |

The balanced multi-task 600-row gate also passed:

| Rows | Candidate score | fp16 reference score | Shared-total saving | Score mismatches | Prediction mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `600` | `0.572944` | `0.570105` | `57.88%` | `25` | `48` |

The full run completed:

| Rows | Candidate score | fp16 Shared-KV score | Shared-total saving | Saving vs independent fp16 | Latency ratio | Score mismatches | Prediction mismatches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2400` | `0.590682` | `0.587374` | `57.82%` | `89.16%` | `4.32x` slower | `105` | `233` |

Task-level full result:

| Task | Candidate score | fp16 Shared-KV score | Delta | Shared-total saving | Latency ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | `0.464145` | `0.457359` | `+0.006786` | `48.94%` | `3.44x` slower |
| `musique` | `0.314151` | `0.309762` | `+0.004389` | `63.44%` | `5.34x` slower |
| `passage_retrieval_en` | `0.993750` | `0.995000` | `-0.001250` | `56.08%` | `3.55x` slower |

### Full-Run Pass Gates

Original full-run pass gates:

- Full workload complete: passed, `2400 / 2400`.
- Score delta vs fp16 full cache no worse than `-0.5` points: passed.
- Score delta vs fp16 Shared-KV reference no worse than `-0.5` points: passed.
- Average latency no worse than fp16 independent full cache: passed, but still slower
  than fp16 Shared-KV reference.
- Persistent KV saving at least `90%` vs independent fp16: narrowly missed, `89.16%`.
- Report exact prediction/score mismatch counts vs fp16 Shared-KV reference: passed.

Decision after full run:

- This is reportable only as a storage-quality result.
- It is not a speed result because it is `4.32x` slower than fp16 Shared-KV reference.
- It is not a strict-equivalence result because it has `105` score mismatches and `233`
  prediction mismatches.
- A stronger follow-up should either exceed `90%` saving vs independent fp16 while
  preserving full quality, or remove Python dequantization/materialization overhead.

### Follow-Up Policy Search Result

The fixed-tail guard search did not produce a stronger full-run candidate:

| Policy | Gate | Score delta | Shared saving | Decision |
| --- | --- | ---: | ---: | --- |
| `start=512/end=1536`, K/V TQ3.5 | balanced 600 rows | `+0.000392` | `61.01%` | Promising but 2Wiki regresses `-0.003600`; estimated full saving only about `89.97%`. Do not full-run yet. |
| `start=512/end=1792`, K/V TQ3.5 | balanced 600 rows | `-0.007280` | `59.44%` | Stop; worse quality than both `1536` and `2048`. |
| `start=512/end=2048`, K fp16/V TQ3.5 | 40 rows | `+0.018750` | `25.32%` | Diagnostic only; storage saving too small. |
| `start=512/end=2048`, K TQ3.5/V fp16 | 40 rows | `+0.031250` | `25.33%` | Diagnostic only; storage saving too small. |

Implication: the next improvement should not be another fixed tail length. It should
change the mechanism: boundary-aware protection, cheap group admission, group-size
scaling, or a fused/non-materializing runtime path.

### Candidate Guards

- Keep the first `N` prefix tokens exact and quantize only the older middle prefix.
- Keep attention-sink and delimiter/question-near prefix regions exact.
- Quantize value only or key only after smoke tests; do not assume both are safe.
- Admit prefix quantization only for groups whose first branch agrees with fp16 on a
  cheap probe.

### Why This Is More Novel Than Independent TurboQuant

Independent TurboQuant compresses every request separately and loses quality/latency on
the repeated-context workload. Guarded prefix-table quantization changes the storage
model: quantization is applied once to selected shared context and amortized across all
requests, while sensitive prefix regions and suffix/decode states remain exact.

## E2: SDPA-Equivalent Prefix-Aware Attention

### Motivation

The current reference cache still materializes prefix+local K/V for stock HuggingFace
attention. The Python split path failed full row-level parity.

A full runtime diagnostic shows that repeated quantized-prefix materialization is the
dominant latency blocker for the guarded prefix table. With one shared dense
materialization of each quantized prefix layer, the full `2400 / 2400`
`start=512/end=2048` run keeps the same quality and persistent storage while reducing
aggregate latency from `4.32x` slower than fp16 Shared-KV to `1.12x`. Task latency ratios
are `1.08x / 1.16x / 1.11x`. The cost is `724,544,651,264` transient dense prefix-cache
bytes.

### Method Target

Implement or bind a prefix-aware attention path that merges prefix and local attention
states with the same semantics as the serving backend. The likely target interface is
online-softmax state merging:

```text
attn(Q, [Kp, Kl], [Vp, Vl]) = merge(attn_state(Q, Kp, Vp), attn_state(Q, Kl, Vl))
```

### Pass Gates

- Guarded shard strict parity first: `40 / 40` score parity vs clone/SDPA.
- Full workload: `2400 / 2400`.
- Full row-level score mismatch count: `0` for strict claim, or a predeclared relaxed
  protocol with mismatch counts reported.
- Latency better than current materialized reference mode.
- Allocator footprint demonstrates removal of the measured `2.52 TB` duplicate prefix
  materialization target.

Updated target after the cached-materialized diagnostic:

- Recover cached-materialized latency without storing a full dense copy of every
  quantized prefix layer.
- Report persistent packed bytes and transient/runtime bytes separately.
- Compare three runtime modes on the same gate: uncached quantized prefix,
  cached-materialized quantized prefix, and fused/non-materializing prefix attention.

## E3: Group-Size Scaling Protocol

### Motivation

The current generated workload uses group size `4`. To strengthen systems novelty, we
need show how the frontier changes with repeat group size.

### Suggested Grid

| Group size | Purpose |
| ---: | --- |
| 1 | No-sharing control |
| 2 | Minimal repeated-context benefit |
| 4 | Current evidence point |
| 8 | Strong sharing regime |

### Pass Gates

- Full quality summary per group size.
- Storage and latency frontier per group size.
- Demonstrate monotonic or explainable scaling of shared-prefix benefit.

## Execution Priority

1. E2 first: it is now the highest-value path because E1 has full storage-quality
   evidence but no speedup.
2. E3 group-size scaling is the next best evidence-building path if kernel work is too
   large for the current iteration.
3. E1 follow-up: try a stronger guarded/selective policy only if it changes the fixed
   guard mechanism and can clear `90%` saving vs independent fp16 without task-level
   regression.
