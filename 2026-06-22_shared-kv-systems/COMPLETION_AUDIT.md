# Completion And Claim Audit

This audit checks the new Shared-KV systems direction against the original objective:
create a new dated direction folder, explore a novel improvement, record the work, and
obtain sufficiently convincing reportable experimental results.

## Objective Requirements

| Requirement | Evidence | Status |
| --- | --- | --- |
| New dated-topic folder under `/home/liying/projects/turboquant` | `2026-06-22_shared-kv-systems/` contains the direction README, design notes, logs, generated summaries, and full result JSONL artifacts. | Satisfied |
| Separate from previous reproduction/incremental exploration | `README.md` states this folder starts a new Shared-KV systems direction and links the new consolidated artifacts. | Satisfied |
| Novelty assessment against related work | `RELATED_WORK.md` explicitly compares against prefix caching, shared-prefix kernels, KV storage/transfer systems, and KV compression work. | Satisfied |
| A method with defensible novelty | The narrow method is a guarded TurboQuant-compressed shared-prefix table: fp16 prefix boundary guards, TurboQuant 3.5 middle shared prefix, fp16 request-local KV, evaluated against an fp16 Shared-KV reference. | Satisfied as storage-quality method |
| Sufficiently convincing reportable experiments | Full `2400 / 2400` guarded-prefix run, full fp16 Shared-KV reference, independent TurboQuant baselines, task-level breakdown, storage decomposition, mismatch counts, and runtime diagnostics. | Satisfied for storage-quality claim |
| Clear non-claims and limitations | `RELATED_WORK.md`, `README.md`, and `claim_frontier.md` state not to claim prefix caching as novel, strict equivalence, or final TurboQuant speedup. | Satisfied |

## Claim-Ready Contributions

### 1. fp16 Shared-KV Reference And Protocol

Evidence:

- Full generated repeated-context workload: `2400 / 2400` rows.
- Clone parity: `2400` compared rows, `0` mismatches.
- Score delta versus independent fp16: `+0.0097` LongBench points.
- Average latency: `1.730s -> 0.336s`, `5.14x` faster.
- Persistent KV storage: `3.40 TB -> 893.45 GB`, `74.30%` lower.

Interpretation:

This is a claim-ready systems baseline/protocol result. It is not novel as generic
prefix caching, but it establishes the answer-preserving repeated-context evaluation
surface and the exact storage accounting used by the new method.

### 2. Guarded TurboQuant Shared-Prefix Storage Method

Evidence:

- Policy: first `512` and last `2048` shared-prefix tokens in fp16; middle shared-prefix
  segment stored with TurboQuant 3.5; request-local suffix/decode KV in fp16.
- Full run: `2400 / 2400` rows, `600` repeat groups.
- Score: `0.590682` versus fp16 Shared-KV `0.587374`, delta `+0.3308` LongBench points.
- Shared-total storage saving versus fp16 Shared-KV: `57.82%`.
- Persistent storage saving versus independent fp16 materialized KV: `89.16%`.
- Score mismatches versus fp16 Shared-KV: `105`; prediction mismatches: `233`.
- Task deltas: `2wikimqa +0.006786`, `musique +0.004389`,
  `passage_retrieval_en -0.001250`.

Interpretation:

This is the strongest new-method result. It is reportable as a storage-quality method:
it preserves full-workload score within the reference band while materially reducing
persistent Shared-KV storage.

### 3. Runtime Diagnostic For The Remaining Speed Gap

Evidence:

- Uncached guarded prefix latency: `1.453345s`, `4.32x` slower than fp16 Shared-KV.
- Cached-materialized diagnostic latency: `0.377205s`, `1.12x` of fp16 Shared-KV.
- Same score and persistent storage as the uncached guarded prefix result.
- Additional transient dense prefix-cache bytes: `724,544,651,264`.

Interpretation:

This does not create a final speed claim. It does show the main runtime blocker is
repeated dequantization/materialization of the same quantized shared prefix, which
justifies the next system target: fused or non-materializing quantized-prefix decode
that recovers the cached-materialized latency without paying the dense transient cache.

## Novelty Boundary

Do claim:

- A guarded shared-prefix-table storage policy that composes TurboQuant with exact
  shared-KV reuse and protects sensitive prefix boundary regions.
- Full answer-preserving LongBench-style evidence against both fp16 Shared-KV and
  independent TurboQuant baselines.
- Explicit persistent/transient storage accounting for shared prefix, request-local KV,
  metadata, and materialized-prefix runtime cache.

Do not claim:

- Prefix caching, shared-prefix attention, prefix trees, or KV block reuse as new.
- A final speedup for guarded TurboQuant Shared-KV.
- Strict output equivalence to fp16 Shared-KV.
- A generic TurboQuant codec improvement independent of the shared-prefix systems setup.

## Remaining Non-Blocking Gaps

- The result narrowly misses the original `90%` persistent-saving gate versus independent
  fp16 at `89.16%`.
- The current compatibility path still materializes quantized prefix tensors for stock
  attention.
- A stronger follow-up would need a fused/non-materializing kernel or a new protection
  mechanism that improves the memory frontier without quality loss.

## Audit Conclusion

The original objective is satisfied for a storage-quality research contribution: the
new direction folder exists, the novelty boundary is documented, the method is narrow
enough to distinguish from related work, and the full-run evidence is sufficient to
support a reportable storage-quality claim.

The objective is not satisfied for a final speedup contribution. The speed-related result
is intentionally classified as a diagnostic and should be presented only as motivation
for the next kernel implementation.

## Stricter Top-Conference Reassessment

After reviewing nearby reusable-KV storage and mixed-precision KV quantization work, this
audit should not be read as saying the current method is ready for a standalone
top-conference methods paper.

The current result is best treated as a strong empirical storage study and a starting
point for a stronger method. Its main novelty risks are:

- applying an existing KV quantizer to a shared-prefix table;
- using a fixed fp16 sink/recent-style guard that resembles prior KV-compression
  protection windows;
- lacking a reusable-KV baseline such as KVTC in the same experimental matrix;
- lacking an end-to-end serving speed/capacity result;
- lacking a principle, objective, or theory for why `512/2048` guards are the right
  protection policy.

For a top-conference submission, the next phase must add a new method module such as
reuse-aware precision allocation, cross-suffix sensitivity protection, or
non-materializing compressed-prefix attention, and it must compare against reusable-KV
storage and mixed-precision KV-cache baselines.

See [TOP_CONFERENCE_ROADMAP.md](TOP_CONFERENCE_ROADMAP.md).
