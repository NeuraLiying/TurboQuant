# Baseline Reproduction Report

Date: 2026-06-23

Workload: `repeated_context_answer_preserving_qa_g4`, Llama-3.1-8B-Instruct, 2400 records / 600 shared-prefix groups.

## Reportable Results

| Method | Status | Score | Shared-total GB | Saving vs FP16 shared | Avg latency s |
|---|---:|---:|---:|---:|---:|
| FP16 shared reference | valid | 0.587374 | 959.330 | 0.00% | 0.336 |
| Guarded shared-prefix TurboQuant 3.5b (ours) | valid | 0.590682 | 404.598 | 57.82% | 0.377 |
| SpectrumKV shared sink-b0.5 | valid | 0.589036 | 498.130 | 48.08% | 0.520 |
| CacheGen q2 shared-prefix quality | valid | 0.569890 | 170.161 | 82.26% | 3.277 |

Complete comparison table:

- `BASELINE_COMPARISON_COMPLETE.md`
- `baseline_comparison_complete.json`
- `baseline_comparison_complete.csv`

Short reportable table:

- `baseline_comparison_current.json`
- `baseline_comparison_current.csv`

The complete table includes independent FP16 full-cache, per-request TurboQuant 2.5b/3.5b, CacheGen storage-only, SKVQ packed, and SKVQ fake-quant partial diagnostics. The short table only includes the main reportable shared-prefix quality rows.

## CacheGen

CacheGen now has a full quality result, not only a storage/reconstruction result.

- Summary: `baseline_runs/cachegen_llama31_8b/repeated_context_shared/cachegen-quality-q2-full-detailed-summary.json`
- Records: 2400 / 2400
- Groups: 600 / 600
- Score: 0.569890
- Mean relative L2: 0.257801
- Max relative L2: 0.261596
- Task scores:
  - 2wikimqa: 0.422218
  - musique: 0.287453
  - passage_retrieval_en: 1.000000

Interpretation: CacheGen gives the strongest storage reduction among valid completed baselines, but it loses 1.748 score points versus the FP16 shared reference on this workload.

## SKVQ

SKVQ does not yet have a valid full quality baseline for this exact setting.

- Completed packed/dequant run: `baseline_runs/skvq_llama31_8b/repeated_context_shared/skvq-shared-k2v2-packed-full-summary.json`
- Packed score: 0.000278
- Packed shared-total storage: 182.235 GB
- Status: invalid quality baseline

Reason: the full completed run uses the packed/dequant path, while the SKVQ README states paper results used fake quantization. The packed path degenerates into repeated punctuation/text artifacts in diagnostics. The official fake-quant path generates normal text on small diagnostics, but OOMs before completing the full repeated-context workload because it materializes dense fake-quant KV tensors in attention.

Current SKVQ fake-quant diagnostics:

- Partial rows: 264 / 2400
- Partial groups: 66 / 600
- Partial score: 0.168144

This partial result is useful for failure attribution, but it should not be reported as a comparable SKVQ baseline.

## Current Takeaway

The current valid comparison supports a storage-quality tradeoff claim, not a full systems speedup claim. Our guarded method preserves quality at a level comparable to FP16 and SpectrumKV while reducing shared-total storage by 57.82% versus FP16 shared and 18.78% versus SpectrumKV. CacheGen compresses more aggressively but has lower quality on the QA tasks.
