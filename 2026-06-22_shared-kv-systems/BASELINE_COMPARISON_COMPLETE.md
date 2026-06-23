# Complete Baseline Comparison

Date: 2026-06-23

Workload: `repeated_context_answer_preserving_qa_g4`; model: Llama-3.1-8B-Instruct; expected records/groups: 2400 / 600.

## Main Comparable Shared-Prefix Results

| Method | Status | Records | Score | 2Wiki | MuSiQue | Passage | Prefix GB | Request GB | Total GB | Save vs shared FP16 | Save vs independent FP16 | Latency s | rel-L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 shared reference | valid_reference | 2400 | 0.587374 | 0.457359 | 0.309762 | 0.995000 | 924.728 | 34.602 | 959.330 | 0.00% | 74.31% | 0.336 | NA |
| Guarded shared-prefix TurboQuant 3.5b (ours) | valid_candidate | 2400 | 0.590682 | 0.464145 | 0.314151 | 0.993750 | 370.009 | 34.589 | 404.598 | 57.82% | 89.16% | 0.377 | NA |
| SpectrumKV shared sink-b0.5 | valid_baseline | 2400 | 0.589036 | 0.462658 | 0.309451 | 0.995000 | 463.247 | 34.883 | 498.130 | 48.08% | 86.66% | 0.520 | NA |
| CacheGen q2 shared-prefix quality | valid_baseline | 2400 | 0.569890 | 0.422218 | 0.287453 | 1.000000 | 135.319 | 34.841 | 170.161 | 82.26% | 95.44% | 3.277 | 0.257801 |

## Diagnostic Or Non-Comparable Rows

| Method | Scope | Status | Records | Score | Total GB | Save vs shared FP16 | Latency s | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Independent FP16 full cache | independent-per-request | valid_reference_nonshared | 2400 | 0.587276 | 3733.822 | -289.21% | 1.730 | Derived logical FP16 KV storage from prompt+generated tokens; not prefix-shared. |
| Per-request TurboQuant 2.5b | independent-per-request | diagnostic_nonshared | 2400 | 0.501528 | 641.845 | 33.09% | 3.896 | Original per-request TurboQuant baseline; no shared-prefix placement. |
| Per-request TurboQuant 3.5b | independent-per-request | diagnostic_nonshared | 2400 | 0.568660 | 875.239 | 8.77% | 9.683 | Original per-request TurboQuant baseline; no shared-prefix placement. |
| CacheGen q2 storage/reconstruction only | shared-prefix | storage_only_not_quality | 2400 | NA | 168.461 | 82.44% | NA | Superseded by full CacheGen quality run; kept for storage/reconstruction audit. |
| SKVQ packed k2-v2-g128-w128-clip-sink5 | shared-prefix | invalid_quality | 2400 | 0.000278 | 182.235 | 81.00% | 5.659 | Completed packed/dequant path degenerates; not paper fake-quant path. |
| SKVQ fake-quant partial | shared-prefix | incomplete_oom_diagnostic | 264 | 0.168144 | NA | NA | NA | Official fake-quant path OOMs before full workload completion; partial rows are not reportable baseline. |

## Source Files

- Independent FP16 full cache: `reproduce/runs/repeated_context/repeated_context_answer_preserving_qa_g4_full_cache_summary.json`
- FP16 shared reference: `reproduce/runs/repeated_context/repeated_context_answer_preserving_qa_g4_fp16_prefix_shared_reference_full_summary.json`
- Guarded shared-prefix TurboQuant 3.5b (ours): `2026-06-22_shared-kv-systems/guarded_prefix_tq35_s512_e2048_cached_materialized_full_0000_2400_summary.json`
- SpectrumKV shared sink-b0.5: `2026-06-22_shared-kv-systems/baseline_runs/spectrumkv_llama31_8b/repeated_context_shared/spectrumkv-shared-sink-b05-full-after-oomfix/summary.json`
- CacheGen q2 shared-prefix quality: `2026-06-22_shared-kv-systems/baseline_runs/cachegen_llama31_8b/repeated_context_shared/cachegen-quality-q2-full-detailed-summary.json`
- Per-request TurboQuant 2.5b: `reproduce/runs/repeated_context/repeated_context_answer_preserving_qa_g4_turboquant_2p5_summary.json`
- Per-request TurboQuant 3.5b: `reproduce/runs/repeated_context/repeated_context_answer_preserving_qa_g4_turboquant_3p5_summary.json`
- CacheGen q2 storage/reconstruction only: `2026-06-22_shared-kv-systems/baseline_runs/cachegen_llama31_8b/repeated_context_shared/cachegen-shared-q2-dualgpu-dedup-summary.json`
- SKVQ packed k2-v2-g128-w128-clip-sink5: `2026-06-22_shared-kv-systems/baseline_runs/skvq_llama31_8b/repeated_context_shared/skvq-shared-k2v2-packed-full-summary.json`
- SKVQ fake-quant partial: `2026-06-22_shared-kv-systems/baseline_runs/skvq_llama31_8b/repeated_context_shared/skvq-fake-q2-shard*/predictions.jsonl`

Interpretation: the main table is now complete for available full shared-prefix quality runs. SKVQ fake-quant remains incomplete/OOM; SKVQ packed remains invalid as a quality baseline. Independent full-cache and per-request TurboQuant are retained for context but should not be mixed with shared-prefix baselines in the main claim.
