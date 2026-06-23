# Baseline Validation Audit

Date: 2026-06-23

This note records the current validity status of the SKVQ and CacheGen repeated-context shared-prefix baselines before starting new method-module experiments.

## SKVQ

Current full packed result:

- Path: `baseline_runs/skvq_llama31_8b/repeated_context_shared/skvq-shared-k2v2-packed-full-summary.json`
- Records: 2400 / 2400
- Score: 0.000278
- Shared-total storage: 182.235 GB
- Average latency: 5.659 s

This result should not be used as a valid SKVQ quality baseline.

Reasons:

- The SKVQ README states that paper results were obtained with fake quantization, and that the current dequantization/GEMV kernel is naive and inefficient.
- The full completed run used `SKVQ_FAKE_QUANT=0`, i.e. the packed quant/dequant path, not the official paper-quality fake-quant path.
- The original packed Llama flash-attention branch failed with a shape mismatch before a local compatibility patch was added.
- A small A/B diagnostic with the same shared-prefix adapter shows fake-quant generates normal text, while packed quant/dequant degenerates into repeated `!` tokens:
  - Fake diagnostic: `skvq-diagnose-g0000-fake/predictions.jsonl`
  - Packed diagnostic: `skvq-diagnose-g0000-packed/predictions.jsonl`
- The fake-quant partial run completed 128 rows with normal text but lower quality than FP16 on the same slice:
  - SKVQ fake first 128: 0.229754
  - FP16 shared reference first 128: 0.437359
- Additional fake-quant shards completed only 264 rows / 66 groups before OOM, with aggregate partial score 0.168144. These partial rows are diagnostic only and are not a reportable full baseline.
- The fake-quant full run OOMs on this long shared-prefix workload because it materializes dense fake-quant KV tensors during prefix processing. That is an implementation/runtime limitation for this workload, not by itself a proof that the SKVQ algorithm has zero quality.
- The OOM point is in the official fake-quant attention path, where `k_sink`, `k_quant`, and `k_window` are concatenated into dense tensors before attention. The local shared-prefix adapter already avoids prefix-cache cloning and prefill logits, so this failure is not explained by an extra adapter-side prefix clone.
- The current Llama-3.1-8B setting does not match the paper-recommended SKVQ LongBench setup. The README recommends `llama3-70b-instruct` with `k2-v2-g128-w128-reorder-pre_rope-clip-sink5-fp8`; the repository calibration config has Llama-3-70B entries but no Llama-3.1-8B reorder/smooth artifacts.

Current attribution:

- The zero-quality full result is primarily attributable to the packed/dequant path and unsupported/unmatched reproduction setting, not to the shared-prefix adapter alone.
- The partial fake-quant degradation may reflect an unsuitable uncalibrated Llama-3.1-8B SKVQ configuration or workload sensitivity, but it is not a complete valid full baseline because it OOMs before the full workload finishes.

Required before reporting SKVQ:

- Either reproduce SKVQ on an official supported model/config with available calibration artifacts, or build a defensible Llama-3.1-8B fake-quant configuration and run it to completion.
- Do not report the packed full result as SKVQ quality; at most report it as a failed packed-kernel diagnostic.

## CacheGen

Previous full repeated-context result:

- Path: `baseline_runs/cachegen_llama31_8b/repeated_context_shared/cachegen-shared-q2-dualgpu-dedup-summary.json`
- Groups: 600 / 600
- Records represented: 2400
- Shared-total storage: 168.461 GB
- Mean relative L2 reconstruction error: 0.257801

This existing full result is storage/reconstruction-only and should not be used as an end-to-end quality baseline.

Reason:

- The current adapter `run_turboquant_repeated_context_shared_cachegen.py` pre-fills the shared prefix, converts KV to a CacheGen blob, runs the official serializer/deserializer, and reports encoded size plus reconstruction error.
- It does not convert the decoded blob back to `past_key_values`, does not run `model.generate`, and does not score predictions.
- Official CacheGen scripts do have a quality path: `run_cachegen.py` decodes KV, converts it to `past_key_values`, calls `model.generate(..., past_key_values=decoded_kv)`, and computes the dataset metric.

This gap is now closed by a full quality run.

Reportable full quality result:

- Path: `baseline_runs/cachegen_llama31_8b/repeated_context_shared/cachegen-quality-q2-full-detailed-summary.json`
- Shards:
  - `cachegen-quality-q2-shard0000-0600`
  - `cachegen-quality-q2-shard0600-1200`
  - `cachegen-quality-q2-shard1200-1800`
  - `cachegen-quality-q2-shard1800-2400`
- Records: 2400 / 2400
- Groups: 600 / 600
- Score: 0.569890
- Shared-total storage: 170.161 GB
- Mean relative L2 reconstruction error: 0.257801
- Average latency: 3.277 s

Implementation note:

- A `--generate-quality` mode was added to `/home/liying/projects/CacheGen/run_turboquant_repeated_context_shared_cachegen.py`.
- The mode keeps the official CacheGen serializer/deserializer, converts decoded shared-prefix blobs to a HF `DynamicCache`, runs greedy suffix generation, and writes `predictions.jsonl`.
- The previous 4-row smoke result validated the path before the full run; the full 2400-row run is now complete.

## Current Comparison Status

Comparable end-to-end quality results currently available:

- FP16 shared reference: 0.587374, shared-total storage 959.330 GB
- SpectrumKV shared baseline: 0.589036, shared-total storage 498.130 GB
- CacheGen q2 shared-prefix quality baseline: 0.569890, shared-total storage 170.161 GB
- Our guarded shared-prefix method: 0.590682, shared-total storage 404.598 GB

Not yet comparable as quality baselines:

- SKVQ packed full: invalid quality baseline due to packed-path degeneration.
- SKVQ fake-quant: partial diagnostics exist, but the official fake-quant path OOMs before completing this 2400-row Llama-3.1-8B repeated-context workload.

Decision:

The CacheGen quality gap is closed. The current reportable baseline table includes FP16 shared reference, SpectrumKV, and CacheGen. SKVQ should be reported separately as unsupported/incomplete for this exact Llama-3.1-8B repeated-context setting unless a faithful fake-quant run is made memory-safe and completed, or the official supported SKVQ model/config is reproduced.
