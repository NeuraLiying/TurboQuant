# Baseline Reproduction Results

Date: 2026-06-22

All runs used the official GitHub repositories under `/home/liying/projects` with thin adapters for local Llama-3.1-8B-Instruct weights and the local LongBench arrow cache. The prompt mode is `turboquant_legacy_prompt`, matching the existing TurboQuant runner: `context + question + answer_prefix`, with middle truncation to `max_length=6144`.

## Repositories

| Baseline | Repo | Commit | Notes |
|---|---:|---|---|
| SKVQ | `/home/liying/projects/SKVQ` | `fdfc7ec315c16293f68dd771850b47f5199787fc` | Official SKVQ model/quantizer/generation path; adapter adds Llama-3.1 RoPE and Transformers 4.53 generation compatibility. |
| CacheGen | `/home/liying/projects/CacheGen` | `6bed34ca9d495289955ed754cf2ad0c43346ee48` | Official LMCache CacheGen serializer/deserializer; adapter measures Llama-3.1 KV storage/error. |
| SpectrumKV | `/home/liying/projects/kvcache-lab` | `e108438df1d903c448a123a43276da88542559d1` | Official SWS/QCBM/KVQuantizer; adapter materializes compressed HF cache for LongBench generation. |

## SKVQ Full Quality

Run:
`baseline_runs/skvq_llama31_8b/longbench_out/pred/skvq-k2v2-g128-w128-clip-sink5-full-l6144-unified`

Configuration:
`k2-v2-g128-w128-clip-sink5`, `max_length=6144`, 200 examples per task.

| Dataset | Count | Score |
|---|---:|---:|
| 2wikimqa | 200 | 13.11 |
| musique | 200 | 6.45 |
| passage_retrieval_en | 200 | 35.76 |
| Mean | 600 | 18.44 |

## SpectrumKV Quality Subset

Run:
`baseline_runs/spectrumkv_llama31_8b/longbench_out/pred/spectrumkv-sink-b05-n20-l6144-unified`

Configuration:
`policy=sink`, `budget=0.5`, `max_length=6144`, 20 examples per task.

| Dataset | Count | Score |
|---|---:|---:|
| 2wikimqa | 20 | 9.18 |
| musique | 20 | 4.70 |
| passage_retrieval_en | 20 | 50.00 |
| Mean | 60 | 21.29 |

Budget realization: for 6144-token inputs, the adapter used about 4 FP16 sink tokens, 6130 INT8 tokens, and 9 INT4 tokens, with actual normalized budget about 0.49996.

## CacheGen Storage/Error Subset

Run:
`baseline_runs/cachegen_llama31_8b/cachegen-q2-n20-l6144-unified`

Configuration:
`QUANT_LEVEL=2`, CacheGen codec config key `lmsys/longchat-7b-16k`, `max_length=6144`, 20 examples per task. This is a storage/error baseline, not a LongBench quality run.

| Dataset | Count | Mean Input Tokens | Mean Storage Saving | Mean Encode s | Mean Decode s | Mean Rel-L2 |
|---|---:|---:|---:|---:|---:|---:|
| 2wikimqa | 20 | 5381.30 | 85.02% | 0.149 | 0.218 | 0.2551 |
| musique | 20 | 6143.95 | 85.22% | 0.115 | 0.263 | 0.2566 |
| passage_retrieval_en | 20 | 6143.00 | 85.26% | 0.122 | 0.265 | 0.2571 |

## Important Caveats

- These are adapter-based official-code reproductions on Llama-3.1-8B. The original repositories do not directly expose this exact Llama-3.1 + local LongBench workflow.
- SKVQ is the only full 600-example LongBench quality baseline completed here.
- SpectrumKV quality is a 60-example subset because the adapter uses materialized cache simulation with greedy Python decode.
- CacheGen is reported as storage/error only; its official generation path targets older KV interfaces and workloads, so quality was not claimed.
- The `max_length=6144` setting is necessary for stable single-RTX-4090 reproduction under these official-code adapters. It must be matched when comparing to our method.
