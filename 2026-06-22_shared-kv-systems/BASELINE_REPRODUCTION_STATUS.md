# Baseline Reproduction Status

Date: 2026-06-22

Status: completed for the three requested baseline projects on the local Llama-3.1-8B-Instruct setup, using official GitHub repositories under `/home/liying/projects` plus thin compatibility adapters.

## Completed Runs

| Method | Official repo | Commit | Run type | Output |
| --- | --- | --- | --- | --- |
| SKVQ | `/home/liying/projects/SKVQ` | `fdfc7ec315c16293f68dd771850b47f5199787fc` | Full LongBench quality, 200 examples/task | `baseline_runs/skvq_llama31_8b/longbench_out/pred/skvq-k2v2-g128-w128-clip-sink5-full-l6144-unified` |
| SpectrumKV | `/home/liying/projects/kvcache-lab` | `e108438df1d903c448a123a43276da88542559d1` | LongBench quality subset, 20 examples/task | `baseline_runs/spectrumkv_llama31_8b/longbench_out/pred/spectrumkv-sink-b05-n20-l6144-unified` |
| CacheGen | `/home/liying/projects/CacheGen` | `6bed34ca9d495289955ed754cf2ad0c43346ee48` | KV storage/error subset, 20 examples/task | `baseline_runs/cachegen_llama31_8b/cachegen-q2-n20-l6144-unified` |

## Unified Setup

- Model: `/home/liying/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`
- Datasets: local LongBench arrow cache under `/home/liying/datasets/turboquant/hf_cache/Xnhyacinth___long_bench`
- Tasks: `2wikimqa`, `musique`, `passage_retrieval_en`
- Prompt mode: `turboquant_legacy_prompt`
- Context truncation: middle truncation to `max_length=6144`

## Main Results

SKVQ full quality:

| Dataset | Count | Score |
| --- | ---: | ---: |
| 2wikimqa | 200 | 13.11 |
| musique | 200 | 6.45 |
| passage_retrieval_en | 200 | 35.76 |
| Mean | 600 | 18.44 |

SpectrumKV quality subset:

| Dataset | Count | Score |
| --- | ---: | ---: |
| 2wikimqa | 20 | 9.18 |
| musique | 20 | 4.70 |
| passage_retrieval_en | 20 | 50.00 |
| Mean | 60 | 21.29 |

CacheGen storage/error subset:

| Dataset | Count | Mean input tokens | Mean storage saving | Mean encode s | Mean decode s | Mean rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wikimqa | 20 | 5381.30 | 85.02% | 0.149 | 0.218 | 0.2551 |
| musique | 20 | 6143.95 | 85.22% | 0.115 | 0.263 | 0.2566 |
| passage_retrieval_en | 20 | 6143.00 | 85.26% | 0.122 | 0.265 | 0.2571 |

The detailed record is in `BASELINE_REPRODUCTION_RESULTS.md`.

## Caveats

- These are official-code adapter runs because the upstream repositories do not directly expose this exact Llama-3.1-8B + local LongBench workflow.
- SKVQ is the only full 600-example LongBench quality baseline completed in this batch.
- SpectrumKV quality is a 60-example subset due to materialized cache simulation overhead.
- CacheGen is reported as a storage/error baseline, not as a LongBench QA quality baseline.
- Future comparisons with our method should use the same prompt mode and `max_length=6144`.
