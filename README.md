# TurboQuant Reproduction

This repository is a reproduction implementation for [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874), focused on the Llama-3.1-8B-Instruct LongBench Table 1 experiments for:

- Full Cache
- TurboQuant 2.5-bit
- TurboQuant 3.5-bit

TurboQuant is an online, data-oblivious vector quantization method for high-dimensional vectors. It randomly rotates vectors, applies scalar Lloyd-Max quantization to the rotated coordinates for near-optimal MSE distortion, and uses a residual 1-bit QJL-style stage when unbiased inner-product estimation is needed. The paper evaluates TurboQuant on KV-cache compression for long-context LLMs and nearest-neighbor search; this repository focuses on reproducing the Llama-3.1-8B-Instruct LongBench Table 1 KV-cache results.

The code implements TurboQuant KV-cache quantization, LongBench prompt formatting, LongBench-compatible scoring, experiment launch scripts, and comparison builders.

## Repository Layout

```text
turboquant/                 Core TurboQuant and LongBench utilities
experiments/longbench/      LongBench generation runner
scripts/                    Data prep, scoring, job queue, and report builders
configs/                    Local model/dataset path configuration
tests/                      Unit tests for quantization and cache behavior
reproduce/                  Reproduction plans, reports, and final comparison tables
```

Important result files:

- `reproduce/TABLE1_OFFICIAL_COMPARISON.md`
- `reproduce/TABLE1_OFFICIAL_COMPARISON.json`
- `reproduce/TABLE1_OFFICIAL_COMPARISON.csv`
- `reproduce/REPRODUCTION_MANIFEST.json`

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate turboquant
```

Run tests:

```bash
python -m pytest tests -q
```

Expected result:

```text
9 passed
```

## Data And Model Preparation

The reproduction uses:

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Benchmark: LongBench English Table 1 tasks

The local run used Hugging Face assets stored outside the repository:

```text
/home/liying/.cache/huggingface/hub/
/home/liying/datasets/turboquant/
```

If using `hf-mirror`, export:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then prepare/update LongBench cache entries:

```bash
python scripts/prepare_longbench_cache.py \
  --cache-root /home/liying/datasets/turboquant/hf_cache \
  --output-report reproduce/logs/longbench_cache_prepare_report.json
```

Check or edit local paths in:

```text
configs/paths.yaml
configs/llama_first.yaml
```

## How To Run

### Single LongBench Task

Full Cache example:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode full \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index 200 \
  --output reproduce/runs/table1_official/longbench_2wikimqa_full_cache_all.jsonl \
  --progress-every 20
```

TurboQuant 2.5-bit example:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index 200 \
  --output reproduce/runs/table1_official/longbench_2wikimqa_turboquant_2_5bit_all.jsonl \
  --progress-every 20
```

TurboQuant 3.5-bit example:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 3.5 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index 200 \
  --output reproduce/runs/table1_official/longbench_2wikimqa_turboquant_3_5bit_all.jsonl \
  --progress-every 20
```

Score a generated JSONL:

```bash
python scripts/summarize_jsonl_accuracy.py \
  reproduce/runs/table1_official/longbench_2wikimqa_turboquant_2_5bit_all.jsonl \
  --recompute-longbench-score \
  --output reproduce/runs/table1_official/longbench_2wikimqa_turboquant_2_5bit_all.aggregate.json
```

### Full Table 1

Run Full Cache jobs:

```bash
bash scripts/run_table1_full_cache_parallel.sh
```

Run TurboQuant queue:

```bash
python scripts/queue_table1_turboquant_jobs.py
```

Recompute official-compatible metrics for all Table 1 outputs:

```bash
python scripts/recompute_table1_official_metrics.py \
  --run-root reproduce/runs/table1_official \
  --workers 8
```

Build final comparison tables:

```bash
python scripts/build_table1_official_comparison.py \
  --run-root reproduce/runs/table1_official \
  --output-prefix reproduce/TABLE1_OFFICIAL_COMPARISON
```

## Experimental Results

Scope: `meta-llama/Llama-3.1-8B-Instruct`, LongBench-V1 English Table 1 tasks.

### Category-Level Table

| Method | KV Size | Source | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Cache | 16.0 | paper | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| Full Cache | 16.0 | local | 44.45 | 44.18 | 29.29 | 69.53 | 52.60 | 62.26 | 50.38 |
| TurboQuant | 2.5 | paper | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |
| TurboQuant | 2.5 | local | 38.61 | 36.01 | 27.54 | 67.12 | 48.22 | 55.02 | 45.42 |
| TurboQuant | 3.5 | paper | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |
| TurboQuant | 3.5 | local | 42.73 | 43.04 | 28.72 | 68.59 | 52.06 | 61.16 | 49.38 |

### Task-Level Local Scores

| Category | Dataset | Full Cache | TurboQuant 2.5-bit | TurboQuant 3.5-bit |
| --- | --- | ---: | ---: | ---: |
| SingleQA | narrativeqa | 31.13 | 25.35 | 29.25 |
| SingleQA | qasper | 46.77 | 42.44 | 46.32 |
| SingleQA | multifieldqa_en | 55.44 | 48.03 | 52.61 |
| MultiQA | hotpotqa | 55.09 | 44.83 | 54.65 |
| MultiQA | 2wikimqa | 46.36 | 37.96 | 44.47 |
| MultiQA | musique | 31.09 | 25.24 | 30.01 |
| Summarization | gov_report | 34.98 | 32.02 | 34.08 |
| Summarization | qmsum | 25.56 | 24.38 | 25.33 |
| Summarization | multi_news | 27.32 | 26.22 | 26.77 |
| Few shot | trec | 72.50 | 70.50 | 72.00 |
| Few shot | triviaqa | 92.48 | 89.39 | 91.34 |
| Few shot | samsum | 43.61 | 41.47 | 42.43 |
| Synthetic | passage_retrieval_en | 99.50 | 93.00 | 100.00 |
| Synthetic | passage_count | 5.71 | 3.45 | 4.12 |
| Code | lcc | 66.61 | 57.04 | 64.04 |
| Code | repobench-p | 57.90 | 53.00 | 58.29 |

### Run Completeness

| Method | Complete Tasks | Expected Tasks |
| --- | ---: | ---: |
| Full Cache | 16 | 16 |
| TurboQuant 2.5-bit | 16 | 16 |
| TurboQuant 3.5-bit | 16 | 16 |

## Notes

- The final comparison table is generated by `scripts/build_table1_official_comparison.py`.
- Scores use LongBench-compatible prompt templates, max generation lengths, and metrics.
- Large generated outputs under `reproduce/runs/` are intentionally not tracked in git.
