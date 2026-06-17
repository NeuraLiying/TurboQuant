# TurboQuant Reproduction

This repository provides a reproduction implementation for [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874), with a focus on the Llama 3.1 8B Instruct LongBench Table 1 KV cache compression experiments.

The reproduced settings include:

- Full Cache
- TurboQuant 2.5 bit
- TurboQuant 3.5 bit

TurboQuant is an online, data-oblivious vector quantization method for high-dimensional vectors. It applies randomized rotation, scalar Lloyd Max quantization on rotated coordinates, and an optional residual 1 bit QJL style correction stage for inner product estimation. The original paper evaluates TurboQuant on KV cache compression for long-context LLM inference and nearest-neighbor search. This repository focuses on the LongBench KV cache compression setting.

## Scope

This repository focuses on reproducing the Llama 3.1 8B Instruct LongBench Table 1 experiments. It does not attempt to reproduce every experiment in the TurboQuant paper.

Implemented components include:

- TurboQuant KV cache quantization
- Full cache baseline evaluation
- LongBench prompt formatting
- LongBench-compatible scoring
- experiment launch scripts
- result aggregation scripts
- comparison table builders
- unit tests for quantization and cache behavior

## Repository Layout

```text
turboquant/                 Core TurboQuant and LongBench utilities
experiments/longbench/      LongBench generation runner
scripts/                    Data preparation, scoring, job queue, and report builders
configs/                    Local model and dataset path configuration
tests/                      Unit tests for quantization and cache behavior
reproduce/                  Reproduction plans, reports, and final comparison tables
```

Important result files:

```text
reproduce/TABLE1_OFFICIAL_COMPARISON.md
reproduce/TABLE1_OFFICIAL_COMPARISON.json
reproduce/TABLE1_OFFICIAL_COMPARISON.csv
reproduce/REPRODUCTION_MANIFEST.json
reproduce/STRUCTURE_ADAPTIVE_LOWBIT_GAIN_RESULTS.md
```

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
40 passed
```

## Data and Model Preparation

The reproduction uses:

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Benchmark: LongBench English Table 1 tasks

Set local model and dataset paths in:

```text
configs/paths.yaml
configs/llama_first.yaml
```

A typical configuration uses the following environment variables:

```bash
export HF_HOME=/path/to/huggingface/cache
export DATA_ROOT=/path/to/turboquant/data
export MODEL_PATH=/path/to/Llama-3.1-8B-Instruct
```

Prepare or update the LongBench cache entries:

```bash
python scripts/prepare_longbench_cache.py \
  --cache-root "$DATA_ROOT/hf_cache" \
  --output-report reproduce/logs/longbench_cache_prepare_report.json
```

## How to Run

### Single LongBench Task

Full Cache example:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode full \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 \
  --end-index 200 \
  --output reproduce/runs/table1_official/longbench_2wikimqa_full_cache_all.jsonl \
  --progress-every 20
```

TurboQuant 2.5 bit example:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 \
  --end-index 200 \
  --output reproduce/runs/table1_official/longbench_2wikimqa_turboquant_2_5bit_all.jsonl \
  --progress-every 20
```

TurboQuant 3.5 bit example:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 3.5 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 \
  --end-index 200 \
  --output reproduce/runs/table1_official/longbench_2wikimqa_turboquant_3_5bit_all.jsonl \
  --progress-every 20
```

Score a generated JSONL file:

```bash
python scripts/summarize_jsonl_accuracy.py \
  reproduce/runs/table1_official/longbench_2wikimqa_turboquant_2_5bit_all.jsonl \
  --recompute-longbench-score \
  --output reproduce/runs/table1_official/longbench_2wikimqa_turboquant_2_5bit_all.aggregate.json
```

### Full Table 1 Reproduction

Run Full Cache jobs:

```bash
bash scripts/run_table1_full_cache_parallel.sh
```

Run TurboQuant jobs:

```bash
python scripts/queue_table1_turboquant_jobs.py
```

Recompute LongBench-compatible metrics for all Table 1 outputs:

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

### 2.5-bit Adaptive LowBit-Gain

This checkpoint also includes a 2.5-bit incremental method over reproduced TurboQuant.
The method uses `lowbit_gain_mse` only for long, non-code prompts whose Passage-style structure is not high risk; otherwise it falls back to standard TurboQuant MSE.

Run a task with the adaptive quantizer gate:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --key-quantizer mse \
  --value-quantizer mse \
  --long-prompt-threshold 6144 \
  --long-prompt-key-quantizer lowbit_gain_mse \
  --long-prompt-value-quantizer lowbit_gain_mse \
  --long-prompt-exclude-code \
  --long-prompt-max-passages 15 \
  --long-prompt-max-question-marks 5 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index 200 \
  --output reproduce/runs/incremental/structure_adaptive_lowbit_gain_2wikimqa_turboquant_2p5_full.jsonl
```

Build the Table 1 comparison from completed TurboQuant and LowBit-Gain runs:

```bash
python scripts/build_content_adaptive_lowbit_table.py \
  --threshold 6144 \
  --max-passages 15 \
  --max-question-marks 5 \
  --feature-table reproduce/incremental/prompt_gate_features_table1.json \
  --output reproduce/incremental/structure_adaptive_lowbit_gain_threshold_6144_p15_q5_table1_comparison.json
```

## Experimental Results

Scope:

```text
Model: meta-llama/Llama-3.1-8B-Instruct
Benchmark: LongBench V1 English Table 1 tasks
Settings: Full Cache, TurboQuant 2.5 bit, TurboQuant 3.5 bit
```

### Category Level Results

| Method | KV Size | Source | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Cache | 16.0 | paper | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| Full Cache | 16.0 | local | 44.45 | 44.18 | 29.29 | 69.53 | 52.60 | 62.26 | 50.38 |
| TurboQuant | 2.5 | paper | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |
| TurboQuant | 2.5 | local | 38.61 | 36.01 | 27.54 | 67.12 | 48.22 | 55.02 | 45.42 |
| TurboQuant | 3.5 | paper | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |
| TurboQuant | 3.5 | local | 42.73 | 43.04 | 28.72 | 68.59 | 52.06 | 61.16 | 49.38 |

### 2.5-bit Incremental Result

| Method | KV Size | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant | 2.5 | 38.61 | 36.01 | 27.54 | 67.12 | 48.22 | 55.02 | 45.42 |
| Structure-Adaptive LowBit-Gain | 2.5 | 40.32 | 38.10 | 27.60 | 67.77 | 50.29 | 55.02 | 46.52 |

Detailed task-level results are in `reproduce/STRUCTURE_ADAPTIVE_LOWBIT_GAIN_RESULTS.md`.

### Task Level Local Scores

| Category | Dataset | Full Cache | TurboQuant 2.5 bit | TurboQuant 3.5 bit |
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
| TurboQuant 2.5 bit | 16 | 16 |
| TurboQuant 3.5 bit | 16 | 16 |

## Reproduction Notes

The final comparison table is generated by:

```bash
python scripts/build_table1_official_comparison.py \
  --run-root reproduce/runs/table1_official \
  --output-prefix reproduce/TABLE1_OFFICIAL_COMPARISON
```

Scores are computed with LongBench-compatible prompt templates, maximum generation lengths, and metrics.

Large generated outputs under the following directory are intentionally excluded from version control:

```text
reproduce/runs/
```

## Limitations

This repository focuses on the Llama 3.1 8B Instruct LongBench Table 1 KV cache experiments. Results may differ from the paper because of environment differences, model checkpoint revisions, tokenizer or chat template behavior, decoding settings, hardware, and dependency versions.

This repository does not currently include optimized low-level kernels for packed low-bit KV cache inference. The current implementation is designed for reproducible evaluation and analysis.
