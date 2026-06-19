# TurboQuant Reproduction and Incremental KV Quantization Experiments

This repository provides a reproduction implementation for [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874), with a focus on the Llama 3.1 8B Instruct LongBench Table 1 KV cache compression experiments. It also records incremental KV quantization experiments built on top of the reproduced TurboQuant baseline.

The repository separates results into two groups:

- Reproduction baselines: Full Cache, TurboQuant 2.5 bit, TurboQuant 3.5 bit. These are the local reproduction of the paper's Table 1 setup.
- Incremental methods: additional methods evaluated against the reproduced TurboQuant outputs. The complete Table 1 increment is **Unified Regular-Gain Gate**. The current core-method direction is **Rate-Hadamard Value MSE**, a rate-aware value Hadamard preconditioning rule with complete MultiQA evidence at both 2.5-bit and 3.5-bit.

The results should be read in two parts. The reproduction table reports Full Cache and TurboQuant against the LongBench Table 1 categories. The incremental tables then use the reproduced TurboQuant numbers as baselines and report additional method results as evidence for the contribution.

TurboQuant is an online, data-oblivious vector quantization method for high-dimensional vectors. It applies randomized rotation, scalar Lloyd Max quantization on rotated coordinates, and an optional residual 1 bit QJL style correction stage for inner product estimation. The original paper evaluates TurboQuant on KV cache compression for long-context LLM inference and nearest-neighbor search. This repository focuses on the LongBench KV cache compression setting.

Unified Regular-Gain Gate keeps the original TurboQuant storage budgets and uses the same prompt-only rule at both bit widths. It falls back to reproduced TurboQuant MSE by default, and switches both K and V to `regular_gain_mse` only when the prompt is not code-like, has at most 12 Passage-style passages, and has at most 20 question marks.

Rate-Regime MSE changes the reconstruction/preconditioning path rather than using a prompt gate. It uses norm-gain scalar TurboQuant in the lower-rate regime and learned unit-block reconstruction in the higher-rate regime. Its 3.5-bit Table 1 run is complete and improves the reproduced TurboQuant average from `49.38` to `49.86`; the 2.5-bit full Table 1 run is incomplete, with completed category/task results recorded as additional evidence rather than a final claim.

Rate-Hadamard Value MSE is the current method-level direction. It keeps K on reproduced TurboQuant MSE and changes only the V path: at low rate it uses a conservative per-vector value Hadamard preconditioner, while at higher rate it applies value Hadamard preconditioning only in later decoder layers. On the full MultiQA group, it improves the reproduced TurboQuant average from `36.01` to `37.16` at 2.5 bits and from `43.04` to `43.73` at 3.5 bits.

## Scope

This repository focuses on reproducing the Llama 3.1 8B Instruct LongBench Table 1 experiments and recording incremental methods on top of that reproduction. It does not attempt to reproduce every experiment in the TurboQuant paper.

Implemented components include:

- TurboQuant KV cache quantization
- prompt-gated incremental KV cache quantization
- rate-aware value Hadamard KV cache quantization
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
reproduce/incremental/UNIFIED_REGULAR_GAIN_GATE_FINAL.md
reproduce/incremental/unified_regular_gain_gate_table1_runner.md
reproduce/incremental/rate_regime_mse_table1.md
reproduce/incremental/RATE_HADAMARD_VALUE_MSE_RESULTS.md
reproduce/incremental/rate_hadamard_value_mse_multiqa_progress.md
reproduce/INCREMENTAL_EXPERIMENTS.md
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
91 passed
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

### Unified Regular-Gain Gate

This checkpoint includes a unified incremental method over reproduced TurboQuant.
The same gate is used for both 2.5-bit and 3.5-bit settings: it uses `regular_gain_mse` for long non-code prompts with bounded Passage-style structure, and otherwise falls back to standard TurboQuant MSE.

Run a task with the adaptive quantizer gate:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --key-quantizer mse \
  --value-quantizer mse \
  --long-prompt-threshold 0 \
  --long-prompt-key-quantizer regular_gain_mse \
  --long-prompt-value-quantizer regular_gain_mse \
  --long-prompt-exclude-code \
  --long-prompt-max-passages 12 \
  --long-prompt-max-question-marks 20 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index 200 \
  --output reproduce/runs/incremental/unified_regular_gain_gate_2wikimqa_turboquant_2p5_full.jsonl
```

Queue the full Table 1 task set for both bit widths:

```bash
python scripts/queue_method_table1_jobs.py \
  --method-name unified_regular_gain_gate \
  --key-quantizer mse \
  --value-quantizer mse \
  --long-prompt-threshold 0 \
  --long-prompt-key-quantizer regular_gain_mse \
  --long-prompt-value-quantizer regular_gain_mse \
  --long-prompt-exclude-code \
  --long-prompt-max-passages 12 \
  --long-prompt-max-question-marks 20
```

Build the Table 1 comparison from completed TurboQuant and Unified Regular-Gain Gate runs:

```bash
python scripts/build_unified_regular_gain_gate_table.py \
  --run-dir reproduce/runs/incremental \
  --baseline-dir reproduce/runs/table1_official \
  --output-prefix reproduce/incremental/unified_regular_gain_gate_table1_runner
```

### Rate-Regime MSE

Rate-Regime MSE is run as a direct quantizer replacement for both K and V:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 3.5 \
  --key-quantizer rate_regime_mse \
  --value-quantizer rate_regime_mse \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index 200 \
  --output reproduce/runs/incremental/rate_regime_mse_2wikimqa_turboquant_3p5_full.jsonl
```

Queue Table 1 jobs for one bit width:

```bash
python scripts/queue_method_table1_jobs.py \
  --method-name rate_regime_mse \
  --key-quantizer rate_regime_mse \
  --value-quantizer rate_regime_mse \
  --bits 3p5
```

Build the current Table 1 comparison:

```bash
python scripts/build_method_table1.py \
  --method-name rate_regime_mse \
  --display-name "Rate-Regime MSE" \
  --description "Rate-regime rotated-domain reconstruction: use norm-gain scalar TurboQuant at low target rate and learned unit block TurboQuant at higher target rate." \
  --output-prefix reproduce/incremental/rate_regime_mse_table1
```

### Rate-Hadamard Value MSE

Rate-Hadamard Value MSE is the current core-method increment. It uses one quantizer name for both bit budgets:

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --key-quantizer rate_hadamard_value_mse \
  --value-quantizer rate_hadamard_value_mse \
  --outlier-hadamard-block-size 16 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index 200 \
  --output reproduce/runs/incremental/rate_hadamard_value_mse_2wikimqa_turboquant_2p5_full.jsonl
```

Queue the full Table 1 task set for both bit widths:

```bash
python scripts/queue_method_table1_jobs.py \
  --method-name rate_hadamard_value_mse \
  --key-quantizer rate_hadamard_value_mse \
  --value-quantizer rate_hadamard_value_mse
```

Build the current comparison table:

```bash
python scripts/build_method_table1.py \
  --method-name rate_hadamard_value_mse \
  --display-name "Rate-Hadamard Value MSE" \
  --description "Rate-aware value Hadamard preconditioning: keep K on TurboQuant MSE; at low rate use conservative per-vector value outlier-Hadamard, and at higher rate apply value outlier-Hadamard only in later decoder layers." \
  --output-prefix reproduce/incremental/rate_hadamard_value_mse_multiqa_progress
```

## Experimental Results

Scope:

```text
Model: meta-llama/Llama-3.1-8B-Instruct
Benchmark: LongBench V1 English Table 1 tasks
Reproduction settings: Full Cache, TurboQuant 2.5 bit, TurboQuant 3.5 bit
Incremental settings: Unified Regular-Gain Gate, Rate-Regime MSE, Rate-Hadamard Value MSE
```

### Reproduction Results

This table is the reproduction target: Full Cache and TurboQuant on the same LongBench Table 1 category grouping used by the paper.

| Method | KV Size | Source | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Cache | 16.0 | paper | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| Full Cache | 16.0 | local | 44.45 | 44.18 | 29.29 | 69.53 | 52.60 | 62.26 | 50.38 |
| TurboQuant | 2.5 | paper | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |
| TurboQuant | 2.5 | local | 38.61 | 36.01 | 27.54 | 67.12 | 48.22 | 55.02 | 45.42 |
| TurboQuant | 3.5 | paper | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |
| TurboQuant | 3.5 | local | 42.73 | 43.04 | 28.72 | 68.59 | 52.06 | 61.16 | 49.38 |

### Incremental Contribution Result

This table is separate from the reproduction target. It reports the additional method results over the reproduced TurboQuant baselines and serves as evidence for the incremental contribution.

Unified Regular-Gain Gate is evaluated on the same 16 LongBench Table 1 tasks and the same reproduced TurboQuant baselines.

| Method | KV Size | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant | 2.5 | 38.61 | 36.01 | 27.54 | 67.12 | 48.22 | 55.02 | 45.42 |
| Unified Regular-Gain Gate | 2.5 | 38.78 | 36.51 | 27.80 | 67.31 | 50.28 | 55.02 | 45.95 |
| TurboQuant | 3.5 | 42.73 | 43.04 | 28.72 | 68.59 | 52.06 | 61.16 | 49.38 |
| Unified Regular-Gain Gate | 3.5 | 43.63 | 43.55 | 28.72 | 68.61 | 51.53 | 61.16 | 49.53 |

Detailed task-level results are in `reproduce/incremental/UNIFIED_REGULAR_GAIN_GATE_FINAL.md`.

### Current Rate-Hadamard Value MSE Evidence

This table is an additional incremental result over the reproduced TurboQuant baseline. It is complete for the MultiQA group and uses the same quantizer rule for 2.5-bit and 3.5-bit runs.

| Method | KV bits | HotpotQA | 2WikiMQA | MuSiQue | MultiQA Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant | 2.5 | 44.83 | 37.96 | 25.24 | 36.01 |
| Rate-Hadamard Value MSE | 2.5 | 49.05 | 38.88 | 23.54 | 37.16 |
| TurboQuant | 3.5 | 54.65 | 44.47 | 30.01 | 43.04 |
| Rate-Hadamard Value MSE | 3.5 | 56.00 | 46.09 | 29.11 | 43.73 |

Detailed method notes and artifacts are in `reproduce/incremental/RATE_HADAMARD_VALUE_MSE_RESULTS.md`.

### Additional Rate-Regime MSE Evidence

Rate-Regime MSE is evaluated against the same reproduced TurboQuant baselines. The 3.5-bit Table 1 run is complete. The 2.5-bit run is still in progress, so the table below reports only complete categories/tasks for 2.5-bit and does not assign a full Table 1 average.

| KV bits | Complete Tasks | TurboQuant Avg | Rate-Regime MSE Avg | Delta |
| ---: | ---: | ---: | ---: | ---: |
| 2.5 | 10 / 16 | in progress | in progress | in progress |
| 3.5 | 16 / 16 | 49.38 | 49.86 | +0.48 |

#### Rate-Regime 3.5-Bit Category Scores

| Category | TurboQuant 3.5 bit | Rate-Regime MSE 3.5 bit | Delta |
| --- | ---: | ---: | ---: |
| SingleQA | 42.73 | 43.41 | +0.68 |
| MultiQA | 43.04 | 43.16 | +0.12 |
| Summarization | 28.72 | 28.67 | -0.05 |
| Few shot | 68.59 | 68.92 | +0.34 |
| Synthetic | 52.06 | 52.97 | +0.91 |
| Code | 61.16 | 62.05 | +0.89 |
| Average | 49.38 | 49.86 | +0.48 |

#### Rate-Regime Task-Level Scores

| KV bits | Category | Dataset | Records | TurboQuant | Rate-Regime MSE | Delta | Complete |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2.5 | SingleQA | narrativeqa | 200/200 | 25.35 | 24.23 | -1.12 | yes |
| 2.5 | SingleQA | qasper | 200/200 | 42.44 | 44.30 | +1.86 | yes |
| 2.5 | SingleQA | multifieldqa_en | 150/150 | 48.03 | 47.03 | -1.00 | yes |
| 2.5 | MultiQA | hotpotqa | 200/200 | 44.83 | 46.49 | +1.67 | yes |
| 2.5 | MultiQA | 2wikimqa | 200/200 | 37.96 | 38.49 | +0.53 | yes |
| 2.5 | MultiQA | musique | 200/200 | 25.24 | 24.68 | -0.56 | yes |
| 2.5 | Summarization | qmsum | 200/200 | 24.38 | 24.34 | -0.04 | yes |
| 2.5 | Few shot | trec | 200/200 | 70.50 | 72.00 | +1.50 | yes |
| 2.5 | Few shot | triviaqa | 200/200 | 89.39 | 89.49 | +0.09 | yes |
| 2.5 | Synthetic | passage_count | 200/200 | 3.45 | 2.64 | -0.81 | yes |
| 3.5 | SingleQA | narrativeqa | 200/200 | 29.25 | 28.22 | -1.03 | yes |
| 3.5 | SingleQA | qasper | 200/200 | 46.32 | 46.04 | -0.29 | yes |
| 3.5 | SingleQA | multifieldqa_en | 150/150 | 52.61 | 55.96 | +3.35 | yes |
| 3.5 | MultiQA | hotpotqa | 200/200 | 54.65 | 55.10 | +0.46 | yes |
| 3.5 | MultiQA | 2wikimqa | 200/200 | 44.47 | 45.24 | +0.77 | yes |
| 3.5 | MultiQA | musique | 200/200 | 30.01 | 29.13 | -0.88 | yes |
| 3.5 | Summarization | gov_report | 200/200 | 34.08 | 34.46 | +0.39 | yes |
| 3.5 | Summarization | qmsum | 200/200 | 25.33 | 24.83 | -0.50 | yes |
| 3.5 | Summarization | multi_news | 200/200 | 26.77 | 26.73 | -0.04 | yes |
| 3.5 | Few shot | trec | 200/200 | 72.00 | 71.50 | -0.50 | yes |
| 3.5 | Few shot | triviaqa | 200/200 | 91.34 | 91.09 | -0.25 | yes |
| 3.5 | Few shot | samsum | 200/200 | 42.43 | 44.18 | +1.75 | yes |
| 3.5 | Synthetic | passage_retrieval_en | 200/200 | 100.00 | 99.50 | -0.50 | yes |
| 3.5 | Synthetic | passage_count | 200/200 | 4.12 | 6.44 | +2.31 | yes |
| 3.5 | Code | lcc | 500/500 | 64.04 | 65.25 | +1.21 | yes |
| 3.5 | Code | repobench-p | 500/500 | 58.29 | 58.85 | +0.57 | yes |

Full Rate-Regime progress and generated CSV/JSON are in `reproduce/incremental/rate_regime_mse_table1.md`.

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

### Incremental Task Level Scores

| Category | Dataset | TurboQuant 2.5 bit | Unified Gate 2.5 bit | TurboQuant 3.5 bit | Unified Gate 3.5 bit |
| --- | --- | ---: | ---: | ---: | ---: |
| SingleQA | narrativeqa | 25.35 | 25.46 | 29.25 | 29.29 |
| SingleQA | qasper | 42.44 | 42.46 | 46.32 | 47.60 |
| SingleQA | multifieldqa_en | 48.03 | 48.41 | 52.61 | 54.01 |
| MultiQA | hotpotqa | 44.83 | 47.78 | 54.65 | 54.60 |
| MultiQA | 2wikimqa | 37.96 | 36.87 | 44.47 | 45.11 |
| MultiQA | musique | 25.24 | 24.87 | 30.01 | 30.93 |
| Summarization | gov_report | 32.02 | 32.48 | 34.08 | 34.10 |
| Summarization | qmsum | 24.38 | 24.41 | 25.33 | 25.29 |
| Summarization | multi_news | 26.22 | 26.50 | 26.77 | 26.77 |
| Few shot | trec | 70.50 | 70.50 | 72.00 | 72.00 |
| Few shot | triviaqa | 89.39 | 89.97 | 91.34 | 91.40 |
| Few shot | samsum | 41.47 | 41.47 | 42.43 | 42.43 |
| Synthetic | passage_retrieval_en | 93.00 | 96.00 | 100.00 | 99.50 |
| Synthetic | passage_count | 3.45 | 4.56 | 4.12 | 3.56 |
| Code | lcc | 57.04 | 57.04 | 64.04 | 64.04 |
| Code | repobench-p | 53.00 | 53.00 | 58.29 | 58.29 |

### Run Completeness

For Rate-Hadamard Value MSE, the rows below report the completed MultiQA evidence currently included in this checkpoint.

| Method | Complete Tasks | Expected Tasks |
| --- | ---: | ---: |
| Full Cache | 16 | 16 |
| TurboQuant 2.5 bit | 16 | 16 |
| TurboQuant 3.5 bit | 16 | 16 |
| Unified Regular-Gain Gate 2.5 bit | 16 | 16 |
| Unified Regular-Gain Gate 3.5 bit | 16 | 16 |
| Rate-Regime MSE 2.5 bit | 10 | 16 |
| Rate-Regime MSE 3.5 bit | 16 | 16 |
| Rate-Hadamard Value MSE 2.5 bit, MultiQA | 3 | 3 |
| Rate-Hadamard Value MSE 3.5 bit, MultiQA | 3 | 3 |

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
