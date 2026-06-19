# Unified Regular-Gain Gate Results

Status: candidate found by offline composition from completed TurboQuant baseline outputs and completed regular-gain outputs. This is not yet the final runner-validated result.

## Method

Use one prompt-only rule at both 2.5-bit and 3.5-bit:

- If the prompt is code-like (`lcc` or `repobench-p`), keep TurboQuant MSE.
- If `passage_count > 12`, keep TurboQuant MSE.
- If `question_marks > 20`, keep TurboQuant MSE.
- Otherwise use `regular_gain_mse` for both K and V.

The same rule is applied to both bit budgets; only the target KV bit budget changes.

## KV 2.5

| Category | TurboQuant | Unified Gate | Delta | Activation |
|---|---:|---:|---:|---:|
| SingleQA | 38.61 | 38.78 | 0.17 | 0.65 |
| MultiQA | 36.01 | 36.51 | 0.50 | 0.86 |
| Summarization | 27.54 | 27.80 | 0.26 | 0.65 |
| Few shot | 67.12 | 67.31 | 0.19 | 0.18 |
| Synthetic | 48.22 | 50.28 | 2.05 | 0.99 |
| Code | 55.02 | 55.02 | 0.00 | 0.00 |
| Average | 45.42 | 45.95 | 0.53 | 0.56 |

| Category | Task | Records | TurboQuant | Unified Gate | Delta | Activation |
|---|---|---:|---:|---:|---:|---:|
| SingleQA | `narrativeqa` | 200/200 | 25.35 | 25.46 | 0.11 | 0.07 |
| SingleQA | `qasper` | 200/200 | 42.44 | 42.46 | 0.02 | 0.99 |
| SingleQA | `multifieldqa_en` | 150/150 | 48.03 | 48.41 | 0.38 | 0.91 |
| MultiQA | `hotpotqa` | 200/200 | 44.83 | 47.78 | 2.95 | 0.98 |
| MultiQA | `2wikimqa` | 200/200 | 37.96 | 36.87 | -1.09 | 0.99 |
| MultiQA | `musique` | 200/200 | 25.24 | 24.87 | -0.36 | 0.61 |
| Summarization | `gov_report` | 200/200 | 32.02 | 32.48 | 0.47 | 0.98 |
| Summarization | `qmsum` | 200/200 | 24.38 | 24.41 | 0.03 | 0.01 |
| Summarization | `multi_news` | 200/200 | 26.22 | 26.50 | 0.28 | 0.97 |
| Few shot | `trec` | 200/200 | 70.50 | 70.50 | 0.00 | 0.00 |
| Few shot | `triviaqa` | 200/200 | 89.39 | 89.97 | 0.57 | 0.53 |
| Few shot | `samsum` | 200/200 | 41.47 | 41.47 | 0.00 | 0.00 |
| Synthetic | `passage_retrieval_en` | 200/200 | 93.00 | 96.00 | 3.00 | 1.00 |
| Synthetic | `passage_count` | 200/200 | 3.45 | 4.56 | 1.11 | 0.98 |
| Code | `lcc` | 500/500 | 57.04 | 57.04 | 0.00 | 0.00 |
| Code | `repobench-p` | 500/500 | 53.00 | 53.00 | 0.00 | 0.00 |

## KV 3.5

| Category | TurboQuant | Unified Gate | Delta | Activation |
|---|---:|---:|---:|---:|
| SingleQA | 42.73 | 43.63 | 0.90 | 0.65 |
| MultiQA | 43.04 | 43.55 | 0.51 | 0.86 |
| Summarization | 28.72 | 28.72 | -0.00 | 0.65 |
| Few shot | 68.59 | 68.61 | 0.02 | 0.18 |
| Synthetic | 52.06 | 51.53 | -0.53 | 0.99 |
| Code | 61.16 | 61.16 | 0.00 | 0.00 |
| Average | 49.38 | 49.53 | 0.15 | 0.56 |

| Category | Task | Records | TurboQuant | Unified Gate | Delta | Activation |
|---|---|---:|---:|---:|---:|---:|
| SingleQA | `narrativeqa` | 200/200 | 29.25 | 29.29 | 0.04 | 0.07 |
| SingleQA | `qasper` | 200/200 | 46.32 | 47.60 | 1.27 | 0.99 |
| SingleQA | `multifieldqa_en` | 150/150 | 52.61 | 54.01 | 1.40 | 0.91 |
| MultiQA | `hotpotqa` | 200/200 | 54.65 | 54.60 | -0.05 | 0.98 |
| MultiQA | `2wikimqa` | 200/200 | 44.47 | 45.11 | 0.65 | 0.99 |
| MultiQA | `musique` | 200/200 | 30.01 | 30.93 | 0.92 | 0.61 |
| Summarization | `gov_report` | 200/200 | 34.08 | 34.10 | 0.03 | 0.98 |
| Summarization | `qmsum` | 200/200 | 25.33 | 25.29 | -0.04 | 0.01 |
| Summarization | `multi_news` | 200/200 | 26.77 | 26.77 | -0.00 | 0.97 |
| Few shot | `trec` | 200/200 | 72.00 | 72.00 | 0.00 | 0.00 |
| Few shot | `triviaqa` | 200/200 | 91.34 | 91.40 | 0.07 | 0.53 |
| Few shot | `samsum` | 200/200 | 42.43 | 42.43 | 0.00 | 0.00 |
| Synthetic | `passage_retrieval_en` | 200/200 | 100.00 | 99.50 | -0.50 | 1.00 |
| Synthetic | `passage_count` | 200/200 | 4.12 | 3.56 | -0.56 | 0.98 |
| Code | `lcc` | 500/500 | 64.04 | 64.04 | 0.00 | 0.00 |
| Code | `repobench-p` | 500/500 | 58.29 | 58.29 | 0.00 | 0.00 |

## Analysis

This candidate satisfies the method-shape requirement: the rule and code path are unified across 2.5-bit and 3.5-bit. The offline Table-1 macro average improves both budgets, but the 3.5-bit gain is small and the Synthetic category regresses. The next required step is runner validation with the same gate arguments, not just offline composition.
