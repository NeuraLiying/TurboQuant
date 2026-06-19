# Structure-Adaptive Regular-Gain Table 1 Comparison

Candidate: one prompt-only gate at both 2.5-bit and 3.5-bit; use `regular_gain_mse` for long, non-code prompts whose Passage-style structure is not blocked, otherwise use TurboQuant MSE.

## KV 2.5

Complete tasks: 16 / 16

| Category | TurboQuant | Regular-Gain | Structure-Adaptive | Delta | Activation | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SingleQA | 38.61 | 39.76 | 39.02 | 0.42 | 0.20 | yes |
| MultiQA | 36.01 | 36.50 | 38.10 | 2.09 | 0.65 | yes |
| Summarization | 27.54 | 27.71 | 27.66 | 0.12 | 0.24 | yes |
| Few shot | 67.12 | 67.48 | 67.12 | 0.00 | 0.00 | yes |
| Synthetic | 48.22 | 50.28 | 50.28 | 2.05 | 0.87 | yes |
| Code | 55.02 | 53.53 | 55.02 | 0.00 | 0.00 | yes |
| Average | 45.42 | 45.88 | 46.20 | 0.78 | 0.33 | yes |

| Category | Task | Records | Source | TurboQuant | Regular-Gain | Structure-Adaptive | Delta | Activation | Complete |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | regular_gain_mse | 25.35 | 27.34 | 25.35 | 0.00 | 0.00 | yes |
| SingleQA | `qasper` | 200/200 | regular_gain_mse | 42.44 | 43.00 | 43.42 | 0.98 | 0.17 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | regular_gain_mse | 48.03 | 48.95 | 48.31 | 0.28 | 0.45 | yes |
| MultiQA | `hotpotqa` | 200/200 | regular_gain_mse | 44.83 | 48.08 | 48.69 | 3.87 | 0.80 | yes |
| MultiQA | `2wikimqa` | 200/200 | regular_gain_mse | 37.96 | 37.21 | 39.04 | 1.08 | 0.43 | yes |
| MultiQA | `musique` | 200/200 | regular_gain_mse | 25.24 | 24.22 | 26.57 | 1.33 | 0.71 | yes |
| Summarization | `gov_report` | 200/200 | lowbit_gain_mse_equivalent | 32.02 | 32.46 | 32.40 | 0.38 | 0.71 | yes |
| Summarization | `qmsum` | 200/200 | regular_gain_mse | 24.38 | 24.13 | 24.38 | 0.00 | 0.00 | yes |
| Summarization | `multi_news` | 200/200 | lowbit_gain_mse_equivalent | 26.22 | 26.53 | 26.20 | -0.01 | 0.03 | yes |
| Few shot | `trec` | 200/200 | regular_gain_mse | 70.50 | 70.00 | 70.50 | 0.00 | 0.00 | yes |
| Few shot | `triviaqa` | 200/200 | regular_gain_mse | 89.39 | 90.06 | 89.39 | 0.00 | 0.00 | yes |
| Few shot | `samsum` | 200/200 | regular_gain_mse | 41.47 | 42.38 | 41.47 | 0.00 | 0.00 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | regular_gain_mse | 93.00 | 96.00 | 96.00 | 3.00 | 0.95 | yes |
| Synthetic | `passage_count` | 200/200 | regular_gain_mse | 3.45 | 4.56 | 4.56 | 1.11 | 0.80 | yes |
| Code | `lcc` | 500/500 | lowbit_gain_mse_equivalent | 57.04 | 57.64 | 57.04 | 0.00 | 0.00 | yes |
| Code | `repobench-p` | 500/500 | lowbit_gain_mse_equivalent | 53.00 | 49.42 | 53.00 | 0.00 | 0.00 | yes |

## KV 3.5

Complete tasks: 14 / 16

| Category | TurboQuant | Regular-Gain | Structure-Adaptive | Delta | Activation | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SingleQA | 42.73 | 43.77 | 42.99 | 0.26 | 0.20 | yes |
| MultiQA | 43.04 | 43.17 | 42.80 | -0.24 | 0.65 | yes |
| Summarization | 28.72 | 28.61 | 28.81 | 0.09 | 0.24 | yes |
| Few shot | 68.59 | 68.37 | 68.59 | 0.00 | 0.00 | yes |
| Synthetic |  |  |  |  |  | no |
| Code |  |  |  |  |  | no |
| Average |  |  |  |  |  | no |

| Category | Task | Records | Source | TurboQuant | Regular-Gain | Structure-Adaptive | Delta | Activation | Complete |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | regular_gain_mse | 29.25 | 29.26 | 29.25 | 0.00 | 0.00 | yes |
| SingleQA | `qasper` | 200/200 | regular_gain_mse | 46.32 | 47.60 | 46.77 | 0.45 | 0.17 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | regular_gain_mse | 52.61 | 54.45 | 52.95 | 0.34 | 0.45 | yes |
| MultiQA | `hotpotqa` | 200/200 | regular_gain_mse | 54.65 | 54.60 | 54.80 | 0.15 | 0.80 | yes |
| MultiQA | `2wikimqa` | 200/200 | regular_gain_mse | 44.47 | 45.11 | 43.90 | -0.57 | 0.43 | yes |
| MultiQA | `musique` | 200/200 | regular_gain_mse | 30.01 | 29.79 | 29.70 | -0.31 | 0.71 | yes |
| Summarization | `gov_report` | 200/200 | regular_gain_mse | 34.08 | 34.07 | 34.27 | 0.19 | 0.71 | yes |
| Summarization | `qmsum` | 200/200 | regular_gain_mse | 25.33 | 25.01 | 25.33 | 0.00 | 0.00 | yes |
| Summarization | `multi_news` | 200/200 | regular_gain_mse | 26.77 | 26.76 | 26.84 | 0.07 | 0.03 | yes |
| Few shot | `trec` | 200/200 | regular_gain_mse | 72.00 | 70.50 | 72.00 | 0.00 | 0.00 | yes |
| Few shot | `triviaqa` | 200/200 | regular_gain_mse | 91.34 | 91.00 | 91.34 | 0.00 | 0.00 | yes |
| Few shot | `samsum` | 200/200 | regular_gain_mse | 42.43 | 43.61 | 42.43 | 0.00 | 0.00 | yes |
| Synthetic | `passage_retrieval_en` | 82/200 | regular_gain_mse |  |  |  |  | 0.94 | no |
| Synthetic | `passage_count` | 200/200 | regular_gain_mse | 4.12 | 3.06 | 3.59 | -0.53 | 0.80 | yes |
| Code | `lcc` | 53/500 | regular_gain_mse |  |  |  |  | 0.00 | no |
| Code | `repobench-p` | 500/500 | regular_gain_mse | 58.29 | 56.84 | 58.29 | 0.00 | 0.00 | yes |
