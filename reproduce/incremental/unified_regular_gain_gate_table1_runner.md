# Unified Regular-Gain Gate Runner Results

Runner-validated results for one prompt-only gate at both 2.5-bit and 3.5-bit. The gate uses TurboQuant MSE by default and switches K/V to `regular_gain_mse` when the prompt is not code-like, has at most 12 Passage-style passages, and has at most 20 question marks.

## KV 2.5

Complete tasks: 16 / 16

| Category | TurboQuant | Unified Gate | Delta | Cache ratio | Activation | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SingleQA | 38.61 | 38.78 | 0.17 | 0.17 | 0.65 | yes |
| MultiQA | 36.01 | 36.51 | 0.50 | 0.17 | 0.86 | yes |
| Summarization | 27.54 | 27.80 | 0.26 | 0.18 | 0.65 | yes |
| Few shot | 67.12 | 67.31 | 0.19 | 0.17 | 0.18 | yes |
| Synthetic | 48.22 | 50.28 | 2.05 | 0.17 | 0.99 | yes |
| Code | 55.02 | 55.02 | 0.00 | 0.17 | 0.00 | yes |
| Average | 45.42 | 45.95 | 0.53 | 0.17 | 0.56 | yes |

| Category | Task | Records | TurboQuant | Unified Gate | Delta | Activation | Complete |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 25.35 | 25.46 | 0.11 | 0.07 | yes |
| SingleQA | `qasper` | 200/200 | 42.44 | 42.46 | 0.02 | 0.99 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 48.03 | 48.41 | 0.38 | 0.91 | yes |
| MultiQA | `hotpotqa` | 200/200 | 44.83 | 47.78 | 2.95 | 0.98 | yes |
| MultiQA | `2wikimqa` | 200/200 | 37.96 | 36.87 | -1.09 | 0.99 | yes |
| MultiQA | `musique` | 200/200 | 25.24 | 24.87 | -0.36 | 0.61 | yes |
| Summarization | `gov_report` | 200/200 | 32.02 | 32.48 | 0.47 | 0.98 | yes |
| Summarization | `qmsum` | 200/200 | 24.38 | 24.41 | 0.03 | 0.01 | yes |
| Summarization | `multi_news` | 200/200 | 26.22 | 26.50 | 0.28 | 0.97 | yes |
| Few shot | `trec` | 200/200 | 70.50 | 70.50 | 0.00 | 0.00 | yes |
| Few shot | `triviaqa` | 200/200 | 89.39 | 89.97 | 0.57 | 0.53 | yes |
| Few shot | `samsum` | 200/200 | 41.47 | 41.47 | 0.00 | 0.00 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | 93.00 | 96.00 | 3.00 | 1.00 | yes |
| Synthetic | `passage_count` | 200/200 | 3.45 | 4.56 | 1.11 | 0.98 | yes |
| Code | `lcc` | 500/500 | 57.04 | 57.04 | 0.00 | 0.00 | yes |
| Code | `repobench-p` | 500/500 | 53.00 | 53.00 | 0.00 | 0.00 | yes |

## KV 3.5

Complete tasks: 16 / 16

| Category | TurboQuant | Unified Gate | Delta | Cache ratio | Activation | Complete |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SingleQA | 42.73 | 43.63 | 0.90 | 0.23 | 0.65 | yes |
| MultiQA | 43.04 | 43.55 | 0.51 | 0.23 | 0.86 | yes |
| Summarization | 28.72 | 28.72 | -0.00 | 0.24 | 0.65 | yes |
| Few shot | 68.59 | 68.61 | 0.02 | 0.24 | 0.18 | yes |
| Synthetic | 52.06 | 51.53 | -0.53 | 0.23 | 0.99 | yes |
| Code | 61.16 | 61.16 | 0.00 | 0.24 | 0.00 | yes |
| Average | 49.38 | 49.53 | 0.15 | 0.24 | 0.56 | yes |

| Category | Task | Records | TurboQuant | Unified Gate | Delta | Activation | Complete |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 29.25 | 29.29 | 0.04 | 0.07 | yes |
| SingleQA | `qasper` | 200/200 | 46.32 | 47.60 | 1.27 | 0.99 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 52.61 | 54.01 | 1.40 | 0.91 | yes |
| MultiQA | `hotpotqa` | 200/200 | 54.65 | 54.60 | -0.05 | 0.98 | yes |
| MultiQA | `2wikimqa` | 200/200 | 44.47 | 45.11 | 0.65 | 0.99 | yes |
| MultiQA | `musique` | 200/200 | 30.01 | 30.93 | 0.92 | 0.61 | yes |
| Summarization | `gov_report` | 200/200 | 34.08 | 34.10 | 0.03 | 0.98 | yes |
| Summarization | `qmsum` | 200/200 | 25.33 | 25.29 | -0.04 | 0.01 | yes |
| Summarization | `multi_news` | 200/200 | 26.77 | 26.77 | -0.00 | 0.97 | yes |
| Few shot | `trec` | 200/200 | 72.00 | 72.00 | 0.00 | 0.00 | yes |
| Few shot | `triviaqa` | 200/200 | 91.34 | 91.40 | 0.07 | 0.53 | yes |
| Few shot | `samsum` | 200/200 | 42.43 | 42.43 | 0.00 | 0.00 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | 100.00 | 99.50 | -0.50 | 1.00 | yes |
| Synthetic | `passage_count` | 200/200 | 4.12 | 3.56 | -0.56 | 0.98 | yes |
| Code | `lcc` | 500/500 | 64.04 | 64.04 | 0.00 | 0.00 | yes |
| Code | `repobench-p` | 500/500 | 58.29 | 58.29 | 0.00 | 0.00 | yes |
