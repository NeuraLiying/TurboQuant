# Unified Regular-Gain Table 1 Comparison

Candidate: `regular_gain_mse` for both K and V, with the same rule at 2.5-bit and 3.5-bit.

## KV 2.5

Complete tasks: 16 / 16

| Category | TurboQuant | Unified Regular-Gain | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | 38.61 | 39.76 | 1.16 | 0.17 | yes |
| MultiQA | 36.01 | 36.50 | 0.50 | 0.17 | yes |
| Summarization | 27.54 | 27.71 | 0.17 | 0.18 | yes |
| Few shot | 67.12 | 67.48 | 0.36 | 0.17 | yes |
| Synthetic | 48.22 | 50.28 | 2.05 | 0.17 | yes |
| Code | 55.02 | 53.53 | -1.49 | 0.17 | yes |
| Average | 45.42 | 45.88 | 0.46 | 0.17 | yes |

| Category | Task | Records | Source | TurboQuant | Unified Regular-Gain | Delta | Complete |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | unified_regular_gain | 25.35 | 27.34 | 2.00 | yes |
| SingleQA | `qasper` | 200/200 | unified_regular_gain | 42.44 | 43.00 | 0.57 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | unified_regular_gain | 48.03 | 48.95 | 0.91 | yes |
| MultiQA | `hotpotqa` | 200/200 | unified_regular_gain | 44.83 | 48.08 | 3.25 | yes |
| MultiQA | `2wikimqa` | 200/200 | unified_regular_gain | 37.96 | 37.21 | -0.75 | yes |
| MultiQA | `musique` | 200/200 | unified_regular_gain | 25.24 | 24.22 | -1.01 | yes |
| Summarization | `gov_report` | 200/200 | lowbit_gain_mse_equivalent | 32.02 | 32.46 | 0.44 | yes |
| Summarization | `qmsum` | 200/200 | unified_regular_gain | 24.38 | 24.13 | -0.26 | yes |
| Summarization | `multi_news` | 200/200 | lowbit_gain_mse_equivalent | 26.22 | 26.53 | 0.32 | yes |
| Few shot | `trec` | 200/200 | unified_regular_gain | 70.50 | 70.00 | -0.50 | yes |
| Few shot | `triviaqa` | 200/200 | unified_regular_gain | 89.39 | 90.06 | 0.66 | yes |
| Few shot | `samsum` | 200/200 | unified_regular_gain | 41.47 | 42.38 | 0.91 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | unified_regular_gain | 93.00 | 96.00 | 3.00 | yes |
| Synthetic | `passage_count` | 200/200 | unified_regular_gain | 3.45 | 4.56 | 1.11 | yes |
| Code | `lcc` | 500/500 | lowbit_gain_mse_equivalent | 57.04 | 57.64 | 0.60 | yes |
| Code | `repobench-p` | 500/500 | lowbit_gain_mse_equivalent | 53.00 | 49.42 | -3.58 | yes |

## KV 3.5

Complete tasks: 10 / 16

| Category | TurboQuant | Unified Regular-Gain | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | 42.73 | 43.77 | 1.04 | 0.23 | yes |
| MultiQA | 43.04 | 43.17 | 0.13 | 0.23 | yes |
| Summarization |  |  |  |  | no |
| Few shot |  |  |  |  | no |
| Synthetic |  |  |  |  | no |
| Code |  |  |  |  | no |
| Average |  |  |  |  | no |

| Category | Task | Records | Source | TurboQuant | Unified Regular-Gain | Delta | Complete |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | unified_regular_gain | 29.25 | 29.26 | 0.01 | yes |
| SingleQA | `qasper` | 200/200 | unified_regular_gain | 46.32 | 47.60 | 1.27 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | unified_regular_gain | 52.61 | 54.45 | 1.83 | yes |
| MultiQA | `hotpotqa` | 200/200 | unified_regular_gain | 54.65 | 54.60 | -0.05 | yes |
| MultiQA | `2wikimqa` | 200/200 | unified_regular_gain | 44.47 | 45.11 | 0.65 | yes |
| MultiQA | `musique` | 200/200 | unified_regular_gain | 30.01 | 29.79 | -0.22 | yes |
| Summarization | `gov_report` | 185/200 | unified_regular_gain |  |  |  | no |
| Summarization | `qmsum` | 200/200 | unified_regular_gain | 25.33 | 25.01 | -0.31 | yes |
| Summarization | `multi_news` | 195/200 | unified_regular_gain |  |  |  | no |
| Few shot | `trec` | 200/200 | unified_regular_gain | 72.00 | 70.50 | -1.50 | yes |
| Few shot | `triviaqa` | 200/200 | unified_regular_gain | 91.34 | 91.00 | -0.33 | yes |
| Few shot | `samsum` | 175/200 | unified_regular_gain |  |  |  | no |
| Synthetic | `passage_retrieval_en` | 0/200 | unified_regular_gain |  |  |  | no |
| Synthetic | `passage_count` | 200/200 | unified_regular_gain | 4.12 | 3.06 | -1.06 | yes |
| Code | `lcc` | 0/500 | unified_regular_gain |  |  |  | no |
| Code | `repobench-p` | 486/500 | unified_regular_gain |  |  |  | no |
