# Unified Regular-Gain Table 1 Comparison

Candidate: regular_gain_mse for both K and V, with the same rule at 2.5-bit and 3.5-bit.

## KV 2.5

Complete tasks: 12 / 16

| Category | TurboQuant | Unified Regular-Gain | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | 38.61 | 39.76 | 1.16 | 0.17 | yes |
| MultiQA | 36.01 | 36.50 | 0.50 | 0.17 | yes |
| Summarization |  |  |  |  | no |
| Few shot | 67.12 | 67.48 | 0.36 | 0.17 | yes |
| Synthetic | 48.22 | 50.28 | 2.05 | 0.17 | yes |
| Code |  |  |  |  | no |
| Average |  |  |  |  | no |

| Category | Task | Records | TurboQuant | Unified Regular-Gain | Delta | Complete |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 25.35 | 27.34 | 2.00 | yes |
| SingleQA | `qasper` | 200/200 | 42.44 | 43.00 | 0.57 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 48.03 | 48.95 | 0.91 | yes |
| MultiQA | `hotpotqa` | 200/200 | 44.83 | 48.08 | 3.25 | yes |
| MultiQA | `2wikimqa` | 200/200 | 37.96 | 37.21 | -0.75 | yes |
| MultiQA | `musique` | 200/200 | 25.24 | 24.22 | -1.01 | yes |
| Summarization | `gov_report` | 91/200 |  |  |  | no |
| Summarization | `qmsum` | 200/200 | 24.38 | 24.13 | -0.26 | yes |
| Summarization | `multi_news` | 146/200 |  |  |  | no |
| Few shot | `trec` | 200/200 | 70.50 | 70.00 | -0.50 | yes |
| Few shot | `triviaqa` | 200/200 | 89.39 | 90.06 | 0.66 | yes |
| Few shot | `samsum` | 200/200 | 41.47 | 42.38 | 0.91 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | 93.00 | 96.00 | 3.00 | yes |
| Synthetic | `passage_count` | 200/200 | 3.45 | 4.56 | 1.11 | yes |
| Code | `lcc` | 464/500 |  |  |  | no |
| Code | `repobench-p` | 24/500 |  |  |  | no |

## KV 3.5

Complete tasks: 15 / 16

| Category | TurboQuant | Unified Regular-Gain | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | 42.73 | 43.77 | 1.04 | 0.23 | yes |
| MultiQA | 43.04 | 43.17 | 0.13 | 0.23 | yes |
| Summarization | 28.72 | 28.61 | -0.11 | 0.24 | yes |
| Few shot | 68.59 | 68.37 | -0.22 | 0.24 | yes |
| Synthetic | 52.06 | 51.28 | -0.78 | 0.23 | yes |
| Code |  |  |  |  | no |
| Average |  |  |  |  | no |

| Category | Task | Records | TurboQuant | Unified Regular-Gain | Delta | Complete |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 29.25 | 29.26 | 0.01 | yes |
| SingleQA | `qasper` | 200/200 | 46.32 | 47.60 | 1.27 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 52.61 | 54.45 | 1.83 | yes |
| MultiQA | `hotpotqa` | 200/200 | 54.65 | 54.60 | -0.05 | yes |
| MultiQA | `2wikimqa` | 200/200 | 44.47 | 45.11 | 0.65 | yes |
| MultiQA | `musique` | 200/200 | 30.01 | 29.79 | -0.22 | yes |
| Summarization | `gov_report` | 200/200 | 34.08 | 34.07 | -0.01 | yes |
| Summarization | `qmsum` | 200/200 | 25.33 | 25.01 | -0.31 | yes |
| Summarization | `multi_news` | 200/200 | 26.77 | 26.76 | -0.01 | yes |
| Few shot | `trec` | 200/200 | 72.00 | 70.50 | -1.50 | yes |
| Few shot | `triviaqa` | 200/200 | 91.34 | 91.00 | -0.33 | yes |
| Few shot | `samsum` | 200/200 | 42.43 | 43.61 | 1.19 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | 100.00 | 99.50 | -0.50 | yes |
| Synthetic | `passage_count` | 200/200 | 4.12 | 3.06 | -1.06 | yes |
| Code | `lcc` | 193/500 |  |  |  | no |
| Code | `repobench-p` | 500/500 | 58.29 | 56.84 | -1.44 | yes |
