# Rate-Regime MSE Table 1 Comparison

Rate-regime rotated-domain reconstruction: use norm-gain scalar TurboQuant at low target rate and learned unit block TurboQuant at higher target rate.

## KV 2.5

Complete tasks: 10 / 16

| Category | TurboQuant | Rate-Regime MSE | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | 38.61 | 38.52 | -0.09 | 0.17 | yes |
| MultiQA | 36.01 | 36.55 | 0.54 | 0.17 | yes |
| Summarization |  |  |  |  | no |
| Few shot |  |  |  |  | no |
| Synthetic |  |  |  |  | no |
| Code |  |  |  |  | no |
| Average |  |  |  |  | no |

| Category | Task | Records | TurboQuant | Rate-Regime MSE | Delta | Complete |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 25.35 | 24.23 | -1.12 | yes |
| SingleQA | `qasper` | 200/200 | 42.44 | 44.30 | 1.86 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 48.03 | 47.03 | -1.00 | yes |
| MultiQA | `hotpotqa` | 200/200 | 44.83 | 46.49 | 1.67 | yes |
| MultiQA | `2wikimqa` | 200/200 | 37.96 | 38.49 | 0.53 | yes |
| MultiQA | `musique` | 200/200 | 25.24 | 24.68 | -0.56 | yes |
| Summarization | `gov_report` | 93/200 |  |  |  | no |
| Summarization | `qmsum` | 200/200 | 24.38 | 24.34 | -0.04 | yes |
| Summarization | `multi_news` | 51/200 |  |  |  | no |
| Few shot | `trec` | 200/200 | 70.50 | 72.00 | 1.50 | yes |
| Few shot | `triviaqa` | 200/200 | 89.39 | 89.49 | 0.09 | yes |
| Few shot | `samsum` | 117/200 |  |  |  | no |
| Synthetic | `passage_retrieval_en` | 54/200 |  |  |  | no |
| Synthetic | `passage_count` | 200/200 | 3.45 | 2.64 | -0.81 | yes |
| Code | `lcc` | 23/500 |  |  |  | no |
| Code | `repobench-p` | 276/500 |  |  |  | no |

## KV 3.5

Complete tasks: 16 / 16

| Category | TurboQuant | Rate-Regime MSE | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | 42.73 | 43.41 | 0.68 | 0.23 | yes |
| MultiQA | 43.04 | 43.16 | 0.12 | 0.23 | yes |
| Summarization | 28.72 | 28.67 | -0.05 | 0.24 | yes |
| Few shot | 68.59 | 68.92 | 0.34 | 0.24 | yes |
| Synthetic | 52.06 | 52.97 | 0.91 | 0.23 | yes |
| Code | 61.16 | 62.05 | 0.89 | 0.24 | yes |
| Average | 49.38 | 49.86 | 0.48 | 0.24 | yes |

| Category | Task | Records | TurboQuant | Rate-Regime MSE | Delta | Complete |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 29.25 | 28.22 | -1.03 | yes |
| SingleQA | `qasper` | 200/200 | 46.32 | 46.04 | -0.29 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 52.61 | 55.96 | 3.35 | yes |
| MultiQA | `hotpotqa` | 200/200 | 54.65 | 55.10 | 0.46 | yes |
| MultiQA | `2wikimqa` | 200/200 | 44.47 | 45.24 | 0.77 | yes |
| MultiQA | `musique` | 200/200 | 30.01 | 29.13 | -0.88 | yes |
| Summarization | `gov_report` | 200/200 | 34.08 | 34.46 | 0.39 | yes |
| Summarization | `qmsum` | 200/200 | 25.33 | 24.83 | -0.50 | yes |
| Summarization | `multi_news` | 200/200 | 26.77 | 26.73 | -0.04 | yes |
| Few shot | `trec` | 200/200 | 72.00 | 71.50 | -0.50 | yes |
| Few shot | `triviaqa` | 200/200 | 91.34 | 91.09 | -0.25 | yes |
| Few shot | `samsum` | 200/200 | 42.43 | 44.18 | 1.75 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | 100.00 | 99.50 | -0.50 | yes |
| Synthetic | `passage_count` | 200/200 | 4.12 | 6.44 | 2.31 | yes |
| Code | `lcc` | 500/500 | 64.04 | 65.25 | 1.21 | yes |
| Code | `repobench-p` | 500/500 | 58.29 | 58.85 | 0.57 | yes |
