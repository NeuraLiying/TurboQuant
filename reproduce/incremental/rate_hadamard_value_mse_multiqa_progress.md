# Rate-Hadamard Value MSE Table 1 Comparison

Rate-aware value Hadamard preconditioning: keep K on TurboQuant MSE; at low rate use conservative per-vector value outlier-Hadamard, and at higher rate apply value outlier-Hadamard only in later decoder layers.

## KV 2.5

Complete tasks: 3 / 16

| Category | TurboQuant | Rate-Hadamard Value MSE | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA |  |  |  |  | no |
| MultiQA | 36.01 | 37.16 | 1.15 | 0.17 | yes |
| Summarization |  |  |  |  | no |
| Few shot |  |  |  |  | no |
| Synthetic |  |  |  |  | no |
| Code |  |  |  |  | no |
| Average |  |  |  |  | no |

| Category | Task | Records | TurboQuant | Rate-Hadamard Value MSE | Delta | Complete |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 0/200 |  |  |  | no |
| SingleQA | `qasper` | 0/200 |  |  |  | no |
| SingleQA | `multifieldqa_en` | 0/150 |  |  |  | no |
| MultiQA | `hotpotqa` | 200/200 | 44.83 | 49.05 | 4.23 | yes |
| MultiQA | `2wikimqa` | 200/200 | 37.96 | 38.88 | 0.92 | yes |
| MultiQA | `musique` | 200/200 | 25.24 | 23.54 | -1.70 | yes |
| Summarization | `gov_report` | 0/200 |  |  |  | no |
| Summarization | `qmsum` | 0/200 |  |  |  | no |
| Summarization | `multi_news` | 0/200 |  |  |  | no |
| Few shot | `trec` | 0/200 |  |  |  | no |
| Few shot | `triviaqa` | 0/200 |  |  |  | no |
| Few shot | `samsum` | 0/200 |  |  |  | no |
| Synthetic | `passage_retrieval_en` | 0/200 |  |  |  | no |
| Synthetic | `passage_count` | 0/200 |  |  |  | no |
| Code | `lcc` | 0/500 |  |  |  | no |
| Code | `repobench-p` | 0/500 |  |  |  | no |

## KV 3.5

Complete tasks: 3 / 16

| Category | TurboQuant | Rate-Hadamard Value MSE | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA |  |  |  |  | no |
| MultiQA | 43.04 | 43.73 | 0.69 | 0.23 | yes |
| Summarization |  |  |  |  | no |
| Few shot |  |  |  |  | no |
| Synthetic |  |  |  |  | no |
| Code |  |  |  |  | no |
| Average |  |  |  |  | no |

| Category | Task | Records | TurboQuant | Rate-Hadamard Value MSE | Delta | Complete |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 0/200 |  |  |  | no |
| SingleQA | `qasper` | 0/200 |  |  |  | no |
| SingleQA | `multifieldqa_en` | 0/150 |  |  |  | no |
| MultiQA | `hotpotqa` | 200/200 | 54.65 | 56.00 | 1.35 | yes |
| MultiQA | `2wikimqa` | 200/200 | 44.47 | 46.09 | 1.62 | yes |
| MultiQA | `musique` | 200/200 | 30.01 | 29.11 | -0.90 | yes |
| Summarization | `gov_report` | 0/200 |  |  |  | no |
| Summarization | `qmsum` | 0/200 |  |  |  | no |
| Summarization | `multi_news` | 0/200 |  |  |  | no |
| Few shot | `trec` | 0/200 |  |  |  | no |
| Few shot | `triviaqa` | 0/200 |  |  |  | no |
| Few shot | `samsum` | 0/200 |  |  |  | no |
| Synthetic | `passage_retrieval_en` | 0/200 |  |  |  | no |
| Synthetic | `passage_count` | 0/200 |  |  |  | no |
| Code | `lcc` | 0/500 |  |  |  | no |
| Code | `repobench-p` | 0/500 |  |  |  | no |
