# Attention-Adaptive Value Hadamard Table 1 Comparison

Value-side attention-adaptive outlier-Hadamard preconditioning: keep K on reproduced TurboQuant; for V, compare TurboQuant and Hadamard-preconditioned paths by current attention-output error and use Hadamard only when the attention objective improves.

## KV 2.5

Complete tasks: 3 / 16

| Category | TurboQuant | Attention-Adaptive Value Hadamard | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA |  |  |  |  | no |
| MultiQA | 36.01 | 34.08 | -1.93 | 0.17 | yes |
| Summarization |  |  |  |  | no |
| Few shot |  |  |  |  | no |
| Synthetic |  |  |  |  | no |
| Code |  |  |  |  | no |
| Average |  |  |  |  | no |

| Category | Task | Records | TurboQuant | Attention-Adaptive Value Hadamard | Delta | Complete |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 0/200 |  |  |  | no |
| SingleQA | `qasper` | 0/200 |  |  |  | no |
| SingleQA | `multifieldqa_en` | 0/150 |  |  |  | no |
| MultiQA | `hotpotqa` | 200/200 | 44.83 | 47.02 | 2.20 | yes |
| MultiQA | `2wikimqa` | 200/200 | 37.96 | 34.87 | -3.09 | yes |
| MultiQA | `musique` | 200/200 | 25.24 | 20.35 | -4.88 | yes |
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

| Category | TurboQuant | Attention-Adaptive Value Hadamard | Delta | Cache ratio | Complete |
| --- | ---: | ---: | ---: | ---: | --- |
| SingleQA |  |  |  |  | no |
| MultiQA | 43.04 | 42.20 | -0.85 | 0.23 | yes |
| Summarization |  |  |  |  | no |
| Few shot |  |  |  |  | no |
| Synthetic |  |  |  |  | no |
| Code |  |  |  |  | no |
| Average |  |  |  |  | no |

| Category | Task | Records | TurboQuant | Attention-Adaptive Value Hadamard | Delta | Complete |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 0/200 |  |  |  | no |
| SingleQA | `qasper` | 0/200 |  |  |  | no |
| SingleQA | `multifieldqa_en` | 0/150 |  |  |  | no |
| MultiQA | `hotpotqa` | 200/200 | 54.65 | 54.53 | -0.12 | yes |
| MultiQA | `2wikimqa` | 200/200 | 44.47 | 44.95 | 0.48 | yes |
| MultiQA | `musique` | 200/200 | 30.01 | 27.11 | -2.90 | yes |
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
