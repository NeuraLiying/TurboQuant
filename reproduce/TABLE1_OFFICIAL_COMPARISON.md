# Table 1 Official Local Comparison

Scope: `meta-llama/Llama-3.1-8B-Instruct`, LongBench-V1 full task splits, Full Cache and TurboQuant only.

Only complete task files with the expected unique example indexes are used for category and average scores.

## Category Comparison

| Method | KV Size | Source | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average | Complete |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Cache | 16.0 | paper | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 | yes |
| Full Cache | 16.0 | local | 44.45 | 44.18 | 29.29 | 69.53 | 52.60 | 62.26 | 50.38 | yes |
| TurboQuant | 2.5 | paper | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 | yes |
| TurboQuant | 2.5 | local | 38.61 | 36.01 | 27.54 | 67.12 | 48.22 | 55.02 | 45.42 | yes |
| TurboQuant | 3.5 | paper | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 | yes |
| TurboQuant | 3.5 | local | 42.73 | 43.04 | 28.72 | 68.59 | 52.06 | 61.16 | 49.38 | yes |

## Task Scores

### Full Cache KV=16.0

| Category | Dataset | Unique Records | Records | Score | Complete |
| --- | --- | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 200 | 31.13 | yes |
| SingleQA | `qasper` | 200/200 | 200 | 46.77 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 150 | 55.44 | yes |
| MultiQA | `hotpotqa` | 200/200 | 200 | 55.09 | yes |
| MultiQA | `2wikimqa` | 200/200 | 200 | 46.36 | yes |
| MultiQA | `musique` | 200/200 | 200 | 31.09 | yes |
| Summarization | `gov_report` | 200/200 | 200 | 34.98 | yes |
| Summarization | `qmsum` | 200/200 | 200 | 25.56 | yes |
| Summarization | `multi_news` | 200/200 | 200 | 27.32 | yes |
| Few shot | `trec` | 200/200 | 200 | 72.50 | yes |
| Few shot | `triviaqa` | 200/200 | 200 | 92.48 | yes |
| Few shot | `samsum` | 200/200 | 200 | 43.61 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | 200 | 99.50 | yes |
| Synthetic | `passage_count` | 200/200 | 200 | 5.71 | yes |
| Code | `lcc` | 500/500 | 500 | 66.61 | yes |
| Code | `repobench-p` | 500/500 | 500 | 57.90 | yes |

### TurboQuant KV=2.5

| Category | Dataset | Unique Records | Records | Score | Complete |
| --- | --- | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 200 | 25.35 | yes |
| SingleQA | `qasper` | 200/200 | 200 | 42.44 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 150 | 48.03 | yes |
| MultiQA | `hotpotqa` | 200/200 | 200 | 44.83 | yes |
| MultiQA | `2wikimqa` | 200/200 | 200 | 37.96 | yes |
| MultiQA | `musique` | 200/200 | 200 | 25.24 | yes |
| Summarization | `gov_report` | 200/200 | 200 | 32.02 | yes |
| Summarization | `qmsum` | 200/200 | 200 | 24.38 | yes |
| Summarization | `multi_news` | 200/200 | 200 | 26.22 | yes |
| Few shot | `trec` | 200/200 | 200 | 70.50 | yes |
| Few shot | `triviaqa` | 200/200 | 200 | 89.39 | yes |
| Few shot | `samsum` | 200/200 | 200 | 41.47 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | 200 | 93.00 | yes |
| Synthetic | `passage_count` | 200/200 | 200 | 3.45 | yes |
| Code | `lcc` | 500/500 | 500 | 57.04 | yes |
| Code | `repobench-p` | 500/500 | 500 | 53.00 | yes |

### TurboQuant KV=3.5

| Category | Dataset | Unique Records | Records | Score | Complete |
| --- | --- | ---: | ---: | ---: | --- |
| SingleQA | `narrativeqa` | 200/200 | 200 | 29.25 | yes |
| SingleQA | `qasper` | 200/200 | 200 | 46.32 | yes |
| SingleQA | `multifieldqa_en` | 150/150 | 150 | 52.61 | yes |
| MultiQA | `hotpotqa` | 200/200 | 200 | 54.65 | yes |
| MultiQA | `2wikimqa` | 200/200 | 200 | 44.47 | yes |
| MultiQA | `musique` | 200/200 | 200 | 30.01 | yes |
| Summarization | `gov_report` | 200/200 | 200 | 34.08 | yes |
| Summarization | `qmsum` | 200/200 | 200 | 25.33 | yes |
| Summarization | `multi_news` | 200/200 | 200 | 26.77 | yes |
| Few shot | `trec` | 200/200 | 200 | 72.00 | yes |
| Few shot | `triviaqa` | 200/200 | 200 | 91.34 | yes |
| Few shot | `samsum` | 200/200 | 200 | 42.43 | yes |
| Synthetic | `passage_retrieval_en` | 200/200 | 200 | 100.00 | yes |
| Synthetic | `passage_count` | 200/200 | 200 | 4.12 | yes |
| Code | `lcc` | 500/500 | 500 | 64.04 | yes |
| Code | `repobench-p` | 500/500 | 500 | 58.29 | yes |
