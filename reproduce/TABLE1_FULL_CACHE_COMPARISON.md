# Table 1 Full Cache Comparison

Last updated: 2026-06-11 20:55 CST

## Scope

- Paper target: Table 1, `Llama-3.1-8B-Instruct`, `Full Cache`, KV Size 16.
- Local dataset: LongBench-V1 full task splits.
- Local output root: `reproduce/runs/table1_official`.
- Progress file: `reproduce/runs/table1_official/table1_full_cache_progress.md`.
- Status: all 16 LongBench-V1 tasks are complete. No smoke-test or capped-example output is included in the table below.

## Category Comparison

| Category | Paper Full Cache | Local Full Cache | Local - Paper | Status |
| --- | ---: | ---: | ---: | --- |
| SingleQA | 45.29 | 44.45 | -0.84 | close |
| MultiQA | 45.16 | 44.18 | -0.98 | close |
| Summarization | 26.55 | 22.01 | -4.54 | needs investigation |
| Few shot | 68.38 | 68.77 | +0.39 | close |
| Synthetic | 59.54 | 52.60 | -6.94 | needs investigation |
| Code | 46.28 | 60.24 | +13.96 | needs investigation |
| Average | 50.06 | 48.71 | -1.35 | close average, category offsets differ |

## Local Task Scores

| Category | Dataset | Records | Local Score |
| --- | --- | ---: | ---: |
| SingleQA | `narrativeqa` | 200/200 | 31.13 |
| SingleQA | `qasper` | 200/200 | 46.77 |
| SingleQA | `multifieldqa_en` | 150/150 | 55.44 |
| MultiQA | `hotpotqa` | 200/200 | 55.09 |
| MultiQA | `2wikimqa` | 200/200 | 46.36 |
| MultiQA | `musique` | 200/200 | 31.09 |
| Summarization | `gov_report` | 200/200 | 24.62 |
| Summarization | `qmsum` | 200/200 | 22.31 |
| Summarization | `multi_news` | 200/200 | 19.09 |
| Few shot | `trec` | 200/200 | 73.00 |
| Few shot | `triviaqa` | 200/200 | 92.48 |
| Few shot | `samsum` | 200/200 | 40.82 |
| Synthetic | `passage_retrieval_en` | 200/200 | 99.50 |
| Synthetic | `passage_count` | 200/200 | 5.71 |
| Code | `lcc` | 500/500 | 64.91 |
| Code | `repobench-p` | 500/500 | 55.58 |

## Interpretation

The local Full Cache baseline is complete and the overall average is near the paper value. However, the category-level shape does not fully match the paper yet: Synthetic is much lower because `passage_count` is low, and Code is much higher than the paper row. These offsets nearly cancel in the final average, so the average alone is not enough evidence that the reproduction exactly matches Table 1.

The next reproduction step is still valid: run TurboQuant 2.5-bit and 3.5-bit on exactly the same 16 tasks, prompts, decoding settings, and scorer. The resulting comparison will show whether the local TurboQuant implementation preserves Full Cache quality under our local evaluation stack. Before claiming paper-level reproduction, investigate the Synthetic and Code mismatches.
