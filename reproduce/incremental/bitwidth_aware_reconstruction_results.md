# Bitwidth-Aware Reconstruction Gate Results
Method: keep the reproduced TurboQuant bit allocation and add a prompt-length gate for reconstruction. At 2.5-bit, use the validated structure-adaptive LowBit-Gain gate. At 3.5-bit, use `learned_unit_mse_block2` when `prompt_tokens <= 14000`, otherwise keep TurboQuant MSE.
## Completed Task Results
| Dataset | Category | TQ 2.5 | Ours 2.5 | Delta 2.5 | TQ 3.5 | Ours 3.5 | Delta 3.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `narrativeqa` | SingleQA | 25.35 | nan | +nan | 29.25 | nan | +nan |
| `qasper` | SingleQA | 42.44 | 44.44 | +2.01 | 46.32 | nan | +nan |
| `multifieldqa_en` | SingleQA | 48.03 | nan | +nan | 52.61 | 55.92 | +3.30 |
| `hotpotqa` | MultiQA | 44.83 | 48.69 | +3.87 | 54.65 | 55.95 | +1.30 |
| `2wikimqa` | MultiQA | 37.96 | 39.04 | +1.08 | 44.47 | 46.32 | +1.86 |
| `musique` | MultiQA | 25.24 | 26.57 | +1.33 | 30.01 | 30.10 | +0.09 |
| `qmsum` | Summarization | 24.38 | nan | +nan | 25.33 | nan | +nan |
| `multi_news` | Summarization | 26.22 | nan | +nan | 26.77 | nan | +nan |
| `trec` | Few shot | 70.50 | nan | +nan | 72.00 | 71.50 | -0.50 |
| `triviaqa` | Few shot | 89.39 | nan | +nan | 91.34 | nan | +nan |
| `samsum` | Few shot | 41.47 | nan | +nan | 42.43 | 43.71 | +1.29 |
| `passage_retrieval_en` | Synthetic | 93.00 | nan | +nan | 100.00 | nan | +nan |

## Completed 3.5-bit Category Averages
| Category | Datasets | TQ 3.5 | Ours 3.5 | Delta |
| --- | --- | ---: | ---: | ---: |
| SingleQA | `multifieldqa_en` | 52.61 | 55.92 | +3.30 |
| MultiQA | `hotpotqa`, `2wikimqa`, `musique` | 43.04 | 44.13 | +1.08 |
| Few shot | `trec`, `samsum` | 57.21 | 57.61 | +0.39 |

## Notes
- Full MultiQA is complete and positive at both 2.5-bit and 3.5-bit.
- `narrativeqa`, `qmsum`, `gov_report`, and full `multi_news` hit 4090 OOM/throughput limits under the 3.5-bit learned-unit gate and are not included as completed 3.5-bit full results.
- `multi_news` partial output was stopped at 74/200 examples because throughput was too low for the current validation block.
