# Structure-Adaptive LowBit-Gain Results

Method: use `lowbit_gain_mse` at the 2.5-bit TurboQuant budget only when the prompt is long, non-code, and not structurally risky as a Passage-style prompt. Otherwise use reproduced TurboQuant `mse`.

Gate: `prompt_tokens > 6144`, not code-completion-like, and if `passage_count > 0` then `passage_count <= 15` and `question_marks <= 5`.

## Category Results

| Category | TurboQuant 2.5 | Structure-Adaptive 2.5 | Delta vs TQ2.5 | TurboQuant 3.5 | Activation |
| --- | ---: | ---: | ---: | ---: | ---: |
| SingleQA | 38.61 | 40.32 | +1.72 | 42.73 | 0.60 |
| MultiQA | 36.01 | 38.10 | +2.09 | 43.04 | 0.65 |
| Summarization | 27.54 | 27.60 | +0.06 | 28.72 | 0.56 |
| Few shot | 67.12 | 67.77 | +0.65 | 68.59 | 0.68 |
| Synthetic | 48.22 | 50.29 | +2.07 | 52.06 | 0.99 |
| Code | 55.02 | 55.02 | +0.00 | 61.16 | 0.00 |
| Average | 45.42 | 46.52 | +1.10 | 49.38 | - |

## Task Results

| Dataset | Category | TurboQuant 2.5 | Structure-Adaptive 2.5 | Delta vs TQ2.5 | TurboQuant 3.5 | Activation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `narrativeqa` | SingleQA | 25.35 | 27.34 | +2.00 | 29.25 | 1.00 |
| `qasper` | SingleQA | 42.44 | 44.44 | +2.01 | 46.32 | 0.21 |
| `multifieldqa_en` | SingleQA | 48.03 | 49.18 | +1.15 | 52.61 | 0.59 |
| `hotpotqa` | MultiQA | 44.83 | 48.69 | +3.87 | 54.65 | 0.80 |
| `2wikimqa` | MultiQA | 37.96 | 39.04 | +1.08 | 44.47 | 0.43 |
| `musique` | MultiQA | 25.24 | 26.57 | +1.33 | 30.01 | 0.71 |
| `gov_report` | Summarization | 32.02 | 32.41 | +0.40 | 34.08 | 0.76 |
| `qmsum` | Summarization | 24.38 | 24.18 | -0.20 | 25.33 | 0.91 |
| `multi_news` | Summarization | 26.22 | 26.20 | -0.01 | 26.77 | 0.03 |
| `trec` | Few shot | 70.50 | 71.50 | +1.00 | 72.00 | 0.56 |
| `triviaqa` | Few shot | 89.39 | 89.48 | +0.09 | 91.34 | 0.80 |
| `samsum` | Few shot | 41.47 | 42.32 | +0.85 | 42.43 | 0.67 |
| `passage_retrieval_en` | Synthetic | 93.00 | 96.00 | +3.00 | 100.00 | 1.00 |
| `passage_count` | Synthetic | 3.45 | 4.59 | +1.14 | 4.12 | 0.98 |
| `lcc` | Code | 57.04 | 57.04 | +0.00 | 64.04 | 0.00 |
| `repobench-p` | Code | 53.00 | 53.00 | +0.00 | 58.29 | 0.00 |

## Full Runner Validation

| Dataset | Examples | Score | Delta vs TQ2.5 | Active quantizer counts | Passage-blocked |
| --- | ---: | ---: | ---: | --- | ---: |
| `2wikimqa` | 200 | 39.04 | +1.08 | 87 lowbit / 113 mse | 35 |
| `hotpotqa` | 200 | 48.69 | +3.87 | 160 lowbit / 40 mse | 23 |
| `musique` | 200 | 26.57 | +1.33 | 142 lowbit / 58 mse | 58 |
| `qasper` | 200 | 44.44 | +2.01 | 42 lowbit / 158 mse | 0 |

## Reproduce Commands

```bash
conda run -n turboquant python scripts/build_content_adaptive_lowbit_table.py \
  --threshold 6144 \
  --max-passages 15 \
  --max-question-marks 5 \
  --feature-table reproduce/incremental/prompt_gate_features_table1.json \
  --output reproduce/incremental/structure_adaptive_lowbit_gain_threshold_6144_p15_q5_table1_comparison.json
```
