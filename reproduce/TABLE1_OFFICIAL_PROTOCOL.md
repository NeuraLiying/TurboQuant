# Table 1 Official Reproduction Protocol

Last updated: 2026-06-11 20:55 CST

## Scope

- Paper target: Table 1, captioned as LongBench-V1 results for `Llama-3.1-8B-Instruct`.
- Local first-stage comparison: `Full Cache` and `TurboQuant` only.
- Excluded baselines: KIVI, PolarQuant, SnapKV, PyramidKV, Product Quantization, RabitQ.
- Dataset family for the official Table 1 comparison: LongBench-V1 task splits, not the earlier LongBench-E-only diagnostic runs.

## Evaluation Rule

For Table 1, a local score is considered official only when the corresponding LongBench-V1 task split is complete.

| Task type | Official local record count |
| --- | ---: |
| Standard LongBench tasks | full task split, usually 200 examples |
| `multifieldqa_en` | 150 examples |
| `lcc` | 500 examples |
| `repobench-p` | 500 examples |

Runs with `--max-examples 1`, `--max-examples 20`, or any other small cap are smoke tests only. They validate model loading, prompt formatting, generation, cache behavior, resume behavior, and scoring. They must not be used as Table 1 comparison numbers.

## Category Aggregation

Table 1 category scores are computed dataset-first:

1. Score every example with the LongBench-style metric for its dataset.
2. Average examples within each dataset.
3. Average dataset scores inside each category.
4. Average the six category scores for the final Table 1 average.

This avoids weighting a category by whichever dataset happens to contain more examples.

## Current Full Cache Status

Current progress file: `reproduce/runs/table1_official/table1_full_cache_progress.md`.
Full Cache comparison note: `reproduce/TABLE1_FULL_CACHE_COMPARISON.md`.

As of this update, the Full Cache baseline pass is complete across all 16 LongBench-V1 tasks. Completed task results can be compared to the corresponding paper categories because every task in every category is complete.

Known current status after the latest refresh:

- Complete Full Cache tasks: 16 / 16.
- Partial Full Cache tasks: none.
- Missing/not-started Full Cache tasks: none.
- Local Full Cache average: 48.71 versus paper 50.06.
- Category offsets remain: Summarization and Synthetic are lower than paper, while Code is higher. See `reproduce/TABLE1_FULL_CACHE_COMPARISON.md`.

## Implication For The 20-Example Runs

The 20-example run was a smoke test. It is not a dataset result and is not comparable to the paper's Table 1. The official reproduction path is now to run the same full splits for `TurboQuant 2.5-bit` and `TurboQuant 3.5-bit`, then compare them against the completed local Full Cache baseline and the paper rows.
