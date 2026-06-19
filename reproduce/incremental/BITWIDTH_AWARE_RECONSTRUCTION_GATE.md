# Bitwidth-Aware Reconstruction Gate

## Purpose

Test a method-level extension over reproduced TurboQuant that improves the original 2.5-bit and 3.5-bit operating points without changing the average KV bit budget.

The method keeps TurboQuant's channel/outlier allocation and changes only the reconstruction rule under a prompt-length gate:

- 2.5-bit: use the previously validated structure-adaptive LowBit-Gain gate.
- 3.5-bit: use learned unit-norm block reconstruction (`learned_unit_mse_block2`) when `prompt_tokens <= 14000`; otherwise keep TurboQuant MSE.

## Full MultiQA Results

Full LongBench MultiQA, Llama-3.1-8B-Instruct.

| Dataset | TurboQuant 2.5 | Ours 2.5 | Delta | TurboQuant 3.5 | Ours 3.5 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2wikimqa` | 37.96 | 39.04 | +1.08 | 44.47 | 46.32 | +1.86 |
| `hotpotqa` | 44.83 | 48.69 | +3.87 | 54.65 | 55.95 | +1.30 |
| `musique` | 25.24 | 26.57 | +1.33 | 30.01 | 30.10 | +0.09 |
| **Average** | **36.01** | **38.10** | **+2.09** | **43.04** | **44.13** | **+1.08** |

## Additional Completed 3.5-bit Checks

| Dataset | Category | TurboQuant 3.5 | Ours 3.5 | Delta |
| --- | --- | ---: | ---: | ---: |
| `multifieldqa_en` | SingleQA | 52.61 | 55.92 | +3.30 |
| `qasper` | SingleQA | 46.32 | 46.53 | +0.21 |
| `trec` | Few shot | 72.00 | 71.50 | -0.50 |
| `samsum` | Few shot | 42.43 | 43.71 | +1.29 |
| `passage_retrieval_en` | Synthetic | 100.00 | 100.00 | +0.00 |

## Negative And Incomplete Checks

- `role_adaptive_mse` failed on 20-example MultiQA slices and was removed from the code path.
- Direct `unit_mse` at 3.5-bit improved `2wikimqa` but reduced `hotpotqa` and `musique`.
- Direct `learned_unit_mse_block2` at 3.5-bit improved `2wikimqa` and `hotpotqa`, but reduced `musique`; the `prompt_tokens <= 14000` gate fixed this on MultiQA.
- `narrativeqa`, `qmsum`, and `gov_report` hit 4090 OOM under the 3.5-bit learned-unit gate.
- Full `multi_news` was stopped at 74/200 examples because throughput was too low for this validation block.

## Reproduce Commands

2.5-bit result uses the existing structure-adaptive LowBit-Gain outputs:

```bash
/home/liying/miniconda3/envs/turboquant/bin/python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --key-quantizer mse \
  --value-quantizer mse \
  --long-prompt-threshold 6144 \
  --long-prompt-key-quantizer lowbit_gain_mse \
  --long-prompt-value-quantizer lowbit_gain_mse \
  --long-prompt-exclude-code \
  --long-prompt-max-passages 15 \
  --long-prompt-max-question-marks 5 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto
```

3.5-bit result uses learned unit block reconstruction under a prompt-length gate:

```bash
/home/liying/miniconda3/envs/turboquant/bin/python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --cache-mode turboquant \
  --kv-bits 3.5 \
  --key-quantizer mse \
  --value-quantizer mse \
  --long-prompt-threshold 0 \
  --long-prompt-max-tokens 14000 \
  --long-prompt-key-quantizer learned_unit_mse_block2 \
  --long-prompt-value-quantizer learned_unit_mse_block2 \
  --long-prompt-exclude-code \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto
```

## Artifacts

- MultiQA JSON summary: `reproduce/incremental/bitwidth_aware_multiqa_results.json`
- Broader completed-task summary: `reproduce/incremental/bitwidth_aware_reconstruction_results.md`
- Full JSONL outputs: `reproduce/runs/incremental/bitwidth_gate_learned_unit_*_turboquant_3p5_full.jsonl`
