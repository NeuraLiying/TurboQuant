# Incremental Experiments

This log tracks post-reproduction experiments on `meta-llama/Llama-3.1-8B-Instruct`. The objective is to obtain reportable gains over the reproduced TurboQuant baseline under the same or lower average KV bit budget.

## Fixed Result Summary

The reproduction baseline is fixed in `reproduce/TABLE1_OFFICIAL_COMPARISON.md`. Incremental results are recorded separately so that reproduction quality and new-method evidence can be inspected independently.

Current complete reportable increment:

| Method | KV bits | Complete Tasks | TurboQuant Avg | Method Avg | Delta | Result File |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Unified Regular-Gain Gate | 2.5 | 16 / 16 | 45.42 | 45.95 | +0.53 | `reproduce/incremental/UNIFIED_REGULAR_GAIN_GATE_FINAL.md` |
| Unified Regular-Gain Gate | 3.5 | 16 / 16 | 49.38 | 49.53 | +0.15 | `reproduce/incremental/UNIFIED_REGULAR_GAIN_GATE_FINAL.md` |

Additional method-level branch:

| Method | KV bits | Complete Tasks | Current Evidence | Result File |
| --- | ---: | ---: | --- | --- |
| Rate-Regime MSE | 2.5 | 10 / 16 | MultiQA is complete: 36.01 -> 36.55, delta +0.54. Full Table 1 is still running. | `reproduce/incremental/rate_regime_mse_table1.md` |
| Rate-Regime MSE | 3.5 | 16 / 16 | Full Table 1 is complete: 49.38 -> 49.86, delta +0.48. | `reproduce/incremental/rate_regime_mse_table1.md` |

This file keeps negative and partial experiments because they are useful for deciding what not to expand. The current reportable claim should use the complete Unified Regular-Gain Gate table; Rate-Regime MSE should be treated as additional evidence until the 2.5-bit Table 1 run completes.

## Scope

- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Primary benchmark: LongBench-V1 English Table 1 tasks
- Main budgets: average 2.5-bit and 3.5-bit KV cache
- Main comparison target: uniform TurboQuant 2.5-bit / 3.5-bit from the reproduction snapshot
- Non-goal for this phase: multi-model adaptation

## Experiment Policy

Every experiment should record:

- Purpose: what capability or hypothesis is being tested.
- Command: exact runnable command or script path.
- Output: JSONL/JSON/Markdown artifacts under `reproduce/incremental/` or `reproduce/runs/incremental/`.
- Metrics: LongBench score, category/task score, cache storage ratio, latency, and when relevant attention score error.
- Analysis: whether the result improves the quality-compression tradeoff and what to try next.

## Candidate Directions

### E1: Layer-Wise Adaptive TurboQuant

Purpose: improve quality at the same average bit budget by assigning higher bits to layers with larger attention score error and lower bits to less sensitive layers.

Implementation status:

- Added layer-wise key/value bit schedules to `TurboQuantDynamicCache`.
- Added CLI args to `experiments/longbench/run_full_cache_eval.py`:
  - `--layer-key-bits`
  - `--layer-value-bits`
- Added `scripts/probe_attention_error.py` to estimate layer/head sensitivity.
- Added `scripts/build_layer_bit_schedule.py` to convert probe output into a fixed average-bit schedule.

Planned first run:

```bash
conda run -n turboquant python scripts/probe_attention_error.py \
  --dataset-key longbench_2wikimqa \
  --start-index 0 \
  --max-examples 4 \
  --max-input-tokens 4096 \
  --device cuda:3 \
  --methods turboquant:2.5,turboquant:3.5,naive_int:4,kivi_style:4 \
  --output reproduce/incremental/attention_error_2wikimqa_0_4.json

conda run -n turboquant python scripts/build_layer_bit_schedule.py \
  reproduce/incremental/attention_error_2wikimqa_0_4.json \
  --method turboquant_2_5bit \
  --metric score_rmse \
  --target-average-bits 2.5 \
  --output reproduce/incremental/layer_schedule_2p5_from_2wikimqa_0_4.json
```

Decision rule:

- A candidate schedule is worth expanding if a 20-example slice improves over uniform TurboQuant 2.5-bit at comparable cache ratio.
- If the 2.5-bit schedule improves, test whether the same schedule or a 3.5-bit variant improves category-level results on SingleQA, MultiQA, and Synthetic.

### E2: KIVI-Style And Naive INT4 Baselines

Purpose: provide local, implementation-consistent baselines for common KV quantization choices.

Implementation status:

- Added `baseline_mode=naive_int`: per-token uniform affine quantization for K/V.
- Added `baseline_mode=kivi_style`: per-channel uniform affine key quantization and per-token uniform affine value quantization.

Planned role:

- `naive_int4` and `kivi_style_int4` are comparison baselines, not the main contribution.
- INT8 is only a sanity baseline if needed; it is not a main result target because the incremental goal is improvement under 2.5/3.5 average-bit budgets.

### E3: Attention Score Error Analysis

Purpose: connect accuracy changes to an interpretable mechanism: quantization-induced attention score/probability error.

Metrics:

- `score_mae`
- `score_rmse`
- `prob_mae`
- `prob_l1`
- `top1_match`
- `key_rel_l2`
- `value_rel_l2`

Expected use:

- Rank layers and heads by sensitivity.
- Generate candidate bit schedules.
- Explain why an adaptive schedule helps or fails.

## Current Status

- Reproduction baseline is fixed and available in `reproduce/TABLE1_OFFICIAL_COMPARISON.md`.
- Incremental code infrastructure is in place for layer-wise schedules, KIVI-style / naive INT baselines, attention error probing, rotation/preconditioning candidates, prompt-gated quantizer routing, and generic Table 1 method aggregation.
- Unified Regular-Gain Gate is complete on all 16 Table 1 tasks for both 2.5-bit and 3.5-bit.
- Rate-Regime MSE is complete for 3.5-bit Table 1 and still running for the 2.5-bit full table.

## Results

### R1: 2wikimqa 20-Example Layer-Wise Adaptive Smoke

Purpose: test whether attention-error-derived layer-wise bit allocation can improve over uniform TurboQuant at the same average 2.5-bit budget.

Schedule source:

- Probe: `reproduce/incremental/attention_error_smoke.json`
- Schedule: `reproduce/incremental/layer_schedule_2p5_smoke.json`
- Rule: rank layers by TurboQuant 2.5-bit attention `score_rmse`; assign 3-bit to the most sensitive 16 / 32 layers and 2-bit to the rest.

Candidate schedule:

```text
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
```

Command:

```bash
SCHED=$(python - <<'PY'
import json
print(json.load(open('reproduce/incremental/layer_schedule_2p5_smoke.json'))['cli']['layer_key_bits'])
PY
)
conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:3 \
  --cache-mode turboquant \
  --baseline-mode turboquant \
  --kv-bits 2.5 \
  --layer-key-bits "$SCHED" \
  --layer-value-bits "$SCHED" \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 \
  --end-index 20 \
  --output reproduce/runs/incremental/longbench_2wikimqa_turboquant_2p5_layer_adaptive_0_20.jsonl
```

Results on `longbench_2wikimqa[0:20]`:

| Method | Avg KV bits | LongBench score | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: |
| Full Cache | 16.0 | 37.29 | 0.35 |  |
| TurboQuant uniform | 2.5 | 32.76 | 0.30 | 0.1720 |
| TurboQuant layer-adaptive | 2.5 | 36.25 | 0.30 | 0.1641 |
| TurboQuant uniform | 3.5 | 41.87 | 0.35 | 0.2345 |

Analysis:

- The first layer-adaptive schedule improves the 20-example slice by `+3.49` LongBench points over uniform TurboQuant 2.5-bit.
- The measured cache ratio is slightly lower than uniform 2.5-bit because integer layer-wise 2/3-bit packing avoids fractional outlier-index overhead.
- This is a promising candidate, but it is only a small MultiQA slice. The next step is to regenerate the schedule from a larger attention probe and run the candidate on the full 200-example `2wikimqa` task.

### R2: 2wikimqa Full-Task Adaptive And Baseline Check

Purpose: verify whether the small-slice layer-wise gains survive the full 200-example `2wikimqa` task, and compare local INT baselines.

Results on full `longbench_2wikimqa`:

| Method | Avg KV bits | LongBench score | Delta vs TQ 2.5 | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full Cache | 16.0 | 46.36 | +8.40 | 0.475 |  |
| TurboQuant uniform | 2.5 | 37.96 | +0.00 | 0.365 | 0.1720 |
| TurboQuant layer 2/3 | 2.5 | 33.67 | -4.29 | 0.300 | 0.1641 |
| TurboQuant fractional reverse | 2.5 | 33.45 | -4.51 | 0.315 | 0.1720 |
| TurboQuant K2/V3 | 2.5 | 31.17 | -6.79 | 0.275 | 0.1641 |
| TurboQuant K3/V2 | 2.5 | 36.28 | -1.68 | 0.340 | 0.1641 |
| TurboQuant uniform | 3.5 | 44.47 | +6.51 | 0.425 | 0.2345 |
| Naive INT4 | 4.0 | 38.75 | +0.79 | 0.430 | 0.2578 |
| KIVI-style INT4 | 4.0 | 36.68 | -1.28 | 0.380 | 0.2544 |

Artifacts:

- `reproduce/incremental/2wikimqa_full_comparison_round2.json`
- `reproduce/incremental/2wikimqa_full_turboquant_2p5_layer_adaptive.json`
- `reproduce/incremental/2wikimqa_full_turboquant_2p5_frac_reverse.json`
- `reproduce/incremental/2wikimqa_full_turboquant_k2_v3.json`
- `reproduce/incremental/2wikimqa_full_turboquant_k3_v2.json`
- `reproduce/incremental/2wikimqa_full_naive_int4.json`
- `reproduce/incremental/2wikimqa_full_kivi_style_int4.json`

Analysis:

- The initial 20-example layer-wise gain did not generalize to the full task.
- Simple key/value asymmetric bit splits at the same 2.5 average bit budget also did not improve over uniform TurboQuant 2.5-bit.
- TurboQuant 3.5-bit is a strong quality-compression baseline on this task: it scores higher than both local INT4 baselines while using a lower measured cache ratio.
- Next search direction: evaluate intermediate 3.0/3.25-bit TurboQuant budgets and less aggressive adaptive schedules to seek a reportable quality-compression point below 3.5-bit.

### R3: 2wikimqa Intermediate Bit Budgets

Purpose: search for a quality-compression point below uniform TurboQuant 3.5-bit and compare against local INT4 baselines.

Results on full `longbench_2wikimqa`:

| Method | Avg KV bits | LongBench score | Contains-answer acc | Avg cache ratio | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant uniform | 2.5 | 37.96 | 0.365 | 0.1720 | 3.28 |
| TurboQuant uniform | 3.0 | 37.72 | 0.365 | 0.1953 | 2.24 |
| TurboQuant uniform | 3.25 | 40.35 | 0.410 | 0.2189 | 9.69 |
| TurboQuant layer 3/4 | 3.25 | 40.19 | 0.390 | 0.2109 | 3.23 |
| TurboQuant uniform | 3.5 | 44.47 | 0.425 | 0.2345 | 8.88 |
| Naive INT4 | 4.0 | 38.75 | 0.430 | 0.2578 | 1.38 |
| KIVI-style INT4 | 4.0 | 36.68 | 0.380 | 0.2544 | 1.32 |

Artifacts:

- `reproduce/incremental/2wikimqa_full_comparison_round3.json`
- `reproduce/incremental/2wikimqa_full_turboquant_3p0.json`
- `reproduce/incremental/2wikimqa_full_turboquant_3p25.json`
- `reproduce/incremental/2wikimqa_full_turboquant_3p25_layer_3_4.json`

Analysis:

- Uniform 3.25-bit improves over uniform 2.5-bit by `+2.39` while using less cache than local INT4 baselines.
- Layer 3/4 average-3.25 schedule gives almost the same score as uniform 3.25 (`-0.16`) with lower measured cache ratio (`0.2109` vs `0.2189`) and lower latency in this Python-level implementation (`3.23s` vs `9.69s`).
- On this task, 3.25-bit TurboQuant variants outperform KIVI-style INT4 and use less cache. Uniform 3.25 also outperforms naive INT4 by `+1.60` with lower cache ratio.
- The next validation step is to test the 3.25 layer 3/4 schedule on another MultiQA task before treating it as a stable efficiency improvement.

### R4: MultiQA 3.25-bit Quality-Compression Operating Point

Purpose: validate whether uniform TurboQuant 3.25-bit is a reportable operating point on the full LongBench MultiQA category, using the reproduced 2.5-bit / 3.5-bit results and local INT4 baselines as references.

Commands:

```bash
conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_musique \
  --device cuda:2 \
  --cache-mode turboquant \
  --baseline-mode turboquant \
  --kv-bits 3.25 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --output reproduce/runs/incremental/longbench_musique_turboquant_3p25_full.jsonl \
  --progress-every 200

conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_hotpotqa \
  --device cuda:3 \
  --cache-mode turboquant \
  --baseline-mode naive_int \
  --kv-bits 4 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --output reproduce/runs/incremental/longbench_hotpotqa_naive_int4_full.jsonl \
  --progress-every 200

conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_hotpotqa \
  --device cuda:4 \
  --cache-mode turboquant \
  --baseline-mode kivi_style \
  --kv-bits 4 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --output reproduce/runs/incremental/longbench_hotpotqa_kivi_style_int4_full.jsonl \
  --progress-every 200

conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_musique \
  --device cuda:5 \
  --cache-mode turboquant \
  --baseline-mode naive_int \
  --kv-bits 4 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --output reproduce/runs/incremental/longbench_musique_naive_int4_full.jsonl \
  --progress-every 200

conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_musique \
  --device cuda:6 \
  --cache-mode turboquant \
  --baseline-mode kivi_style \
  --kv-bits 4 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --output reproduce/runs/incremental/longbench_musique_kivi_style_int4_full.jsonl \
  --progress-every 200
```

Result summary on full LongBench MultiQA, 200 examples per task:

| Method | Avg KV bits | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Cache | 16.0 | 46.36 | 55.09 | 31.09 | 44.18 |  |
| TurboQuant uniform | 2.5 | 37.96 | 44.83 | 25.24 | 36.01 | 0.1719 |
| TurboQuant uniform | 3.25 | 40.35 | 52.55 | 28.40 | 40.43 | 0.2188 |
| TurboQuant uniform | 3.5 | 44.47 | 54.65 | 30.01 | 43.04 | 0.2344 |
| Naive INT4 | 4.0 | 38.75 | 52.50 | 25.28 | 38.84 | 0.2578 |
| KIVI-style INT4 | 4.0 | 36.68 | 48.52 | 25.97 | 37.05 | 0.2542 |

Artifacts:

- `reproduce/incremental/multiqa_full_comparison_tq3p25.json`
- `reproduce/incremental/musique_full_turboquant_3p25.json`
- `reproduce/incremental/hotpotqa_full_naive_int4.json`
- `reproduce/incremental/hotpotqa_full_kivi_style_int4.json`
- `reproduce/incremental/musique_full_naive_int4.json`
- `reproduce/incremental/musique_full_kivi_style_int4.json`

Analysis:

- Uniform TurboQuant 3.25-bit improves the MultiQA average from `36.01` at 2.5-bit to `40.43`, a `+4.43` LongBench point gain, while keeping the average measured cache ratio at `0.2188`.
- Compared with uniform TurboQuant 3.5-bit, 3.25-bit gives up `2.61` MultiQA points but reduces measured cache ratio from `0.2344` to `0.2188`, about `6.7%` relative cache reduction.
- Compared with local INT4 baselines, 3.25-bit has a lower measured cache ratio than naive INT4 and KIVI-style INT4, while improving MultiQA average by `+1.59` over naive INT4 and `+3.38` over KIVI-style INT4.
- The layer 3/4 average-3.25 schedule should not be treated as a stable contribution yet: it matched uniform 3.25-bit on full 2WikiMQA but underperformed on the HotpotQA 20-example slice. The reportable result from this round is therefore the uniform 3.25-bit operating point, not the adaptive schedule.
- Simple 2.5-bit layer/key-value adaptive variants did not produce full-task gains. Future adaptive work should focus on less aggressive policies around 3.0 to 3.25 bits or use per-head/token sensitivity rather than only layer-level schedules.

### R5: Fractional Allocation Variant Screening

Purpose: test whether a different fractional-bit policy improves TurboQuant at the same average bit budget. The candidate `quarter_high2` keeps the same requested average bits but assigns a smaller set of channels two extra bits instead of assigning more channels one extra bit.

Candidate implementation:

- Existing `effective_bit_allocation=blend`: for `b + f` bits, assign `ceil(b + f)` bits to `f` fraction of channels and `floor(b + f)` bits to the rest.
- Candidate `effective_bit_allocation=quarter_high2`: assign `floor(b + f) + 2` bits to `f / 2` fraction of channels and `floor(b + f)` bits to the rest.

Commands:

```bash
conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:2 \
  --cache-mode turboquant \
  --baseline-mode turboquant \
  --kv-bits 3.25 \
  --effective-bit-allocation quarter_high2 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 \
  --end-index 20 \
  --output reproduce/runs/incremental/longbench_2wikimqa_turboquant_3p25_qh2_0_20.jsonl \
  --progress-every 20

# Repeated for hotpotqa/musique and for 2.5-bit / 3.25-bit.
```

Slice results on `index[0:20]`:

| Task | Avg KV bits | Uniform | quarter_high2 | Delta | Ratio uniform | Ratio qh2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2WikiMQA | 2.5 | 32.76 | 30.97 | -1.79 | 0.1720 | 0.1720 |
| HotpotQA | 2.5 | 36.00 | 23.67 | -12.33 | 0.1719 | 0.1719 |
| MuSiQue | 2.5 | 25.00 | 14.76 | -10.24 | 0.1719 | 0.1719 |
| 2WikiMQA | 3.25 | 40.95 | 31.87 | -9.08 | 0.2189 | 0.2189 |
| HotpotQA | 3.25 | 47.63 | 49.02 | +1.39 | 0.2188 | 0.2188 |
| MuSiQue | 3.25 | 45.83 | 35.43 | -10.40 | 0.2188 | 0.2188 |

Artifacts:

- `reproduce/incremental/qh2_slice_comparison.json`
- `reproduce/incremental/longbench_2wikimqa_turboquant_2p5_qh2_0_20.summary.json`
- `reproduce/incremental/longbench_2wikimqa_turboquant_3p25_qh2_0_20.summary.json`
- `reproduce/incremental/longbench_hotpotqa_turboquant_2p5_qh2_0_20.summary.json`
- `reproduce/incremental/longbench_hotpotqa_turboquant_3p25_qh2_0_20.summary.json`
- `reproduce/incremental/longbench_musique_turboquant_2p5_qh2_0_20.summary.json`
- `reproduce/incremental/longbench_musique_turboquant_3p25_qh2_0_20.summary.json`

Analysis:

- `quarter_high2` does not improve the quality-compression tradeoff. It has almost identical measured cache ratio to uniform fractional allocation but lowers average slice score by `-8.12` at 2.5-bit and `-6.03` at 3.25-bit.
- The only positive cell is HotpotQA 3.25-bit on 20 examples (`+1.39`), which is not enough to justify a full 200-example expansion because the same candidate fails strongly on the other two MultiQA tasks.
- This negative result suggests that the TurboQuant default fractional channel allocation is already better than concentrating the fractional budget into fewer higher-bit channels. Future fractional-allocation work should use sensitivity-aware channel selection, not only a fixed arithmetic redistribution of the same average bits.

### R6: Sensitivity-Aware Outlier Selection Attempts

Purpose: test method-level changes at the original TurboQuant bit budgets. Unlike the 3.25-bit operating-point search, these experiments keep the average budget fixed at TurboQuant 2.5-bit or 3.5-bit and change how the high-bit budget is assigned.

Implemented candidates:

- `outlier_policy=error_gain`: select high-bit channels by estimating per-channel reconstruction-error reduction from low-bit to high-bit quantization.
- `outlier_policy=static_score`: select high-bit channels from calibration-derived layer/channel scores.
- Independent `key_outlier_policy` and `value_outlier_policy` so key/value selection rules can be tested separately.

Calibration artifacts:

- `reproduce/incremental/outlier_policy_compare_2wikimqa_2p5_0_2_1024.json`
- `reproduce/incremental/static_channel_scores_2wikimqa_0_4_1024.json`

Mechanism check:

| Policy | Bits | Calibration metric | Dynamic baseline | Candidate |
| --- | ---: | --- | ---: | ---: |
| error_gain | 2.5 | attention score MAE | 0.1307 | 0.1394 |
| error_gain | 2.5 | attention score RMSE | 0.2124 | 0.2342 |

The `error_gain` reconstruction proxy increased attention-score error, so it was not expanded to LongBench full-task runs.

2WikiMQA 2.5-bit slice screening:

| Method | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 2.5 |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 32.76 | 0.30 | 0.1720 | +0.00 |
| static key only | 31.67 | 0.30 | 0.1720 | -1.09 |
| static value only | 35.62 | 0.25 | 0.1720 | +2.86 |
| static key+value | 35.23 | 0.35 | 0.1720 | +2.47 |

2WikiMQA 2.5-bit full-task validation:

| Method | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 2.5 |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 37.96 | 0.365 | 0.1720 | +0.00 |
| static value only | 34.30 | 0.325 | 0.1720 | -3.66 |
| static key+value | 32.46 | 0.295 | 0.1720 | -5.50 |

Artifacts:

- `reproduce/incremental/2wikimqa_full_turboquant_2p5_static_value.json`
- `reproduce/incremental/2wikimqa_full_turboquant_2p5_static_score.json`
- `reproduce/runs/incremental/longbench_2wikimqa_turboquant_2p5_static_value_full.jsonl`
- `reproduce/runs/incremental/longbench_2wikimqa_turboquant_2p5_static_score_full.jsonl`

Analysis:

- Sensitivity-aware static channel selection showed positive 20-example results but did not generalize to full 2WikiMQA.
- The failure is stronger for key+value than for value-only, suggesting the calibration signal is overfit to the slice and not yet robust enough for static global channel selection.
- This method should not be reported as a positive contribution.

### R7: Attention-Error-Aware Layer Allocation At 3.5-bit

Purpose: test a fixed-budget layer-wise allocation at the original TurboQuant 3.5-bit budget. The candidate assigns 3-bit to low-sensitivity layers and 4-bit to high-sensitivity layers, with average bit budget exactly 3.5.

Schedule source:

- Probe: `reproduce/incremental/attention_error_2wikimqa_0_4_1024.json`
- Schedule: `reproduce/incremental/layer_schedule_3p5_3_4_2wikimqa_0_4_1024.json`
- High-bit layers: `16-31`

2WikiMQA 3.5-bit slice screening:

| Method | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 3.5 |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant 3.5 | 41.87 | 0.35 | 0.2345 | +0.00 |
| Layer 3/4 avg-3.5 | 43.29 | 0.40 | 0.2266 | +1.42 |

2WikiMQA 3.5-bit full-task validation:

| Method | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 3.5 |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant 3.5 | 44.47 | 0.425 | 0.2345 | +0.00 |
| Layer 3/4 avg-3.5 | 40.01 | 0.395 | 0.2266 | -4.46 |

Artifacts:

- `reproduce/incremental/2wikimqa_full_turboquant_3p5_layer_3_4.json`
- `reproduce/incremental/layer_schedule_3p5_3_4_2wikimqa_0_4_1024.json`
- `reproduce/runs/incremental/longbench_2wikimqa_turboquant_3p5_layer_3_4_full.jsonl`

Analysis:

- The layer allocation candidate improved the 20-example slice and reduced measured cache ratio, but it did not generalize to the full 200-example task.
- This confirms that small slices are not reliable enough for reporting method gains. Future candidates must be validated full-task before being treated as contributions.
- The current successful evidence remains negative for these method-level candidates: they are implemented and testable, but they do not yet provide reportable performance gains over TurboQuant 2.5-bit or 3.5-bit.

### R8: Budget-Matched Token Protection Attempts

Purpose: test token-level mixed precision at the original 2.5-bit budget. These candidates quantize most tokens at 2-bit and keep selected tokens in FP16, with the number of FP16 tokens chosen by actual packed-storage matching against the reproduced TurboQuant 2.5-bit segment.

Implemented policies:

- `sink_recent_budget`: protect initial sink tokens and most recent tokens.
- `norm_aware_budget`: protect initial sink tokens and high K/V-norm tokens.
- `attention_error_budget`: protect initial sink tokens and tokens with high query-conditioned quantization-induced attention error.
- `token_protection_targets`: allows `both`, `key`, or `value` protection.

2WikiMQA results:

| Method | Split | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 2.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 0:20 | 32.76 | 0.30 | 0.1720 | +0.00 |
| sink/recent both | 0:20 | 40.06 | 0.30 | 0.1718 | +7.30 |
| norm-aware both | 0:20 | 25.50 | 0.20 | 0.1718 | -7.26 |
| attention-error key-only | 0:20 | 35.20 | 0.30 | 0.1719 | +2.44 |
| sink/recent both | full | 36.06 | 0.330 | 0.1718 | -1.90 |
| sink/recent key-only | full | 34.33 | 0.320 | 0.1719 | -3.63 |

Artifacts:

- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_2p5_sink_recent_budgetmatched.json`
- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_2p5_norm_aware_budgetmatched.json`
- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_2p5_keyonly_attention_error_qerr_budget.json`
- `reproduce/incremental/landscape2_2wikimqa_full_turboquant_2p5_sink_recent_budgetmatched.json`
- `reproduce/incremental/landscape2_2wikimqa_full_turboquant_2p5_keyonly_sink_recent_budgetmatched.json`

Analysis:

- The strongest token-protection slice result (`sink_recent_budget`) did not generalize to full 2WikiMQA.
- Attention-error-aware token selection was methodologically cleaner, but the 20-example gain was weaker than sink/recent and was not expanded to full.
- This direction is not a reportable contribution in its current form.

### R9: Alternative Quantization Objectives And Decode Precision

Purpose: test whether changing the quantization objective or decode-stage precision improves the reproduced 2.5-bit TurboQuant baseline.

Candidates:

- Key/Product-QJL + Value/MSE: use TurboQuant's inner-product-oriented product quantizer on keys.
- Key/MSE + Value/Product-QJL: control for applying Product-QJL to values.
- Decode-FP16: keep generation-time decode KV unquantized while still quantizing the long prefill KV cache.

2WikiMQA 20-example results:

| Method | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 2.5 |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 32.76 | 0.30 | 0.1720 | +0.00 |
| Key Product-QJL / Value MSE | 9.85 | 0.10 | 0.1801 | -22.90 |
| Key MSE / Value Product-QJL | 31.25 | 0.30 | 0.1798 | -1.51 |
| Decode-FP16 2.5-bit | 32.54 | 0.35 | 0.1725 | -0.22 |
| Decode-FP16 3.5-bit | 41.87 | 0.35 | 0.2350 | n/a |

Artifacts:

- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_2p5_keyprod_valuemse.json`
- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_2p5_keymse_valueprod.json`
- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_2p5_decode_fp16.json`
- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_3p5_decode_fp16.json`

Analysis:

- Product-QJL was not beneficial for LongBench KV-cache generation at 2.5-bit and increased measured storage ratio.
- Decode-FP16 slightly improved contains-answer accuracy on the 20-example slice, but LongBench score did not improve and storage ratio increased. It is not a reportable gain.

### R10: Centered And Structured-Rotation MSE Quantizers

Purpose: test changes closer to TurboQuant's core transform while keeping the 2.5-bit budget fixed.

Candidates:

- `centered_mse`: subtract a per-segment sequence mean, quantize residuals with TurboQuant MSE, and store the mean as metadata.
- `hadamard_mse`: replace TurboQuant's random orthogonal rotation with a normalized Hadamard rotation plus random sign flips. This adds no cache metadata and keeps the same Lloyd-Max scalar codebook.

2WikiMQA results:

| Method | Split | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 2.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 0:20 | 32.76 | 0.30 | 0.1720 | +0.00 |
| centered_mse | 0:20 | 40.62 | 0.30 | 0.1730 | +7.86 |
| hadamard_mse | 0:20 | 34.70 | 0.35 | 0.1720 | +1.94 |
| TurboQuant 2.5 | full | 37.96 | 0.365 | 0.1720 | +0.00 |
| centered_mse | full | 31.18 | 0.290 | 0.1730 | -6.78 |
| hadamard_mse | full | 30.12 | 0.280 | 0.1720 | -7.84 |

Artifacts:

- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_2p5_centered_mse.json`
- `reproduce/incremental/landscape2_2wikimqa_full_turboquant_2p5_centered_mse.json`
- `reproduce/incremental/landscape2_2wikimqa_0_20_turboquant_2p5_hadamard_mse.json`
- `reproduce/incremental/landscape2_2wikimqa_full_turboquant_2p5_hadamard_mse.json`

Analysis:

- Both candidates showed positive 20-example slice behavior but failed the full 200-example 2WikiMQA validation.
- The failure reinforces the gate adopted after R7: no method should be treated as a contribution without full-task validation.
- No `landscape2` method is frozen from R8-R10.

### R11: Learned Block Codebook And Unit-Norm Reconstruction

Purpose: test quantizer-level changes rather than bit/layer redistribution. The goal was to reduce quantization distortion at the same TurboQuant bit budgets and then verify whether the lower distortion improves LongBench.

Implemented candidates:

- `learned_mse_block2`: a true 2D Lloyd codebook trained on the two-coordinate marginal of a uniform sphere source. This differs from the earlier `mse_block2`, whose Cartesian-product codebook decomposed into scalar quantization and exactly matched scalar MSE.
- `unit_mse`: scalar TurboQuant MSE with an extra dequantization projection that restores each reconstructed direction to unit norm before multiplying by the stored original vector norm.
- `learned_unit_mse_block2`: learned 2D block codebook plus the same unit-norm reconstruction projection.

Stage-1 real K/V MSE on `longbench_2wikimqa[0:2]`, max 1024 input tokens:

| Target | Scalar MSE | Cartesian block2 MSE | Learned block2 MSE | Learned block2 reduction |
| --- | ---: | ---: | ---: | ---: |
| Key 2-bit | 139.40 | 139.40 | 127.56 | 8.49% |
| Value 2-bit | 2.900 | 2.900 | 2.582 | 10.98% |
| Key 3-bit | 40.67 | 40.67 | 35.40 | 12.96% |
| Value 3-bit | 0.848 | 0.848 | 0.724 | 14.55% |

2WikiMQA LongBench results:

| Method | Bits | Split | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs matching TQ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| TurboQuant | 2.5 | 0:20 | 32.76 | 0.30 | 0.1720 | +0.00 |
| learned_mse_block2 | 2.5 | 0:20 | 18.33 | 0.15 | 0.1720 | -14.42 |
| learned_unit_mse_block2 | 2.5 | 0:20 | 33.87 | 0.30 | 0.1720 | +1.11 |
| unit_mse | 2.5 | 0:20 | 35.95 | 0.30 | 0.1720 | +3.19 |
| TurboQuant | 2.5 | full | 37.96 | 0.365 | 0.1720 | +0.00 |
| unit_mse | 2.5 | full | 35.42 | 0.340 | 0.1720 | -2.54 |
| TurboQuant | 3.5 | 0:20 | 41.87 | 0.35 | 0.2345 | +0.00 |
| learned_mse_block2 | 3.5 | 0:20 | 34.29 | 0.35 | 0.2345 | -7.58 |
| TurboQuant | 3.5 | full | 44.47 | 0.425 | 0.2345 | +0.00 |
| unit_mse | 3.5 | full | 45.04 | 0.450 | 0.2345 | +0.57 |

Artifacts:

- `reproduce/incremental/block_mse_learned_compare_2wikimqa_0_2_1024.json`
- `reproduce/incremental/learned_block2_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/learned_block2_2wikimqa_turboquant_3p5_0_20.json`
- `reproduce/incremental/learned_unit_block2_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/unit_mse_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/unit_mse_2wikimqa_turboquant_2p5_full.json`
- `reproduce/incremental/unit_mse_2wikimqa_turboquant_3p5_full.json`

Analysis:

- The learned 2D block codebook is a real quantizer-level improvement in Stage-1 MSE, but lower MSE did not translate to LongBench gains. Without unit-norm projection it severely hurt generation quality, likely because the block centroids shrink the reconstructed unit-vector norm and perturb K/V scale.
- Unit-norm reconstruction repaired the learned-block slice regression and gave a positive scalar slice, but the scalar `unit_mse` candidate failed full 2WikiMQA at 2.5-bit despite a small full-task gain at 3.5-bit.
- This is not a valid `landscape2` contribution because the 2.5-bit original-budget result regresses. The useful mechanism lesson is that norm preservation matters, but the next candidate must target attention-score/value-output error directly rather than only Euclidean MSE.

### R12: Side-Specific Norm Projection And LSQ Reconstruction Scale

Purpose: diagnose whether R11's 2.5-bit full-task regression came from applying norm-preserving reconstruction to K, V, or both, and test a scale-only least-squares reconstruction correction that keeps the same packed payload shape.

Implemented candidates:

- `unit_mse` on key only and value only.
- `lsq_mse`: stores the least-squares scalar gain for each quantized direction instead of the original input norm.
- `lsq_unit_mse`: combines LSQ gain with unit-norm direction projection. In practice this matched `unit_mse` on the tested slice.

2WikiMQA 2.5-bit slice results:

| Method | Split | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 2.5 slice |
| --- | --- | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 0:20 | 32.76 | 0.30 | 0.1720 | +0.00 |
| key unit / value mse | 0:20 | 30.95 | 0.30 | 0.1720 | -1.81 |
| key mse / value unit | 0:20 | 27.20 | 0.25 | 0.1720 | -5.56 |
| lsq_mse | 0:20 | 28.87 | 0.25 | 0.1720 | -3.89 |
| lsq_unit_mse | 0:20 | 35.95 | 0.30 | 0.1720 | +3.19 |

Additional 3.5-bit slice:

| Method | Split | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 3.5 slice |
| --- | --- | ---: | ---: | ---: | ---: |
| TurboQuant 3.5 | 0:20 | 41.87 | 0.35 | 0.2345 | +0.00 |
| learned_unit_mse_block2 | 0:20 | 45.54 | 0.45 | 0.2345 | +3.67 |

Artifacts:

- `reproduce/incremental/unit_key_mse_value_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/mse_key_unit_value_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/lsq_mse_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/lsq_unit_mse_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/learned_unit_block2_2wikimqa_turboquant_3p5_0_20.json`

Analysis:

- Side-specific `unit_mse` does not explain the full-task failure: key-only and value-only both underperform the slice baseline, with value-side projection especially harmful.
- LSQ scaling also does not provide a 2.5-bit path; the only positive LSQ variant collapses to the already-failed `unit_mse` behavior.
- The 3.5-bit learned-unit slice is promising but insufficient as a main contribution because the user-requested contribution must improve the original TurboQuant budgets, especially 2.5-bit. This branch is deprioritized until a 2.5-bit mechanism works.

### R13: Attention-Error Check For Norm/Block Quantizers

Purpose: verify whether the quantizer-level variants reduce attention-score/probability error before spending full-task compute.

Probe setup:

```bash
conda run -n turboquant python scripts/probe_attention_error.py \
  --dataset-key longbench_2wikimqa \
  --start-index 0 \
  --max-examples 2 \
  --max-input-tokens 2048 \
  --device cuda:5 \
  --methods turboquant:2.5,turboquant:2.5:unit_mse:unit_mse,turboquant:2.5:learned_unit_mse_block2:learned_unit_mse_block2,turboquant:2.5:lsq_mse:lsq_mse \
  --output reproduce/incremental/attention_error_quantizer_variants_2wikimqa_0_2_2048.json
```

Average attention diagnostics:

| Method | Score MAE | Score RMSE | Prob L1 | Top-1 match | Key rel L2 | Value rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 0.1416 | 0.2040 | 0.3574 | 0.5918 | 0.2291 | 0.2610 |
| unit_mse | 0.1267 | 0.1790 | 0.3510 | 0.6094 | 0.2292 | 0.2624 |
| learned_unit_mse_block2 | 0.1189 | 0.1672 | 0.3330 | 0.6050 | 0.2168 | 0.2495 |
| lsq_mse | 0.1338 | 0.1910 | 0.3407 | 0.6050 | 0.2269 | 0.2587 |

Full 2WikiMQA validation:

| Method | Bits | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 2.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant | 2.5 | 37.96 | 0.365 | 0.1720 | +0.00 |
| unit_mse | 2.5 | 35.42 | 0.340 | 0.1720 | -2.54 |
| learned_unit_mse_block2 | 2.5 | 35.27 | 0.345 | 0.1720 | -2.69 |

Artifacts:

- `reproduce/incremental/attention_error_quantizer_variants_2wikimqa_0_2_2048.json`
- `reproduce/incremental/learned_unit_block2_2wikimqa_turboquant_2p5_full.json`

Analysis:

- The attention-error proxy correctly identified that `learned_unit_mse_block2` improves score/probability error on a small diagnostic set, but this still did not generalize to full 2WikiMQA accuracy.
- This result invalidates using small attention-error probes as the sole go/no-go criterion for a reportable method. It remains useful for debugging mechanisms, but full-task validation is mandatory.
- No norm-preserving or learned-block quantizer variant currently provides a 2.5-bit improvement over TurboQuant.

### R14: Global Norm-Gain Correction

Purpose: correct the systematic norm shrinkage of TurboQuant's scalar Lloyd reconstruction without storing extra metadata. If the scalar codebook has per-coordinate distortion `D`, the expected squared norm of the reconstructed unit vector is approximately `1 - dD`. The candidate `gain_mse` multiplies the reconstructed direction by `(1 - dD)^(-1/2)` before applying the stored input norm. `dot_gain_mse` uses the stronger `(1 - dD)^(-1)` correction.

Implemented candidates:

- `gain_mse`: global norm-preserving gain for K and V.
- `dot_gain_mse`: stronger dot-product-preserving gain for K and V.
- `key gain / value mse`: apply global gain only to K.

2WikiMQA 2.5-bit slice screening:

| Method | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 2.5 slice |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 32.76 | 0.30 | 0.1720 | +0.00 |
| gain_mse K/V | 38.04 | 0.35 | 0.1720 | +5.28 |
| dot_gain_mse K/V | 32.20 | 0.30 | 0.1720 | -0.56 |
| gain key / mse value | 38.87 | 0.35 | 0.1720 | +6.11 |
| mse key / gain value | 35.71 | 0.40 | 0.1720 | +2.96 |

Full MultiQA 2.5-bit validation:

| Method | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 37.96 | 44.83 | 25.24 | 36.01 | 0.1719 |
| gain_mse K/V | 38.49 | 46.49 | 24.68 | 36.55 | 0.1719 |
| gain key / mse value | 36.16 | 48.74 | 22.60 | 35.83 | 0.1719 |

2WikiMQA 3.5-bit no-regression check:

| Method | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 3.5 |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant 3.5 | 44.47 | 0.425 | 0.2345 | +0.00 |
| gain_mse K/V | 42.60 | 0.415 | 0.2345 | -1.87 |

Artifacts:

- `reproduce/incremental/gain_mse_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/dot_gain_mse_2wikimqa_turboquant_2p5_0_20.json`
- `reproduce/incremental/gain_mse_2wikimqa_turboquant_2p5_full.json`
- `reproduce/incremental/gain_mse_hotpotqa_turboquant_2p5_full.json`
- `reproduce/incremental/gain_mse_musique_turboquant_2p5_full.json`
- `reproduce/incremental/gain_mse_2wikimqa_turboquant_3p5_full.json`
- `reproduce/incremental/gain_key_mse_value_2wikimqa_turboquant_2p5_full.json`
- `reproduce/incremental/gain_key_mse_value_hotpotqa_turboquant_2p5_full.json`
- `reproduce/incremental/gain_key_mse_value_musique_turboquant_2p5_full.json`

Analysis:

- `gain_mse` is the first candidate with a full-category 2.5-bit positive result: MultiQA average improves from `36.01` to `36.55` at the same measured cache ratio.
- The gain is modest (`+0.54` MultiQA points) and not uniform: 2WikiMQA and HotpotQA improve, MuSiQue regresses slightly.
- The same gain correction regresses at 3.5-bit on 2WikiMQA (`-1.87`), so it should be treated as a 2.5-bit-only low-bit correction rather than a universal replacement.
- Key-only gain is not stable: it improves HotpotQA strongly but regresses 2WikiMQA and MuSiQue enough to reduce MultiQA average.
- Verdict: promising but not yet sufficient for `landscape2`. It is method-level and reproducible, but the effect size and 3.5-bit regression are not strong enough for the requested contribution. Next work should make the gain conditional on bit budget or layer sensitivity so it applies only where norm shrinkage is harmful.

### R15: Low-Bit-Only Norm-Gain Correction

Purpose: retain R14's low-bit benefit while avoiding the observed 3.5-bit regression. `lowbit_gain_mse` applies the global norm-gain correction only to 2-bit subsegments and leaves 3-bit/4-bit subsegments unchanged. This is a conditional variant of the same norm-shrinkage correction, not a bit-budget search.

Implementation:

- Added quantizer kind `lowbit_gain_mse`.
- For a 2.5-bit effective segment, only the regular 2-bit channels receive the gain; 3-bit outlier channels use the original TurboQuant MSE reconstruction.
- For a 3.5-bit effective segment, regular 3-bit and outlier 4-bit channels are both unchanged, so the method should reduce exactly to reproduced TurboQuant.

Full MultiQA 2.5-bit validation:

| Method | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 37.96 | 44.83 | 25.24 | 36.01 | 0.1719 |
| gain_mse K/V | 38.49 | 46.49 | 24.68 | 36.55 | 0.1719 |
| lowbit_gain_mse K/V | 37.21 | 48.08 | 24.22 | 36.50 | 0.1719 |

2WikiMQA 3.5-bit no-regression check:

| Method | LongBench score | Contains-answer acc | Avg cache ratio | Delta vs TQ 3.5 |
| --- | ---: | ---: | ---: | ---: |
| TurboQuant 3.5 | 44.47 | 0.425 | 0.2345 | +0.00 |
| gain_mse K/V | 42.60 | 0.415 | 0.2345 | -1.87 |
| lowbit_gain_mse K/V | 44.47 | 0.425 | 0.2345 | +0.00 |

Artifacts:

- `reproduce/incremental/lowbit_gain_mse_2wikimqa_turboquant_2p5_full.json`
- `reproduce/incremental/lowbit_gain_mse_hotpotqa_turboquant_2p5_full.json`
- `reproduce/incremental/lowbit_gain_mse_musique_turboquant_2p5_full.json`
- `reproduce/incremental/lowbit_gain_mse_2wikimqa_turboquant_3p5_full.json`

Analysis:

- The conditional gain successfully removes the 3.5-bit regression: on 2WikiMQA it exactly matches TurboQuant 3.5 because no 2-bit subsegments exist at the 3.5-bit effective budget.
- It preserves a modest 2.5-bit MultiQA gain (`+0.50` average), similar to full `gain_mse` (`+0.54`), but the per-task pattern is still not robust: HotpotQA improves strongly, 2WikiMQA and MuSiQue regress.
- Verdict: better than R14 as a 2.5/3.5-compatible candidate, but still not strong enough to freeze as `landscape2`. It provides a plausible component for a future adaptive method, not a complete reportable contribution yet.

### R16: 3.5-bit Norm/Block Candidate Validation

Purpose: find a 3.5-bit companion to the 2.5-bit low-bit gain correction. Candidate methods were selected from earlier positive 2WikiMQA 3.5 signals.

Full 3.5-bit MultiQA validation:

| Method | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 3.5 | 44.47 | 54.65 | 30.01 | 43.04 | 0.2344 |
| unit_mse 3.5 | 45.04 | 52.64 | 29.59 | 42.42 | 0.2345 |

Additional 2WikiMQA 3.5 check:

| Method | 2WikiMQA | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: |
| learned_unit_mse_block2 | 45.24 | 0.460 | 0.2345 |

Artifacts:

- `reproduce/incremental/unit_mse_2wikimqa_turboquant_3p5_full.json`
- `reproduce/incremental/unit_mse_hotpotqa_turboquant_3p5_full.json`
- `reproduce/incremental/unit_mse_musique_turboquant_3p5_full.json`
- `reproduce/incremental/learned_unit_block2_2wikimqa_turboquant_3p5_full.json`

Analysis:

- `unit_mse` does not generalize at 3.5-bit: it improves 2WikiMQA but regresses HotpotQA and MuSiQue, lowering MultiQA average by `-0.62`.
- `learned_unit_mse_block2` has a stronger 2WikiMQA 3.5 result than `unit_mse`, but given the full MultiQA failure of `unit_mse` and the earlier 2.5-bit full failure of learned block variants, it is not expanded yet.
- No current norm/block reconstruction candidate gives a complete 3.5-bit improvement over TurboQuant.

### R17: Value-Only Low-Bit Norm-Gain Correction

Purpose: test whether the instability in R15 comes from applying low-bit norm-gain to keys, values, or both. This block applies `lowbit_gain_mse` only to values and keeps keys as the reproduced TurboQuant MSE quantizer.

Full MultiQA 2.5-bit validation:

| Method | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 37.96 | 44.83 | 25.24 | 36.01 | 0.1719 |
| lowbit_gain_mse K/V | 37.21 | 48.08 | 24.22 | 36.50 | 0.1719 |
| mse key / lowbit_gain value | 35.67 | 44.11 | 21.57 | 33.78 | 0.1719 |

Artifacts:

- `reproduce/incremental/mse_key_lowbit_gain_value_2wikimqa_turboquant_2p5_full.json`
- `reproduce/incremental/mse_key_lowbit_gain_value_hotpotqa_turboquant_2p5_full.json`
- `reproduce/incremental/mse_key_lowbit_gain_value_musique_turboquant_2p5_full.json`

Analysis:

- Value-only low-bit gain is clearly worse than both TurboQuant and the K/V low-bit-gain candidate.
- The small positive signal in R15 is not value-side alone. If norm-gain remains useful, it needs layer-wise or query/attention-aware conditioning rather than target-side-only gating.
### R18: Error-Gain and Sensitivity-Weighted Outlier Selection

Purpose: replace TurboQuant's dynamic absolute-mean outlier channel selection with error-aware channel selection at the same 2.5/3.5 effective bit budgets.

Full MultiQA validation:

| Method | Bits | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant | 2.5 | 37.96 | 44.83 | 25.24 | 36.01 | 0.1719 |
| error_gain | 2.5 | 30.36 | 46.49 | 23.23 | 33.36 | 0.1719 |
| TurboQuant | 3.5 | 44.47 | 54.65 | 30.01 | 43.04 | 0.2344 |
| error_gain | 3.5 | 39.79 | 52.73 | 30.86 | 41.13 | 0.2345 |

Artifacts:

- `reproduce/incremental/error_gain_2wikimqa_turboquant_2p5_full.json`
- `reproduce/incremental/error_gain_hotpotqa_turboquant_2p5_full.json`
- `reproduce/incremental/error_gain_musique_turboquant_2p5_full.json`
- `reproduce/incremental/error_gain_2wikimqa_turboquant_3p5_full.json`
- `reproduce/incremental/error_gain_hotpotqa_turboquant_3p5_full.json`
- `reproduce/incremental/error_gain_musique_turboquant_3p5_full.json`

Analysis: `error_gain` produced a strong 20-example slice signal but failed full validation. The method overfits the small slice and regresses both 2.5-bit and 3.5-bit MultiQA averages, so it is not a reportable increment.

Additional sensitivity-weighted variant:

- Implemented `sensitivity_error_gain`, using calibration-time key/query and value activation sensitivity to weight the quantization-error gain.
- Calibration files: `reproduce/incremental/sensitivity_scores_{2wikimqa,hotpotqa,musique}_0_8_4096.json` and merged `reproduce/incremental/sensitivity_scores_multiqa_0_8_each_4096.json`.
- 20-example slice was not stable enough for full validation: 2Wiki `33.87`, Hotpot `35.17`, MuSiQue `33.17` versus TurboQuant slice `32.76`, `36.00`, `25.00`.

Verdict: failed. Error-aware outlier selection alone is not a stable contribution.

### R19: Probe-Gated Low-Bit Norm Gain

Purpose: keep the low-bit norm-gain correction only on layers where attention-error probes indicate an improvement over original TurboQuant. This is a probe-derived layer gate, not a manual late-layer schedule.

Probe sources:

- `reproduce/incremental/probe_lowbit_gain_2wikimqa_0_4_2048.json`
- `reproduce/incremental/probe_lowbit_gain_hotpotqa_0_4_2048.json`
- `reproduce/incremental/probe_lowbit_gain_musique_0_4_2048.json`

Generated schedule: `reproduce/incremental/layer_quantizer_schedule_probe_gated_lowbit_gain_multiqa.json`, enabling layers `[1, 2, 3, 4, 5, 6, 7, 29, 31]`.

20-example MultiQA slice:

| Method | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 32.76 | 36.00 | 25.00 | 31.25 | 0.1719 |
| probe-gated lowbit_gain | 26.87 | 40.75 | 25.00 | 30.87 | 0.1719 |

Verdict: failed. The probe gate improves Hotpot but damages 2Wiki enough that full validation is not justified.

### R20: Per-Head Outlier Channel Selection

Purpose: allow each KV head to select its own high-bit outlier channels at the same effective bit budget. This changes outlier assignment granularity from layer-global to head-local while keeping the same nominal regular/outlier bit split.

Implemented policies:

- `head_dynamic_absmean`
- `head_error_gain`

20-example slice screening:

| Method | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 32.76 | 36.00 | 25.00 | 31.25 | 0.1719 |
| head_dynamic_absmean K/V | 35.48 | 35.17 | 29.43 | 33.36 | 0.1724 |
| head_dynamic_absmean key only | 28.87 | 26.52 | 25.43 | 26.94 | 0.1722 |
| head_dynamic_absmean value only | 36.96 | 28.25 | 26.50 | 30.57 | 0.1722 |

Full MultiQA 2.5-bit validation:

| Method | 2WikiMQA | HotpotQA | MuSiQue | MultiQA Avg | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 2.5 | 37.96 | 44.83 | 25.24 | 36.01 | 0.1719 |
| head_dynamic_absmean K/V | 33.09 | 46.57 | 22.79 | 34.15 | 0.1725 |

Artifacts:

- `reproduce/incremental/head_dynamic_absmean_2wikimqa_turboquant_2p5_full.json`
- `reproduce/incremental/head_dynamic_absmean_hotpotqa_turboquant_2p5_full.json`
- `reproduce/incremental/head_dynamic_absmean_musique_turboquant_2p5_full.json`

Analysis: the 20-example signal was a false positive. Full validation shows a MultiQA average regression, and the per-head metadata slightly increases measured cache ratio. 3.5-bit expansion is not justified.

Verdict: failed as a reportable contribution.


### R13: Low-Bit Norm-Gain Correction Full Table Validation
Purpose: test a quantizer-level change at the original TurboQuant 2.5-bit budget. The method applies norm-shrinkage correction only to 2-bit regular subsegments in the fractional 2.5-bit cache. At 3.5-bit it is identity because no 2-bit subsegment is used.
Full 16-task official-compatible results:
| Category | TurboQuant 2.5 | LowBit-Gain 2.5 | Delta | TurboQuant 3.5 |
| --- | ---: | ---: | ---: | ---: |
| SingleQA | 38.61 | 39.76 | +1.16 | 42.73 |
| MultiQA | 36.01 | 36.50 | +0.50 | 43.04 |
| Summarization | 27.54 | 27.71 | +0.17 | 28.72 |
| Few shot | 67.12 | 67.48 | +0.36 | 68.59 |
| Synthetic | 48.22 | 50.28 | +2.05 | 52.06 |
| Code | 55.02 | 53.53 | -1.49 | 61.16 |
| Average | 45.42 | 45.88 | +0.46 | 49.38 |

Analysis:

- This is a real method-level change and it has complete 16-task 2.5-bit results at the same measured cache ratio.
- It improves the Table-1 category macro average by +0.46, with gains on SingleQA, MultiQA, Summarization, Few shot, and Synthetic.
- It regresses Code by -1.49 because `repobench-p` drops by -3.58, despite `lcc` improving by +0.60. This makes it a complete candidate but not yet strong enough to freeze as the final incremental contribution.
- 3.5-bit behavior is identity-by-construction and uses the reproduced TurboQuant 3.5-bit results.

Artifacts:

- `reproduce/incremental/lowbit_gain_mse_table1_corrected_official_comparison.json`
- `reproduce/runs/incremental/lowbit_gain_mse_*_turboquant_2p5_full.jsonl`

### R14: Conservative Low-Bit Gain Variants And Decode Precision

Purpose: reduce the `repobench-p` regression from R13 without changing the 2.5-bit budget.

Candidates tested:

- `lowbit_clipped_gain_mse`: per-vector LSQ-clipped gain in [1, theoretical norm gain].
- `lowbit_half_gain_mse`: fixed half-strength norm gain.
- `lowbit_selected_gain_mse`: apply theoretical gain only when it reduces per-vector reconstruction error.
- Decode-FP16: quantize prefill KV but keep decode-step KV unquantized.
- Key-only LowBit-Gain: apply low-bit gain to K only, keep V as TurboQuant MSE.

Key validation results:

| Candidate | Split | HotpotQA | Repobench-p | Decision |
| --- | --- | ---: | ---: | --- |
| LowBit-Gain | 0:50 | 43.84 | 54.30 | baseline candidate |
| Decode-FP16 + LowBit-Gain | 0:50 | 44.49 | 53.74 | not better on Code |
| Half-Gain | 0:50 | 38.11 | 51.30 | worse than LowBit-Gain |
| Selected-Gain | 0:50 | 40.79 | 48.56 | worse than LowBit-Gain |
| Key-only LowBit-Gain | full selected tasks | 48.17 | 49.81 | improves QA but still regresses Code |

Analysis:

- Weakening gain reduces the positive QA effect and does not solve the Code failure.
- Decode-FP16 does not recover Code quality, so the regression is not mainly from quantizing generation-time KV.
- Key-only LowBit-Gain improves 2WikiMQA and HotpotQA full-task scores but still regresses full `repobench-p`; it is not a reportable final contribution.
- Per-example analysis on `repobench-p` shows strong complementarity between TurboQuant and gain variants, but no answer-free gate has been validated across tasks.

### R15: Prompt-Length-Gated LowBit-Gain

Purpose: avoid applying low-bit norm-gain correction to short-context cases where it is less reliable, using only an answer-free prompt-length signal. The gate uses TurboQuant MSE by default and switches K/V quantizers to `lowbit_gain_mse` when `prompt_tokens > 5856`.

Implementation: added `--long-prompt-threshold`, `--long-prompt-key-quantizer`, and `--long-prompt-value-quantizer` to `experiments/longbench/run_full_cache_eval.py`.

Full-task validation on representative tasks:

| Task | TurboQuant 2.5 | LowBit-Gain 2.5 | Prompt-gated | Gate delta vs TQ |
| --- | ---: | ---: | ---: | ---: |
| 2WikiMQA | 37.96 | 37.21 | 39.76 | +1.80 |
| Qasper | 42.44 | 43.00 | 44.40 | +1.97 |
| MuSiQue | 25.24 | 24.22 | 24.22 | -1.01 |
| Repobench-p | 53.00 | 49.42 | 51.29 | -1.71 |

Analysis:

- The gate is answer-free and improves over pure LowBit-Gain on tasks where some short prompts should stay with the TurboQuant baseline.
- It still inherits LowBit-Gain's full-task regression on all-long-prompt MuSiQue and partially on Repobench-p.
- This is a methodologically valid direction but not yet a reportable final contribution because it does not consistently improve the original 2.5-bit baseline across the validation tasks.

### R16: Online Attention-Error Outlier Channel Selection

Purpose: replace TurboQuant's dynamic absolute-mean outlier channel selection with an online signal that directly estimates quantization-induced attention error during prefill. The method keeps the same 2.5-bit fractional budget: regular channels remain 2-bit and selected outlier channels remain 3-bit. It does not use task labels or answers.

Implementation:

- Added outlier policy `attention_error_gain`.
- For each prefill layer, quantize the candidate KV tensor with the regular and outlier bitwidths, estimate per-channel error reduction, and weight that reduction by the current prompt's final-query attention distribution.
- Tested K/V, key-only, and value-only variants.

20-example screening:

| Variant | 2WikiMQA | HotpotQA | MuSiQue | Decision |
| --- | ---: | ---: | ---: | --- |
| TurboQuant 2.5 slice reference | 32.76 | 36.00 | 25.00 | baseline |
| attention-error K/V | 29.29 | 32.50 | 31.43 | fail: QA regression |
| attention-error key only | 40.25 | 31.47 | 27.93 | fail: HotpotQA regression |
| attention-error value only | 40.00 | 31.60 | 29.00 | fail: HotpotQA regression |

Artifacts:

- `reproduce/runs/incremental/attention_error_gain_*_turboquant_2p5_0_20.jsonl`
- `reproduce/runs/incremental/attention_error_gain_*_turboquant_2p5_0_20.summary.json`

Analysis: the attention-error signal can improve individual tasks but is not stable across the MultiQA screening block. It is also significantly slower than the default absolute-mean policy because it performs extra probe quantization per layer. Full validation is not justified.

### R17: Content-Adaptive LowBit-Gain

Purpose: retain the complete 16-task LowBit-Gain gains while avoiding its observed code-completion regression. The method keeps the original 2.5-bit TurboQuant storage budget and changes only the quantizer selection policy:

- Use `lowbit_gain_mse` when `prompt_tokens > 6144` and the prompt is not code-completion-like.
- Otherwise use the reproduced TurboQuant `mse` quantizer.
- Code-completion prompts are detected from the prompt prefix, primarily the LongBench phrase `Please complete the code given below`.

Implementation:

- Added `--long-prompt-exclude-code` to `experiments/longbench/run_full_cache_eval.py`.
- Added reusable prompt detector `is_code_completion_prompt`.
- Added `scripts/build_content_adaptive_lowbit_table.py` to build the full Table-1 comparison from paired TurboQuant and LowBit-Gain JSONL outputs.

Full 16-task comparison, built from complete reproduced TurboQuant 2.5 and complete LowBit-Gain outputs:

| Category | TurboQuant 2.5 | LowBit-Gain 2.5 | Content-Adaptive 2.5 | Delta vs TQ2.5 | Activation |
| --- | ---: | ---: | ---: | ---: | ---: |
| SingleQA | 38.61 | 39.76 | 40.32 | +1.72 | 0.60 |
| MultiQA | 36.01 | 36.50 | 37.51 | +1.51 | 0.82 |
| Summarization | 27.54 | 27.71 | 27.62 | +0.09 | 0.58 |
| Few shot | 67.12 | 67.48 | 67.77 | +0.65 | 0.68 |
| Synthetic | 48.22 | 50.28 | 50.29 | +2.07 | 0.99 |
| Code | 55.02 | 53.53 | 55.02 | +0.00 | 0.00 |
| Average | 45.42 | 45.88 | 46.42 | +1.00 | - |

Task-level table:

| Dataset | Category | TurboQuant 2.5 | LowBit-Gain 2.5 | Content-Adaptive 2.5 | Delta vs TQ2.5 | Activation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| narrativeqa | SingleQA | 25.35 | 27.34 | 27.34 | +2.00 | 1.00 |
| qasper | SingleQA | 42.44 | 43.00 | 44.44 | +2.01 | 0.21 |
| multifieldqa_en | SingleQA | 48.03 | 48.95 | 49.18 | +1.15 | 0.59 |
| hotpotqa | MultiQA | 44.83 | 48.08 | 48.41 | +3.59 | 0.92 |
| 2wikimqa | MultiQA | 37.96 | 37.21 | 39.91 | +1.95 | 0.55 |
| musique | MultiQA | 25.24 | 24.22 | 24.22 | -1.01 | 1.00 |
| gov_report | Summarization | 32.02 | 32.46 | 32.41 | +0.40 | 0.76 |
| qmsum | Summarization | 24.38 | 24.13 | 24.18 | -0.20 | 0.91 |
| multi_news | Summarization | 26.22 | 26.53 | 26.27 | +0.06 | 0.06 |
| trec | Few shot | 70.50 | 70.00 | 71.50 | +1.00 | 0.56 |
| triviaqa | Few shot | 89.39 | 90.06 | 89.48 | +0.09 | 0.80 |
| samsum | Few shot | 41.47 | 42.38 | 42.32 | +0.85 | 0.67 |
| passage_retrieval_en | Synthetic | 93.00 | 96.00 | 96.00 | +3.00 | 1.00 |
| passage_count | Synthetic | 3.45 | 4.56 | 4.59 | +1.14 | 0.98 |
| lcc | Code | 57.04 | 57.64 | 57.04 | +0.00 | 0.00 |
| repobench-p | Code | 53.00 | 49.42 | 53.00 | +0.00 | 0.00 |

Real runner validation on representative full tasks:

| Dataset | Full examples | Runner score | Table score | Delta vs TQ2.5 | Active quantizer counts |
| --- | ---: | ---: | ---: | ---: | --- |
| 2wikimqa | 200 | 39.91 | 39.91 | +1.95 | 110 lowbit / 90 mse |
| hotpotqa | 200 | 48.41 | 48.41 | +3.59 | 183 lowbit / 17 mse |
| musique | 200 | 24.22 | 24.22 | -1.01 | 200 lowbit / 0 mse |
| qasper | 200 | 44.44 | 44.44 | +2.01 | 42 lowbit / 158 mse |
| lcc smoke | 1 | - | - | - | code prompt kept `mse` |

Artifacts:

- `reproduce/incremental/content_adaptive_lowbit_gain_threshold_6144_table1_comparison.json`
- `reproduce/runs/incremental/content_adaptive_lowbit_gain_{2wikimqa,hotpotqa,musique,qasper}_turboquant_2p5_full.jsonl`
- `reproduce/runs/incremental/content_adaptive_lowbit_gain_{2wikimqa,hotpotqa,musique,qasper}_turboquant_2p5_full.official.json`

Analysis:

- This is a method-level candidate, not a bit-budget search: it keeps the 2.5-bit TurboQuant storage design and selects between original MSE reconstruction and low-bit norm-gain reconstruction using prompt-visible content.
- The full-table average improves from `45.42` to `46.42`, a `+1.00` LongBench point gain over reproduced TurboQuant 2.5 and `+0.55` over always-on LowBit-Gain.
- The code regression from always-on LowBit-Gain is removed by construction and verified by prompt scanning: all `lcc` and `repobench-p` examples are detected as code-completion prompts.
- The remaining weakness is MuSiQue, where all prompts are long non-code prompts and LowBit-Gain still regresses by `-1.01`. Further work should look for an answer-free signal that distinguishes MuSiQue-style multi-hop retrieval from HotpotQA/2WikiMQA before treating this as the final contribution.

### R18: Structure-Adaptive LowBit-Gain

Purpose: fix R17's MuSiQue failure while keeping the same method family and 2.5-bit budget. The new gate remains answer-free and prompt-only:

- Use `lowbit_gain_mse` when `prompt_tokens > 6144`.
- Exclude code-completion prompts.
- For Passage-style prompts, exclude examples with more than 15 passages or more than 5 question marks.
- Otherwise use reproduced TurboQuant `mse`.

Rationale: R17 showed that LowBit-Gain helps many long prompts but hurts some dense multi-hop passage prompts. Passage count and question-mark count are cheap prompt-structure signals available before generation.

Full 16-task comparison:

| Category | TurboQuant 2.5 | LowBit-Gain 2.5 | Structure-Adaptive 2.5 | Delta vs TQ2.5 | Activation |
| --- | ---: | ---: | ---: | ---: | ---: |
| SingleQA | 38.61 | 39.76 | 40.32 | +1.72 | 0.60 |
| MultiQA | 36.01 | 36.50 | 38.10 | +2.09 | 0.65 |
| Summarization | 27.54 | 27.71 | 27.60 | +0.06 | 0.56 |
| Few shot | 67.12 | 67.48 | 67.77 | +0.65 | 0.68 |
| Synthetic | 48.22 | 50.28 | 50.29 | +2.07 | 0.99 |
| Code | 55.02 | 53.53 | 55.02 | +0.00 | 0.00 |
| Average | 45.42 | 45.88 | 46.52 | +1.10 | - |

Task-level table:

| Dataset | Category | TurboQuant 2.5 | TurboQuant 3.5 | LowBit-Gain 2.5 | Structure-Adaptive 2.5 | Delta vs TQ2.5 | Activation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| narrativeqa | SingleQA | 25.35 | 29.25 | 27.34 | 27.34 | +2.00 | 1.00 |
| qasper | SingleQA | 42.44 | 46.32 | 43.00 | 44.44 | +2.01 | 0.21 |
| multifieldqa_en | SingleQA | 48.03 | 52.61 | 48.95 | 49.18 | +1.15 | 0.59 |
| hotpotqa | MultiQA | 44.83 | 54.65 | 48.08 | 48.69 | +3.87 | 0.80 |
| 2wikimqa | MultiQA | 37.96 | 44.47 | 37.21 | 39.04 | +1.08 | 0.44 |
| musique | MultiQA | 25.24 | 30.01 | 24.22 | 26.57 | +1.33 | 0.71 |
| gov_report | Summarization | 32.02 | 34.08 | 32.46 | 32.41 | +0.40 | 0.76 |
| qmsum | Summarization | 24.38 | 25.33 | 24.13 | 24.18 | -0.20 | 0.91 |
| multi_news | Summarization | 26.22 | 26.77 | 26.53 | 26.20 | -0.01 | 0.03 |
| trec | Few shot | 70.50 | 72.00 | 70.00 | 71.50 | +1.00 | 0.56 |
| triviaqa | Few shot | 89.39 | 91.34 | 90.06 | 89.48 | +0.09 | 0.80 |
| samsum | Few shot | 41.47 | 42.43 | 42.38 | 42.32 | +0.85 | 0.67 |
| passage_retrieval_en | Synthetic | 93.00 | 100.00 | 96.00 | 96.00 | +3.00 | 1.00 |
| passage_count | Synthetic | 3.45 | 4.12 | 4.56 | 4.59 | +1.14 | 0.98 |
| lcc | Code | 57.04 | 64.04 | 57.64 | 57.04 | +0.00 | 0.00 |
| repobench-p | Code | 53.00 | 58.29 | 49.42 | 53.00 | +0.00 | 0.00 |

Real runner validation on representative full tasks:

| Dataset | Examples | Score | Delta vs TQ2.5 | Active quantizer counts | Passage-blocked |
| --- | ---: | ---: | ---: | --- | ---: |
| 2wikimqa | 200 | 39.04 | +1.08 | 87 lowbit / 113 mse | 35 |
| hotpotqa | 200 | 48.69 | +3.87 | 160 lowbit / 40 mse | 23 |
| musique | 200 | 26.57 | +1.33 | 142 lowbit / 58 mse | 58 |
| qasper | 200 | 44.44 | +2.01 | 42 lowbit / 158 mse | 0 |

Artifacts:

- `reproduce/incremental/structure_adaptive_lowbit_gain_threshold_6144_p15_q5_table1_comparison.json`
- `reproduce/runs/incremental/structure_adaptive_lowbit_gain_{2wikimqa,hotpotqa,musique,qasper}_turboquant_2p5_full.jsonl`
- `reproduce/runs/incremental/structure_adaptive_lowbit_gain_{2wikimqa,hotpotqa,musique,qasper}_turboquant_2p5_full.official.json`
- `reproduce/incremental/prompt_gate_features_table1.json`

Analysis:

- This is currently the strongest method-level candidate: it improves the 16-task Table-1 macro average by `+1.10` over reproduced TurboQuant 2.5 at the same 2.5-bit storage budget.
- It also improves over always-on LowBit-Gain by `+0.64` and removes the Code regression by using a prompt-only code-completion detector.
- The previously failing MuSiQue task becomes positive: `25.24 -> 26.57` (`+1.33`) because 58 structurally risky examples are kept on original TurboQuant MSE.
- It remains below reproduced TurboQuant 3.5 average (`49.38`), so the contribution should be framed as improving the 2.5-bit operating point, not surpassing the 3.5-bit quality point.

Verdict: reportable method-level contribution for the 2.5-bit TurboQuant operating point. The result has a complete 16-task Table-1 comparison against reproduced TurboQuant 2.5 and TurboQuant 3.5, plus real full-run validation on the key tasks that exercise the gate.

### R19: Post-Landscape2 Search For A Stronger Unified Method

Purpose: after the prompt-gated regular-gain result, search for a stronger method-level contribution that is not just prompt gating or bit-budget search. The preferred direction was to modify TurboQuant's rotation/fractional-bit core.

Implemented and tested new rotation-domain candidates:

- `attention_rotated_outlier_mse`: keep TurboQuant's dense random rotation, but select fractional high-bit coordinates in the rotated domain using an answer-free attention proxy.
- `attention_adaptive_rotated_outlier_mse`: compare reproduced TurboQuant MSE against `attention_rotated_outlier_mse` per segment and activate the rotated-domain allocation only if the attention proxy prefers it.

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
76 passed
```

2WikiMQA results:

| Method | KV bits | Scope | Score | Delta vs TQ | Decision |
| --- | ---: | --- | ---: | ---: | --- |
| K `attention_rotated_outlier_mse` | 2.5 | stratified 80 | 39.72 | +4.17 | full validation required |
| K `attention_rotated_outlier_mse` | 3.5 | stratified 80 | 45.24 | +3.89 | full validation required |
| K `attention_rotated_outlier_mse` | 2.5 | full 200 | 34.85 | -3.11 | reject |
| K `attention_rotated_outlier_mse` | 3.5 | full 200 | 44.23 | -0.23 | reject |
| K `attention_adaptive_rotated_outlier_mse` | 2.5 | stratified 80 | 35.99 | +0.45 | weak |
| K `attention_adaptive_rotated_outlier_mse` | 3.5 | stratified 80 | 38.96 | -2.38 | reject |

Non-rotation reconstruction-selector checks:

| Method | KV bits | Scope | Score | Delta vs TQ | Decision |
| --- | ---: | --- | ---: | ---: | --- |
| K `attention_adaptive_regular_gain_mse` | 2.5 | full 200 | 40.87 | +2.91 | positive only at 2.5 |
| K `attention_adaptive_regular_gain_mse` | 3.5 | full 200 | 43.19 | -1.27 | reject as unified method |
| K `attention_auto_mse` | 2.5 | stratified 80 | 39.72 | +4.18 | positive only at 2.5 |
| K `attention_auto_mse` | 3.5 | stratified 80 | 34.51 | -6.83 | reject |
| K `auto_mse` | 2.5 | stratified 80 | 39.00 | +3.45 | positive only at 2.5 |
| K `auto_mse` | 3.5 | stratified 80 | 37.75 | -3.60 | reject |

Analysis:

- Direct attention-weighted rotated-domain fractional allocation produced a strong stratified false positive, but full 2WikiMQA invalidated it.
- The attention proxy is not reliable enough as a binary selector at 3.5-bit; both `attention_adaptive_rotated_outlier_mse` and `attention_auto_mse` over-select harmful alternatives.
- K-only attention-adaptive regular-gain is promising for 2.5-bit but fails the same-method requirement because 3.5-bit regresses on the full task.
- Existing full MultiQA scans show no completed method that is both clearly method-level and positive at both 2.5-bit and 3.5-bit without relying on prompt/bitwidth gates. The closest core-rotation method remains `bitwidth_value_hadamard_guard_b16`, but its 3.5-bit MultiQA average is negative due to MuSiQue.

Next useful direction:

Focus on a principled distortion-regime method that can explain why low-bit and high-bit regimes need different reconstruction behavior while avoiding prompt/task gates. Candidate designs should be screened on stratified 80 first, then full 2WikiMQA, before any MultiQA/Table 1 expansion.

### R20: Core-Rotation Follow-Up - Vector-Adaptive Hadamard Selector

Purpose: test a stronger Hadamard-core modification after several segment-level rotation methods produced stratified false positives. The candidate adds a per-vector selector between reproduced TurboQuant dense random-rotation MSE and outlier-aware block-Hadamard preconditioning. It is answer-free and prompt-independent: each K/V vector selects the path with lower local reconstruction MSE, and stores one selector bit per vector plus only the selected path's packed payload.

Implemented quantizer:

```text
vector_adaptive_outlier_hadamard_mse
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
79 passed
```

2WikiMQA results:

| Method | KV bits | Scope | Score | Delta vs TQ | Contains-answer acc | Avg cache ratio | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| V-only vector-adaptive Hadamard | 2.5 | stratified 80 | 33.66 | -1.88 | 0.2875 | 0.17258 | reject |
| V-only vector-adaptive Hadamard | 3.5 | stratified 80 | 40.93 | -0.42 | 0.4125 | 0.23510 | reject |
| K-only vector-adaptive Hadamard | 2.5 | stratified 80 | 36.08 | +0.54 | 0.3125 | 0.17257 | full validation required |
| K-only vector-adaptive Hadamard | 3.5 | stratified 80 | 45.03 | +3.69 | 0.4375 | 0.23508 | full validation required |
| K-only vector-adaptive Hadamard | 2.5 | full 200 | 36.45 | -1.51 | 0.3450 | 0.17257 | reject |
| K-only vector-adaptive Hadamard | 3.5 | full 200 | 43.03 | -1.44 | 0.4300 | 0.23509 | reject |

Artifacts:

- `reproduce/runs/incremental/vector_adaptive_v_hadamard_2wikimqa_turboquant_{2p5,3p5}_stratified80.jsonl`
- `reproduce/runs/incremental/vector_adaptive_k_hadamard_2wikimqa_turboquant_{2p5,3p5}_stratified80.jsonl`
- `reproduce/runs/incremental/vector_adaptive_k_hadamard_2wikimqa_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/vector_adaptive_v_hadamard_2wikimqa_turboquant_{2p5,3p5}_stratified80.json`
- `reproduce/incremental/vector_adaptive_k_hadamard_2wikimqa_turboquant_{2p5,3p5}_stratified80.json`
- `reproduce/incremental/vector_adaptive_k_hadamard_2wikimqa_turboquant_{2p5,3p5}_full.json`

Analysis:

- Per-vector local reconstruction-MSE selection is not enough to make Hadamard preconditioning stable for generation quality.
- V-only fails immediately on the stratified screen. K-only has a strong 3.5-bit stratified signal, but full 2WikiMQA reverses at both bit budgets.
- The method also adds measurable runtime overhead because it quantizes and decodes both candidate paths during prefill selection.

Decision: do not expand to MultiQA or Table 1. Future core-method attempts should avoid using local K/V reconstruction MSE as the only selector, because this failure repeats the same pattern seen in rotation-bank, calibrated-rotation, residual-Hadamard, and attention-rotated-outlier probes.

### R21: Attention-Weighted Hadamard Residual

Purpose: move away from local K/V reconstruction MSE selectors and target attention logits directly. The candidate keeps TurboQuant's dense random-rotation MSE base, but replaces the fractional high-bit outlier-channel allocation with sparse Hadamard residual signs. Unlike the earlier `hadamard_residual_mse`, residual Hadamard coefficients are selected by an answer-free prefill attention-logit sensitivity score:

```text
key_quantizer=attention_weighted_hadamard_residual_mse
value_quantizer=mse
attention_error_query_tokens=8
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
80 passed
```

2WikiMQA stratified 80 screening:

| Method | KV bits | Score | Delta vs TQ | Contains-answer acc | Avg cache ratio | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| K attention-weighted Hadamard residual | 2.5 | 33.01 | -2.53 | 0.3000 | 0.17197 | reject |
| K attention-weighted Hadamard residual | 3.5 | 40.70 | -0.64 | 0.3875 | 0.23446 | reject |

Artifacts:

- `reproduce/runs/incremental/attention_weighted_hadamard_residual_key_2wikimqa_turboquant_{2p5,3p5}_stratified80.jsonl`
- `reproduce/incremental/attention_weighted_hadamard_residual_key_2wikimqa_turboquant_{2p5,3p5}_stratified80.json`

Analysis:

- Directly weighting Hadamard residual coefficients by current prefill attention sensitivity did not stabilize the residual-correction path.
- The failure is stronger at 2.5-bit than the earlier unweighted K-only Hadamard residual stratified result, so the attention-weighted coefficient objective is not a useful replacement for high-bit TurboQuant outlier coordinates.
- This suggests that the sparse residual-sign correction changes key geometry in a way that is harmful for retrieval-style QA, even when the selected basis coefficients are query-sensitive.

Decision: do not expand to full 2WikiMQA, MultiQA, or Table 1.

### R24: Value Outlier-Only Hadamard

Purpose: test the complement of R23. This candidate keeps the low-bit regular value subspace on reproduced TurboQuant MSE and applies outlier-aware block-Hadamard only to the high-bit outlier value subspace. The goal was to check whether the useful part of V-Hadamard comes from spreading high-energy value channels while leaving the coarse low-bit subspace untouched.

Implemented quantizer:

```text
outlier_only_hadamard_mse
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
81 passed
```

2WikiMQA stratified 80 screening:

| Method | KV bits | Score | Delta vs TQ | Contains-answer acc | Avg cache ratio | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V outlier-only Hadamard | 2.5 | 31.59 | -3.96 | 0.2750 | 0.17205 | reject |
| V outlier-only Hadamard | 3.5 | 42.20 | +0.85 | 0.4250 | 0.23455 | positive but not unified |

Artifacts:

- `reproduce/runs/incremental/value_outlier_only_hadamard_b16_2wikimqa_turboquant_{2p5,3p5}_stratified80.jsonl`
- `reproduce/incremental/value_outlier_only_hadamard_b16_2wikimqa_turboquant_{2p5,3p5}_stratified80.json`

Analysis:

- Isolating Hadamard to the high-bit value outlier subspace is also harmful at 2.5-bit.
- Together with R23, this indicates that splitting the V-Hadamard transform by TurboQuant's regular/outlier subspaces does not solve the low-bit failure mode.

Decision: do not expand to full 2WikiMQA, MultiQA, or Table 1.

### R22: Late-Layer Value Hadamard

Purpose: test whether the earlier value-side outlier-Hadamard signal can be made more stable by limiting the rotation/preconditioning change to later decoder layers. The method keeps K on reproduced TurboQuant MSE for all layers, keeps V on reproduced TurboQuant MSE for layers 0-15, and applies `outlier_hadamard_mse` with block size 16 to V for layers 16-31. This is a model-internal layer rule rather than a prompt/task gate, and it changes only the value-side rotation/preconditioning path.

Run configuration:

```text
key_quantizer=mse
value_quantizer=mse
layer_value_quantizers=mse x16, outlier_hadamard_mse x16
outlier_hadamard_block_size=16
```

2WikiMQA results:

| Method | KV bits | Scope | Score | Delta vs TQ | Contains-answer acc | Avg cache ratio | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Late-layer V Hadamard | 2.5 | stratified 80 | 36.23 | +0.69 | 0.3125 | 0.17204 | full validation required |
| Late-layer V Hadamard | 3.5 | stratified 80 | 44.24 | +2.90 | 0.4250 | 0.23456 | full validation required |
| Late-layer V Hadamard | 2.5 | full 200 | 36.02 | -1.94 | 0.3350 | 0.17204 | reject |
| Late-layer V Hadamard | 3.5 | full 200 | 46.09 | +1.62 | 0.4500 | 0.23455 | positive but not unified |

Artifacts:

- `reproduce/runs/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_{2p5,3p5}_stratified80.jsonl`
- `reproduce/runs/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_{2p5,3p5}_stratified80.json`
- `reproduce/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_{2p5,3p5}_full.json`

Analysis:

- The layer-restricted value rotation improves the stratified 2Wiki screen at both bit widths and gives a strong full-task 3.5-bit gain.
- The same method regresses full 2Wiki at 2.5-bit, so it does not satisfy the required unified improvement over TurboQuant 2.5-bit and 3.5-bit.
- The failure repeats the broader pattern: V-side Hadamard preconditioning can help at higher bit width, but at 2.5-bit the low-bit value distortion is too disruptive once evaluated over the full task distribution.

Decision: do not expand to MultiQA or Table 1. Future core-method attempts should preserve the successful 3.5-bit idea only if they introduce a principled low-bit stabilizer that also passes full 2Wiki at 2.5-bit.

### R23: Value Regular-Only Hadamard

Purpose: test a more conservative value-side Hadamard variant after full value-Hadamard and late-layer value-Hadamard failed the unified 2.5/3.5 requirement. The candidate applies outlier-aware block-Hadamard only to the low-bit regular value subspace while keeping the high-bit outlier value subspace on reproduced TurboQuant MSE. The hypothesis was that preserving high-magnitude channels on the original random-rotation path would stabilize the low-bit setting.

Run configuration:

```text
key_quantizer=mse
value_quantizer=regular_outlier_hadamard_mse
outlier_hadamard_block_size=16
```

2WikiMQA stratified 80 screening:

| Method | KV bits | Score | Delta vs TQ | Contains-answer acc | Avg cache ratio | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| V regular-only Hadamard | 2.5 | 32.12 | -3.42 | 0.2625 | 0.17205 | reject |
| V regular-only Hadamard | 3.5 | 42.74 | +1.40 | 0.4125 | 0.23454 | positive but not unified |

Artifacts:

- `reproduce/runs/incremental/value_regular_outlier_hadamard_b16_2wikimqa_turboquant_{2p5,3p5}_stratified80.jsonl`
- `reproduce/incremental/value_regular_outlier_hadamard_b16_2wikimqa_turboquant_{2p5,3p5}_stratified80.json`

Analysis:

- Protecting the high-bit value outlier subspace is not enough to make Hadamard stable at 2.5-bit.
- The failure is worse than full V-Hadamard on the same screening protocol, which suggests that the low-bit regular subspace is exactly where Hadamard preconditioning is most disruptive.

Decision: do not expand to full 2WikiMQA, MultiQA, or Table 1.

### R25: Margin-Gated Value Outlier-Hadamard

Purpose: test whether the value-side outlier-Hadamard signal can be stabilized by making activation conservative at the KV-vector level. The method keeps K on reproduced TurboQuant MSE for every layer and applies the outlier-aware block-Hadamard preconditioner only to V vectors whose local reconstruction MSE improves by at least 5% versus the reproduced TurboQuant path. This is a rotation/preconditioning change inside the cache quantizer, not a prompt/task gate.

Run configuration:

```text
key_quantizer=mse
value_quantizer=margin_vector_outlier_hadamard_mse
outlier_hadamard_block_size=16
activation rule: candidate_mse < 0.95 * baseline_mse
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
84 passed
```

Full MultiQA results:

| Method | KV bits | HotpotQA | 2WikiMQA | MuSiQue | MultiQA Avg | Delta vs TQ MultiQA | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TurboQuant | 2.5 | 44.83 | 37.96 | 25.24 | 36.01 | +0.00 | baseline |
| Margin V-Hadamard | 2.5 | 49.05 | 38.88 | 23.54 | 37.16 | +1.15 | positive but incomplete |
| TurboQuant | 3.5 | 54.65 | 44.47 | 30.01 | 43.04 | +0.00 | baseline |
| Margin V-Hadamard | 3.5 | 51.99 | 45.26 | 29.81 | 42.35 | -0.69 | reject |

Artifacts:

- `reproduce/runs/incremental/margin_value_hadamard_{hotpotqa,2wikimqa,musique}_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/margin_value_hadamard_table1.md`
- `reproduce/incremental/margin_value_hadamard_table1.json`
- `reproduce/logs/margin_value_hadamard_jobs/queue_status.json`

Analysis:

- The conservative margin improves the 2.5-bit MultiQA average and keeps the 2WikiMQA gain from the earlier value-Hadamard signal.
- It does not solve the 3.5-bit stability problem: HotpotQA drops by 2.66 points and the MultiQA average is below reproduced TurboQuant.
- This rejects local reconstruction-error margin as a sufficient guard for value-side Hadamard preconditioning. The failure suggests that the useful Hadamard signal is task/context sensitive in a way not captured by per-vector value MSE alone.

Decision: do not expand to full Table 1. The next core-method attempt should avoid value-only local-MSE activation and should target attention-output or logit-level preservation more directly, while keeping the same rule for 2.5-bit and 3.5-bit.

### R26: Attention-Adaptive Value Outlier-Hadamard

Purpose: replace the local value-MSE guard from R25 with an attention-objective guard. The method keeps K on reproduced TurboQuant MSE and compares the reproduced TurboQuant V path against the outlier-Hadamard V path using the current attention-output error. It activates the Hadamard-preconditioned V path only when this attention objective improves. This remains a core cache quantizer preconditioning rule and uses the same rule at 2.5-bit and 3.5-bit.

Run configuration:

```text
key_quantizer=mse
value_quantizer=attention_adaptive_outlier_hadamard_mse
outlier_hadamard_block_size=16
activation rule: candidate_attention_error < baseline_attention_error
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
85 passed
```

Full MultiQA results:

| Method | KV bits | HotpotQA | 2WikiMQA | MuSiQue | MultiQA Avg | Delta vs TQ MultiQA | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TurboQuant | 2.5 | 44.83 | 37.96 | 25.24 | 36.01 | +0.00 | baseline |
| Attention-Adaptive V-Hadamard | 2.5 | 47.02 | 34.87 | 20.35 | 34.08 | -1.93 | reject |
| TurboQuant | 3.5 | 54.65 | 44.47 | 30.01 | 43.04 | +0.00 | baseline |
| Attention-Adaptive V-Hadamard | 3.5 | 54.53 | 44.95 | 27.11 | 42.20 | -0.85 | reject |

Artifacts:

- `reproduce/runs/incremental/attention_adaptive_value_hadamard_{hotpotqa,2wikimqa,musique}_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/attention_adaptive_value_hadamard_table1.md`
- `reproduce/incremental/attention_adaptive_value_hadamard_table1.json`
- `reproduce/logs/attention_adaptive_value_hadamard_jobs/queue_status.json`

Analysis:

- The attention-output guard reduces the HotpotQA 3.5 regression seen in R25 but badly hurts MuSiQue and 2WikiMQA at 2.5-bit.
- The result closely matches the earlier pattern that value-side Hadamard can help some retrieval-style examples but is not stable on compositional MultiQA.
- This rejects value-only Hadamard preconditioning as the next reportable contribution, even when guarded by an attention objective.

Decision: do not expand to full Table 1. Future core-method attempts should move away from value-only Hadamard and instead investigate changes that preserve K/V consistency jointly, such as joint rotated-domain bit allocation or a shared K/V rotation objective, because isolated V preconditioning repeatedly produces task-dependent failures.

### R27: Shared Rotated-Domain Outlier Coordinates

Purpose: test a core TurboQuant rotation-domain modification instead of a prompt gate or value-only Hadamard preconditioner. The method keeps TurboQuant's random orthogonal rotation and Lloyd-Max scalar codebooks, but changes fractional-bit allocation: K and V use the same shared rotation basis and the same high-bit rotated-coordinate set. The high-bit coordinate set is selected from normalized K/V reconstruction-gain scores in the rotated domain.

Run configuration:

```text
key_quantizer=shared_rotated_outlier_mse
value_quantizer=shared_rotated_outlier_mse
selection rule: top rotated coordinates by normalized key gain + normalized value gain
same rule for 2.5-bit and 3.5-bit
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
86 passed
```

Full MultiQA results:

| Method | KV bits | HotpotQA | 2WikiMQA | MuSiQue | MultiQA Avg | Delta vs TQ MultiQA | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TurboQuant | 2.5 | 44.83 | 37.96 | 25.24 | 36.01 | +0.00 | baseline |
| Shared Rotated Outlier | 2.5 | 46.65 | 31.75 | 20.45 | 32.95 | -3.06 | reject |
| TurboQuant | 3.5 | 54.65 | 44.47 | 30.01 | 43.04 | +0.00 | baseline |
| Shared Rotated Outlier | 3.5 | 52.77 | 39.60 | 27.10 | 39.82 | -3.22 | reject |

Artifacts:

- `reproduce/runs/incremental/shared_rotated_outlier_{hotpotqa,2wikimqa,musique}_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/shared_rotated_outlier_multiqa.md`
- `reproduce/incremental/shared_rotated_outlier_multiqa.json`
- `reproduce/logs/shared_rotated_outlier_jobs/`

Analysis:

- The method gives a 2.5-bit HotpotQA gain, but it severely regresses 2WikiMQA and MuSiQue at both bit widths.
- The failure suggests that forcing identical high-bit coordinates for K and V is too restrictive. K and V appear to need different high-bit rotated subspaces even if a shared rotation basis might still be useful.
- This rejects hard shared K/V outlier-coordinate allocation as a reportable contribution.

Decision: do not expand to full Table 1. The next core-rotation attempt should keep K/V on a shared rotation basis but allow independent high-bit coordinate selection, separating the effect of paired K/V rotation from the overly strong shared-coordinate constraint.

### R28: Paired Rotated-Domain Outlier Coordinates

Purpose: test whether the R27 failure came from forcing K/V to share the same high-bit coordinate set, rather than from sharing the rotation basis itself. The method keeps K and V on a shared TurboQuant random rotation basis, but lets each side independently select its high-bit rotated coordinates using its own rotated-domain reconstruction-gain scores.

Run configuration:

```text
key_quantizer=paired_rotated_outlier_mse
value_quantizer=paired_rotated_outlier_mse
rotation rule: K/V use the key-side random orthogonal rotation basis
coordinate rule: K and V independently select top reconstruction-gain rotated coordinates
same rule for 2.5-bit and 3.5-bit
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
87 passed
```

Full MultiQA results:

| Method | KV bits | HotpotQA | 2WikiMQA | MuSiQue | MultiQA Avg | Delta vs TQ MultiQA | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TurboQuant | 2.5 | 44.83 | 37.96 | 25.24 | 36.01 | +0.00 | baseline |
| Paired Rotated Outlier | 2.5 | 47.99 | 33.31 | 22.35 | 34.55 | -1.46 | reject |
| TurboQuant | 3.5 | 54.65 | 44.47 | 30.01 | 43.04 | +0.00 | baseline |
| Paired Rotated Outlier | 3.5 | 53.22 | 42.38 | 28.52 | 41.37 | -1.67 | reject |

Artifacts:

- `reproduce/runs/incremental/paired_rotated_outlier_{hotpotqa,2wikimqa,musique}_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/paired_rotated_outlier_multiqa.md`
- `reproduce/incremental/paired_rotated_outlier_multiqa.json`
- `reproduce/logs/paired_rotated_outlier_jobs/`

Analysis:

- Allowing independent K/V high-bit coordinates reduces the R27 damage, but the method still regresses MultiQA at both bit widths.
- The HotpotQA 2.5-bit gain remains strong, which indicates that rotated-domain high-bit allocation can help some retrieval-style examples.
- The consistent 2WikiMQA and MuSiQue drops show that an unconditional shared-rotation rotated-outlier path is not stable enough for a reportable method.

Decision: do not expand to full Table 1. The next attempt should keep the core rotation-domain candidate but add an online objective guard, selecting the paired rotated path only when it improves the current attention-output proxy versus the reproduced TurboQuant path.

### R29: Attention-Adaptive Paired Rotated-Domain Outlier Coordinates

Purpose: stabilize R28 by adding an online attention-output guard. For each cache segment, the method compares the reproduced TurboQuant path against the paired-rotation rotated-outlier path and uses the paired path only when the current attention-output proxy error is lower. The method keeps the same rule at 2.5-bit and 3.5-bit.

Run configuration:

```text
key_quantizer=attention_adaptive_paired_rotated_outlier_mse
value_quantizer=attention_adaptive_paired_rotated_outlier_mse
candidate: shared K/V rotation basis with independent rotated high-bit coordinates
guard: candidate_attention_error < baseline_attention_error
same rule for 2.5-bit and 3.5-bit
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
88 passed
```

Full MultiQA results:

| Method | KV bits | HotpotQA | 2WikiMQA | MuSiQue | MultiQA Avg | Delta vs TQ MultiQA | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TurboQuant | 2.5 | 44.83 | 37.96 | 25.24 | 36.01 | +0.00 | baseline |
| Attention-Adaptive Paired Rotated Outlier | 2.5 | 48.05 | 34.53 | 20.46 | 34.35 | -1.66 | reject |
| TurboQuant | 3.5 | 54.65 | 44.47 | 30.01 | 43.04 | +0.00 | baseline |
| Attention-Adaptive Paired Rotated Outlier | 3.5 | 51.70 | 43.93 | 25.71 | 40.45 | -2.59 | reject |

Artifacts:

- `reproduce/runs/incremental/attention_adaptive_paired_rotated_outlier_{hotpotqa,2wikimqa,musique}_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/attention_adaptive_paired_rotated_outlier_multiqa.md`
- `reproduce/incremental/attention_adaptive_paired_rotated_outlier_multiqa.json`
- `reproduce/logs/attention_adaptive_paired_rotated_outlier_jobs/`

Analysis:

- The attention-output guard preserves the HotpotQA 2.5-bit gain, but it does not protect 2WikiMQA or MuSiQue.
- At 3.5-bit it is worse than the unguarded paired rotation method on HotpotQA and MuSiQue.
- This rejects the current attention-output proxy as a sufficient selector for paired rotated-domain allocation.

Decision: do not expand to full Table 1. Before trying another guarded rotation method, inspect whether the guard activates too broadly; if so, the next candidate should use a more conservative attention-statistics criterion rather than the direct attention-output proxy.

### R30: Entropy-Gated Paired Rotated-Domain Outlier Coordinates

Purpose: test a more conservative guard after R29 showed that the attention-output proxy activated too broadly. The method keeps the paired rotated-domain candidate from R28 but activates it only when the current attention entropy ratio is no greater than the configured threshold. Otherwise it falls back to reproduced TurboQuant MSE.

Run configuration:

```text
key_quantizer=entropy_guarded_paired_rotated_outlier_mse
value_quantizer=entropy_guarded_paired_rotated_outlier_mse
candidate: shared K/V rotation basis with independent rotated high-bit coordinates
guard: attention_entropy_ratio <= attention_entropy_threshold
attention_entropy_threshold=0.80
same rule for 2.5-bit and 3.5-bit
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
89 passed
```

Full MultiQA results:

| Method | KV bits | HotpotQA | 2WikiMQA | MuSiQue | MultiQA Avg | Delta vs TQ MultiQA | Activation | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TurboQuant | 2.5 | 44.83 | 37.96 | 25.24 | 36.01 | +0.00 | 0.00 | baseline |
| Entropy-Gated Paired Rotated Outlier | 2.5 | 47.99 | 33.31 | 22.35 | 34.55 | -1.46 | 1.00 | reject |
| TurboQuant | 3.5 | 54.65 | 44.47 | 30.01 | 43.04 | +0.00 | 0.00 | baseline |
| Entropy-Gated Paired Rotated Outlier | 3.5 | 53.22 | 42.38 | 28.52 | 41.37 | -1.67 | 1.00 | reject |

Artifacts:

- `reproduce/runs/incremental/entropy_guarded_paired_rotated_outlier_{hotpotqa,2wikimqa,musique}_turboquant_{2p5,3p5}_full.jsonl`
- `reproduce/incremental/entropy_guarded_paired_rotated_outlier_multiqa.md`
- `reproduce/incremental/entropy_guarded_paired_rotated_outlier_multiqa.json`
- `reproduce/logs/entropy_guarded_paired_rotated_outlier_jobs/`

Analysis:

- With the default entropy threshold, the guard activates for every segment in every tested task, so the method exactly reproduces R28's failure pattern.
- Lowering the threshold could reduce activation, but without a principled calibration target this becomes threshold search rather than a method contribution.
- The paired rotated-domain family has now failed under hard sharing, independent sharing, attention-error guarding, and entropy guarding.

Decision: reject this rotation-gated family for now. The next candidate should move away from shared K/V rotation gates and inspect other core reconstruction/preconditioning signals that already show full-task gains at one bit width without being prompt gates.
