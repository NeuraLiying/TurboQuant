# Rotation-Bank MSE Experiments

This file records a rotation-level incremental direction over reproduced TurboQuant.

## Method Candidate

`rotation_bank_mse` changes TurboQuant at its core random-rotation step:

- TurboQuant baseline: one random orthogonal rotation per layer/KV side, then scalar Lloyd-Max quantization.
- Candidate: keep the same scalar Lloyd-Max quantizer and requested bit budget, but build a small bank of random orthogonal rotations.
- For each K/V vector segment, quantize with each bank rotation and select the rotation with the lowest reconstruction MSE.
- Store the selected rotation id as packed metadata. With `rotation_bank_size=4`, this adds 2 bits per K/V vector, not per channel.

The method is answer-free and label-free. It directly modifies TurboQuant's rotation mechanism, unlike prompt gates or task-specific hyperparameter search.

## Implementation Notes

Implemented quantizer:

```text
rotation_bank_mse
```

Relevant files:

```text
turboquant/kv_cache.py
experiments/longbench/run_full_cache_eval.py
tests/test_core.py
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
48 passed
```

Mechanism tests verify:

- `rotation_bank_size=1` exactly matches baseline `mse`.
- `rotation_bank_size=4` has no larger reconstruction MSE than baseline `mse` on tested K/V tensors.
- Rotation ids are bit-packed, so metadata overhead is small.

## Probe: 2WikiMQA 0:20

Model: `meta-llama/Llama-3.1-8B-Instruct`

Dataset slice: `longbench_2wikimqa`, examples `0:20`

Runner settings:

```text
--cache-mode turboquant
--key-quantizer rotation_bank_mse
--value-quantizer rotation_bank_mse
--rotation-bank-size 4
--turboquant-fast-materialized-eval
```

| Method | KV bits | Examples | LongBench score | Delta | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| Rotation-Bank MSE | 2.5 | 20 | 40.95 | +8.19 | 0.40 | 0.17395 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| Rotation-Bank MSE | 3.5 | 20 | 45.20 | +3.33 | 0.40 | 0.23647 |

Artifacts:

```text
reproduce/runs/incremental/rotation_bank_mse_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/rotation_bank_mse_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/incremental/rotation_bank_mse_2wikimqa_tq25_baseline_0_20.json
reproduce/incremental/rotation_bank_mse_2wikimqa_tq25_0_20.json
reproduce/incremental/rotation_bank_mse_2wikimqa_tq35_baseline_0_20.json
reproduce/incremental/rotation_bank_mse_2wikimqa_tq35_0_20.json
```

## Full 2WikiMQA Validation

The 20-example slice signal did not hold on the full 200-example task.

| Method | KV bits | Examples | LongBench score | Delta | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| Rotation-Bank MSE | 2.5 | 200 | 33.53 | -4.43 | 0.335 | 0.17395 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| Rotation-Bank MSE | 3.5 | 200 | 43.56 | -0.91 | 0.430 | 0.23645 |

Full-task artifacts:

```text
reproduce/runs/incremental/rotation_bank_mse_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/rotation_bank_mse_2wikimqa_turboquant_3p5_full_merged.jsonl
reproduce/incremental/rotation_bank_mse_2wikimqa_tq25_baseline_full.json
reproduce/incremental/rotation_bank_mse_2wikimqa_tq25_full.json
reproduce/incremental/rotation_bank_mse_2wikimqa_tq35_baseline_full.json
reproduce/incremental/rotation_bank_mse_2wikimqa_tq35_full.json
```

## Analysis

This is a stronger novelty direction than the previous prompt-gated reconstruction method because it modifies TurboQuant's main random-rotation mechanism.

The first slice signal was positive at both 2.5-bit and 3.5-bit, but full 2WikiMQA invalidated the method. Metadata overhead is small after bit-packing rotation ids:

- 2.5-bit: cache ratio `0.17199 -> 0.17395`
- 3.5-bit: cache ratio `0.23450 -> 0.23647`

The failure suggests that minimizing reconstruction MSE over rotation candidates can overfit local K/V reconstruction without preserving task-relevant attention behavior. This is consistent with earlier rotation-related probes, especially `hadamard_mse`, which were also false positives on small slices and failed full 2WikiMQA validation.

Decision: do not expand `rotation_bank_mse` to MultiQA or Table 1.

The next rotation-level candidate should use an attention-aware selection criterion rather than pure reconstruction MSE. A plausible direction is to choose the rotation that minimizes a prefill attention-score or attention-output proxy using the current query states, while still keeping the same TurboQuant scalar codebook and bit budget.

## Attention-Aware Rotation-Bank Probe

Candidate:

```text
attention_rotation_bank_mse
```

This variant still changes the random-rotation step, but chooses the rotation bank candidate with a prefill attention-error proxy rather than reconstruction MSE.

2WikiMQA `0:20` result:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| Attention Rotation-Bank MSE | 2.5 | 20 | 30.20 | -2.56 | 0.25 | 0.17396 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| Attention Rotation-Bank MSE | 3.5 | 20 | 43.54 | +1.67 | 0.40 | 0.23646 |

Artifacts:

```text
reproduce/runs/incremental/attention_rotation_bank_mse_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/attention_rotation_bank_mse_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/incremental/attention_rotation_bank_mse_2wikimqa_tq25_0_20.json
reproduce/incremental/attention_rotation_bank_mse_2wikimqa_tq35_0_20.json
```

Decision: do not expand `attention_rotation_bank_mse`. It fails the 2.5-bit slice and is significantly slower than TurboQuant because it decodes and scores every rotation candidate.

Current lesson: rotation-candidate selection alone is not enough. Pure reconstruction-MSE selection overfits local reconstruction, while the simple attention proxy is too noisy at 2.5-bit. A stronger rotation-level contribution likely needs a deterministic outlier-aware rotation transform that improves the coordinate distribution before Lloyd-Max quantization, rather than choosing among several full random rotations online.

## Value-Only Outlier-Hadamard Block16 Probe

Candidate:

```text
key_quantizer=mse
value_quantizer=outlier_hadamard_mse
outlier_hadamard_block_size=16
```

This variant changes the value-side rotation/preconditioning path. It keeps TurboQuant's key path unchanged, but applies a deterministic outlier-spreading block Hadamard transform to V before the scalar Lloyd-Max quantizer. The goal is to improve value reconstruction geometry under the same effective bit budget without using prompt labels or answer feedback.

Stratified 80-example screening used indexes `0:20`, `50:70`, `100:120`, and `150:170`.

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| V Outlier-Hadamard Block16 | 2.5 | 80 | 36.20 | +0.66 | 0.325 | 0.17208 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| V Outlier-Hadamard Block16 | 3.5 | 80 | 43.38 | +2.04 | 0.413 | 0.23459 |

Full 2WikiMQA validation:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| V Outlier-Hadamard Block16 | 2.5 | 200 | 38.72 | +0.76 | 0.365 | 0.17208 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| V Outlier-Hadamard Block16 | 3.5 | 200 | 44.33 | -0.14 | 0.430 | 0.23459 |

Artifacts:

```text
reproduce/runs/incremental/value_outlier_hadamard_b16_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/value_outlier_hadamard_b16_2wikimqa_turboquant_3p5_full.jsonl
reproduce/incremental/value_outlier_hadamard_b16_2wikimqa_turboquant_2p5_full.json
reproduce/incremental/value_outlier_hadamard_b16_2wikimqa_turboquant_3p5_full.json
```

Decision: do not expand to Table 1. The method is a legitimate core rotation/preconditioning modification and improves 2.5-bit full 2WikiMQA, but it does not beat TurboQuant at 3.5-bit on the same full task.

## MSE-Safeguarded Value Outlier-Hadamard Probe

Candidate:

```text
key_quantizer=mse
value_quantizer=adaptive_outlier_hadamard_mse
outlier_hadamard_block_size=16
```

This variant keeps K on reproduced TurboQuant and applies a conservative rotation choice only to V. For each value segment, it computes both the original TurboQuant random-rotation reconstruction and the outlier-aware block-Hadamard reconstruction, then uses the Hadamard segment only when its segment-level reconstruction MSE is lower. The method is answer-free and modifies the rotation/preconditioning path, but uses a local reconstruction-MSE safeguard instead of a prompt gate.

Stratified 80-example screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| V Adaptive Outlier-Hadamard | 2.5 | 80 | 35.60 | +0.06 | 0.300 | 0.17205 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| V Adaptive Outlier-Hadamard | 3.5 | 80 | 43.01 | +1.67 | 0.413 | 0.23454 |

Full 2WikiMQA validation:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| V Adaptive Outlier-Hadamard | 2.5 | 200 | 35.88 | -2.08 | 0.350 | 0.17207 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| V Adaptive Outlier-Hadamard | 3.5 | 200 | 43.49 | -0.98 | 0.425 | 0.23454 |

Artifacts:

```text
reproduce/runs/incremental/value_adaptive_outlier_hadamard_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/value_adaptive_outlier_hadamard_2wikimqa_turboquant_3p5_full.jsonl
reproduce/incremental/value_adaptive_outlier_hadamard_2wikimqa_turboquant_2p5_full.json
reproduce/incremental/value_adaptive_outlier_hadamard_2wikimqa_turboquant_3p5_full.json
```

Decision: do not expand. Even the MSE safeguard does not align with full-task accuracy, so the next candidate should not rely only on local K/V reconstruction error.

## Grouped-Head Hadamard Rotation Probe

Candidate family:

```text
head_hadamard_mse
key_quantizer=head_hadamard_mse, value_quantizer=mse
key_quantizer=mse, value_quantizer=head_hadamard_mse
```

Rationale: instead of rotating only within each 128D KV head, mix the KV heads with an orthogonal Hadamard transform before applying the usual TurboQuant scalar quantizer, then invert the head transform after dequantization. This is a structured core-rotation change inspired by grouped-head rotation ideas in recent KV quantization work, and it adds no per-token metadata.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| Head-Hadamard K/V | 2.5 | 80 | 30.66 | -4.88 | 0.275 | 0.17200 |
| K-only Head-Hadamard | 2.5 | 80 | 37.53 | +1.99 | 0.363 | 0.17198 |
| V-only Head-Hadamard | 2.5 | 80 | 34.08 | -1.46 | 0.313 | 0.17201 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| Head-Hadamard K/V | 3.5 | 80 | 38.38 | -2.96 | 0.363 | 0.23451 |
| K-only Head-Hadamard | 3.5 | 80 | 40.95 | -0.39 | 0.413 | 0.23451 |
| V-only Head-Hadamard | 3.5 | 80 | 40.57 | -0.77 | 0.388 | 0.23450 |

Artifacts:

```text
reproduce/runs/incremental/head_hadamard_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/head_hadamard_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/runs/incremental/key_head_hadamard_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/key_head_hadamard_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/runs/incremental/value_head_hadamard_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/value_head_hadamard_2wikimqa_turboquant_3p5_stratified80.jsonl
```

Decision: do not expand standalone. K-only head-Hadamard has a useful 2.5-bit signal but fails 3.5-bit. The next probe combines this K-side transform with the V-side block outlier-Hadamard signal.

### K-Head / V-Outlier Hadamard Combination

Candidate:

```text
key_quantizer=head_hadamard_mse
value_quantizer=outlier_hadamard_mse
outlier_hadamard_block_size=16
```

Rationale: combine the positive 2.5-bit K-only head-Hadamard signal with the positive 3.5-bit V-only block outlier-Hadamard signal under one K/V role-aware rotation rule.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| K-head / V-outlier Hadamard | 2.5 | 80 | 24.58 | -10.96 | 0.213 | 0.17209 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| K-head / V-outlier Hadamard | 3.5 | 80 | 39.17 | -2.17 | 0.400 | 0.23461 |

Decision: do not expand. The two positive partial signals are not additive.

## Sparse Hadamard Residual Correction Probe

Candidate family:

```text
key_quantizer=hadamard_residual_mse, value_quantizer=mse
key_quantizer=mse, value_quantizer=hadamard_residual_mse
```

Rationale: replace fractional high-bit scalar outlier coordinates with a low-bit TurboQuant MSE base plus sparse Hadamard residual-sign correction. For 2.5-bit this stores 2-bit MSE coordinates plus 0.5 residual-sign bits per dimension; for 3.5-bit it stores 3-bit MSE coordinates plus 0.5 residual-sign bits per dimension. The K-only variant targets attention inner-product errors while keeping V on the reproduced path.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| K-only Hadamard residual | 2.5 | 80 | 38.75 | +3.21 | 0.325 | 0.17197 |
| V-only Hadamard residual | 2.5 | 80 | 23.49 | -12.05 | 0.188 | 0.17197 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| K-only Hadamard residual | 3.5 | 80 | 42.43 | +1.08 | 0.400 | 0.23446 |
| V-only Hadamard residual | 3.5 | 80 | 40.22 | -1.12 | 0.413 | 0.23447 |

Full 2WikiMQA validation:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| K-only Hadamard residual | 2.5 | 200 | 37.28 | -0.68 | 0.340 | 0.17197 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| K-only Hadamard residual | 3.5 | 200 | 41.87 | -2.60 | 0.420 | 0.23447 |

Artifacts:

```text
reproduce/runs/incremental/key_hadamard_residual_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/key_hadamard_residual_2wikimqa_turboquant_3p5_full.jsonl
reproduce/incremental/key_hadamard_residual_2wikimqa_turboquant_2p5_full.json
reproduce/incremental/key_hadamard_residual_2wikimqa_turboquant_3p5_full.json
```

Decision: do not expand to Table 1. This is a stronger core-method idea than prompt gating, but full-task validation again reverses the stratified screening signal.

### Attention-Aware Hadamard Residual Safeguard

Candidate:

```text
key_quantizer=attention_hadamard_residual_mse
value_quantizer=mse
```

Rationale: choose between reproduced TurboQuant MSE and K-only sparse Hadamard residual per segment by an answer-free prefill attention-score error proxy. This was intended to avoid the severe `150:200` failures from the ungated residual correction while remaining an internal cache-statistics method rather than a prompt gate.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| Attention-aware K Hadamard residual | 2.5 | 80 | 37.98 | +2.44 | 0.325 | 0.17198 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| Attention-aware K Hadamard residual | 3.5 | 80 | 40.55 | -0.79 | 0.388 | 0.23448 |

Decision: do not expand. The attention proxy improves the 2.5-bit screening score but fails the required unified 2.5/3.5 criterion.

## Attention-Safeguarded Value Outlier-Hadamard Probe

Candidate:

```text
key_quantizer=mse
value_quantizer=attention_outlier_hadamard_mse
outlier_hadamard_block_size=16
```

Rationale: start from the V-only block outlier-Hadamard method, but choose between reproduced TurboQuant MSE and outlier-Hadamard per value segment using an answer-free attention-output proxy. This is intended to preserve the 3.5-bit setting where unconditional V Hadamard was slightly negative.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| Attention-safeguarded V Hadamard | 2.5 | 80 | 37.78 | +2.24 | 0.363 | 0.17206 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| Attention-safeguarded V Hadamard | 3.5 | 80 | 42.34 | +1.00 | 0.425 | 0.23456 |

Full 2WikiMQA validation:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| Attention-safeguarded V Hadamard | 2.5 | 200 | 34.87 | -3.09 | 0.330 | 0.17205 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| Attention-safeguarded V Hadamard | 3.5 | 200 | 44.95 | +0.48 | 0.450 | 0.23455 |

Artifacts:

```text
reproduce/runs/incremental/value_attention_outlier_hadamard_b16_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/value_attention_outlier_hadamard_b16_2wikimqa_turboquant_3p5_full.jsonl
reproduce/incremental/value_attention_outlier_hadamard_b16_2wikimqa_turboquant_2p5_full.json
reproduce/incremental/value_attention_outlier_hadamard_b16_2wikimqa_turboquant_3p5_full.json
```

Decision: not sufficient as a standalone method because it fails 2.5-bit full validation. However, together with the unconditional V Hadamard result, it motivates a single bitwidth-aware value-rotation rule: use unconditional outlier-Hadamard when the low-bit base quantizer is very coarse, and use attention-safeguarded outlier-Hadamard when the base quantizer is less noisy.

## Bitwidth-Aware Value Hadamard Guard

Candidate:

```text
key_quantizer=mse
value_quantizer=bitwidth_attention_outlier_hadamard_mse
outlier_hadamard_block_size=16
```

Rationale: use one V-side rotation rule for both reported bit budgets. When the fractional-bit base quantizer is very coarse (`regular_bits <= 2`, as in 2.5-bit), apply outlier-aware block-Hadamard to V unconditionally. When the base quantizer is less noisy (`regular_bits >= 3`, as in 3.5-bit), apply the same V-side block-Hadamard only when an answer-free attention-output proxy prefers it over reproduced TurboQuant MSE.

This is still a core KV-cache rotation/preconditioning method: K remains reproduced TurboQuant to preserve attention logits, while V gets a bitwidth-aware Hadamard rotation guard.

Full 2WikiMQA validation:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| Bitwidth-aware V Hadamard guard | 2.5 | 200 | 38.72 | +0.76 | 0.365 | 0.17208 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| Bitwidth-aware V Hadamard guard | 3.5 | 200 | 44.95 | +0.48 | 0.450 | 0.23455 |

Full MultiQA validation:

| Task | TQ 2.5 | Method 2.5 | Delta | TQ 3.5 | Method 3.5 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hotpotqa` | 44.83 | 48.04 | +3.21 | 54.65 | 54.53 | -0.12 |
| `2wikimqa` | 37.96 | 38.72 | +0.76 | 44.47 | 44.95 | +0.48 |
| `musique` | 25.24 | 23.59 | -1.65 | 30.01 | 27.11 | -2.90 |
| MultiQA avg | 36.01 | 36.78 | +0.77 | 43.04 | 42.20 | -0.85 |

Artifacts:

```text
reproduce/runs/incremental/bitwidth_value_hadamard_guard_b16_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/bitwidth_value_hadamard_guard_b16_2wikimqa_turboquant_3p5_full.jsonl
reproduce/runs/incremental/bitwidth_value_hadamard_guard_b16_hotpotqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/bitwidth_value_hadamard_guard_b16_hotpotqa_turboquant_3p5_full.jsonl
reproduce/runs/incremental/bitwidth_value_hadamard_guard_b16_musique_turboquant_2p5_full.jsonl
reproduce/runs/incremental/bitwidth_value_hadamard_guard_b16_musique_turboquant_3p5_full.jsonl
```

Decision: do not expand to full Table 1. The method passes full 2WikiMQA at both bit budgets, but fails the broader MultiQA requirement at 3.5-bit because Musique regresses substantially.

### Wider Attention-Proxy Check

Follow-up:

```text
key_quantizer=mse
value_quantizer=attention_outlier_hadamard_mse
attention_error_query_tokens=8
outlier_hadamard_block_size=16
```

Rationale: test whether the Musique failure and earlier full-task reversals were caused by the default attention proxy looking only at the last query token. This keeps the same V-side Hadamard rotation and only widens the answer-free attention-output proxy.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| 8-query attention V Hadamard | 2.5 | 80 | 35.76 | +0.22 | 0.300 | 0.17205 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| 8-query attention V Hadamard | 3.5 | 80 | 41.32 | -0.02 | 0.400 | 0.23454 |

Decision: do not expand. Widening the proxy removes most of the positive 2WikiMQA signal and does not satisfy the unified 2.5/3.5 screening criterion.

## RMS-Balanced Rotation Preconditioning Probe

Candidate:

```text
rms_rotation_mse
```

This variant applies deterministic per-segment channel RMS preconditioning before TurboQuant's random rotation and scalar Lloyd-Max quantization, then multiplies the scales back during reconstruction. It is a rotation/preconditioning-level change, not a prompt gate or seed search.

2WikiMQA `0:20` result:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| RMS Rotation MSE | 2.5 | 20 | 38.87 | +6.11 | 0.35 | 0.17204 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| RMS Rotation MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23451 |

Artifacts:

```text
reproduce/runs/incremental/rms_rotation_mse_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/rms_rotation_mse_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/incremental/rms_rotation_mse_2wikimqa_tq25_0_20.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq35_0_20.json
```

Decision: do not run full validation yet. The 2.5-bit signal is positive and metadata overhead is negligible, but the 3.5-bit slice only ties TurboQuant. The next probe is RMS preconditioning plus the previously useful regular-subspace gain, applied uniformly to both bit budgets.

## RMS Preconditioning Plus Regular Gain Probe

Candidate:

```text
rms_regular_gain_mse
```

This combines RMS-balanced preconditioning with the regular-subspace norm-gain correction used in fractional-bit settings. The intent was to keep the rotation/preconditioning novelty while adding the reconstruction correction that previously helped some 2.5-bit cases.

2WikiMQA `0:20` result:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| RMS Regular-Gain MSE | 2.5 | 20 | 30.95 | -1.81 | 0.30 | 0.17200 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| RMS Regular-Gain MSE | 3.5 | 20 | 40.54 | -1.33 | 0.40 | 0.23450 |

Artifacts:

```text
reproduce/runs/incremental/rms_regular_gain_mse_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/rms_regular_gain_mse_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/incremental/rms_regular_gain_mse_2wikimqa_tq25_0_20.json
reproduce/incremental/rms_regular_gain_mse_2wikimqa_tq35_0_20.json
```

Decision: do not expand. RMS preconditioning and regular-subspace norm gain are not additive on this probe.

## Current Status

No rotation-level candidate in this file is reportable yet:

- `rotation_bank_mse`: positive slice, failed full 2WikiMQA at both 2.5-bit and 3.5-bit.
- `attention_rotation_bank_mse`: failed 2.5-bit slice and is too slow.
- `rms_rotation_mse`: positive 2.5-bit slice but only tied 3.5-bit.
- `rms_regular_gain_mse`: failed both 2.5-bit and 3.5-bit slices.
- K-only `rms_rotation_mse`: positive 20-example slice at both bit widths, but failed full 2WikiMQA.

The most promising partial signal is `rms_rotation_mse` at 2.5-bit because it is deterministic, cheap, and has negligible metadata overhead. However, it does not yet satisfy the requirement of a unified improvement over TurboQuant at both reported bit budgets.

## K-Only RMS Rotation Preconditioning Probe

Candidate:

```text
key_quantizer=rms_rotation_mse
value_quantizer=mse
```

Rationale: KV-cache attention scores are directly sensitive to key direction errors. This variant applies RMS-balanced rotation preconditioning only to K, while leaving V on the reproduced TurboQuant MSE path. It is a unified rule across 2.5-bit and 3.5-bit and stays within the same effective bit budgets.

2WikiMQA `0:20` screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| K-only RMS Rotation | 2.5 | 20 | 40.54 | +7.78 | 0.40 | 0.17200 |
| V-only RMS Rotation | 2.5 | 20 | 18.47 | -14.29 | 0.15 | 0.17201 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| K-only RMS Rotation | 3.5 | 20 | 47.20 | +5.33 | 0.45 | 0.23451 |
| V-only RMS Rotation | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |

Full 2WikiMQA validation:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| K-only RMS Rotation | 2.5 | 200 | 35.41 | -2.55 | 0.320 | 0.17201 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| K-only RMS Rotation | 3.5 | 200 | 42.48 | -1.98 | 0.420 | 0.23451 |

Artifacts:

```text
reproduce/runs/incremental/rms_key_only_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/rms_key_only_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/runs/incremental/rms_value_only_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/rms_value_only_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/runs/incremental/rms_key_only_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/rms_key_only_2wikimqa_turboquant_3p5_full_merged.jsonl
reproduce/incremental/rms_key_only_2wikimqa_turboquant_2p5_full.json
reproduce/incremental/rms_key_only_2wikimqa_turboquant_3p5_full.json
```

Decision: do not expand. The 20-example slice was a strong false positive at both bit widths. Full-task validation confirms that this preconditioning does not improve TurboQuant on 2WikiMQA.

## Rotated-Coordinate Outlier Allocation Probe

Candidate:

```text
rotated_outlier_mse
```

Rationale: reproduced TurboQuant fractional bits choose high-bit channels in the original head coordinate system, then quantize the regular and outlier channel subsets separately. This candidate moves the fractional-bit decision into TurboQuant's own random-rotation coordinate system: it rotates the full vector once, computes lower-bit and upper-bit Lloyd-Max assignments per rotated coordinate, and allocates the upper-bit budget to rotated coordinates with the largest quantization-error reduction.

This directly modifies the TurboQuant rotation-coordinate quantization path and does not use labels, answers, prompt gates, or dataset-specific bit schedules.

2WikiMQA `0:20` screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| Rotated-Coordinate Outlier | 2.5 | 20 | 40.92 | +8.16 | 0.30 | 0.17199 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| Rotated-Coordinate Outlier | 3.5 | 20 | 42.29 | +0.42 | 0.40 | 0.23449 |

Full 2WikiMQA validation at 2.5-bit:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| Rotated-Coordinate Outlier | 2.5 | 200 | 32.21 | -5.75 | 0.285 | 0.17200 |

Artifacts:

```text
reproduce/runs/incremental/rotated_outlier_mse_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/rotated_outlier_mse_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/runs/incremental/rotated_outlier_mse_2wikimqa_turboquant_2p5_full_merged.jsonl
reproduce/incremental/rotated_outlier_mse_2wikimqa_turboquant_2p5_full.json
```

Decision: do not expand. The method is novel at the rotation-coordinate level, but full 2.5-bit 2WikiMQA fails substantially, so full 3.5-bit and Table 1 expansion are not justified.

## Rotation-Domain Reconstruction Gain Checks

Candidate family:

```text
selected_gain_mse
clipped_gain_mse
```

Rationale: TurboQuant's Lloyd-Max scalar reconstruction after random rotation has a predictable unit-vector norm shrinkage. `gain_mse` applies the theoretical correction globally, while `selected_gain_mse` applies it per vector only when it reduces reconstruction MSE, and `clipped_gain_mse` uses a per-vector LSQ gain clipped to the theoretical gain range. These variants preserve the bit budget and modify the rotation-domain reconstruction rule rather than prompt formatting or task-specific bit allocation.

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | 0.30 | 0.17199 |
| Selected Gain MSE | 2.5 | 50 | 32.21 | -3.73 | 0.26 | 0.17199 |
| Clipped Gain MSE | 2.5 | 50 | 29.84 | -6.10 | 0.26 | 0.17201 |
| TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | 0.34 | 0.23450 |
| Selected Gain MSE | 3.5 | 50 | 42.33 | -0.40 | 0.36 | 0.23453 |
| Clipped Gain MSE | 3.5 | 50 | 42.72 | +0.00 | 0.36 | 0.23451 |

Artifacts:

```text
reproduce/runs/incremental/unified_selected_gain_mse_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/unified_selected_gain_mse_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/runs/incremental/unified_clipped_gain_mse_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/unified_clipped_gain_mse_2wikimqa_turboquant_3p5_0_50.jsonl
```

Decision: do not expand. The per-vector gain variants are theoretically motivated by TurboQuant's rotated-domain reconstruction shrinkage, but the more stable 50-example screening block already regresses at 2.5-bit.

## Segment-Level Rotation Bank Probe

Candidate:

```text
segment_rotation_bank_mse
```

Rationale: `rotation_bank_mse` selected a different rotation per vector and failed full-task validation. This variant is more conservative: for each KV update segment, it evaluates a small bank of random rotations and uses one rotation for the whole segment, selecting by segment-level reconstruction MSE. It keeps attention geometry more coherent than per-vector rotation selection while still modifying TurboQuant's random rotation mechanism.

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | 0.30 | 0.17199 |
| Segment Rotation Bank | 2.5 | 50 | 38.51 | +2.57 | 0.32 | 0.17396 |
| TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | 0.34 | 0.23450 |
| Segment Rotation Bank | 3.5 | 50 | 34.39 | -8.33 | 0.28 | 0.23647 |

Artifacts:

```text
reproduce/runs/incremental/segment_rotation_bank_mse_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/segment_rotation_bank_mse_2wikimqa_turboquant_3p5_0_50_merged.jsonl
reproduce/incremental/segment_rotation_bank_mse_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/segment_rotation_bank_mse_2wikimqa_turboquant_3p5_0_50_merged.json
```

Decision: do not expand. The method has a 2.5-bit slice signal but fails the unified 2.5/3.5 requirement because the 3.5-bit screening regression is large.

## Outlier-Aware Block-Hadamard Rotation Probe

Candidate family:

```text
outlier_hadamard_mse
regular_outlier_hadamard_mse
```

Rationale: TurboQuant's core compression path relies on an orthogonal rotation before scalar Lloyd-Max quantization. This candidate replaces the fully random rotation with a structured, data-dependent block-Hadamard rotation. Channels are scored by segment-level absolute mean, high-score channels are spread across Hadamard blocks by a deterministic permutation, and per-segment random signs are applied before block-Hadamard mixing. `regular_outlier_hadamard_mse` is a conservative fractional-bit variant that applies this rotation only to the low-bit regular subspace while keeping high-bit outlier coordinates on reproduced TurboQuant MSE.

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | 0.30 | 0.17199 |
| Outlier Block-Hadamard | 2.5 | 50 | 39.00 | +3.06 | 0.32 | 0.17218 |
| Regular-only Outlier Block-Hadamard | 2.5 | 50 | 38.81 | +2.87 | 0.32 | 0.17209 |
| TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | 0.34 | 0.23450 |
| Outlier Block-Hadamard | 3.5 | 50 | 40.35 | -2.38 | 0.32 | 0.23471 |
| Regular-only Outlier Block-Hadamard | 3.5 | 50 | 39.32 | -3.41 | 0.36 | 0.23465 |

Artifacts:

```text
reproduce/runs/incremental/outlier_hadamard_mse_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/outlier_hadamard_mse_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/runs/incremental/regular_outlier_hadamard_mse_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/regular_outlier_hadamard_mse_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/incremental/outlier_hadamard_mse_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/outlier_hadamard_mse_2wikimqa_turboquant_3p5_0_50.json
reproduce/incremental/regular_outlier_hadamard_mse_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/regular_outlier_hadamard_mse_2wikimqa_turboquant_3p5_0_50.json
```

Decision: do not expand. The candidate is method-level and directly modifies the TurboQuant rotation construction, but it fails the unified 2.5/3.5 requirement because both Hadamard variants regress the 3.5-bit screening block.

## Attention-Score Calibrated Scale Probe

Candidate:

```text
regular_attention_scale_mse
```

Rationale: prior rotation-bank and Hadamard probes showed that lower reconstruction MSE does not reliably preserve LongBench accuracy. This candidate keeps TurboQuant's original random orthogonal rotation and scalar Lloyd-Max codebook, but changes the key reconstruction scale using a prefill attention-score proxy. For each key vector, the scale is chosen by one-dimensional least squares so the quantized key direction best matches original attention scores under the current grouped query states. For fractional-bit settings, this is applied to the low-bit regular subspace and the high-bit outlier subspace remains reproduced TurboQuant MSE.

This is still an algorithmic change in the TurboQuant reconstruction path, not a prompt gate, label-conditioned choice, or bit search.

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | 0.30 | 0.17199 |
| Regular Attention-Scale MSE | 2.5 | 50 | 35.94 | +0.00 | 0.30 | 0.17199 |
| TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | 0.34 | 0.23450 |
| Regular Attention-Scale MSE | 3.5 | 50 | 42.72 | +0.00 | 0.34 | 0.23450 |

Artifacts:

```text
reproduce/runs/incremental/regular_attention_scale_mse_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/regular_attention_scale_mse_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/incremental/regular_attention_scale_mse_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/regular_attention_scale_mse_2wikimqa_turboquant_3p5_0_50.json
```

Decision: do not expand. The method is safe and preserves both screening scores exactly, but it does not improve either bit budget and therefore does not satisfy the required incremental contribution.

### Reconstruction-Gated Regular-Gain Probe

Candidate:

```text
adaptive_regular_gain_mse
```

Rationale: keep TurboQuant's dense random rotation and scalar Lloyd-Max quantizer, but adapt the rotated-domain reconstruction scale per cache update segment. For each K/V segment, construct both reproduced TurboQuant MSE and `regular_gain_mse`, decode both, and choose the lower local reconstruction-MSE segment. This removes the previous prompt/task gate and uses only internal quantization error. It preserves the bit budget and the original outlier allocation.

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
73 passed
```

Stratified 2WikiMQA screening uses indices `0:20`, `50:70`, `100:120`, and `150:170`.

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| Adaptive Regular-Gain | 2.5 | 80 | 35.65 | +0.11 | 0.325 | 0.17200 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| Adaptive Regular-Gain | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |

Artifacts:

```text
reproduce/runs/incremental/adaptive_regular_gain_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/adaptive_regular_gain_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/incremental/adaptive_regular_gain_2wikimqa_turboquant_2p5_stratified80.json
reproduce/incremental/adaptive_regular_gain_2wikimqa_turboquant_3p5_stratified80.json
```

Decision: do not expand yet. The method is safer than prompt gating and improves the 2.5-bit stratified screen slightly, but it only ties 3.5-bit. It does not yet satisfy the requirement of a clear unified improvement over TurboQuant at both reported bit budgets.

## Calibrated Fixed Rotation Schedule Probe

Candidate:

```text
layer_key_rotation_indices
layer_value_rotation_indices
```

Rationale: instead of selecting a rotation online for each segment, calibrate one fixed random-rotation index per layer and KV side from a small unlabeled prompt set, then use the selected rotations for all examples. This directly modifies TurboQuant's random orthogonal rotation construction while adding no per-token or per-vector cache metadata. The calibration criterion is layer-wise K/V reconstruction MSE on four 2WikiMQA prompts without using answers.

Calibration artifact:

```text
reproduce/incremental/calibrated_rotations_2wikimqa_0_4_bits2_bank8.json
```

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs reproduced TQ | Delta vs fixed-seed TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reproduced TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | -1.32 | 0.30 | 0.17199 |
| Fixed-seed TurboQuant MSE | 2.5 | 50 | 37.26 | +1.32 | +0.00 | 0.30 | 0.17200 |
| Calibrated Fixed Rotations | 2.5 | 50 | 29.70 | -6.24 | -7.56 | 0.24 | 0.17201 |
| Reproduced TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | -5.52 | 0.34 | 0.23450 |
| Fixed-seed TurboQuant MSE | 3.5 | 50 | 48.25 | +5.52 | +0.00 | 0.40 | 0.23450 |
| Calibrated Fixed Rotations | 3.5 | 50 | 49.55 | +6.82 | +1.30 | 0.46 | 0.23452 |

Artifacts:

```text
reproduce/runs/incremental/fixed_seed0_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/fixed_seed0_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/runs/incremental/calibrated_rotations_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/calibrated_rotations_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/incremental/fixed_seed0_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/fixed_seed0_2wikimqa_turboquant_3p5_0_50.json
reproduce/incremental/calibrated_rotations_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/calibrated_rotations_2wikimqa_turboquant_3p5_0_50.json
```

Decision: do not expand. The 3.5-bit signal is strong, but the same schedule fails 2.5-bit badly. The fixed-seed control also shows that rotation seed variance alone can dominate a small slice, so this is not yet a robust method-level contribution.

### Joint-Bit Calibration Variant

Follow-up: regenerate the same kind of fixed layer-wise rotation schedule, but choose each layer/side rotation by the mean normalized reconstruction error over both 2-bit and 3-bit quantization. The intent is to avoid overfitting the schedule to the 2.5-bit regular subspace while still serving the 3.5-bit setting.

Calibration artifact:

```text
reproduce/incremental/calibrated_rotations_2wikimqa_0_4_bits2_3_bank8.json
```

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs reproduced TQ | Delta vs fixed-seed TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reproduced TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | -1.32 | 0.30 | 0.17199 |
| Fixed-seed TurboQuant MSE | 2.5 | 50 | 37.26 | +1.32 | +0.00 | 0.30 | 0.17200 |
| Joint-Bit Calibrated Rotations | 2.5 | 50 | 29.25 | -6.69 | -8.01 | 0.22 | 0.17200 |
| Reproduced TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | -5.52 | 0.34 | 0.23450 |
| Fixed-seed TurboQuant MSE | 3.5 | 50 | 48.25 | +5.52 | +0.00 | 0.40 | 0.23450 |
| Joint-Bit Calibrated Rotations | 3.5 | 50 | 43.51 | +0.79 | -4.73 | 0.38 | 0.23452 |

Artifacts:

```text
reproduce/runs/incremental/joint_calibrated_rotations_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/joint_calibrated_rotations_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/incremental/joint_calibrated_rotations_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/joint_calibrated_rotations_2wikimqa_turboquant_3p5_0_50.json
```

Decision: do not expand. Joint-bit reconstruction calibration does not fix the 2.5-bit failure and loses most of the 3.5-bit benefit. This reinforces that K/V reconstruction MSE is a poor calibration objective for task-level LongBench accuracy.

### Selective Top-4 Calibration Variant

Follow-up: apply the original 2-bit calibrated rotations only to the four key layers and four value layers with the largest calibration-set reconstruction gain, leaving all other layers at rotation index 0. The purpose was to test whether the all-layer schedule mainly failed because it perturbed too many layers.

Selected layers:

```text
key:   4, 15, 21, 22
value: 5, 8, 11, 31
```

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs reproduced TQ | Delta vs fixed-seed TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reproduced TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | -1.32 | 0.30 | 0.17199 |
| Fixed-seed TurboQuant MSE | 2.5 | 50 | 37.26 | +1.32 | +0.00 | 0.30 | 0.17200 |
| Selective Top-4 Calibrated Rotations | 2.5 | 50 | 32.22 | -3.72 | -5.04 | 0.26 | 0.17200 |
| Reproduced TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | -5.52 | 0.34 | 0.23450 |
| Fixed-seed TurboQuant MSE | 3.5 | 50 | 48.25 | +5.52 | +0.00 | 0.40 | 0.23450 |
| Selective Top-4 Calibrated Rotations | 3.5 | 50 | 36.68 | -6.04 | -11.57 | 0.30 | 0.23451 |

Artifacts:

```text
reproduce/runs/incremental/selective_calibrated_rotations_top4_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/selective_calibrated_rotations_top4_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/incremental/selective_calibrated_rotations_top4_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/selective_calibrated_rotations_top4_2wikimqa_turboquant_3p5_0_50.json
```

Decision: do not expand. Selectively applying the highest reconstruction-gain layer rotations still fails both bit settings. The issue is not just excessive all-layer perturbation; reconstruction-MSE-selected rotations are misaligned with task accuracy.

### Attention-Output Calibrated Fixed Rotation Variant

Follow-up: replace the reconstruction-MSE calibration objective with an answer-free attention-output proxy. For each layer and K/V side, a small calibration set stores the last query states, pre-RoPE keys, and values. Candidate rotations are selected by minimizing the MSE between the original attention output and the output produced when only K or only V is quantized under that candidate rotation.

Calibration artifact:

```text
reproduce/incremental/attention_output_rotations_2wikimqa_0_4_bits2_3_q8_bank8.json
```

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs reproduced TQ | Delta vs fixed-seed TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reproduced TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | -1.32 | 0.30 | 0.17199 |
| Fixed-seed TurboQuant MSE | 2.5 | 50 | 37.26 | +1.32 | +0.00 | 0.30 | 0.17200 |
| Attention-Output Calibrated Rotations | 2.5 | 50 | 31.36 | -4.58 | -5.90 | 0.24 | 0.17199 |
| Reproduced TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | -5.52 | 0.34 | 0.23450 |
| Fixed-seed TurboQuant MSE | 3.5 | 50 | 48.25 | +5.52 | +0.00 | 0.40 | 0.23450 |
| Attention-Output Calibrated Rotations | 3.5 | 50 | 42.40 | -0.32 | -5.85 | 0.36 | 0.23451 |

Artifacts:

```text
reproduce/runs/incremental/attention_output_rotations_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/attention_output_rotations_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/incremental/attention_output_rotations_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/attention_output_rotations_2wikimqa_turboquant_3p5_0_50.json
```

Decision: do not expand. This was a more task-aligned calibration objective than reconstruction MSE, but it still regresses the 2.5-bit screening block substantially and slightly regresses 3.5-bit versus the reproduced TurboQuant baseline. Fixed layer-wise rotation calibration remains too brittle for a reportable unified method.

## Head-Wise Independent Rotation Probe

Candidate:

```text
head_rotation_mse
```

Rationale: reproduced TurboQuant uses one random orthogonal rotation per layer and KV side. This candidate keeps TurboQuant's orthogonal-rotation plus scalar Lloyd-Max structure, but uses a deterministic independent random rotation for each KV head. The method is label-free, prompt-independent, and adds no per-token or per-vector rotation metadata. It directly changes the granularity of TurboQuant's core random rotation mechanism to better match the head-structured KV distribution.

2WikiMQA `0:50` screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | 0.30 | 0.17199 |
| Head-Wise Independent Rotation | 2.5 | 50 | 34.50 | -1.44 | 0.28 | 0.17295 |

Artifacts:

```text
reproduce/runs/incremental/head_rotation_mse_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/incremental/head_rotation_mse_2wikimqa_turboquant_2p5_0_50.json
```

Decision: do not expand. The method is a clean core-rotation modification, but it regresses the 2.5-bit screening block and is much slower than the reproduced path because each head is quantized through a separate segment. The 3.5-bit screening job was stopped after this 2.5-bit failure because the unified 2.5/3.5 requirement was already not satisfied.

## Orthogonal ACAR / Second-Hadamard Rotation Probe

Candidate:

```text
calibrated_rotation_mse
```

Rationale: a previous ACAR proposal used non-orthogonal PCA whitening, but that breaks TurboQuant's unit-norm rotation assumption and Stage-1 MSE is much worse than random rotation. This corrected probe keeps the transform orthogonal: estimate the second moment of unit-normalized K/V vectors on a small calibration set, rotate into its PCA basis, then apply a Hadamard mixing matrix to spread anisotropic directions across scalar Lloyd-Max coordinates. This directly changes TurboQuant's core rotation matrix while preserving orthogonality, the scalar codebook, and the cache bit budget.

Stage-1 MSE kill-switch on held-out 2WikiMQA vectors:

| Target | Bits | Random rotation MSE | PCA | Non-orthogonal whiten | Orthogonal second-Hadamard | Reduction vs random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Key | 2 | 138.19 | 249.90 | 988.27 | 135.50 | +1.95% |
| Value | 2 | 2.804 | 5.642 | 19.466 | 2.662 | +5.06% |
| Key | 3 | 41.12 | 106.01 | 910.34 | 39.03 | +5.09% |
| Value | 3 | 0.824 | 2.971 | 17.874 | 0.789 | +4.23% |

Integer-path 2WikiMQA `0:20` smoke at 3.0 bits:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 3.0 | 20 | 39.29 | +0.00 | 0.40 | 0.19531 |
| Orthogonal ACAR second-Hadamard | 3.0 | 20 | 43.95 | +4.67 | 0.40 | 0.19531 |

Artifacts:

```text
reproduce/incremental/acar/stage1_mse_2wikimqa_calib0_4_eval4_2_1024_second_hadamard.json
reproduce/incremental/acar/second_hadamard_rotations_2wikimqa_calib0_4_1024.json
reproduce/runs/incremental/acar_second_hadamard_2wikimqa_turboquant_3p0_0_20.jsonl
reproduce/incremental/acar_second_hadamard_2wikimqa_turboquant_3p0_0_20.json
```

Decision: continue, but do not report yet. The 3.0-bit integer-path smoke has a positive downstream signal, but the user-required budgets are 2.5-bit and 3.5-bit. The current fractional-bit implementation cannot directly apply a 128x128 calibrated rotation after the existing original-coordinate regular/outlier split. The next implementation should quantize the full vector in the calibrated rotation domain and allocate the fractional high-bit coordinates there.

### Fractional Rotated-Outlier ACAR Follow-Up

Candidate:

```text
calibrated_rotated_outlier_mse
```

Rationale: apply the calibrated orthogonal second-Hadamard matrix to the full 128D K/V vector before fractional-bit allocation. Low-bit and high-bit coordinates are selected in the rotated domain, then the full vector is reconstructed through the inverse calibrated rotation. This directly tests whether ACAR can improve the user-required 2.5-bit and 3.5-bit TurboQuant settings rather than only the integer 3.0-bit path.

2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| ACAR rotated-outlier | 2.5 | 20 | 33.54 | +0.78 | 0.30 | 0.17200 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| ACAR rotated-outlier | 3.5 | 20 | 48.95 | +7.08 | 0.45 | 0.23449 |
| TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | 0.30 | 0.17199 |
| ACAR rotated-outlier | 2.5 | 50 | 35.12 | -0.82 | 0.30 | 0.17200 |
| TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | 0.34 | 0.23450 |
| ACAR rotated-outlier | 3.5 | 50 | 39.35 | -3.38 | 0.32 | 0.23451 |

Artifacts:

```text
reproduce/incremental/acar/second_hadamard_rotations_2wikimqa_calib0_4_1024.json
reproduce/runs/incremental/acar_second_hadamard_rotated_outlier_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/acar_second_hadamard_rotated_outlier_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/runs/incremental/acar_second_hadamard_rotated_outlier_2wikimqa_turboquant_2p5_0_50.jsonl
reproduce/runs/incremental/acar_second_hadamard_rotated_outlier_2wikimqa_turboquant_3p5_0_50.jsonl
reproduce/incremental/acar_second_hadamard_rotated_outlier_2wikimqa_turboquant_2p5_0_20.json
reproduce/incremental/acar_second_hadamard_rotated_outlier_2wikimqa_turboquant_3p5_0_20.json
reproduce/incremental/acar_second_hadamard_rotated_outlier_2wikimqa_turboquant_2p5_0_50.json
reproduce/incremental/acar_second_hadamard_rotated_outlier_2wikimqa_turboquant_3p5_0_50.json
```

Decision: do not expand. The 0:20 positive signal does not survive the 0:50 block at either bit budget.

## Sign-Balanced Hadamard Rotation Probe

Candidate:

```text
sign_balanced_hadamard + calibrated_rotated_outlier_mse
```

Rationale: preserve the fast orthogonal Hadamard structure, but calibrate the Rademacher sign vector before the Hadamard transform for each layer and KV side. The fixed calibration objective selects signs that reduce the shared 2.5-bit and 3.5-bit rotated-domain reconstruction error on unlabeled calibration prompts.

Calibration artifact:

```text
reproduce/incremental/acar/sign_balanced_hadamard_2wikimqa_calib0_4_1024_bank16.json
```

Calibration objective summary:

| Side | Avg best normalized objective | Min | Max | Non-default layers |
| --- | ---: | ---: | ---: | ---: |
| Key | 0.935 | 0.818 | 1.000 | 30/32 |
| Value | 0.951 | 0.842 | 1.000 | 27/32 |

2WikiMQA `0:20` screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| Sign-balanced Hadamard | 2.5 | 20 | 32.20 | -0.56 | 0.25 | 0.17197 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| Sign-balanced Hadamard | 3.5 | 20 | 44.29 | +2.42 | 0.45 | 0.23453 |

Artifacts:

```text
reproduce/runs/incremental/sign_balanced_hadamard_2wikimqa_turboquant_2p5_0_20.jsonl
reproduce/runs/incremental/sign_balanced_hadamard_2wikimqa_turboquant_3p5_0_20.jsonl
reproduce/incremental/sign_balanced_hadamard_2wikimqa_turboquant_2p5_0_20.json
reproduce/incremental/sign_balanced_hadamard_2wikimqa_turboquant_3p5_0_20.json
```

Decision: do not expand. This is a clean Hadamard-core modification and improves 3.5-bit on the small block, but it regresses 2.5-bit and therefore fails the unified 2.5/3.5 requirement.

## RMS-Balanced Rotation Preconditioning Probe

Candidate:

```text
rms_rotation_mse
```

Rationale: before TurboQuant's random orthogonal rotation, scale each coordinate by its segment RMS normalized to geometric mean. This attempts to make the vector distribution more isotropic before scalar Lloyd-Max quantization while preserving the same bit budget and the same base rotation quantizer.

2WikiMQA screening and full validation:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 20 | 32.76 | +0.00 | 0.30 | 0.17199 |
| RMS-balanced rotation | 2.5 | 20 | 38.87 | +6.11 | 0.35 | 0.17204 |
| TurboQuant MSE | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23450 |
| RMS-balanced rotation | 3.5 | 20 | 41.87 | +0.00 | 0.35 | 0.23451 |
| TurboQuant MSE | 2.5 | 50 | 35.94 | +0.00 | 0.30 | 0.17199 |
| RMS-balanced rotation | 2.5 | 50 | 37.29 | +1.34 | 0.32 | 0.17206 |
| TurboQuant MSE | 3.5 | 50 | 42.72 | +0.00 | 0.34 | 0.23450 |
| RMS-balanced rotation | 3.5 | 50 | 44.68 | +1.96 | 0.38 | 0.23452 |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.355 | 0.17199 |
| RMS-balanced rotation | 2.5 | 200 | 34.75 | -3.21 | 0.315 | 0.17202 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23450 |
| RMS-balanced rotation | 3.5 | 200 | 42.33 | -2.14 | 0.425 | 0.23452 |

Artifacts:

```text
reproduce/runs/incremental/rms_rotation_mse_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/rms_rotation_mse_2wikimqa_turboquant_3p5_full.jsonl
reproduce/incremental/rms_rotation_mse_2wikimqa_tq25_0_20.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq35_0_20.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq25_0_50.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq35_0_50.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq25_full.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq35_full.json
```

Decision: do not expand to Table 1. The 0:20 and 0:50 slices were promising, but the full 2WikiMQA task regresses both bit budgets.

## Seed Protocol and Slice-Stability Note

Observation: the current runner uses `seed=index` when `--fixed-kv-seed` is not set. Passing `--fixed-kv-seed 0` uses one shared rotation seed for every example. This is not an incremental method, but it is important for interpreting any rotation experiment.

2WikiMQA full validation:

| Method | KV bits | Examples | LongBench score | Delta vs reproduced TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reproduced TurboQuant MSE (`seed=index`) | 2.5 | 200 | 37.96 | +0.00 | 0.355 | 0.17200 |
| Fixed-seed TurboQuant MSE (`seed=0`) | 2.5 | 200 | 34.22 | -3.73 | 0.310 | 0.17199 |
| Reproduced TurboQuant MSE (`seed=index`) | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23450 |
| Fixed-seed TurboQuant MSE (`seed=0`) | 3.5 | 200 | 46.57 | +2.11 | 0.460 | 0.23450 |

2WikiMQA 50-example block scores:

| Method | KV bits | 0:50 | 50:100 | 100:150 | 150:200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reproduced TurboQuant MSE | 2.5 | 35.94 | 35.19 | 48.86 | 31.84 |
| RMS-balanced rotation | 2.5 | 37.29 | 38.00 | 34.55 | 29.16 |
| Fixed-seed TurboQuant MSE | 2.5 | 37.26 | 41.93 | 39.10 | 18.60 |
| Reproduced TurboQuant MSE | 3.5 | 42.72 | 44.72 | 46.91 | 43.51 |
| RMS-balanced rotation | 3.5 | 44.68 | 43.52 | 45.04 | 36.09 |
| Fixed-seed TurboQuant MSE | 3.5 | 48.25 | 45.06 | 57.81 | 35.18 |

Artifacts:

```text
reproduce/runs/incremental/fixed_seed0_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/fixed_seed0_2wikimqa_turboquant_3p5_full.jsonl
reproduce/incremental/fixed_seed0_2wikimqa_turboquant_2p5_full.json
reproduce/incremental/fixed_seed0_2wikimqa_turboquant_3p5_full.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq25_050_100.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq25_100_150.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq25_150_200.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq35_050_100.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq35_100_150.json
reproduce/incremental/rms_rotation_mse_2wikimqa_tq35_150_200.json
```

Decision: future rotation-method screening should not rely on the leading 0:20 or 0:50 block. The 150:200 block reverses several apparently positive candidates, so the next screening protocol should either run the full task or use a stratified slice spanning the full index range.

## Full-Head Outlier-Aware Hadamard Probe

Candidate:

```text
outlier_hadamard_mse, outlier_hadamard_block_size=128
```

Rationale: the earlier outlier-aware Hadamard probe used local 16D blocks. This variant uses one full 128D Hadamard block per KV head, making the transform closer to a full-head structured rotation while preserving the data-dependent outlier-spreading permutation and random sign flip.

Stratified 2WikiMQA screening uses indices `0:20`, `50:70`, `100:120`, and `150:170` to avoid the leading-slice bias observed above.

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| Full-head outlier Hadamard | 2.5 | 80 | 29.62 | -5.92 | 0.2375 | 0.17215 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23450 |
| Full-head outlier Hadamard | 3.5 | 80 | 42.59 | +1.25 | 0.400 | 0.23468 |

Artifacts:

```text
reproduce/runs/incremental/outlier_hadamard_b128_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/outlier_hadamard_b128_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/incremental/outlier_hadamard_b128_2wikimqa_turboquant_2p5_stratified80.json
reproduce/incremental/outlier_hadamard_b128_2wikimqa_turboquant_3p5_stratified80.json
```

Decision: do not expand. The full-head Hadamard rotation improves 3.5-bit on the stratified slice, but it badly regresses 2.5-bit and therefore fails the unified 2.5/3.5 requirement.

### Value-Only Full-Head Outlier-Aware Hadamard

Candidate:

```text
key_quantizer=mse
value_quantizer=outlier_hadamard_mse
outlier_hadamard_block_size=128
```

Rationale: preserve K on reproduced TurboQuant to avoid perturbing attention logits, and apply the structured Hadamard rotation only to V.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| Value-only full-head Hadamard | 2.5 | 80 | 32.24 | -3.30 | 0.2875 | 0.17209 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23450 |
| Value-only full-head Hadamard | 3.5 | 80 | 44.01 | +2.67 | 0.4375 | 0.23460 |

Artifacts:

```text
reproduce/runs/incremental/value_outlier_hadamard_b128_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/value_outlier_hadamard_b128_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/incremental/value_outlier_hadamard_b128_2wikimqa_turboquant_2p5_stratified80.json
reproduce/incremental/value_outlier_hadamard_b128_2wikimqa_turboquant_3p5_stratified80.json
```

Decision: do not expand. Keeping K on baseline improves the 3.5-bit signal, but 2.5-bit remains negative.

### Attention-Entropy Guarded Value Outlier-Aware Hadamard

Candidate:

```text
key_quantizer=mse
value_quantizer=entropy_guarded_outlier_hadamard_mse
outlier_hadamard_block_size=16
attention_error_query_tokens=8
attention_entropy_threshold=0.80
```

Rationale: keep K on reproduced TurboQuant and apply the structured outlier-Hadamard V rotation only when the current prefill attention is sufficiently concentrated. The method uses internal Q/K attention entropy rather than prompt text, task name, or answer feedback. This is a core rotation/preconditioning change because the active path changes the value-side coordinates before the scalar Lloyd-Max quantizer, but it tries to avoid applying the transform when attention is diffuse and value reconstruction noise is broadly averaged across many tokens.

Implementation:

```text
turboquant/kv_cache.py
experiments/longbench/run_full_cache_eval.py
tests/test_core.py
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
71 passed
```

Full LongBench validation on the two MultiQA tasks that previously exposed false positives:

| Method | KV bits | Dataset | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 2wikimqa | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| Entropy-Guarded V Outlier-Hadamard | 2.5 | 2wikimqa | 200 | 35.94 | -2.02 | 0.330 | 0.17209 |
| TurboQuant MSE | 2.5 | musique | 200 | 25.24 | +0.00 | 0.145 | 0.17199 |
| Entropy-Guarded V Outlier-Hadamard | 2.5 | musique | 200 | 20.22 | -5.02 | 0.130 | 0.17195 |

Artifacts:

```text
reproduce/runs/incremental/entropy_guarded_hadamard_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/entropy_guarded_hadamard_musique_turboquant_2p5_full.jsonl
reproduce/incremental/entropy_guarded_hadamard_2wikimqa_turboquant_2p5_full.json
reproduce/incremental/entropy_guarded_hadamard_musique_turboquant_2p5_full.json
```

Decision: stop. The method fails full 2.5-bit validation on both 2WikiMQA and Musique, so it cannot satisfy the unified 2.5-bit/3.5-bit requirement. The remaining 3.5-bit jobs were terminated after the 2.5-bit failure to avoid spending GPU time on a method that already fails the required setting.

Analysis: attention entropy alone is not a reliable guard for the outlier-Hadamard V rotation. The cache ratio indicates that the active path still changes many segments, and full-task accuracy drops more than the earlier unconditional block-16 V outlier-Hadamard on Musique. Future rotation-level candidates should avoid a binary activation rule based only on a global attention-diffuseness statistic. A more plausible direction is to preserve TurboQuant's random rotation but alter the fractional-bit coordinate allocation inside the rotated domain, because the repeated failures suggest the transform itself is often more disruptive than the 2/3-bit coordinate assignment.

### Structured Randomized Hadamard Rotation Probe

Candidate:

```text
srht_mse
```

Rationale: replace TurboQuant's dense random QR rotation with a structured randomized Hadamard transform: random Rademacher signs, normalized Hadamard mixing, and a random coordinate permutation. This directly changes the core orthogonal rotation matrix family while keeping the same scalar Lloyd-Max quantizer, norm restoration, bit budget, and K/V treatment. Unlike prompt gates or task-specific bit schedules, this is a rotation-kernel modification.

Implementation:

```text
turboquant/kv_cache.py
experiments/longbench/run_full_cache_eval.py
tests/test_core.py
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
72 passed
```

The unit test verifies that the SRHT transform is orthogonal and has the expected Hadamard equal-magnitude entries.

Stratified 2WikiMQA screening uses indices `0:20`, `50:70`, `100:120`, and `150:170`.

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| SRHT MSE | 2.5 | 80 | 29.36 | -6.19 | 0.250 | 0.17198 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| SRHT MSE | 3.5 | 80 | 34.52 | -6.82 | 0.350 | 0.23452 |

Artifacts:

```text
reproduce/runs/incremental/srht_mse_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/srht_mse_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/incremental/srht_mse_2wikimqa_turboquant_2p5_stratified80.json
reproduce/incremental/srht_mse_2wikimqa_turboquant_3p5_stratified80.json
```

Decision: do not expand. SRHT is a clean core-rotation change, but it regresses both bit budgets on the stratified screen. This also explains why the earlier plain `hadamard_mse` failure was not just missing a random permutation; the dense random orthogonal rotation appears important for this KV cache setting.

## Attention-Weighted Rotated-Domain Fractional Allocation

Candidate:

```text
key_quantizer=attention_rotated_outlier_mse
value_quantizer=mse
attention_error_query_tokens=8
```

Rationale: keep TurboQuant's dense random orthogonal rotation, but move the fractional-bit high-precision coordinate allocation into the rotated domain. The high-bit coordinates are selected by an answer-free attention proxy: key coordinates are weighted by query sensitivity in the same rotated basis and by current attention probability, while value coordinates are weighted by attention-output probability. This is a core rotation-domain change rather than a prompt/task gate.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| K attention-rotated outlier | 2.5 | 80 | 39.72 | +4.17 | 0.363 | 0.17200 |
| V attention-rotated outlier | 2.5 | 80 | 39.39 | +3.85 | 0.350 | 0.17200 |
| K/V attention-rotated outlier | 2.5 | 80 | 31.37 | -4.17 | 0.288 | 0.17200 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| K attention-rotated outlier | 3.5 | 80 | 45.24 | +3.89 | 0.413 | 0.23449 |
| V attention-rotated outlier | 3.5 | 80 | 39.95 | -1.39 | 0.413 | 0.23450 |
| K/V attention-rotated outlier | 3.5 | 80 | 40.95 | -0.39 | 0.400 | 0.23450 |

Full 2WikiMQA validation for the only passing stratified variant:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| K attention-rotated outlier | 2.5 | 200 | 34.85 | -3.11 | 0.340 | 0.17200 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| K attention-rotated outlier | 3.5 | 200 | 44.23 | -0.23 | 0.425 | 0.23450 |

Decision: do not expand to MultiQA. The stratified 80 screen was a false positive, especially at 2.5-bit. Selecting high-bit coordinates directly in the rotated basis can protect some query-sensitive slices, but it is not stable across the full 2WikiMQA distribution.

## Attention-Safeguarded Rotated-Domain Fractional Allocation

Candidate:

```text
key_quantizer=attention_adaptive_rotated_outlier_mse
value_quantizer=mse
attention_error_query_tokens=8
```

Rationale: add a conservative segment-level guard to the previous candidate. The method compares reproduced TurboQuant MSE against attention-weighted rotated-domain fractional allocation and activates the rotated-domain allocation only when the same answer-free attention proxy prefers it.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| K attention-adaptive rotated outlier | 2.5 | 80 | 35.99 | +0.45 | 0.338 | 0.17199 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| K attention-adaptive rotated outlier | 3.5 | 80 | 38.96 | -2.38 | 0.375 | 0.23450 |

Decision: do not expand. The attention proxy is not reliable enough as a segment-level binary guard, and it removes the useful 3.5-bit stratified signal from direct attention-weighted allocation.

## Vector-Adaptive Outlier-Hadamard Selector

Candidate:

```text
vector_adaptive_outlier_hadamard_mse
```

Rationale: previous Hadamard/preconditioning variants used segment-level activation and often moved many helpful and harmful vectors together. This candidate selects between reproduced TurboQuant dense random-rotation MSE and outlier-aware block-Hadamard preconditioning independently for each KV vector, using local reconstruction MSE as the selector. It stores one selector bit per vector and stores only the selected path's packed payload, so it remains a core rotation/preconditioning method rather than a prompt or task gate.

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
79 passed
```

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| V-only vector-adaptive Hadamard | 2.5 | 80 | 33.66 | -1.88 | 0.288 | 0.17258 |
| K-only vector-adaptive Hadamard | 2.5 | 80 | 36.08 | +0.54 | 0.313 | 0.17257 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| V-only vector-adaptive Hadamard | 3.5 | 80 | 40.93 | -0.42 | 0.413 | 0.23510 |
| K-only vector-adaptive Hadamard | 3.5 | 80 | 45.03 | +3.69 | 0.438 | 0.23508 |

Full 2WikiMQA validation for the only passing stratified variant:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| K-only vector-adaptive Hadamard | 2.5 | 200 | 36.45 | -1.51 | 0.345 | 0.17257 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| K-only vector-adaptive Hadamard | 3.5 | 200 | 43.03 | -1.44 | 0.430 | 0.23509 |

Artifacts:

```text
reproduce/runs/incremental/vector_adaptive_v_hadamard_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/vector_adaptive_v_hadamard_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/runs/incremental/vector_adaptive_k_hadamard_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/vector_adaptive_k_hadamard_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/runs/incremental/vector_adaptive_k_hadamard_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/vector_adaptive_k_hadamard_2wikimqa_turboquant_3p5_full.jsonl
```

Decision: do not expand. The vector-level selector removes some segment-level coarseness, but local reconstruction MSE is still not aligned with full downstream QA quality. This reinforces that the next core-method attempt needs an objective closer to attention behavior than K/V reconstruction error alone.

## Attention-Weighted Hadamard Residual

Candidate:

```text
key_quantizer=attention_weighted_hadamard_residual_mse
value_quantizer=mse
attention_error_query_tokens=8
```

Rationale: keep TurboQuant's dense random rotation and scalar Lloyd-Max base, but use the fractional residual budget for sparse Hadamard residual signs chosen by an attention-logit sensitivity objective. For each candidate Hadamard coefficient, the score combines residual coefficient magnitude, the current grouped query's coefficient magnitude in the same Hadamard basis, and causal attention probability. This avoids prompt/task gates and avoids selecting residual coefficients by K/V reconstruction error alone.

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
80 passed
```

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| K attention-weighted Hadamard residual | 2.5 | 80 | 33.01 | -2.53 | 0.300 | 0.17197 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| K attention-weighted Hadamard residual | 3.5 | 80 | 40.70 | -0.64 | 0.388 | 0.23446 |

Artifacts:

```text
reproduce/runs/incremental/attention_weighted_hadamard_residual_key_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/attention_weighted_hadamard_residual_key_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/incremental/attention_weighted_hadamard_residual_key_2wikimqa_turboquant_2p5_stratified80.json
reproduce/incremental/attention_weighted_hadamard_residual_key_2wikimqa_turboquant_3p5_stratified80.json
```

Decision: do not expand. A more attention-aligned coefficient selector still regresses both bit budgets, so sparse Hadamard residual correction is not a promising path for the requested unified 2.5/3.5 improvement.

## Late-Layer Value Hadamard

Candidate:

```text
key_quantizer=mse
value_quantizer=mse
layer_value_quantizers=mse x16, outlier_hadamard_mse x16
outlier_hadamard_block_size=16
```

Rationale: preserve K and the early value layers on reproduced TurboQuant, while applying outlier-aware block-Hadamard preconditioning only to later value layers. This is a core rotation/preconditioning change with a fixed model-internal layer rule. It is intended to keep early retrieval-style value information stable while testing whether later layers benefit from the value-side Hadamard signal observed in prior probes.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| Late-layer V Hadamard | 2.5 | 80 | 36.23 | +0.69 | 0.313 | 0.17204 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| Late-layer V Hadamard | 3.5 | 80 | 44.24 | +2.90 | 0.425 | 0.23456 |

Full 2WikiMQA validation:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 200 | 37.96 | +0.00 | 0.365 | 0.17200 |
| Late-layer V Hadamard | 2.5 | 200 | 36.02 | -1.94 | 0.335 | 0.17204 |
| TurboQuant MSE | 3.5 | 200 | 44.47 | +0.00 | 0.425 | 0.23449 |
| Late-layer V Hadamard | 3.5 | 200 | 46.09 | +1.62 | 0.450 | 0.23455 |

Artifacts:

```text
reproduce/runs/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/runs/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_2p5_full.jsonl
reproduce/runs/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_3p5_full.jsonl
reproduce/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_2p5_stratified80.json
reproduce/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_3p5_stratified80.json
reproduce/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_2p5_full.json
reproduce/incremental/late16_value_hadamard_b16_2wikimqa_turboquant_3p5_full.json
```

Decision: do not expand. The method gives a strong 3.5-bit full-task gain but fails 2.5-bit full validation, so it does not satisfy the requested same-method improvement over TurboQuant at both reported bit budgets.

## Value Regular-Only Hadamard

Candidate:

```text
key_quantizer=mse
value_quantizer=regular_outlier_hadamard_mse
outlier_hadamard_block_size=16
```

Rationale: apply the outlier-aware block-Hadamard transform only to the low-bit regular value subspace, while leaving the high-bit outlier value subspace on the reproduced TurboQuant random-rotation MSE path. This tests whether the 2.5-bit failures come from perturbing high-magnitude value channels.

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| V regular-only Hadamard | 2.5 | 80 | 32.12 | -3.42 | 0.263 | 0.17205 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| V regular-only Hadamard | 3.5 | 80 | 42.74 | +1.40 | 0.413 | 0.23454 |

Artifacts:

```text
reproduce/runs/incremental/value_regular_outlier_hadamard_b16_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/value_regular_outlier_hadamard_b16_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/incremental/value_regular_outlier_hadamard_b16_2wikimqa_turboquant_2p5_stratified80.json
reproduce/incremental/value_regular_outlier_hadamard_b16_2wikimqa_turboquant_3p5_stratified80.json
```

Decision: do not expand. Keeping the outlier subspace on TurboQuant does not stabilize the 2.5-bit path; the regular low-bit value subspace appears to be the source of the Hadamard-related degradation.

## Value Outlier-Only Hadamard

Candidate:

```text
key_quantizer=mse
value_quantizer=outlier_only_hadamard_mse
outlier_hadamard_block_size=16
```

Rationale: preserve the low-bit regular value subspace on reproduced TurboQuant MSE, and apply outlier-aware block-Hadamard only to the high-bit value outlier subspace. This is the complement of the previous regular-only Hadamard test.

Implementation:

```text
turboquant/kv_cache.py
experiments/longbench/run_full_cache_eval.py
tests/test_core.py
```

Validation:

```text
/home/liying/miniconda3/envs/turboquant/bin/python -m pytest tests/test_core.py -q
81 passed
```

Stratified 2WikiMQA screening:

| Method | KV bits | Examples | LongBench score | Delta vs TQ | Contains-answer acc | Avg cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TurboQuant MSE | 2.5 | 80 | 35.54 | +0.00 | 0.325 | 0.17200 |
| V outlier-only Hadamard | 2.5 | 80 | 31.59 | -3.96 | 0.275 | 0.17205 |
| TurboQuant MSE | 3.5 | 80 | 41.34 | +0.00 | 0.375 | 0.23449 |
| V outlier-only Hadamard | 3.5 | 80 | 42.20 | +0.85 | 0.425 | 0.23455 |

Artifacts:

```text
reproduce/runs/incremental/value_outlier_only_hadamard_b16_2wikimqa_turboquant_2p5_stratified80.jsonl
reproduce/runs/incremental/value_outlier_only_hadamard_b16_2wikimqa_turboquant_3p5_stratified80.jsonl
reproduce/incremental/value_outlier_only_hadamard_b16_2wikimqa_turboquant_2p5_stratified80.json
reproduce/incremental/value_outlier_only_hadamard_b16_2wikimqa_turboquant_3p5_stratified80.json
```

Decision: do not expand. Both regular-only and outlier-only V-Hadamard variants fail the 2.5-bit screen, so subspace isolation is not a viable low-bit stabilizer.

## Reproduce Full Validation

```bash
python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --key-quantizer rotation_bank_mse \
  --value-quantizer rotation_bank_mse \
  --rotation-bank-size 4 \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index 200 \
  --output reproduce/runs/incremental/rotation_bank_mse_2wikimqa_turboquant_2p5_full.jsonl
```

Repeat with `--kv-bits 3.5` and output:

```text
reproduce/runs/incremental/rotation_bank_mse_2wikimqa_turboquant_3p5_full.jsonl
```
