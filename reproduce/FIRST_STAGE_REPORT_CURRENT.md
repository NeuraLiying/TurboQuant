# TurboQuant First-Stage Reproduction Report

Generated: 2026-06-11T09:14:39+08:00

Scope: `meta-llama/Llama-3.1-8B-Instruct`; compare Full Cache / Full-Precision vs TurboQuant only.

## Completion

- Table 1 full reproduction: incomplete
- Figure 4 full reproduction: incomplete
- Figure 5 DBpedia reproduction: complete
- Figure 5 GloVe reproduction: incomplete
- Core algorithm validation: complete

Current blocking data gaps remain explicit in the refresh summary below.

## Refresh Summary

- Asset audit: `{'longbench': {'metadata_or_lock_only': 30, 'materialized': 2}, 'needle': {'missing': 5, 'materialized': 1}, 'dbpedia': {'materialized': 2}}`
- Table 1 manifest: `{'complete': 6, 'missing_dataset': 90}`
- Table 1 plan: `{'manifest_status_counts': {'complete': 6, 'missing_dataset': 90}, 'num_planned_entries': 0, 'num_planned_jobs': 0, 'planned_status_counts': {}, 'skipped_status_counts': {'missing_dataset': 90, 'complete': 6}}`
- Figure 4 plan: `{'num_planned_entries': 0, 'planned_status_counts': {}, 'skipped_status_counts': {'missing_dataset': 10, 'complete': 2}, 'all_status_counts': {'missing_dataset': 10, 'complete': 2}}`

## Table 1

| Method | KV | Paper SingleQA | Paper MultiQA | Paper Summ. | Paper Few shot | Paper Synthetic | Paper Code | Paper Avg | Local SingleQA | Local MultiQA | Local Summ. | Local Few shot | Local Synthetic | Local Code | Local Available Avg | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full Cache | 16.0 | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |  | 11.33 |  |  |  |  | 11.33 | SingleQA: none; MultiQA: 2wikimqa; Summarization: none; Few shot: none; Synthetic: none; Code: none |
| TurboQuant | 2.5 | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |  | 11.86 |  |  |  |  | 11.86 | SingleQA: none; MultiQA: 2wikimqa; Summarization: none; Few shot: none; Synthetic: none; Code: none |
| TurboQuant | 3.5 | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |  | 12.56 |  |  |  |  | 12.56 | SingleQA: none; MultiQA: 2wikimqa; Summarization: none; Few shot: none; Synthetic: none; Code: none |

Table 1 local values are partial until all LongBench / LongBench-E datasets are materialized.

## Figure 4

| Method | KV | Paper Score | Local Examples | Local Score | Token Limits | Depth Percents | Cache Ratio |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Full-Precision | 16.0 | 0.997 | 24 | 1.0000 | 8192,16384 | 0,50,100 |  |
| TurboQuant | 2.5 | 0.997 | 24 | 1.0000 | 8192,16384 | 0,50,100 | 0.1723 |

Figure 4 local values currently cover only materialized 16k EN data; missing token lengths remain absent.

## Figure 5 DBpedia

| Dataset | Dim | Bits | Top-k | Recall | Database | Queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dbpedia_openai3_1536 | 1536 | 2 | 1 | 0.9000 | 100000 | 1000 |
| dbpedia_openai3_1536 | 1536 | 4 | 1 | 0.9660 | 100000 | 1000 |
| dbpedia_openai3_3072 | 3072 | 2 | 1 | 0.9040 | 100000 | 1000 |
| dbpedia_openai3_3072 | 3072 | 4 | 1 | 0.9730 | 100000 | 1000 |

GloVe 200d is still unavailable, so Figure 5 is complete only for DBpedia.

## Table 2 Timing

| Dimension | Source | Local Seconds | Paper Seconds | Local/Paper |
| ---: | --- | ---: | ---: | ---: |
| 200 | random_unit | 0.000147 | 0.000700 | 0.210 |
| 1536 | dbpedia_cache | 0.001425 | 0.001300 | 1.096 |
| 3072 | dbpedia_cache | 0.003256 | 0.002100 | 1.551 |

## KV Compression Policy

| Run | Regular bits | Outlier bits | Outlier channels | Effective KV bits | Cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| longbench_e_2wikimqa_tq_2_5 | 2 | 3 | 64 | 2.50 | 0.1727 |
| longbench_e_2wikimqa_tq_3_5 | 3 | 4 | 64 | 3.50 | 0.2353 |
| longbench_2wikimqa_tq_2_5 | 2 | 3 | 64 | 2.50 | 0.1729 |
| longbench_2wikimqa_tq_3_5 | 3 | 4 | 64 | 3.50 | 0.2355 |
| needle_16k_tq_2_5 | 2 | 3 | 64 | 2.50 | 0.1723 |
| needle_generated_4k_tq_2_5 | 2 | 3 | 64 | 2.50 | 0.1728 |

The current fractional-bit implementation stores packed regular channels plus higher-bit outlier channels, then dequantizes before attention. It is a faithful Python-level reproduction path, not a fused-kernel speed reproduction.

## Core Validation

| Bits | MSE mean | Inner-product error mean |
| ---: | ---: | ---: |
| 1 | 0.3611 | 0.000926 |
| 2 | 0.1167 | 0.000184 |
| 3 | 0.0341 | -0.000254 |
| 4 | 0.0094 | 0.000023 |

## Sources

- `refresh`: `/home/liying/projects/turboquant/reproduce/logs/reproduction_state_refresh_goal_resume_paths_fixed_2026_06_11.json`
- `table1_summary`: `/home/liying/projects/turboquant/reproduce/runs/table1_llama_available_summary.json`
- `figure4_heatmap`: `/home/liying/projects/turboquant/reproduce/runs/figure4_needle_available_heatmap.json`
- `figure5_summary`: `/home/liying/projects/turboquant/reproduce/runs/figure5_turboquant_dbpedia_fullscale.json`
- `table2_summary`: `/home/liying/projects/turboquant/reproduce/runs/table2_quantization_time_summary.json`
- `kv_policy`: `/home/liying/projects/turboquant/reproduce/runs/kv_compression_policy_summary.json`
- `figure3_summary`: `/home/liying/projects/turboquant/reproduce/runs/figure3_core_d256_summary.json`
