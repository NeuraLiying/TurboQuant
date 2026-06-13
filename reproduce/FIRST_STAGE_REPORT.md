# TurboQuant 第一阶段复现报告

> 当前可重复生成的第一阶段汇总入口是 `reproduce/FIRST_STAGE_REPORT_CURRENT.md` 和 `reproduce/FIRST_STAGE_REPORT_CURRENT.json`，生成脚本为 `scripts/build_first_stage_report.py`。本文件保留为 2026-06-11 08:17 的长版手写记录。

生成时间：2026-06-11 08:17 Asia/Shanghai

论文文件：`/home/liying/projects/turboquant/paper/turboquant.pdf`

本报告记录当前第一阶段复现状态。复现目标限定为论文主要实验结果中 TurboQuant 方法本身的代码复现和结果对齐；不复现 motivation、理论证明，也不复现 KIVI、PolarQuant、SnapKV、PyramidKV、PQ、RabitQ 等 baseline。当前模型限定为 `meta-llama/Llama-3.1-8B-Instruct`。

## 1. 当前结论

当前已经完成：

- TurboQuant core 算法实现与 Figure 3-style MSE sanity check。
- Llama LongBench-E `2wikimqa` Full Cache 300/300 本地完整任务。
- Llama 标准 LongBench `2wikimqa` Full Cache 200/200 本地完整任务。
- LongBench 数据发现与评测入口已支持 datasets cache Arrow 和 Hub snapshot parquet。
- TurboQuant packed KV cache 的整数 bit、2.5-bit、3.5-bit smoke 验证。
- Llama LongBench-E `2wikimqa` TurboQuant 2.5-bit / 3.5-bit 300/300 本地完整任务。
- Llama 标准 LongBench `2wikimqa` TurboQuant 2.5-bit / 3.5-bit 200/200 本地完整任务。
- Needle 16k Full-Precision 与 TurboQuant 24-example 本地网格。
- Needle 16k ar/de/en/es/hi/vi/zh 七个本地 split 发现、注册和多 Arrow 加载验证。
- Generated Needle length-grid 数据构造与分目标 token 评测入口。
- Generated Needle 4k Full-Precision / TurboQuant 诊断切片。
- Figure-4-style Needle heatmap 生成脚本和本地 16k / generated 4k heatmap 验证产物。
- TurboQuant KV compression policy 汇总，记录 2.5-bit / 3.5-bit 的 outlier channel 策略证据。
- DBpedia 1536d / 3072d 论文规模 100k database / 1k query 的 TurboQuant ANN recall 曲线。
- Table 2-style 4-bit quantization timing summary。

当前不能声称完整复现的部分：

- Table 1 还不是完整复现，因为 2026-06-11 07:16 复查后，本地 LongBench / LongBench-E cache 仍只有 `2wikimqa` / `2wikimqa_e`；Table 1 manifest 当前是 6 个已完成条目、90 个 missing-dataset 条目。
- Table 1 还不是完整复现，因为本地只有 `2wikimqa_e` 和标准 `2wikimqa` 两个 MultiQA 子任务；不过这两个本地任务的 Full Cache、TurboQuant 2.5-bit、TurboQuant 3.5-bit 都已跑完。
- Figure 4 还不是完整 heatmap，因为本地 Needle cache 只有 16k 七个语言 split，缺少论文覆盖的真实 4k 到 104k token 网格。
- Generated Needle 4k 诊断切片不能替代论文 Figure 4：Full-Precision 自身在该裁剪数据上只有 0.3333，说明裁剪后的数据分布/位置结构不可靠。
- Figure 5 缺 GloVe 200d；DBpedia 两个维度已经完成论文规模 TurboQuant 曲线。

## 2. 环境和资产

运行环境：

- Conda env：`turboquant`
- 创建方式：从已有 `layer_skip` clone
- `torch 2.2.1+cu121`
- `transformers 4.53.0`
- `datasets 3.6.0`
- `accelerate 1.8.1`
- `flash-attn 2.5.6`
- GPU：8 x NVIDIA GeForce RTX 4090

本地资产：

- Llama snapshot：`/home/liying/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`
- LongBench-E `2wikimqa`：300 test examples
- LongBench `2wikimqa`：200 test examples
- Needle 16k ar/de/en/es/hi/vi/zh：每个 split 2400 examples，合计 16800 examples
- DBpedia OpenAI3 1536d / 3072d embeddings
- GloVe 200d：未提供
- 机器可复查数据清单：`reproduce/logs/hf_cache_inventory_enhanced_2026-06-11.json`
- 数据覆盖审计：`reproduce/DATA_COVERAGE_AUDIT.md`

环境记录见：`reproduce/ENVIRONMENT.md`  
运行状态记录见：`reproduce/STATUS.md`

## 3. Table 1：LongBench / LongBench-E

论文中第一阶段关注的 Llama-3.1-8B-Instruct 目标行：

| Model | Method | KV Size | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | Full Cache | 16 | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| Llama-3.1-8B-Instruct | TurboQuant | 2.5 | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |
| Llama-3.1-8B-Instruct | TurboQuant | 3.5 | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |

已实现：

- `experiments/longbench/run_full_cache_eval.py`
- `turboquant/kv_cache.py`
- `turboquant/longbench_metrics.py`
- `scripts/summarize_jsonl_accuracy.py`
- `scripts/backfill_longbench_scores.py`
- `scripts/build_table1_summary.py`
- `scripts/merge_jsonl_by_index.py`
- `scripts/prepare_longbench_cache.py`
- `scripts/build_table1_manifest.py`

当前本地完整任务结果：

| Dataset | Method | Examples | Contains-answer accuracy | LongBench-style score | Avg prompt tokens | Avg latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LongBench-E `2wikimqa` | Full Cache | 300/300 | 0.65 | 11.2523 | 8904.37 | 2.76s/example |
| LongBench `2wikimqa` | Full Cache | 200/200 | 0.6450 | 11.4448 | 7168.24 | 2.39s/example |

当前本地 TurboQuant 完整任务结果：

| Dataset | Method | Examples | Contains-answer accuracy | LongBench-style score | Avg prompt tokens | Avg latency | Cache ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench-E `2wikimqa` | TurboQuant 2.5-bit | 300/300 | 0.5233 | 12.2449 | 8904.37 | 43.89s/example | 0.1727 |
| LongBench-E `2wikimqa` | TurboQuant 3.5-bit | 300/300 | 0.6233 | 12.9650 | 8904.37 | 50.21s/example | 0.2353 |
| LongBench `2wikimqa` | TurboQuant 2.5-bit | 200/200 | 0.4750 | 11.2938 | 7168.24 | 41.26s/example | 0.1729 |
| LongBench `2wikimqa` | TurboQuant 3.5-bit | 200/200 | 0.6000 | 11.9544 | 7168.24 | 51.48s/example | 0.2355 |

产物：

- `reproduce/runs/longbench_e_2wikimqa_full_cache_all.jsonl`
- `reproduce/runs/longbench_e_2wikimqa_full_cache_all.summary.json`
- `reproduce/runs/longbench_e_2wikimqa_full_cache_all.aggregate.json`
- `reproduce/runs/longbench_e_2wikimqa_turboquant_2_5bit_chunked.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/longbench_e_2wikimqa_turboquant_3_5bit_chunked.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/longbench_2wikimqa_full_cache_all.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/longbench_2wikimqa_turboquant_2_5bit_chunked.{jsonl,aggregate.json}`
- `reproduce/runs/longbench_2wikimqa_turboquant_3_5bit_chunked.{jsonl,aggregate.json}`
- `reproduce/runs/table1_llama_longbench_e_2wikimqa_current.{json,csv,md}`
- `reproduce/runs/table1_llama_longbench_2wikimqa_current.{json,csv,md}`

当前本地 `2wikimqa_e` Table-1-shaped 结果：

| Model | Method | KV Size | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | Full Cache | 16.0 |  | 11.25 |  |  |  |  | 11.25 |
| Llama-3.1-8B-Instruct | TurboQuant | 2.5 |  | 12.24 |  |  |  |  | 12.24 |
| Llama-3.1-8B-Instruct | TurboQuant | 3.5 |  | 12.96 |  |  |  |  | 12.96 |

当前本地标准 `2wikimqa` Table-1-shaped 结果：

| Model | Method | KV Size | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | Full Cache | 16.0 |  | 11.44 |  |  |  |  | 11.44 |
| Llama-3.1-8B-Instruct | TurboQuant | 2.5 |  | 11.29 |  |  |  |  | 11.29 |
| Llama-3.1-8B-Instruct | TurboQuant | 3.5 |  | 11.95 |  |  |  |  | 11.95 |

注意：这两张 Table-1-shaped 表都只覆盖 MultiQA 里的 `2wikimqa`，仍不能作为最终 Table 1，因为其余 LongBench / LongBench-E 任务未在本地 cache 中。

Table 1 数据准备状态：

- 本地 cache 准备报告：`reproduce/logs/longbench_cache_prepare_local_only_2026-06-11.json`
- Paths 更新报告：`reproduce/logs/longbench_cache_prepare_update_paths_2026-06-11.json`
- Table 1 manifest：`reproduce/logs/table1_manifest_2026-06-11.json`
- 用户数据可用性提示后的复查报告：`reproduce/logs/longbench_cache_prepare_user_claim_refresh_2026-06-11.json`
- 用户数据可用性提示后的 Table 1 manifest：`reproduce/logs/table1_manifest_user_claim_refresh_2026-06-11.json`
- Snapshot-aware LongBench 复查报告：`reproduce/logs/longbench_cache_prepare_snapshot_aware_2026-06-11.json`
- Snapshot-aware Table 1 manifest：`reproduce/logs/table1_manifest_snapshot_aware_2026-06-11.json`
- 数据覆盖审计：`reproduce/DATA_COVERAGE_AUDIT.md`
- Manifest 当前统计：6 complete，90 missing_dataset。
- 尝试下载标准 LongBench `narrativeqa` 的报告：`reproduce/logs/longbench_cache_prepare_download_attempt_narrativeqa_2026-06-11.json`
- 用户数据可用性提示后的 `narrativeqa` 下载/缓存重试报告：`reproduce/logs/longbench_cache_prepare_download_attempt_narrativeqa_user_claim_refresh_2026-06-11.json`
- 下载尝试未成功；当前环境只能看到已缓存的 `2wikimqa` config，缺失任务无法从 Hugging Face Hub 获取。
- Snapshot-aware 复查也只在 `/home/liying/.cache/huggingface/hub` 中发现 LongBench/LongBench-E `2wikimqa` parquet，没有发现其他 Table 1 任务数据。

Table 1 下一步：

- 补齐 LongBench / LongBench-E 其余任务 cache 后，先运行 `scripts/prepare_longbench_cache.py --update-paths` 和 `scripts/build_table1_manifest.py`，再按 manifest 复跑完整三行 Table 1。
- 根据完整任务结果决定是否调整 2.5-bit / 3.5-bit outlier-channel policy。

## 4. Figure 4：Needle-In-A-Haystack

论文 Figure 4 第一阶段只复现两种方法：

| Method | Paper Score |
| --- | ---: |
| Full-Precision | 0.997 |
| TurboQuant | 0.997 |

已实现：

- `experiments/needle/run_needle_eval.py`
- `scripts/summarize_needle_results.py`
- `scripts/build_figure4_needle_summary.py`
- `scripts/build_needle_length_grid.py`
- `scripts/prepare_needle_cache.py`
- `scripts/build_needle_heatmap.py`

本地 16k 24-example grid 结果：

| Method | KV Size | Examples | Local score | Paper score | Avg prompt tokens | Avg latency | Cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-Precision | 16.0 | 24 | 1.0000 | 0.997 | 8927.12 | 2.11s/example |  |
| TurboQuant | 2.5 | 24 | 1.0000 | 0.997 | 8927.12 | 20.28s/example | 0.1723 |

本地 16k grid 覆盖：

- start/middle/end 三个 needle position。
- 8 个 distractor language：ar/de/en/es/hi/multilingual/vi/zh。
- 每个 `needle_position x distractor_lang` 组合 1 个样本，每种方法共 24 个样本。
- 两种方法在 start、middle、end 三组上的 answer-contains accuracy 均为 1.0000。

产物：

- `reproduce/runs/needle_16k_full_precision_grid_24.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/needle_16k_turboquant_2_5bit_grid_24.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/figure4_needle_16k_local_summary.{json,csv,md}`
- `reproduce/runs/figure4_needle_16k_local_heatmap.{json,csv,md,png}`
- `reproduce/runs/needle_generated_4k_full_precision.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/needle_generated_4k_turboquant_2_5bit.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/figure4_needle_generated_4k_summary.{json,csv,md}`
- `reproduce/runs/figure4_needle_generated_4k_heatmap.{json,csv,md,png}`
- `reproduce/logs/needle_cache_prepare_2026-06-11.json`
- `reproduce/runs/needle_16k_all_full_precision_smoke_3.{jsonl,summary.json,aggregate.json}`
- `reproduce/generated_data/needle_length_grid/needle_generated_length_grid_en.arrow`
- `reproduce/generated_data/needle_length_grid/needle_generated_length_grid_en.hf/`
- `reproduce/logs/needle_length_grid_report_2026-06-11.json`
- `reproduce/generated_data/needle_length_grid_smoke/needle_generated_length_grid_en_smoke.arrow`
- `reproduce/logs/needle_length_grid_smoke_report_2026-06-11.json`
- `reproduce/runs/needle_full_precision_smoke_1.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/needle_full_precision_position_balanced_smoke_3.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/needle_turboquant_2_5bit_smoke_1.{jsonl,summary.json,aggregate.json}`

Generated length-grid 状态：

- `configs/paths.yaml` 已注册 `needle_generated_length_grid_en` 和 `needle_generated_length_grid_en_smoke`。
- `needle_generated_length_grid_en` 共 144 条：6 个 target token lengths x 3 个 position x 8 个 distractor language x 1 个样本。
- 目标长度：4096、8192、16384、32768、65536、104000。
- `experiments/needle/run_needle_eval.py` 已支持 `--target-prompt-tokens`，可按目标长度分块运行。

Needle 16k multi-split 状态：

- `configs/paths.yaml` 已注册 `needle_16k_ar/de/en/es/hi/vi/zh` 和 `needle_16k_all`。
- 每个语言 split 2400 条；`needle_16k_all` 共 16800 条。
- `experiments/needle/run_needle_eval.py` 和 `scripts/build_needle_length_grid.py` 已支持同一 dataset key 下多个 Arrow 文件。
- `needle_16k_all_full_precision_smoke_3` 验证了多 Arrow 加载路径；该 smoke 不作为 accuracy 结果，因为前 3 条是 Arabic hidden answer，而当前 exact substring scorer 会把英文转述判错。

Generated 4k 诊断切片结果：

| Method | KV Size | Examples | Local score | Avg prompt tokens | Avg latency | Cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-Precision | 16.0 | 24 | 0.3333 | 4095.88 | 1.20s/example |  |
| TurboQuant | 2.5 | 24 | 0.3333 | 4095.88 | 20.06s/example | 0.1728 |

按位置看，两种方法都是 `middle=1.0000`、`start=0.0000`、`end=0.0000`。这说明 generated 4k 裁剪切片能用于验证评测入口和同数据对比，但不能作为 Figure 4 的论文级复现结果。

Heatmap 输出状态：

- `figure4_needle_16k_local_heatmap` 根据实际 prompt tokens 推断出 8192 和 16384 两个 token bucket；两种方法在 0/50/100 depth percent 上都是 1.0000。
- `figure4_needle_generated_4k_heatmap` 是诊断图；4096 token bucket 上两种方法都是 `depth 0 = 0.0000`、`depth 50 = 1.0000`、`depth 100 = 0.0000`。
- 该脚本已经能输出 Figure-4-style JSON/CSV/Markdown/PNG；完整论文 Figure 4 仍等待真实 4k 到 104k token/depth 网格数据。

当前限制：

- 本地 cache 只有 Needle 16k 七个语言 split。
- 论文 Figure 4 覆盖约 4k 到 104k token limit 与多个 depth percent。当前 generated grid 可提供分长度运行入口，但源 prompt 实际约 7k-11k tokens，16k 以上 target 不能替代真实 32k/65k/104k 上下文。

Figure 4 下一步：

- 准备或生成 4k 到 104k 的 Needle 网格数据。
- 跑 Full-Precision / TurboQuant 两种方法的完整 grid。
- 输出 `figure4_needle_summary.{json,csv}` 和 `figure4_needle_heatmap.{png,pdf}`。

## 5. TurboQuant 算法 sanity check

已实现：

- `turboquant/codebook.py`
- `turboquant/core.py`
- `tests/test_core.py`
- `scripts/validate_figure3_core.py`

验证结果：

- Unit test：`6 passed`
- Figure 3-style local MSE：

| Bit | Local MSE |
| ---: | ---: |
| 1 | 0.3611 |
| 2 | 0.1167 |
| 3 | 0.0341 |
| 4 | 0.0094 |

产物：

- `reproduce/runs/figure3_core_d256.csv`
- `reproduce/runs/figure3_core_d256_summary.json`

这些数值与论文报告的小 bit MSE 参考值接近，可作为 core 实现的 sanity check。

## 6. Figure 5：Nearest Neighbor Search

第一阶段只记录 TurboQuant，不复现 PQ/RabitQ baseline。

已实现：

- `experiments/ann_search/run_turboquant_ann.py`
- `scripts/build_ann_summary.py`
- `scripts/plot_figure5_dbpedia.py`

DBpedia 论文规模设置：

- Database：100,000
- Queries：1,000
- Top-k：1, 2, 4, 8, 16, 32, 64
- Bits：2 and 4

DBpedia 结果：

| Dataset | Bits | 1@1 | 1@2 | 1@4 | 1@8 | 1@16 | 1@32 | 1@64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DBpedia 1536d | 2 | 0.900 | 0.973 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| DBpedia 1536d | 4 | 0.966 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DBpedia 3072d | 2 | 0.904 | 0.984 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |
| DBpedia 3072d | 4 | 0.973 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

产物：

- `reproduce/runs/ann_dbpedia_1536_fullscale_100k_1k.{json,csv}`
- `reproduce/runs/ann_dbpedia_3072_fullscale_100k_1k.{json,csv}`
- `reproduce/runs/figure5_turboquant_dbpedia_fullscale.{json,csv,md,png,pdf}`

当前限制：

- GloVe 200d 尚未提供，Figure 5 的 GloVe 子图未复现。
- 当前 ANN scoring 是先 dequantize 再矩阵乘法，验证 recall 行为，不代表 optimized compressed-domain ANN kernel。

## 7. Table 2：4-bit Quantization Runtime

已实现：

- `scripts/benchmark_quantization_time.py`
- `scripts/build_table2_summary.py`

当前结果：

| d | Source | Vectors | Local mean seconds | Paper seconds | Local / paper |
| ---: | --- | ---: | ---: | ---: | ---: |
| 200 | random_unit | 2048 | 0.000147 | 0.000700 | 0.21 |
| 1536 | dbpedia_cache | 2048 | 0.001425 | 0.001300 | 1.10 |
| 3072 | dbpedia_cache | 2048 | 0.003256 | 0.002100 | 1.55 |

产物：

- `reproduce/runs/table2_quantization_time_smoke.{json,csv}`
- `reproduce/runs/table2_quantization_time_summary.{json,csv,md}`

计时说明：

- 当前计时测量 `TurboQuantMSE.quantize()`。
- 不包含 codebook 构造、rotation 构造、数据加载和 dequantization。
- d=200 使用 random unit vectors，因为 GloVe 200d 不在本地。

## 8. 关键实现差异和风险

- TurboQuant KV cache 是 Python-level reproduction：内部存储 packed indices + norms，但 attention 调用前会 materialize dequantized K/V，因此速度不是论文优化 kernel 水平。
- LongBench 评测入口现在支持 `arrow_files` 与 `parquet_files`；当前仍只有 `2wikimqa` 数据可用，不是读取格式问题。
- 当前 2.5-bit / 3.5-bit policy 为动态 outlier-channel：
  - 2.5-bit：64/128 channels 用 3-bit，64/128 channels 用 2-bit。
  - 3.5-bit：64/128 channels 用 4-bit，64/128 channels 用 3-bit。
- KV policy 证据产物：`reproduce/runs/kv_compression_policy_summary.{json,csv,md}`。
  - 覆盖 LongBench-E `2wikimqa`、LongBench `2wikimqa`、Needle 16k、generated Needle 4k 的已有 TurboQuant runs。
  - 2.5-bit runs：average effective index bits = 2.5000，outlier count = 64，regular/outlier bits = 2/3。
  - 3.5-bit runs：average effective index bits = 3.5000，outlier count = 64，regular/outlier bits = 3/4。
- 论文文字称 2.5-bit 使用 32 个 3-bit outlier channels 与 96 个 2-bit channels，但 `(32 * 3 + 96 * 2) / 128 = 2.25`，不是 2.5。当前实现优先匹配 Table 1 的 reported KV Size 语义。
- Full Table 1 的最大阻塞是数据 cache 不完整；当前脚本已在本地 `2wikimqa_e` 300-example 和标准 `2wikimqa` 200-example 上完成 Full Cache 与 TurboQuant 2.5-bit / 3.5-bit 对比。`narrativeqa` 下载重试未成功，直接加载 `THUDM/LongBench/narrativeqa` 也因为网络不可达且数据未 materialized 而失败。
- Full Figure 4 的最大阻塞是缺少真实 4k 到 104k Needle grid；当前已验证 16k 24-example 本地网格、16k 多 split 加载和 heatmap 输出管线，但还不是完整论文 heatmap。

## 9. 下一步执行顺序

1. 补齐 LongBench / LongBench-E 完整 Table 1 所需任务数据。
2. 数据补齐后，复跑完整 Table 1 的 Full Cache、TurboQuant 2.5-bit、TurboQuant 3.5-bit。
3. 补齐或生成 Needle 4k 到 104k grid，复现 Figure 4 的 Full-Precision 与 TurboQuant heatmap。
4. 如需要完整 Figure 5 / Table 2，补齐 GloVe 200d。
