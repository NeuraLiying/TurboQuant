# TurboQuant 第一阶段复现 TODO

创建时间：2026-06-10
最近更新：2026-06-11 20:55 Asia/Shanghai

论文文件：`/home/liying/projects/turboquant/paper/turboquant.pdf`

当前自动汇总报告：

- Markdown：`reproduce/FIRST_STAGE_REPORT_CURRENT.md`
- JSON：`reproduce/FIRST_STAGE_REPORT_CURRENT.json`
- 生成脚本：`scripts/build_first_stage_report.py`

本文件是后续复现进度的主计划。第一阶段的目标是复现论文报告的主要实验结果中与 TurboQuant 方法直接相关的部分，而不是复现 motivation、理论证明或所有 baseline。

## 0. 范围约束

- 所有代码、脚本、配置和实验记录只放在 `/home/liying/projects/turboquant`。
- 实验进度、日志、原始输出和汇总结果统一放在 `/home/liying/projects/turboquant/reproduce`。
- 第一轮模型只使用 `meta-llama/Llama-3.1-8B-Instruct`。
- 第一轮硬件按 RTX 4090 规划。8B 模型的 Full Cache 与 TurboQuant KV cache 复现应可在当前机器上推进。
- 第一阶段只比较：
  - `Full Cache` / `Full-Precision`
  - `TurboQuant`
- 第一阶段不复现：
  - KIVI
  - PolarQuant
  - SnapKV
  - PyramidKV
  - Product Quantization
  - RabitQ

### 0.1 执行分工

用户侧准备：

- [x] 提供论文 PDF：`paper/turboquant.pdf`。
- [x] 提供第一轮模型权重：`meta-llama/Llama-3.1-8B-Instruct` 本地 snapshot 已可用。
- [x] 提供 RTX 4090 运行环境。当前 8B Full Cache / TurboQuant 复现可在 4090 上推进。
- [x] 补齐完整 Table 1 所需的 LongBench-V1 数据 cache。
- [ ] 补齐或生成 Figure 4 所需的 Needle 4k 到 104k token-length/depth grid。
- [ ] 如需完整 Figure 5，补齐 GloVe 200d 数据。

Codex 侧执行：

- [x] 创建并验证 `turboquant` conda 环境，记录环境文件。
- [x] 阅读论文实验设置并把第一阶段复现范围落到可执行计划。
- [x] 编写 TurboQuant core、KV cache、LongBench、Needle、ANN、timing 相关脚本。
- [x] 在已有本地 cache 上运行 Full Cache / Full-Precision 与 TurboQuant 的可复查实验。
- [x] 生成当前 partial Table 1、本地 Figure 4 16k summary、DBpedia Figure 5、Table 2-style 结果文件。
- [x] 生成当前自动化第一阶段报告：`reproduce/FIRST_STAGE_REPORT_CURRENT.{md,json}`。
- [ ] 运行完整 Table 1 / Figure 4，并把论文数值和本地数值逐项对齐。
- [ ] 复现完成后，基于结果差异和实现瓶颈提出增量研究方向。

当前阻塞/进行中项：

- Table 1 的数据缺口已通过 HF mirror 补齐，当前正式口径切换为 LongBench-V1 全量 task split。
- 正式 Table 1 协议见 `reproduce/TABLE1_OFFICIAL_PROTOCOL.md`：`--max-examples 1`、`--max-examples 20` 等小样本运行只作为 smoke test，不作为论文对比结果。
- Full Cache 全量评测已完成；最新进度见 `reproduce/runs/table1_official/table1_full_cache_progress.md`。
- Full Cache 与论文对比见 `reproduce/TABLE1_FULL_CACHE_COMPARISON.md`。
- 2026-06-11 20:55 最新进度：Full Cache 16/16 tasks complete；本地平均 48.71，论文 Full Cache 平均 50.06。
- Full Cache caveat：平均值接近，但 Summarization / Synthetic 偏低、Code 偏高，后续需要按 category 解释，而不是只看 average。
- TurboQuant 2.5-bit / 3.5-bit 的完整 Table 1 运行现在进入执行阶段，必须使用同一套 16 个 LongBench-V1 full split、prompt 和 scorer。
- 历史数据缺口记录保留在下面的审计文件中，用于追踪当时为什么只跑了 `2wikimqa` 和 smoke tests。
- 强审计脚本：`scripts/audit_reproduction_assets.py`；报告：`reproduce/logs/reproduction_asset_audit_2026-06-11.{json,md}`。
- 强审计结果：LongBench/LongBench-E Table 1 的 32 个数据入口中 2 个 materialized、30 个 metadata-or-lock-only；Needle Figure 4 目标长度中只有 16k materialized；DBpedia 1536d/3072d materialized。
- 非模型刷新脚本：`scripts/refresh_reproduction_state.py`；当前汇总报告：`reproduce/logs/reproduction_state_refresh_goal_resume_paths_fixed_2026_06_11.{json,md}`。
- 自动第一阶段报告脚本：`scripts/build_first_stage_report.py`；当前报告：`reproduce/FIRST_STAGE_REPORT_CURRENT.{json,md}`。
- Symlink-aware 强审计报告：`reproduce/logs/reproduction_asset_audit_symlink_aware_2026-06-11.{json,md}`。Hugging Face snapshot 中的 parquet/csv symlink 会被计入真实数据文件。
- Raw data 探测脚本：`scripts/probe_raw_data_assets.py`；报告：`reproduce/logs/raw_data_asset_probe_2026-06-11.{json,md}`。它会扫描项目、`/home/liying/datasets/turboquant`、Hub snapshots/blobs 中未注册的原始数据；当前未发现 `2wikimqa` 和 Needle `16k` 之外的目标数据。
- Table 1 manifest 当前是 6 个已完成条目、90 个 missing-dataset 条目；最新报告见 `reproduce/logs/table1_manifest_current_2026-06-11.json`。
- 已尝试下载标准 LongBench `narrativeqa`，但当前环境只能看到本地 cached config `2wikimqa`，缺失数据无法从 Hugging Face Hub 获取；报告见 `reproduce/logs/longbench_cache_prepare_download_attempt_narrativeqa_2026-06-11.json`。
- 用户数据可用性提示后又重试了一次 `narrativeqa` 下载/缓存，仍未新增数据；报告见 `reproduce/logs/longbench_cache_prepare_download_attempt_narrativeqa_user_claim_refresh_2026-06-11.json`。
- 2026-06-11 直接 `load_dataset` 验证显示：`Xnhyacinth/LongBench` 和 `Xnhyacinth/LongBench-e` 只能命中本地 `2wikimqa`；Needle 只能命中本地 `16k`。对 `huggingface.co:443` 的公开 URL 探测超时，因此当前环境不能在线补齐缺失数据。
- 当前数据覆盖审计见 `reproduce/DATA_COVERAGE_AUDIT.md`。
- 已增加 snapshot-aware 数据发现、parquet 读取和多 Arrow shard 合并支持；`/home/liying/.cache/huggingface/hub` 当前也只发现 LongBench/LongBench-E `2wikimqa` parquet，报告见 `reproduce/logs/longbench_cache_prepare_snapshot_aware_2026-06-11.json`。
- 完整 Figure 4 阻塞在 Needle 长度网格不全；当前确认 Needle 16k 有 ar/de/en/es/hi/vi/zh 七个 split，并已完成 16k 本地 24-example EN grid。
- 已实现本地 generated Needle length-grid 管线，可从 16k cache 生成按目标 token 分组的评测入口；真实 32k/65k/104k 仍需要更长源数据。
- 已完成 generated 4k 诊断切片；Full-Precision 与 TurboQuant 2.5-bit 都是 0.3333，因此该裁剪数据不能替代论文 Figure 4。
- 完整 Figure 5 阻塞在 GloVe 200d 缺失；DBpedia 1536d / 3072d 已完成论文规模 TurboQuant 曲线。

## 1. 第一阶段验收目标

### 1.1 Table 1：LongBench / LongBench-E 端到端生成

论文报告的 Llama-3.1-8B-Instruct 核心目标行：

| Model | Method | KV Size | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | Full Cache | 16 | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| Llama-3.1-8B-Instruct | TurboQuant | 2.5 | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |
| Llama-3.1-8B-Instruct | TurboQuant | 3.5 | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |

复现 TODO：

- [x] 建立 Llama full-cache LongBench 最小评测脚本。
- [x] 建立可 resume 的 LongBench 评测脚本。
- [x] 在本地 LongBench-E `2wikimqa` cache 上跑通 Full Cache smoke。
- [x] 在本地 LongBench-E `2wikimqa` cache 上跑通 TurboQuant cache prototype smoke。
- [x] 实现整数 bit-width 的 packed TurboQuant KV cache，并跑通 Llama smoke。
- [x] 实现 2.5-bit / 3.5-bit fractional effective-bit outlier-channel 策略，并跑通 2.5-bit / 3.5-bit Llama smoke。
- [x] 实现 LongBench 官方风格 scorer 和 Table-1-shaped 聚合脚本。
- [x] 基于当前 `2wikimqa` smoke 生成 partial Table 1，验证聚合结构。
- [x] 跑完本地已缓存 LongBench-E `2wikimqa` Full Cache 300/300 全量任务。
- [x] 实现 Needle 16k evaluation / summary 脚本，并跑通 Full-Precision 与 TurboQuant smoke。
- [x] 跑完本地 Needle 16k Full-Precision / TurboQuant 2.5-bit 24-example grid。
- [x] 生成本地 Figure-4-style Needle 16k summary。
- [x] 实现 DBpedia TurboQuant ANN recall / quantization runtime 脚本，并跑通 1536d/3072d smoke。
- [x] 跑通 DBpedia 1536d / 3072d 论文规模 100k/1k TurboQuant ANN 曲线。
- [x] 生成 DBpedia Figure 5 图和 Table 2 summary。
- [x] 生成机器可复查的 HF cache inventory。
- [x] 增加强审计脚本，逐项检查 Table 1 / Figure 4 / Figure 5 所需本地资产是否真正 materialized。
- [x] 给 LongBench 评测脚本增加 `--start-index` / `--end-index` 分块运行能力。
- [x] 跑完本地 LongBench-E `2wikimqa` TurboQuant 2.5-bit 300/300 任务。
- [x] 跑完本地 LongBench-E `2wikimqa` TurboQuant 3.5-bit 300/300 任务。
- [x] 跑完本地标准 LongBench `2wikimqa` Full Cache 200/200 任务。
- [x] 跑完本地标准 LongBench `2wikimqa` TurboQuant 2.5-bit 200/200 任务。
- [x] 跑完本地标准 LongBench `2wikimqa` TurboQuant 3.5-bit 200/200 任务。
- [x] 增加按 `index` 合并 JSONL shard 的工具，支持多 GPU 分块实验汇总。
- [x] 增加 Table 1 多 GPU 运行 planner，支持按 manifest 生成分块评测、merge、aggregate 和最终 summary 脚本。
- [x] 增加 LongBench / LongBench-E cache 准备与 `paths.yaml` 更新工具。
- [x] 增强 LongBench 数据发现：支持 datasets cache、Hub snapshot parquet、THUDM `_e` config alias。
- [x] 增强 LongBench 评测入口：支持 `arrow_files` 与 `parquet_files`，并可合并多个 Arrow/Parquet shard。
- [x] 增加 LongBench loader regression smoke，确认多 shard loader 修改后现有 `2wikimqa` 单 shard 仍可正常评测。
- [x] 增加 Table 1 manifest，用于跟踪 dataset/method 覆盖和生成下一步命令。
- [x] 增加非模型刷新脚本，一次性刷新 LongBench/Needle 注册、强审计、Table 1 manifest、Table 1/Figure 4 planner。
- [x] 对缺失的标准 LongBench `narrativeqa` 做下载尝试并记录失败原因。
- [x] 补齐 LongBench-V1 Table 1 所需任务数据 cache。
- [x] 跑完整 Llama Full Cache Table 1 全任务评测。
- [ ] 用完整 Table 1 评测校准 2.5-bit / 3.5-bit outlier-channel 策略。
- [ ] 跑完整 Llama TurboQuant 2.5-bit Table 1 全任务评测。
- [ ] 跑完整 Llama TurboQuant 3.5-bit Table 1 全任务评测。
- [ ] 生成 Table 1 复现表格，并记录与论文数值的差异。

验收产物：

- `reproduce/runs/longbench_full_cache_*.jsonl`
- `reproduce/runs/longbench_turboquant_2_5bit_*.jsonl`
- `reproduce/runs/longbench_turboquant_3_5bit_*.jsonl`
- `reproduce/runs/table1_llama_summary.{json,csv,md}`

当前结构验证产物：

- `reproduce/runs/table1_llama_partial.json`
- `reproduce/runs/table1_llama_partial.csv`
- `reproduce/runs/table1_llama_partial.md`
- `reproduce/runs/table1_llama_partial_current.json`
- `reproduce/runs/table1_llama_partial_current.csv`
- `reproduce/runs/table1_llama_partial_current.md`
- `reproduce/runs/table1_llama_2wikimqa_mixed_current.json`
- `reproduce/runs/table1_llama_2wikimqa_mixed_current.csv`
- `reproduce/runs/table1_llama_2wikimqa_mixed_current.md`
- `reproduce/runs/table1_llama_available_summary.{json,csv,md}`
- `reproduce/logs/table1_run_plan_current_2026-06-11.{json,md,sh}`
- `reproduce/logs/table1_run_plan_complete_rechunk_demo_2026-06-11.{json,md,sh}`
- `scripts/refresh_reproduction_state.py`
- `reproduce/logs/reproduction_state_refresh_2026-06-11_current.{json,md}`

当前本地完整任务产物：

- `reproduce/runs/longbench_e_2wikimqa_full_cache_all.jsonl`
- `reproduce/runs/longbench_e_2wikimqa_full_cache_all.summary.json`
- `reproduce/runs/longbench_e_2wikimqa_full_cache_all.aggregate.json`
- `reproduce/runs/longbench_2wikimqa_full_cache_all.jsonl`
- `reproduce/runs/longbench_2wikimqa_full_cache_all.summary.json`
- `reproduce/runs/longbench_2wikimqa_full_cache_all.aggregate.json`
- `reproduce/runs/longbench_2wikimqa_loader_regression_smoke_1.{jsonl,summary.json}`
- `reproduce/runs/longbench_2wikimqa_turboquant_2_5bit_chunked.{jsonl,aggregate.json}`
- `reproduce/runs/longbench_2wikimqa_turboquant_3_5bit_chunked.{jsonl,aggregate.json}`
- `reproduce/runs/table1_llama_longbench_e_2wikimqa_current.{json,csv,md}`
- `reproduce/runs/table1_llama_longbench_2wikimqa_current.{json,csv,md}`
- `reproduce/logs/longbench_cache_prepare_local_only_2026-06-11.json`
- `reproduce/logs/longbench_cache_prepare_update_paths_2026-06-11.json`
- `reproduce/logs/longbench_cache_prepare_download_attempt_narrativeqa_2026-06-11.json`
- `reproduce/logs/table1_manifest_2026-06-11.json`

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

当前限制：

- 重新扫描 `/home/liying/datasets/turboquant/hf_cache` 后，本地 LongBench cache 目前只确认有 `2wikimqa` / `2wikimqa_e`，不足以计算完整 Table 1 分类平均。
- 机器可复查清单：`reproduce/logs/hf_cache_inventory_2026-06-11.json`、`reproduce/logs/hf_cache_inventory_2026-06-11_refresh.json` 与 `reproduce/logs/hf_cache_inventory_2026-06-11_user_claim_refresh.json`。最新刷新版当前仍只有 5 个 `dataset_info.json`：DBpedia 1536d、DBpedia 3072d、LongBench-E `2wikimqa`、LongBench `2wikimqa`、Needle 16k。
- 强审计清单：`reproduce/logs/reproduction_asset_audit_2026-06-11.{json,md}`。它会跟随 `/home/liying/datasets/turboquant/hf_cache` 和 `/home/liying/.cache/huggingface/hub`，按 Table 1 / Figure 4 / Figure 5 需要的条目判断是否存在真实数据文件，而不是只看 loader、README、lock 或 `dataset_info.json`。
- Snapshot-aware LongBench 报告：`reproduce/logs/longbench_cache_prepare_snapshot_aware_2026-06-11.json`。当前 Hub snapshot 也只提供 LongBench/LongBench-E `2wikimqa` parquet。
- 数据覆盖审计：`reproduce/DATA_COVERAGE_AUDIT.md`。
- Full Cache 与 TurboQuant 2.5-bit / 3.5-bit 均已完成本地 `2wikimqa_e` 300/300 和标准 `2wikimqa` 200/200。
- Table 1 manifest 当前统计：6 complete，90 missing_dataset。
- TurboQuant cache 已经改为内部存储 packed indices + norms，但当前是 Python-level 复现实现，会在 attention 调用前临时反量化，速度还不是论文级优化 kernel。
- 2.5-bit / 3.5-bit 已有可复现 outlier-channel 规则；还需要完整 Table 1 评测确认是否需要进一步校准。

### 1.2 Figure 4：Needle-In-A-Haystack

论文设置：

- Model：`Llama-3.1-8B-Instruct`
- Token limit：约 `4k` 到 `104k`
- Depth percent：`0` 到 `100`
- Metric：hidden sentence retrieval score / recall

第一阶段只复现两种方法：

| Method | Paper Score |
| --- | ---: |
| Full-Precision | 0.997 |
| TurboQuant | 0.997 |

复现 TODO：

- [x] 确认本地 Needle 16k EN cache 可读。
- [x] 编写 Needle 数据加载与 prompt 构造脚本。
- [x] 跑本地 16k Full-Precision / Full Cache smoke。
- [x] 跑本地 16k TurboQuant 2.5-bit smoke。
- [x] 跑本地 16k start/middle/end balanced Full-Precision smoke。
- [x] 跑本地 16k Full-Precision 24-example grid：start/middle/end x 8 distractor languages。
- [x] 跑本地 16k TurboQuant 2.5-bit 24-example grid：start/middle/end x 8 distractor languages。
- [x] 生成本地 Figure-4-style 16k summary。
- [x] 实现本地 generated Needle length-grid 数据构造脚本。
- [x] 生成本地 4k/8k/16k/32k/65k/104k target-token grid Arrow 和 report。
- [x] 给 Needle 评测脚本增加 `--target-prompt-tokens` 分长度过滤。
- [x] 跑 generated local 4k Full-Precision / TurboQuant 2.5-bit 诊断切片，并记录其不能替代 Figure 4 的原因。
- [x] 发现并注册 Needle 16k 的 ar/de/en/es/hi/vi/zh 七个本地 split，增加多 Arrow 加载支持。
- [x] 扩展 Needle cache 准备脚本，支持一次扫描/注册 4k/8k/16k/32k/65k/104k 多个真实长度 config。
- [x] 实现 Figure-4-style Needle heatmap 生成脚本，并用本地 16k / generated 4k 结果生成验证产物。
- [x] 增加 Figure 4 Needle run planner，按真实 materialized Needle 数据生成 Full-Precision/TurboQuant 运行计划和 available heatmap。
- [ ] 固定论文 token limit / depth percent 网格。
- [ ] 跑完整 Full-Precision / Full Cache 网格。
- [ ] 跑完整 TurboQuant 网格。
- [ ] 输出 heatmap 原始数据和 Figure 4 复现图。

验收产物：

- `reproduce/runs/needle_full_precision_*.jsonl`
- `reproduce/runs/needle_turboquant_*.jsonl`
- `reproduce/runs/figure4_needle_summary.{json,csv}`
- `reproduce/runs/figure4_needle_heatmap.{png,pdf}`

当前本地 16k grid 产物：

- `reproduce/runs/needle_16k_full_precision_grid_24.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/needle_16k_turboquant_2_5bit_grid_24.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/figure4_needle_16k_local_summary.{json,csv,md}`

当前 generated length-grid 产物：

- `scripts/build_needle_length_grid.py`
- `reproduce/generated_data/needle_length_grid/needle_generated_length_grid_en.arrow`
- `reproduce/generated_data/needle_length_grid/needle_generated_length_grid_en.hf/`
- `reproduce/logs/needle_length_grid_report_2026-06-11.json`
- `reproduce/generated_data/needle_length_grid_smoke/needle_generated_length_grid_en_smoke.arrow`
- `reproduce/logs/needle_length_grid_smoke_report_2026-06-11.json`
- `configs/paths.yaml` keys: `needle_generated_length_grid_en` and `needle_generated_length_grid_en_smoke`
- `reproduce/runs/needle_generated_4k_full_precision.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/needle_generated_4k_turboquant_2_5bit.{jsonl,summary.json,aggregate.json}`
- `reproduce/runs/figure4_needle_generated_4k_summary.{json,csv,md}`
- `scripts/build_needle_heatmap.py`
- `scripts/plan_figure4_runs.py`
- `reproduce/runs/figure4_needle_16k_local_heatmap.{json,csv,md,png}`
- `reproduce/runs/figure4_needle_generated_4k_heatmap.{json,csv,md,png}`
- `reproduce/runs/figure4_needle_available_heatmap.{json,csv,md,png}`
- `reproduce/logs/figure4_run_plan_current_2026-06-11.{json,md,sh}`
- `reproduce/logs/figure4_run_plan_16k_replay_demo_2026-06-11.{json,md,sh}`
- `reproduce/logs/needle_cache_prepare_2026-06-11.json`
- `reproduce/logs/needle_cache_prepare_all_configs_2026-06-11.json`
- `reproduce/logs/needle_cache_prepare_all_configs_update_paths_2026-06-11.json`
- `reproduce/runs/needle_16k_all_full_precision_smoke_3.{jsonl,summary.json,aggregate.json}`

当前 Needle 16k multi-split 状态：

- `configs/paths.yaml` keys: `needle_16k_ar`, `needle_16k_de`, `needle_16k_en`, `needle_16k_es`, `needle_16k_hi`, `needle_16k_vi`, `needle_16k_zh`, `needle_16k_all`
- 每个语言 split 2400 条，`needle_16k_all` 共 16800 条。
- `scripts/prepare_needle_cache.py --configs 4k 8k 16k 32k 65k 104k --update-paths` 当前只发现并注册 16k；未发现真实 `needle_4k_*`、`needle_8k_*`、`needle_32k_*`、`needle_65k_*`、`needle_104k_*`。
- `needle_16k_all_full_precision_smoke_3` 只用于验证多 Arrow 加载；它不作为 Figure 4 accuracy 结果，因为前 3 条是 Arabic hidden answer，当前 exact substring scorer 会把英文转述判错。

当前本地 16k grid 结果：

| Method | KV Size | Examples | Local score | Paper score | Avg prompt tokens | Avg latency | Cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-Precision | 16.0 | 24 | 1.0000 | 0.997 | 8927.12 | 2.11s/example |  |
| TurboQuant | 2.5 | 24 | 1.0000 | 0.997 | 8927.12 | 20.28s/example | 0.1723 |

当前 generated 4k 诊断结果：

| Method | KV Size | Examples | Local score | Avg prompt tokens | Avg latency | Cache ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-Precision | 16.0 | 24 | 0.3333 | 4095.88 | 1.20s/example |  |
| TurboQuant | 2.5 | 24 | 0.3333 | 4095.88 | 20.06s/example | 0.1728 |

当前 heatmap 产物说明：

- `figure4_needle_16k_local_heatmap` 根据实际 prompt tokens 推断出 8192 和 16384 两个 token bucket；两种方法在 0/50/100 depth percent 上都是 1.0000。
- `figure4_needle_available_heatmap` 由 `scripts/plan_figure4_runs.py` 汇总当前 materialized 的 16k EN Full-Precision/TurboQuant 输出生成；当前计划为 2 complete、10 missing_dataset、0 planned jobs。
- `figure4_needle_generated_4k_heatmap` 是诊断图；4096 token bucket 上两种方法都是 `depth 0 = 0.0000`、`depth 50 = 1.0000`、`depth 100 = 0.0000`。
- 这些 heatmap 脚本产物验证了 Figure 4 输出管线，但完整论文 Figure 4 仍需要真实 4k/16k/32k/65k/104k token limit 和 0-100 depth 网格。

当前限制：

- 本地 cache 当前确认的是 Needle 16k 七个语言 split。论文 Figure 4 覆盖约 4k 到 104k token limit，完整复现前需要额外长度 cache，或者生成对应网格数据。
- 当前已完成的是本地 16k 管线、24-example balanced grid、以及 generated length-grid 评测入口。由于源数据实际 prompt 约 7k-11k tokens，generated grid 中 16k 以上 target 不能替代真实 32k/65k/104k Needle 数据。
- generated 4k 诊断切片中 Full-Precision 也只有 0.3333，失败集中在 start/end，因此不能把该裁剪数据作为论文 Figure 4 的 4k 网格替代。

### 1.3 TurboQuant 算法正确性验证

这部分不是最终论文主结果，但用于保证后续 Table 1 / Figure 4 的 TurboQuant 实现没有明显偏差。

论文算法要点：

- `TurboQuantMSE`：随机旋转输入向量，然后使用 Lloyd-Max scalar codebook 逐坐标量化。
- `TurboQuantProd`：用 `b-1` bit 的 MSE quantizer 量化主体，再对 residual 做 1-bit QJL。
- 论文报告的 MSE 小 bit 参考值约为：
  - 1-bit：0.36
  - 2-bit：0.117
  - 3-bit：0.03
  - 4-bit：0.009

复现 TODO：

- [x] 实现 random QR rotation。
- [x] 实现 Lloyd-Max scalar codebook。
- [x] 实现 `TurboQuantMSE.quantize/dequantize`。
- [x] 实现 `TurboQuantProd.quantize/dequantize`。
- [x] 添加 core unit tests。
- [x] 跑 Figure 3-style MSE sanity check。
- [ ] 为 `TurboQuantProd` 增加 inner-product unbiased / variance sanity test。
- [ ] 给 codebook 增加磁盘缓存，避免每次实验重复构造。

已得到的本地 sanity 结果：

| Bit | Local MSE |
| ---: | ---: |
| 1 | 0.3611 |
| 2 | 0.1167 |
| 3 | 0.0341 |
| 4 | 0.0094 |

输出文件：

- `reproduce/runs/figure3_core_d256.csv`
- `reproduce/runs/figure3_core_d256_summary.json`

### 1.4 Figure 5：Nearest Neighbor Search

论文设置：

- GloVe `d=200`
- DBpedia OpenAI3 `d=1536`
- DBpedia OpenAI3 `d=3072`
- Database：随机采样 `100,000`
- Query：DBpedia 随机采样 `1,000`
- Top-k：`1, 2, 4, 8, 16, 32, 64`
- 第一阶段只复现 TurboQuant 曲线，不复现 PQ/RabitQ。

复现 TODO：

- [x] 确认 DBpedia 1536d / 3072d local Arrow schema。
- [x] 编写 TurboQuant ANN recall 脚本。
- [x] 跑通 DBpedia 1536d smoke。
- [x] 跑通 DBpedia 3072d smoke。
- [x] 跑论文规模 DBpedia 1536d：database 100,000 / query 1,000。
- [x] 跑论文规模 DBpedia 3072d：database 100,000 / query 1,000。
- [ ] 获取 GloVe 200d 并复现 GloVe recall。
- [x] 输出 DBpedia Figure 5 TurboQuant recall 曲线数据。
- [x] 输出 DBpedia Figure 5 TurboQuant recall 图。
- [ ] 输出包含 GloVe 的完整 Figure 5 TurboQuant recall 曲线。

当前 smoke 产物：

- `reproduce/runs/ann_dbpedia_1536_smoke.{json,csv}`
- `reproduce/runs/ann_dbpedia_3072_smoke.{json,csv}`
- `reproduce/runs/ann_dbpedia_1536_fullscale_100k_1k.{json,csv}`
- `reproduce/runs/ann_dbpedia_3072_fullscale_100k_1k.{json,csv}`
- `reproduce/runs/figure5_turboquant_dbpedia_fullscale.{json,csv,md}`
- `reproduce/runs/figure5_turboquant_dbpedia_fullscale.{png,pdf}`

### 1.5 Table 2：4-bit Quantization Runtime

论文报告 TurboQuant 4-bit quantization time：

| d | Paper TurboQuant seconds |
| ---: | ---: |
| 200 | 0.0007 |
| 1536 | 0.0013 |
| 3072 | 0.0021 |

复现 TODO：

- [x] 编写 quantization-only timing 脚本。
- [x] 跑通 d=200 / 1536 / 3072 smoke。
- [x] 生成 Table 2 summary。
- [ ] 在最终报告中固定 batch size、warmup、repeat、GPU、是否包含 codebook/rotation 构造。
- [ ] 若需要严格对齐论文，进一步校准 batch size 和计时对象。

当前 smoke 产物：

- `reproduce/runs/table2_quantization_time_smoke.json`
- `reproduce/runs/table2_quantization_time_smoke.csv`
- `reproduce/runs/table2_quantization_time_summary.{json,csv,md}`

当前 smoke 结果：

| d | Source | Local mean seconds |
| ---: | --- | ---: |
| 200 | random unit vectors | 0.000147 |
| 1536 | DBpedia cache | 0.001425 |
| 3072 | DBpedia cache | 0.003256 |

限制：

- GloVe 200d 尚未提供，因此 d=200 runtime smoke 使用随机 unit vectors。
- DBpedia ANN recall 已经有论文 100k/1k 规模结果；GloVe recall 仍缺数据。
- 当前 ANN scoring 会先 dequantize 后矩阵乘法，验证 recall 行为，不代表 optimized compressed-domain serving kernel。

## 2. 论文 2.5-bit / 3.5-bit 策略

论文文字说明 2.5-bit 例子：

- 每个 head dimension 为 `128`。
- `32` 个 outlier channels 使用 `3-bit`。
- 剩余 `96` 个普通 channels 使用 `2-bit`。
- 论文写成有效 bit：`(32 * 3 + 96 * 2) / 128 = 2.5`。

注意：这个算式实际等于 `2.25`，不是 `2.5`。当前实现为了匹配 Table 1 的 KV Size 语义，采用按目标 effective bit 自动分配 outlier 比例的可复现规则。

需要实现的 TODO：

- [x] 定义 outlier channel 选择规则。
  - 建议先按 calibration batch 中 K/V 每个 channel 的绝对值均值或方差排序。
  - 需要分别记录 key/value、layer/head 的选择结果。
  - 当前实现：对每个 K/V segment，按最后一维 channel 的绝对值均值选 top-k outlier。
  - 当前实现：key/value 分别选择，layer 分别选择，prefill/decode segment 分别选择。
- [x] 2.5-bit：
  - 当前实现：64/128 channels 用 3-bit，64/128 channels 用 2-bit。
  - 有效 bit：`(64 * 3 + 64 * 2) / 128 = 2.5`。
  - 已跑通 1 条 Llama LongBench-E smoke。
- [x] 3.5-bit：
  - 当前实现：64/128 channels 用 4-bit，64/128 channels 用 3-bit。
  - 有效 bit：`(64 * 4 + 64 * 3) / 128 = 3.5`。
  - 已跑通 1 条 Llama LongBench-E smoke。
- [x] 在实验输出里记录 outlier channel 数量、选择规则、calibration 数据和 seed。
  - 每条 TurboQuant JSONL 记录包含 `cache_compression_summary`。
  - 汇总产物：`reproduce/runs/kv_compression_policy_summary.{json,csv,md}`。
  - 当前已有 run 证据显示 2.5-bit 平均 effective index bits 为 2.5000，outlier count 为 64，regular/outlier bits 为 2/3。
  - 当前已有 run 证据显示 3.5-bit 平均 effective index bits 为 3.5000，outlier count 为 64，regular/outlier bits 为 3/4。
- [ ] 用完整 Table 1 数值判断是否需要改成论文文字中的 32-channel outlier 变体，或记录论文算术差异。

## 3. 数据和模型准备分工

### 你负责

- [x] Llama-3.1-8B-Instruct 权重已在本地 cache。
- [x] LongBench / LongBench-E 部分数据已在本地 cache。
- [x] Needle 16k EN 已在本地 cache。
- [x] DBpedia OpenAI3 embedding 1536d / 3072d 已在本地 cache。
- [ ] 补齐完整 Table 1 所需 LongBench / LongBench-E 任务 cache。
- [ ] 如后续要复现 Figure 5 / Table 2，再提供 GloVe 200d。
- [ ] 如后续迁移 Ministral，再确认模型名称、权限和本地 cache。

### 我负责

- [x] 建立项目结构、配置文件和环境记录。
- [x] 创建并验证 `turboquant` conda 环境。
- [x] 实现 TurboQuant core。
- [x] 实现 Llama full-cache LongBench smoke。
- [x] 实现 TurboQuant KV cache prototype。
- [x] 实现整数 bit-width 的 packed TurboQuant KV cache。
- [x] 实现 2.5-bit / 3.5-bit outlier-channel policy 的可复现版本。
- [x] 实现 LongBench 官方风格 scorer、旧 JSONL score backfill 和 Table 1 聚合脚本。
- [x] 实现 Needle 16k eval/summary 脚本。
- [x] 实现 Needle 本地 16k Figure-4-style summary 脚本。
- [x] 实现 Needle generated length-grid 构造脚本和分 target-token 评测过滤。
- [x] 实现 DBpedia ANN recall 和 Table 2 quantization timing 脚本。
- [x] 跑 DBpedia 1536d/3072d 论文规模 ANN recall。
- [x] 生成 Figure 5 DBpedia plot 和 Table 2 summary artifacts。
- [x] 跑完本地 LongBench-E `2wikimqa` Full Cache 300/300。
- [x] 跑完本地 LongBench-E `2wikimqa` TurboQuant 2.5-bit / 3.5-bit 300/300。
- [x] 跑完本地标准 LongBench `2wikimqa` Full Cache / TurboQuant 2.5-bit / 3.5-bit 200/200。
- [x] 实现本地可用 LongBench 任务的 Table-1-shaped 聚合与输出。
- [x] 实现 JSONL shard 按 `index` 合并工具，支持多 GPU 分块运行。
- [x] 实现 LongBench / LongBench-E cache 准备和 `paths.yaml` 更新工具。
- [x] 实现 LongBench Hub snapshot parquet 发现与评测读取支持。
- [x] 实现 Table 1 manifest，跟踪 complete / missing_dataset 状态并生成后续命令。
- [x] 实现本地 Needle 16k Figure-4-style 汇总脚本。
- [x] 实现 Needle Figure-4-style heatmap 复现脚本。
- [ ] 数据补齐后运行完整 Needle Figure 4 heatmap 复现。
- [ ] 维护 `reproduce/STATUS.md` 中的每次运行记录。

## 4. 当前环境

- Conda env：`turboquant`
- 创建方式：从已有 `layer_skip` clone。
- 关键版本：
  - `torch 2.2.1+cu121`
  - `transformers 4.53.0`
  - `datasets 3.6.0`
  - `accelerate 1.8.1`
  - `flash-attn 2.5.6`
- GPU：8 x NVIDIA GeForce RTX 4090。
- 环境细节见：`reproduce/ENVIRONMENT.md`。

## 5. 推荐执行顺序

- [x] Step 1：创建并验证 `turboquant` conda 环境。
- [x] Step 2：确认本地 Llama snapshot、LongBench-E、Needle、DBpedia cache 可读取。
- [x] Step 3：实现 TurboQuant core，并用 Figure 3-style sanity check 验证。
- [x] Step 4：跑通 Llama full-cache LongBench smoke。
- [x] Step 5：接入 TurboQuant KV cache prototype，并跑通 smoke。
- [x] Step 6：实现整数 bit-width 的 packed TurboQuant KV cache。
- [x] Step 7：实现 2.5-bit / 3.5-bit outlier-channel 策略的可复现版本。
- [x] Step 8：实现 LongBench scoring 和 Table-1-shaped partial aggregation。
- [x] Step 9：实现 Needle 16k smoke 管线。
- [x] Step 10：实现 DBpedia ANN recall / quantization runtime smoke 管线。
- [x] Step 11：跑 DBpedia 1536d/3072d 论文规模 ANN recall。
- [x] Step 12：生成 DBpedia Figure 5 plot 和 Table 2 summary。
- [x] Step 13：跑完本地 LongBench-E `2wikimqa` Full Cache 300/300。
- [x] Step 14：跑完本地 LongBench-E `2wikimqa` TurboQuant 2.5-bit / 3.5-bit 300/300。
- [x] Step 15：跑完本地标准 LongBench `2wikimqa` Full Cache / TurboQuant 2.5-bit / 3.5-bit 200/200。
- [x] Step 16：实现 LongBench / LongBench-E cache 准备工具和 Table 1 manifest。
- [x] Step 17：增强 LongBench 数据发现与评测读取，支持 Hub snapshot parquet。
- [x] Step 18：跑完本地 Needle 16k 24-example Full-Precision / TurboQuant grid 并生成 summary。
- [x] Step 19：实现并生成本地 Needle generated length-grid，补齐 Figure 4 评测入口。
- [x] Step 20：实现 Figure-4-style heatmap artifact builder 并用现有结果验证。
- [ ] Step 21：补齐 LongBench / LongBench-E 数据 cache。
- [ ] Step 22：复跑完整 Llama Table 1 三行。
- [ ] Step 23：补齐真实 Needle 32k/65k/104k 源数据并复现完整 Figure 4 的 Full-Precision 与 TurboQuant。
- [ ] Step 24：补齐 GloVe 并完成完整 Figure 5。
- [x] Step 25：整理当前第一阶段复现报告，记录已有结果、数据缺口和与论文数值的对齐状态。
- [ ] Step 26：数据补齐并完成完整 Table 1 / Figure 4 后，更新最终第一阶段复现报告。

## 6. 常用命令

运行 core tests：

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python -m pytest tests/test_core.py -q
```

运行 Figure 3-style sanity：

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/validate_figure3_core.py \
  --dimension 256 \
  --num-vectors 512 \
  --output-csv reproduce/runs/figure3_core_d256.csv \
  --summary-output reproduce/runs/figure3_core_d256_summary.json
```

运行 Full Cache LongBench smoke：

```bash
cd /home/liying/projects/turboquant
CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --max-examples 2 \
  --device cuda:0 \
  --cache-mode full \
  --output reproduce/runs/longbench_full_cache_debug_2.jsonl
```

运行 TurboQuant packed cache smoke：

```bash
cd /home/liying/projects/turboquant
CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --max-examples 1 \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 3 \
  --codebook-grid-size 10001 \
  --output reproduce/runs/longbench_turboquant_packed_cache_smoke_1_3bit.jsonl
```

运行 LongBench-E `2wikimqa` TurboQuant 分块：

```bash
cd /home/liying/projects/turboquant
CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_e_2wikimqa \
  --start-index 0 \
  --end-index 5 \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --codebook-grid-size 10001 \
  --resume \
  --output reproduce/runs/longbench_e_2wikimqa_turboquant_2_5bit_chunked.jsonl \
  --progress-every 1

CUDA_VISIBLE_DEVICES=1 conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_e_2wikimqa \
  --start-index 0 \
  --end-index 5 \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 3.5 \
  --codebook-grid-size 10001 \
  --resume \
  --output reproduce/runs/longbench_e_2wikimqa_turboquant_3_5bit_chunked.jsonl \
  --progress-every 1
```

运行本地已缓存 LongBench-E `2wikimqa` Full Cache 300/300：

```bash
cd /home/liying/projects/turboquant
CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --device cuda:0 \
  --cache-mode full \
  --output reproduce/runs/longbench_e_2wikimqa_full_cache_all.jsonl

conda run -n turboquant python scripts/summarize_jsonl_accuracy.py \
  reproduce/runs/longbench_e_2wikimqa_full_cache_all.jsonl \
  --output reproduce/runs/longbench_e_2wikimqa_full_cache_all.aggregate.json
```

生成当前 partial Table 1：

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/build_table1_summary.py \
  --run "Full Cache" 16 "Llama-3.1-8B-Instruct" reproduce/runs/longbench_e_2wikimqa_full_cache_all.jsonl \
  --run "TurboQuant" 2.5 "Llama-3.1-8B-Instruct" reproduce/runs/longbench_turboquant_packed_cache_smoke_1_2_5bit.jsonl \
  --run "TurboQuant" 3.5 "Llama-3.1-8B-Instruct" reproduce/runs/longbench_turboquant_packed_cache_smoke_1_3_5bit.jsonl \
  --output-prefix reproduce/runs/table1_llama_partial_current
```

补齐 LongBench / LongBench-E 数据后的 Table 1 运行编排：

```bash
cd /home/liying/projects/turboquant

conda run -n turboquant python scripts/refresh_reproduction_state.py \
  --tag after_data_add \
  --gpus 0 1 4 5

# 如果 refresh 报告显示有新的 not_started / partial 条目，再运行对应 plan shell。
bash reproduce/logs/reproduction_state_refresh_after_data_add_table1_plan.sh
```

也可以手动分步刷新 LongBench：

```bash
cd /home/liying/projects/turboquant

conda run -n turboquant python scripts/prepare_longbench_cache.py \
  --cache-root /home/liying/datasets/turboquant/hf_cache \
  --hub-cache-root /home/liying/.cache/huggingface/hub \
  --update-paths \
  --output-report reproduce/logs/longbench_cache_prepare_after_data_add.json

conda run -n turboquant python scripts/build_table1_manifest.py \
  --output reproduce/logs/table1_manifest_after_data_add.json

conda run -n turboquant python scripts/plan_table1_runs.py \
  --manifest reproduce/logs/table1_manifest_after_data_add.json \
  --output-prefix reproduce/logs/table1_run_plan_after_data_add \
  --gpus 0 1 4 5 \
  --turboquant-chunk-size 50

bash reproduce/logs/table1_run_plan_after_data_add.sh
```

生成当前第一阶段自动报告：

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/build_first_stage_report.py
```

当前已有数据的 planner 验证产物：

- `reproduce/logs/table1_run_plan_current_2026-06-11.{json,md,sh}`：当前没有新任务可跑，shell 只汇总已有 complete 输出。
- `reproduce/logs/table1_run_plan_complete_rechunk_demo_2026-06-11.{json,md,sh}`：用已完成的 `longbench_2wikimqa` TurboQuant 2.5-bit 演示 4 个 50-example shard、merge、aggregate 的命令结构。

运行 Needle 16k smoke：

```bash
cd /home/liying/projects/turboquant
CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/needle/run_needle_eval.py \
  --examples-per-position 1 \
  --positions start middle end \
  --device cuda:0 \
  --cache-mode full \
  --max-new-tokens 32 \
  --output reproduce/runs/needle_full_precision_position_balanced_smoke_3.jsonl

CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/needle/run_needle_eval.py \
  --max-examples 1 \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --codebook-grid-size 10001 \
  --max-new-tokens 32 \
  --output reproduce/runs/needle_turboquant_2_5bit_smoke_1.jsonl
```

运行 Needle 16k 本地 24-example grid 并生成 Figure-4-style summary：

```bash
cd /home/liying/projects/turboquant
CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/needle/run_needle_eval.py \
  --examples-per-position-lang 1 \
  --positions start middle end \
  --distractor-langs ar de en es hi multilingual vi zh \
  --device cuda:0 \
  --cache-mode full \
  --max-new-tokens 32 \
  --output reproduce/runs/needle_16k_full_precision_grid_24.jsonl

CUDA_VISIBLE_DEVICES=1 conda run -n turboquant python experiments/needle/run_needle_eval.py \
  --examples-per-position-lang 1 \
  --positions start middle end \
  --distractor-langs ar de en es hi multilingual vi zh \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --codebook-grid-size 10001 \
  --max-new-tokens 32 \
  --output reproduce/runs/needle_16k_turboquant_2_5bit_grid_24.jsonl

conda run -n turboquant python scripts/summarize_needle_results.py \
  reproduce/runs/needle_16k_full_precision_grid_24.jsonl \
  --output reproduce/runs/needle_16k_full_precision_grid_24.aggregate.json

conda run -n turboquant python scripts/summarize_needle_results.py \
  reproduce/runs/needle_16k_turboquant_2_5bit_grid_24.jsonl \
  --output reproduce/runs/needle_16k_turboquant_2_5bit_grid_24.aggregate.json

conda run -n turboquant python scripts/build_figure4_needle_summary.py \
  --run Full-Precision 16 reproduce/runs/needle_16k_full_precision_grid_24.aggregate.json \
  --run TurboQuant 2.5 reproduce/runs/needle_16k_turboquant_2_5bit_grid_24.aggregate.json \
  --output-prefix reproduce/runs/figure4_needle_16k_local_summary
```

生成当前 Figure 4 可用数据运行计划和 available heatmap：

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/refresh_reproduction_state.py \
  --tag after_needle_data_add \
  --gpus 0 1 4 5

# 如果 refresh 报告显示 Figure 4 有新的 not_started / partial 条目，再运行对应 plan shell。
bash reproduce/logs/reproduction_state_refresh_after_needle_data_add_figure4_plan.sh
```

也可以手动分步刷新 Needle：

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/prepare_needle_cache.py \
  --configs 4k 8k 16k 32k 65k 104k \
  --update-paths \
  --output-report reproduce/logs/needle_cache_prepare_all_configs_after_data_add.json

conda run -n turboquant python scripts/plan_figure4_runs.py \
  --output-prefix reproduce/logs/figure4_run_plan_current_2026-06-11 \
  --plot

bash reproduce/logs/figure4_run_plan_current_2026-06-11.sh
```

补齐真实 Needle 4k/8k/32k/65k/104k 数据并注册到 `configs/paths.yaml` 后，重新运行同一 planner；它会对 `not_started` / `partial` 的真实数据入口生成 Full-Precision 和 TurboQuant 2.5-bit 命令。16k replay demo 可用下面命令生成，输出带 `replay_demo` 后缀，不覆盖当前基准结果：

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/plan_figure4_runs.py \
  --output-prefix reproduce/logs/figure4_run_plan_16k_replay_demo_2026-06-11 \
  --statuses not_started partial \
  --token-configs 16k \
  --methods full_precision turboquant_2_5bit \
  --output-tag replay_demo \
  --no-resume
```

构建 generated Needle length-grid，并按目标长度运行：

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/build_needle_length_grid.py \
  --target-tokens 4096 8192 16384 32768 65536 104000 \
  --examples-per-cell 1 \
  --output-dataset-key needle_generated_length_grid_en \
  --output-dir reproduce/generated_data/needle_length_grid \
  --report reproduce/logs/needle_length_grid_report_2026-06-11.json \
  --update-paths

CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/needle/run_needle_eval.py \
  --dataset-key needle_generated_length_grid_en \
  --target-prompt-tokens 4096 \
  --device cuda:0 \
  --cache-mode full \
  --max-new-tokens 32 \
  --output reproduce/runs/needle_generated_4k_full_precision.jsonl

CUDA_VISIBLE_DEVICES=1 conda run -n turboquant python experiments/needle/run_needle_eval.py \
  --dataset-key needle_generated_length_grid_en \
  --target-prompt-tokens 4096 \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits 2.5 \
  --codebook-grid-size 10001 \
  --max-new-tokens 32 \
  --output reproduce/runs/needle_generated_4k_turboquant_2_5bit.jsonl

conda run -n turboquant python scripts/summarize_needle_results.py \
  reproduce/runs/needle_generated_4k_full_precision.jsonl \
  --output reproduce/runs/needle_generated_4k_full_precision.aggregate.json

conda run -n turboquant python scripts/summarize_needle_results.py \
  reproduce/runs/needle_generated_4k_turboquant_2_5bit.jsonl \
  --output reproduce/runs/needle_generated_4k_turboquant_2_5bit.aggregate.json

conda run -n turboquant python scripts/build_figure4_needle_summary.py \
  --run Full-Precision 16 reproduce/runs/needle_generated_4k_full_precision.aggregate.json \
  --run TurboQuant 2.5 reproduce/runs/needle_generated_4k_turboquant_2_5bit.aggregate.json \
  --output-prefix reproduce/runs/figure4_needle_generated_4k_summary \
  --title "Figure 4 Needle Generated 4k Diagnostic Summary" \
  --description "This is a generated 4k diagnostic slice built from the local Needle 16k cache. It compares Full-Precision and TurboQuant on the same 24 examples, but it is not a paper Figure 4 reproduction because the generated crop changes the original Needle distribution and the local cache still lacks true 4k-104k length grids."
```

运行 DBpedia ANN / Table 2 smoke：

```bash
cd /home/liying/projects/turboquant
CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/ann_search/run_turboquant_ann.py \
  --dataset-key dbpedia_openai3_1536 \
  --num-database 2048 \
  --num-queries 128 \
  --bits 2 4 \
  --device cuda:0 \
  --output-json reproduce/runs/ann_dbpedia_1536_smoke.json \
  --output-csv reproduce/runs/ann_dbpedia_1536_smoke.csv

CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python scripts/benchmark_quantization_time.py \
  --dimensions 200 1536 3072 \
  --num-vectors 2048 \
  --bits 4 \
  --device cuda:0 \
  --output-json reproduce/runs/table2_quantization_time_smoke.json \
  --output-csv reproduce/runs/table2_quantization_time_smoke.csv
```

## 7. 后续阶段

第一阶段完成后，再考虑：

- Ministral-7B-Instruct 的 Table 1 迁移复现。
- Figure 5 最近邻检索实验中 TurboQuant 的 recall 曲线。
- Table 2 中 TurboQuant quantization runtime。
- 在复现基础上探索增量研究，例如更明确的 outlier policy、更快的 rotation/codebook 路径、packed KV cache kernel、或对长上下文任务的自适应 bit allocation。
