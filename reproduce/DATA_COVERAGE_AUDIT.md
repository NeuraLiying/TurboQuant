# TurboQuant Data Coverage Audit

Updated: 2026-06-11 09:07 Asia/Shanghai

This audit records the current contents visible under `/home/liying/datasets/turboquant/hf_cache` for the first-stage Llama-3.1-8B-Instruct reproduction.

## Commands Run

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/inventory_hf_cache.py \
  --cache-root /home/liying/datasets/turboquant/hf_cache \
  --output reproduce/logs/hf_cache_inventory_2026-06-11_user_claim_refresh.json

conda run -n turboquant python scripts/prepare_longbench_cache.py \
  --cache-root /home/liying/datasets/turboquant/hf_cache \
  --update-paths \
  --output-report reproduce/logs/longbench_cache_prepare_user_claim_refresh_2026-06-11.json

conda run -n turboquant python scripts/build_table1_manifest.py \
  --output reproduce/logs/table1_manifest_user_claim_refresh_2026-06-11.json

conda run -n turboquant python scripts/prepare_longbench_cache.py \
  --cache-root /home/liying/datasets/turboquant/hf_cache \
  --hub-cache-root /home/liying/.cache/huggingface/hub \
  --update-paths \
  --output-report reproduce/logs/longbench_cache_prepare_snapshot_aware_2026-06-11.json

conda run -n turboquant python scripts/build_table1_manifest.py \
  --output reproduce/logs/table1_manifest_snapshot_aware_2026-06-11.json

conda run -n turboquant python scripts/prepare_needle_cache.py \
  --update-paths \
  --output-report reproduce/logs/needle_cache_prepare_2026-06-11.json

conda run -n turboquant python scripts/inventory_hf_cache.py \
  --cache-root /home/liying/datasets/turboquant/hf_cache \
  --output reproduce/logs/hf_cache_inventory_enhanced_2026-06-11.json

conda run -n turboquant python scripts/build_table1_manifest.py \
  --output reproduce/logs/table1_manifest_current_2026-06-11.json

conda run -n turboquant python scripts/audit_reproduction_assets.py \
  --output reproduce/logs/reproduction_asset_audit_2026-06-11.json \
  --markdown-output reproduce/logs/reproduction_asset_audit_2026-06-11.md

conda run -n turboquant python scripts/probe_raw_data_assets.py \
  --output reproduce/logs/raw_data_asset_probe_2026-06-11.json \
  --markdown-output reproduce/logs/raw_data_asset_probe_2026-06-11.md

find /home/liying/datasets/turboquant -type f \
  \( -iname '*longbench*' -o -iname '*long_bench*' -o -iname '*narrative*' \
     -o -iname '*qasper*' -o -iname '*hotpot*' -o -iname '*musique*' \
     -o -iname '*gov_report*' -o -iname '*multi_news*' -o -iname '*repobench*' \
     -o -iname '*needle*' -o -iname '*haystack*' \)
```

## Current Cache Contents

| Dataset family | Config / split | Examples | Evidence |
| --- | --- | ---: | --- |
| LongBench | `2wikimqa` / `test` | 200 | `reproduce/logs/hf_cache_inventory_2026-06-11_user_claim_refresh.json` |
| LongBench-E | `2wikimqa` / `test` | 300 | `reproduce/logs/hf_cache_inventory_2026-06-11_user_claim_refresh.json` |
| Needle-In-A-Haystack | `16k` / ar,de,en,es,hi,vi,zh | 2400 per language, 16800 total | `reproduce/logs/needle_cache_prepare_2026-06-11.json` |
| DBpedia OpenAI3 | 1536d / `train` | 1,000,000 | `reproduce/logs/hf_cache_inventory_2026-06-11_user_claim_refresh.json` |
| DBpedia OpenAI3 | 3072d / `train` | 1,000,000 | `reproduce/logs/hf_cache_inventory_2026-06-11_user_claim_refresh.json` |

The refreshed inventory reports:

- `dataset_info.json` files: 5
- Arrow files: 98
- Lock files: 17
- Incomplete marker files: 5
- `downloads` directory: empty
- LongBench cache prep: 2 cached entries, 30 missing entries
- Table 1 manifest: 6 complete entries, 90 missing-dataset entries

The strong reproduction asset audit reports:

- Script: `scripts/audit_reproduction_assets.py`
- JSON: `reproduce/logs/reproduction_asset_audit_2026-06-11.json`
- Markdown: `reproduce/logs/reproduction_asset_audit_2026-06-11.md`
- Symlink-aware refresh: `reproduce/logs/reproduction_asset_audit_symlink_aware_2026-06-11.{json,md}`
- LongBench/LongBench-E Table 1 entries: 2 materialized, 30 metadata-or-lock-only.
- Needle Figure 4 target configs: only `16k` materialized; `4k`, `8k`, `32k`, `65k`, and `104k` are missing.
- DBpedia Figure 5 configs: 1536d and 3072d materialized.
- The symlink-aware audit follows Hugging Face snapshot symlinks, so parquet/csv symlink files count as data when present.

The raw data asset probe reports:

- Script: `scripts/probe_raw_data_assets.py`
- JSON: `reproduce/logs/raw_data_asset_probe_2026-06-11.json`
- Markdown: `reproduce/logs/raw_data_asset_probe_2026-06-11.md`
- It checks project files, `/home/liying/datasets/turboquant`, and Hugging Face Hub snapshots/blobs for unregistered raw data.
- Result: `narrativeqa` has 0 data matches and 2 non-data lock matches; only `2wikimqa` has LongBench data matches. Needle `4k`, `8k`, `32k`, `65k`, and `104k` have 0 data matches; only `16k` has data matches.

There is a zero-byte lock file named like `THUDM___long_bench_narrativeqa` in the cache root, but there is no corresponding dataset directory, `dataset_info.json`, or Arrow file. It is evidence of an attempted cache/build operation, not usable `narrativeqa` data.

A wider filename scan under `/home/liying/datasets/turboquant` found no additional materialized LongBench task files beyond `2wikimqa` / `2wikimqa_e`. The only task-like LongBench miss hit is the zero-byte `THUDM___long_bench_narrativeqa...lock` file.

`configs/paths.yaml` now registers each local Needle 16k language split and an aggregate key:

- `needle_16k_ar`, `needle_16k_de`, `needle_16k_en`, `needle_16k_es`, `needle_16k_hi`, `needle_16k_vi`, `needle_16k_zh`: 2400 examples each.
- `needle_16k_all`: 16800 examples across the seven Arrow files.

`experiments/needle/run_needle_eval.py` and `scripts/build_needle_length_grid.py` now support multiple Arrow files in one dataset key. A small loading smoke was run:

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n turboquant python experiments/needle/run_needle_eval.py \
  --dataset-key needle_16k_all \
  --device cuda:0 \
  --cache-mode full \
  --max-examples 3 \
  --max-new-tokens 32 \
  --output reproduce/runs/needle_16k_all_full_precision_smoke_3.jsonl
```

The smoke verifies that multi-Arrow loading works. It is not an accuracy benchmark: the first three aggregate examples use Arabic hidden answers, while the model may answer in English paraphrase, so exact substring matching is not appropriate evidence for the Figure 4 score.

## Hub Snapshot Check

The project now checks both datasets cache directories and Hugging Face Hub dataset snapshots:

- `/home/liying/datasets/turboquant/hf_cache`
- `/home/liying/.cache/huggingface/hub`

Current Hub snapshots contain:

| Repo snapshot | Data files found |
| --- | --- |
| `datasets--Xnhyacinth--LongBench` | `2wikimqa/test-00000-of-00001.parquet` |
| `datasets--Xnhyacinth--LongBench-e` | `2wikimqa/test-00000-of-00001.parquet` |
| `datasets--THUDM--LongBench` | `LongBench.py`, `README.md`; no task data files |
| `datasets--ameyhengle--Multilingual-Needle-in-a-Haystack` | `data/16k/{ar,de,en,es,hi,vi,zh}.csv` |

The 2026-06-11 09:06 symlink-aware scan confirms the same result. Hugging Face snapshot data files are symlinks in this cache layout; even when symlinks are included, LongBench/LongBench-E still only have `2wikimqa`, and Needle still only has `16k`.

Snapshot-aware LongBench preparation still reports 2 cached entries and 30 missing entries:

- `reproduce/logs/longbench_cache_prepare_snapshot_aware_2026-06-11.json`
- `reproduce/logs/table1_manifest_snapshot_aware_2026-06-11.json`
- `reproduce/logs/longbench_cache_prepare_snapshot_only_probe_2026-06-11.json`

The LongBench evaluation script now supports one or more `arrow_files` and one or more `parquet_files` in `configs/paths.yaml`, so future multi-shard Arrow caches and parquet snapshots can be evaluated without first converting or merging them manually.

A loader regression smoke was run after adding multi-Arrow loading support:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key longbench_2wikimqa \
  --device cuda:0 \
  --cache-mode full \
  --max-examples 1 \
  --progress-every 1 \
  --output reproduce/runs/longbench_2wikimqa_loader_regression_smoke_1.jsonl
```

The smoke succeeded with `full_dataset_len=200` and 1/1 answer contains. It is a loader regression check, not a new benchmark result.

The snapshot-only probe used an empty project-local cache root and still found LongBench and LongBench-E `2wikimqa` from Hub snapshot parquet files, while `narrativeqa` remained missing. This verifies the snapshot fallback path itself.

## Missing For Full Paper Reproduction

Full Table 1 still needs the LongBench and LongBench-E configs other than `2wikimqa`:

- `narrativeqa`, `qasper`, `multifieldqa_en`
- `hotpotqa`, `musique`
- `gov_report`, `qmsum`, `multi_news`
- `trec`, `triviaqa`, `samsum`
- `passage_retrieval_en`, `passage_count`
- `lcc`, `repobench-p`

Full Figure 4 still needs Needle token-length/depth coverage beyond the local `16k` cache. The current local Figure-4-style run covers only a small 16k grid; the local cache has seven 16k language splits, but not true 4k/32k/65k/104k contexts.

Full Figure 5 still needs GloVe 200d. DBpedia 1536d and 3072d are already available and have been run at paper scale.

## Download Check

A fresh attempt to cache `Xnhyacinth/LongBench` `narrativeqa` was run:

```bash
conda run -n turboquant python scripts/prepare_longbench_cache.py \
  --cache-root /home/liying/datasets/turboquant/hf_cache \
  --repos longbench \
  --datasets narrativeqa \
  --download-missing \
  --update-paths \
  --output-report reproduce/logs/longbench_cache_prepare_download_attempt_narrativeqa_user_claim_refresh_2026-06-11.json
```

It did not add data. The report says the environment used the latest cached dataset version and only found local config `2wikimqa`.

Additional direct checks on 2026-06-11:

- `get_dataset_config_names("THUDM/LongBench")` can list configs from the cached loader module.
- `load_dataset("THUDM/LongBench", "narrativeqa", split="test")` fails with `Network is unreachable` because the actual data is not materialized locally.
- `load_dataset("Xnhyacinth/LongBench", "narrativeqa", split="test")` fails because only `2wikimqa` is cached for that repo.
- `load_dataset("Xnhyacinth/LongBench-e", "hotpotqa", split="test")` fails because only `2wikimqa` is cached for that repo.
- `load_dataset("ameyhengle/Multilingual-Needle-in-a-Haystack", <4k|8k|32k|65k|104k>, split="en")` fails because only `16k` is cached for that repo.

Network probes on 2026-06-11 also timed out:

- `curl -I -L --max-time 20 https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip`
- `curl -I -L --max-time 20 https://huggingface.co/datasets/Xnhyacinth/LongBench/resolve/main/README.md`
- `curl -I -L --max-time 20 https://huggingface.co/datasets/ameyhengle/Multilingual-Needle-in-a-Haystack/resolve/main/data/4k/en.csv`

All three failed to connect to `huggingface.co:443`, so the missing datasets cannot currently be filled from the network in this environment.

## Next Action

To continue full Table 1, copy or cache the missing LongBench / LongBench-E configs under `/home/liying/datasets/turboquant/hf_cache`, then rerun:

```bash
cd /home/liying/projects/turboquant
conda run -n turboquant python scripts/prepare_longbench_cache.py \
  --cache-root /home/liying/datasets/turboquant/hf_cache \
  --update-paths \
  --output-report reproduce/logs/longbench_cache_prepare_after_data_add.json

conda run -n turboquant python scripts/build_table1_manifest.py \
  --output reproduce/logs/table1_manifest_after_data_add.json
```
