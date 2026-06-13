#!/usr/bin/env bash
set -euo pipefail

cd /home/liying/projects/turboquant

RUN_ROOT="reproduce/runs/table1_official"
LOG_ROOT="reproduce/logs/table1_official_full_cache_parallel"
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/inputs" "${LOG_ROOT}"

run_eval() {
  local gpu="$1"
  local dataset_key="$2"
  local expected_end="$3"
  local output="${RUN_ROOT}/${dataset_key}_full_cache_all.jsonl"
  local aggregate="${RUN_ROOT}/${dataset_key}_full_cache_all.aggregate.json"
  local log="${LOG_ROOT}/${dataset_key}.log"

  {
    echo "[$(date --iso-8601=seconds)] start ${dataset_key} gpu=${gpu} expected=${expected_end}"
    CUDA_VISIBLE_DEVICES="${gpu}" conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
      --dataset-key "${dataset_key}" \
      --device cuda:0 \
      --cache-mode full \
      --prompt-mode longbench \
      --chat-template-mode auto \
      --start-index 0 \
      --end-index "${expected_end}" \
      --resume \
      --output "${output}" \
      --progress-every 10
    conda run -n turboquant python scripts/summarize_jsonl_accuracy.py "${output}" --output "${aggregate}"
    echo "[$(date --iso-8601=seconds)] done ${dataset_key}"
  } >"${log}" 2>&1
}

# Full LongBench-V1 English Table 1 tasks. These are full splits, not smoke subsets.
run_eval 0 longbench_narrativeqa 200 &
run_eval 1 longbench_qasper 200 &
run_eval 2 longbench_multifieldqa_en 150 &
run_eval 3 longbench_hotpotqa 200 &
run_eval 5 longbench_2wikimqa 200 &
wait

run_eval 0 longbench_musique 200 &
run_eval 1 longbench_gov_report 200 &
run_eval 2 longbench_qmsum 200 &
run_eval 3 longbench_multi_news 200 &
run_eval 5 longbench_trec 200 &
wait

run_eval 0 longbench_triviaqa 200 &
run_eval 1 longbench_samsum 200 &
run_eval 2 longbench_passage_retrieval_en 200 &
run_eval 3 longbench_passage_count 200 &
run_eval 5 longbench_lcc 500 &
wait

run_eval 0 longbench_repobench-p 500

cat \
  "${RUN_ROOT}/longbench_narrativeqa_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_qasper_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_multifieldqa_en_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_hotpotqa_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_2wikimqa_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_musique_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_gov_report_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_qmsum_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_multi_news_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_trec_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_triviaqa_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_samsum_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_passage_retrieval_en_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_passage_count_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_lcc_full_cache_all.jsonl" \
  "${RUN_ROOT}/longbench_repobench-p_full_cache_all.jsonl" \
  > "${RUN_ROOT}/inputs/full_available.jsonl"

conda run -n turboquant python scripts/build_table1_summary.py \
  --output-prefix "${RUN_ROOT}/table1_llama_full_cache_summary" \
  --run 'Full Cache' 16.0 Llama-3.1-8B-Instruct "${RUN_ROOT}/inputs/full_available.jsonl"

