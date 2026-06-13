#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 DATASET_KEY GPU METHOD_STEM KV_BITS" >&2
  exit 2
fi

dataset_key="$1"
gpu="$2"
method_stem="$3"
kv_bits="$4"

case "$dataset_key" in
  longbench_multifieldqa_en) end_index=150 ;;
  longbench_lcc|longbench_repobench-p) end_index=500 ;;
  *) end_index=200 ;;
esac

dataset_name="${dataset_key#longbench_}"
run_root="reproduce/runs/table1_official"
log_root="reproduce/logs/table1_turboquant_jobs"
mkdir -p "$run_root" "$log_root"

output="$run_root/longbench_${dataset_name}_${method_stem}_all.jsonl"
aggregate="$run_root/longbench_${dataset_name}_${method_stem}_all.aggregate.json"
log="$log_root/${dataset_name}_${method_stem}.log"
pidfile="$log_root/${dataset_name}_${method_stem}.pid"
jobscript="$log_root/${dataset_name}_${method_stem}.sh"

if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "already running: $dataset_key pid=$(cat "$pidfile")"
  exit 0
fi

cat > "$jobscript" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /home/liying/projects/turboquant
export CUDA_VISIBLE_DEVICES=$gpu
/home/liying/miniconda3/bin/conda run -n turboquant python experiments/longbench/run_full_cache_eval.py \
  --dataset-key '$dataset_key' \
  --device cuda:0 \
  --cache-mode turboquant \
  --kv-bits '$kv_bits' \
  --turboquant-fast-materialized-eval \
  --prompt-mode longbench \
  --chat-template-mode auto \
  --start-index 0 --end-index '$end_index' \
  --resume \
  --output '$output' \
  --progress-every 20
/home/liying/miniconda3/bin/conda run -n turboquant python scripts/summarize_jsonl_accuracy.py '$output' --output '$aggregate'
EOF
chmod +x "$jobscript"

setsid "$jobscript" > "$log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$pidfile"
echo "launched: $dataset_key gpu=$gpu pid=$pid log=$log"
