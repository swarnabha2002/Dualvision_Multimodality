#!/bin/bash
# DualVision evaluation on DualVision-500 across 13 conditions:
#   original (clean) + 12 RGB degradations (blur/bright/fog). IR is always clean.
#
# Usage:
#   bash scripts/eval.sh <ckpt_path> <data_version> [answer_file]
#     ckpt_path     : trained LoRA dir (must contain 'llava' and 'lora' in its name)
#     data_version  : v1 | v2   (default: v1)
#     answer_file   : optional; defaults to <ckpt_parent>/exps/eval_<version>.jsonl
#
# Runs NUM_GPUS conditions in parallel, then the remainder — never more than NUM_GPUS at once.
set +e

ckpt_path="${1:?usage: eval.sh <ckpt_path> <v1|v2> [answer_file]}"
version="${2:-v1}"

DATA_ROOT="${DATA_ROOT:-/path/to/images}"   # contains HDRT/{visible,infrared}, LLVIP/{visible,infrared}
REPO_DATA="${REPO_DATA:-$(cd "$(dirname "$0")/../../data" && pwd)}"
NUM_GPUS="${NUM_GPUS:-8}"

test_jsonl="${REPO_DATA}/${version}/dv500.jsonl"
answer_file="${3:-$(dirname "${ckpt_path}")/exps/eval_${version}.jsonl}"
mkdir -p "$(dirname "${answer_file}")"

subfolders=("" blur5 blur10 blur15 blur20 bright10 bright20 bright30 bright50 fog070 fog085 fog092 fog097)

# Track PIDs per batch and wait only on those (bare `wait` would also wait on
# unrelated background children such as a tee from a parent script).
batch_pids=()
for i in "${!subfolders[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    subfolder="${subfolders[$i]}"
    echo "Launching '${subfolder:-original}' on GPU ${gpu_id}"

    CUDA_VISIBLE_DEVICES=$gpu_id python -m llava.eval.my_model_vqa \
        --model-path "${ckpt_path}" \
        --model-base liuhaotian/llava-v1.5-7b \
        --question-file "${test_jsonl}" \
        --data_root_dir "${DATA_ROOT}" \
        --answers-file "${answer_file}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --subfolder "${subfolder}" \
        --use_features False &
    batch_pids+=($!)

    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait "${batch_pids[@]}"
        batch_pids=()
    fi
done
if (( ${#batch_pids[@]} > 0 )); then
    wait "${batch_pids[@]}"
fi

echo "=== All evals done: $(date) ==="
echo "Predictions written next to: ${answer_file}"
