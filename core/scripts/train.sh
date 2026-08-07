#!/bin/bash
# DualVision training (LoRA finetune of LLaVA-1.5-7b with RGB-IR fusion).
#
# Usage:
#   bash scripts/train.sh <strategy> <data_version>
#     strategy      : local_xattn | concat   (default: local_xattn)
#     data_version  : v1 | v2                (default: v1)
#
# v1 = the annotation set used in the paper.
# v2 = the larger, Opus-verified annotation set (recommended).
#
# Set DATA_ROOT to the directory that contains HDRT/ and LLVIP/ image folders,
# and REPO_DATA to the released annotations dir.
set -e

strategy="${1:-local_xattn}"
version="${2:-v1}"

# ---- paths (edit these for your machine) ----
DATA_ROOT="${DATA_ROOT:-/path/to/images}"          # contains HDRT/{visible,infrared}, LLVIP/{visible,infrared}
REPO_DATA="${REPO_DATA:-$(cd "$(dirname "$0")/../../data" && pwd)}"
RUNS_ROOT="${RUNS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/runs}"
NUM_GPUS="${NUM_GPUS:-8}"

train_json="${REPO_DATA}/${version}/train.json"
test_jsonl="${REPO_DATA}/${version}/dv500.jsonl"

# Free the deepspeed launch port if an old run left it bound (no sudo needed)
fuser -k 29500/tcp 2>/dev/null || true

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${RUNS_ROOT}/${strategy}_${version}/${timestamp}"
ckpt_path="${run_dir}/${strategy}-llava-v1.5-7b-task-lora"   # must contain 'llava' and 'lora'
answer_file="${run_dir}/exps/${strategy}_${version}.jsonl"
mkdir -p "${run_dir}/logs" "${run_dir}/exps"

# Tee all output to run.log inside the timestamped dir
exec > >(tee -a "${run_dir}/logs/run.log") 2>&1
echo "=== Run started: $(date) ==="
echo "strategy=${strategy} version=${version}"
echo "train=${train_json}"
echo "run_dir=${run_dir}"

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path "${train_json}" \
    --data_root_dir "${DATA_ROOT}" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fusion_strategy "${strategy}" \
    --learned_emb_mask False \
    --fusion_inp_dim 1024 \
    --fusion_proj_dim 1024 \
    --fusion_nheads 1 \
    --fusion_nblocks 3 \
    --degradation_prob 0.25 \
    --fusion_window_radius 1 \
    --n_rgb_tokens 577 \
    --n_m1_tokens 577 \
    --use_features False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "${ckpt_path}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none

echo "=== Training done. Evaluating on dv500_${version} ==="
DATA_ROOT="${DATA_ROOT}" NUM_GPUS="${NUM_GPUS}" bash "$(dirname "$0")/eval.sh" "${ckpt_path}" "${version}" "${answer_file}"
