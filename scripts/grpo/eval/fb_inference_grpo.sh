#!/bin/bash

# GRPO evaluation inference wrapper for FB.
# Run from repo root.

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
CKPT_ROOT=${CKPT_ROOT:-./checkpoints/verl_hm/qwen2_5_vl_7b_grpo_FB_CDEpaper}
LOG_PREFIX=${LOG_PREFIX:-qwen25vl7b_grpo_fb}
BATCH_SIZE=${BATCH_SIZE:-1000}

# If MODEL_DIRS is set, use it (space-separated). Otherwise auto-discover merged checkpoints.
if [ -n "${MODEL_DIRS:-}" ]; then
  mapfile -t MODEL_LIST < <(printf "%s\n" ${MODEL_DIRS})
else
  mapfile -t MODEL_LIST < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "checkpoint-*-merged" | sort -V)
fi

if [ "${#MODEL_LIST[@]}" -eq 0 ]; then
  echo "No merged checkpoints found."
  echo "Set CKPT_ROOT or MODEL_DIRS and retry."
  exit 1
fi

QUESTION_PC="<image>Does this meme express hate or negative bias toward any of the following protected groups? Religion, Race, Sex, Nationality, Disability. Please respond with one or more protected categories if applicable. If the meme does not contain hateful content, respond with Benign."
QUESTION_ATTACK="<image>Does this meme use any of the following attack types against a group? Dehumanizing, Inferiority, Inciting violence, Mocking, Contempt, Slurs, Exclusion. Please respond with one or more attack types if applicable. If the meme does not contain hateful content, respond with Benign."
FORMAT_PROMPT="Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output format should be: <think> ... </think> <answer>...</answer>. Please strictly follow the format."

QUERY_PC="${QUESTION_PC} ${FORMAT_PROMPT}"
QUERY_ATTACK="${QUESTION_ATTACK} ${FORMAT_PROMPT}"

for model_path in "${MODEL_LIST[@]}"; do
  model_tag=$(basename "${model_path}")
  log_name="${LOG_PREFIX}_${model_tag}"

  echo "Running binary inference for ${model_tag}"
  python eval/inference/qwen2_hm_inference_vllm.py \
    --model_path "${model_path}" \
    --processor_path "${BASE_MODEL}" \
    --dataset "FB" \
    --data_split "dev_seen test_seen test_unseen" \
    --batch_size "${BATCH_SIZE}" \
    --log_name "${log_name}"

  echo "Running fine-grained PC inference for ${model_tag}"
  python eval/inference/qwen2_hm_inference_vllm_fine_grained.py \
    --model_path "${model_path}" \
    --processor_path "${BASE_MODEL}" \
    --dataset "FB-fine-grained-PC" \
    --data_split "dev_seen dev_unseen" \
    --batch_size "${BATCH_SIZE}" \
    --log_name "${log_name}" \
    --query "${QUERY_PC}"

  echo "Running fine-grained attack inference for ${model_tag}"
  python eval/inference/qwen2_hm_inference_vllm_fine_grained.py \
    --model_path "${model_path}" \
    --processor_path "${BASE_MODEL}" \
    --dataset "FB-fine-grained-attack" \
    --data_split "dev_seen dev_unseen" \
    --batch_size "${BATCH_SIZE}" \
    --log_name "${log_name}" \
    --query "${QUERY_ATTACK}"
done

