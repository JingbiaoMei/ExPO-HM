#!/bin/bash
set -x

# Merge an FSDP "actor" checkpoint shard directory into a single HF-style directory.
#
# Usage:
#   bash scripts/grpo/tools/merge_actor_checkpoint.sh [CKPT_ROOT] [STEP]
#
# Examples:
#   bash scripts/grpo/tools/merge_actor_checkpoint.sh ./checkpoints/verl_hm/qwen2_5_vl_7b_grpo_FB 180
#   CKPT_ROOT=./checkpoints/verl_hm/qwen2_5_vl_7b_grpo_FB STEP=180 bash scripts/grpo/tools/merge_actor_checkpoint.sh
#   CKPT_ROOT=./checkpoints/verl_hm/qwen2_5_vl_7b_grpo_FB bash scripts/grpo/tools/merge_actor_checkpoint.sh  # auto-pick latest
#
# Notes:
# - If STEP is omitted, the script finds the latest `global_step_*/actor` under CKPT_ROOT.
# - Override TARGET_DIR to control output location.

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}./verl"

CKPT_ROOT="${1:-${CKPT_ROOT:-}}"
STEP="${2:-${STEP:-}}"

if [ -z "$CKPT_ROOT" ]; then
  echo "Missing CKPT_ROOT. Usage: bash scripts/grpo/tools/merge_actor_checkpoint.sh [CKPT_ROOT] [STEP]"
  exit 1
fi

if [ -n "$STEP" ]; then
  ACTOR_DIR="${CKPT_ROOT}/global_step_${STEP}/actor"
else
  ACTOR_DIR="$(ls -d "${CKPT_ROOT}"/global_step_*/actor 2>/dev/null | sort -V | tail -n1)"
  if [ -z "$ACTOR_DIR" ]; then
    echo "Cannot find any actor checkpoints under: ${CKPT_ROOT}/global_step_*/actor"
    exit 1
  fi
  STEP_DIR="$(basename "$(dirname "$ACTOR_DIR")")"  # global_step_XXX
  STEP="${STEP_DIR#global_step_}"
fi

if [ ! -d "$ACTOR_DIR" ]; then
  echo "Actor checkpoint dir does not exist: $ACTOR_DIR"
  exit 1
fi

TARGET_DIR="${TARGET_DIR:-${CKPT_ROOT}/checkpoint-${STEP}-merged}"

python3 scripts/grpo/tools/merge_checkpoint.py \
  --local_dir "$ACTOR_DIR" \
  --target_dir "$TARGET_DIR"

echo "Merged checkpoint written to: $TARGET_DIR"

