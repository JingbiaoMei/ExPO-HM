#!/bin/bash
set -x
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}./verl"

# Optional first positional arg for rollout engine.
if [ "$#" -gt 0 ] && { [ "$1" = "vllm" ] || [ "$1" = "sglang" ] || [ "$1" = "hf" ]; }; then
  ENGINE="$1"
  shift
else
  ENGINE="${ENGINE:-vllm}"
fi

EXTRA_ARGS=("$@")

DATA_ROOT=${DATA_ROOT:-./data/verl}
FG_TRAIN_PATH=${FG_TRAIN_PATH:-$DATA_ROOT/CDE/fb-fine-grained-combined/train.parquet}
BINARY_TRAIN_PATH=${BINARY_TRAIN_PATH:-$DATA_ROOT/CDE/FB/train.parquet}
TEST_PATH=${TEST_PATH:-$DATA_ROOT/FB/test.parquet}

# Warm start from merged SFT fine-grained checkpoint.
# Override with MODEL_PATH if your merged checkpoint lives elsewhere.
MODEL_PATH=${MODEL_PATH:-./checkpoints/fb/qwen2_5vl_7b/sft_merged}

PROJECT_NAME=${PROJECT_NAME:-verl_hm}
BASE_EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_7b_grpo_FB_CDEpaper}
STAGE1_EXPERIMENT_NAME=${STAGE1_EXPERIMENT_NAME:-${BASE_EXPERIMENT_NAME}_fg_stage}
STAGE2_EXPERIMENT_NAME=${STAGE2_EXPERIMENT_NAME:-${BASE_EXPERIMENT_NAME}}

FG_EPOCHS=${FG_EPOCHS:-3}
BINARY_EPOCHS=${BINARY_EPOCHS:-3}

NGPUS=${NGPUS:-8}
NNODES=${NNODES:-1}

if [ ! -f "$FG_TRAIN_PATH" ]; then
  echo "Missing FG CDE train parquet: $FG_TRAIN_PATH"
  exit 1
fi

if [ ! -f "$BINARY_TRAIN_PATH" ]; then
  echo "Missing binary CDE train parquet: $BINARY_TRAIN_PATH"
  exit 1
fi

if [ ! -f "$TEST_PATH" ]; then
  echo "Missing FB test parquet: $TEST_PATH"
  exit 1
fi

run_stage() {
  local train_path="$1"
  local model_path="$2"
  local experiment_name="$3"
  local total_epochs="$4"

  # Paper-aligned CDE defaults: a=0.1, b=0.5, w=0.2, rho=0.25
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="$train_path" \
      data.val_files="$TEST_PATH" \
      data.train_batch_size=512 \
      data.max_prompt_length=1024 \
      data.max_response_length=2048 \
      data.filter_overlong_prompts=False \
      data.truncation='error' \
      data.image_key=images \
      actor_rollout_ref.model.path="$model_path" \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=128 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.01 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
      actor_rollout_ref.rollout.name="$ENGINE" \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
      actor_rollout_ref.rollout.enable_chunked_prefill=False \
      actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.free_cache_engine=False \
      actor_rollout_ref.rollout.n=5 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.use_kl_in_reward=False \
      trainer.critic_warmup=0 \
      trainer.logger=['console','wandb'] \
      trainer.project_name="$PROJECT_NAME" \
      trainer.experiment_name="$experiment_name" \
      trainer.n_gpus_per_node="$NGPUS" \
      trainer.nnodes="$NNODES" \
      trainer.save_freq=20 \
      trainer.test_freq=5 \
      trainer.total_epochs="$total_epochs" \
      +reward_model.reward_kwargs.cde_weight=0.2 \
      +reward_model.reward_kwargs.format_score=0.1 \
      +reward_model.reward_kwargs.low_entropy_cutoff=0.10 \
      +reward_model.reward_kwargs.high_entropy_cutoff_correct=0.50 \
      +reward_model.reward_kwargs.high_entropy_cutoff_wrong=0.50 \
      +reward_model.reward_kwargs.confidence_penalty_ratio=0.25 \
      +reward_model.reward_kwargs.use_sigmoid_smoothing=False \
      +reward_model.reward_kwargs.sigmoid_steepness=8.0 \
      +actor_rollout_ref.rollout.reward_logprobs=50 \
      "${EXTRA_ARGS[@]}"
}

echo "[Curriculum] Stage 1/2: FG CDE for ${FG_EPOCHS} epochs"
run_stage "$FG_TRAIN_PATH" "$MODEL_PATH" "$STAGE1_EXPERIMENT_NAME" "$FG_EPOCHS"

STAGE1_CKPT_ROOT="./checkpoints/${PROJECT_NAME}/${STAGE1_EXPERIMENT_NAME}"
LATEST_ACTOR_DIR=$(ls -d "${STAGE1_CKPT_ROOT}"/global_step_*/actor 2>/dev/null | sort -V | tail -n1)
if [ -z "$LATEST_ACTOR_DIR" ]; then
  echo "Cannot find stage-1 actor checkpoint under ${STAGE1_CKPT_ROOT}"
  exit 1
fi

STAGE1_MERGED_MODEL=${STAGE1_MERGED_MODEL:-${STAGE1_CKPT_ROOT}/stage1_latest_merged}
python3 scripts/grpo/tools/merge_checkpoint.py \
  --local_dir "$LATEST_ACTOR_DIR" \
  --target_dir "$STAGE1_MERGED_MODEL"

if [ ! -f "${STAGE1_MERGED_MODEL}/config.json" ]; then
  echo "Merged model is missing config.json: ${STAGE1_MERGED_MODEL}"
  exit 1
fi

echo "[Curriculum] Stage 2/2: Binary CDE for ${BINARY_EPOCHS} epochs"
run_stage "$BINARY_TRAIN_PATH" "$STAGE1_MERGED_MODEL" "$STAGE2_EXPERIMENT_NAME" "$BINARY_EPOCHS"
