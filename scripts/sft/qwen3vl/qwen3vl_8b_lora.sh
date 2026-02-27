model_name=qwen3vl_8b
dataset_name=FB
export WANDB_PROJECT="LLAMAFACTORY_hateful_qwen3vl"

export WANDB_RUN_GROUP="Finetuning_${dataset_name}_${model_name}_sft"

export WANDB_NAME=${dataset_name}_${model_name}_lora_sft_warmup_fg_choice
llamafactory-cli train scripts/sft/qwen3vl/qwen3vl_8b_lora.yaml


python eval/inference/qwen2_hm_inference_vllm.py \
    --base_model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --model_path "checkpoints/fb/qwen3_vl_8b" \
    --processor_path "Qwen/Qwen3-VL-8B-Instruct" \
    --data_split "dev_seen test_seen" \
    --dataset "FB" \
    --batch_size 1000 \
    --log_name "qwen3vl_8b_lora_sft_warmup_fg_choice"
