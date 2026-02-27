model_name=qwen3vl_8b
dataset_name=MAMI
export WANDB_PROJECT="LLAMAFACTORY_hateful_qwen3vl"

export WANDB_RUN_GROUP="Finetuning_${dataset_name}_${model_name}_sft"

export WANDB_NAME=${dataset_name}_${model_name}_lora_sft_warmup_fg_choice
llamafactory-cli train scripts/sft/qwen3vl/qwen3vl_8b_lora_mami.yaml


python eval/inference/qwen2_hm_inference_vllm.py \
    --base_model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --model_path "checkpoints/mami/qwen3_vl_8b" \
    --processor_path "Qwen/Qwen3-VL-8B-Instruct" \
    --data_split "val test" \
    --dataset "MAMI" \
    --batch_size 1000 \
    --log_name "qwen3vl_8b_lora_sft_warmup_fg_choice_mami"
