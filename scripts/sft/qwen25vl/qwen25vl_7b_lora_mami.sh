model_name=qwen25vl_7b
dataset_name=MAMI
export WANDB_PROJECT="LLAMAFACTORY_hateful_qwen25vl"

export WANDB_RUN_GROUP="Finetuning_${dataset_name}_${model_name}_sft"
export WANDB_NAME=${dataset_name}_${model_name}_lora_sft_warmup_fg_choice

llamafactory-cli train scripts/sft/qwen25vl/qwen25vl_7b_lora_mami.yaml

python eval/inference/qwen2_hm_inference_vllm.py \
    --base_model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --model_path "checkpoints/mami/qwen2_5vl_7b/sft" \
    --processor_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --data_split "val test" \
    --dataset "MAMI" \
    --batch_size 1000 \
    --log_name "qwen25vl_7b_lora_sft_warmup_fg_choice_mami"
