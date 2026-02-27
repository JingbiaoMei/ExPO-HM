model_name=qwen3vl_4b
dataset_name=PrideMM
export WANDB_PROJECT="LLAMAFACTORY_hateful_qwen3vl"

export WANDB_RUN_GROUP="Finetuning_${dataset_name}_${model_name}_sft"

export WANDB_NAME=${dataset_name}_${model_name}_lora_sft_warmup_fg_choice
llamafactory-cli train scripts/sft/qwen3vl/qwen3vl_4b_lora_pridemm.yaml


python eval/inference/qwen2_hm_inference_vllm.py \
    --base_model_path "Qwen/Qwen3-VL-4B-Instruct" \
    --model_path "checkpoints/pridemm/qwen3_vl_4b" \
    --processor_path "Qwen/Qwen3-VL-4B-Instruct" \
    --data_split "val test" \
    --dataset "PrideMM" \
    --batch_size 1000 \
    --log_name "qwen3vl_4b_lora_sft_warmup_fg_choice_pridemm"
