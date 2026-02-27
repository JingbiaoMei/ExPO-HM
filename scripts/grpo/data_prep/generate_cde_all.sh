BASE_DIR=${BASE_DIR:-./data/gt}
IMAGE_DIR=${IMAGE_DIR:-./data/image}
OUTPUT_DIR=${OUTPUT_DIR:-./data/verl/CDE}

echo "[CDE] Generating binary CDE parquet..."
python3 scripts/grpo/data_prep/hatefulmemes.py --base_dir "$BASE_DIR" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR" --dataset_name FB --split "train.jsonl" --reward_type meme_cde
python3 scripts/grpo/data_prep/hatefulmemes.py --base_dir "$BASE_DIR" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR" --dataset_name MAMI --split "train.jsonl" --max_pixels 401408 --reward_type meme_cde
python3 scripts/grpo/data_prep/hatefulmemes.py --base_dir "$BASE_DIR" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR" --dataset_name PrideMM --split "train.jsonl" --max_pixels 401408 --reward_type meme_cde

echo "[CDE] Generating fine-grained CDE parquet..."
python3 scripts/grpo/data_prep/hatefulmemes.py --base_dir "$BASE_DIR" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR" --dataset_name fb-fine-grained-pc --split "train.json dev_seen.json" --reward_type meme_fg_cde
python3 scripts/grpo/data_prep/hatefulmemes.py --base_dir "$BASE_DIR" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR" --dataset_name fb-fine-grained-attack --split "train.json dev_seen.json" --reward_type meme_fg_cde
python3 scripts/grpo/data_prep/merge_parquest.py "$OUTPUT_DIR/fb-fine-grained-pc/train.parquet" "$OUTPUT_DIR/fb-fine-grained-attack/train.parquet" -o "$OUTPUT_DIR/fb-fine-grained-combined" --info

echo "[CDE] Done. Outputs are under: $OUTPUT_DIR"
