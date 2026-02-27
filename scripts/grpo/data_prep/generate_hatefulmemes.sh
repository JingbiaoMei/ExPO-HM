BASE_DIR=${BASE_DIR:-./data/gt}
IMAGE_DIR=${IMAGE_DIR:-./data/image}
OUTPUT_DIR=${OUTPUT_DIR:-./data/verl}

python3 scripts/grpo/data_prep/hatefulmemes.py --base_dir "$BASE_DIR" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR" --dataset_name FB --split "train.jsonl test_seen.jsonl"
python3 scripts/grpo/data_prep/hatefulmemes.py --base_dir "$BASE_DIR" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR" --dataset_name MAMI --split "train.jsonl test.jsonl" --max_pixels 401408
python3 scripts/grpo/data_prep/hatefulmemes.py --base_dir "$BASE_DIR" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR" --dataset_name PrideMM --split "train.jsonl test.jsonl" --max_pixels 401408
