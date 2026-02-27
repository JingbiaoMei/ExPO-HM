# Judge Reasoning Evaluation

This package now keeps only **LLM-as-judge** evaluation.

It compares model-generated reasoning against reference human reasoning and outputs a score (0–10) plus a short explanation per example.
This is optional and separate from standard classification metrics (accuracy/F1).

## Install

```bash
pip install -r eval/judge_reasoning/requirements.txt
```

## Required Input Formats

### Model predictions JSON

Expected list entries:

```json
[
  {
    "image_id": "08291",
    "thinking": "Model reasoning text...",
    "text_prediction": 1,
    "correct": true
  }
]
```

### Ground-truth reasoning JSONL

Example line:

```json
{"id":1456,"img":"01456.png","target":["the jews"],"reasonings":["Human reasoning text..."]}
```

This repository does **not** ship any ground-truth reasoning annotations. Provide your own JSONL file in the above format via `--gt_file`.

## Usage

### LLM Judge (OpenAI)

```bash
export OPENAI_API_KEY="your-api-key"
python3 eval/judge_reasoning/llm_judge_eval.py \
  --model_file logging/FB/model_predictions.json \
  --gt_file /path/to/your_ground_truth_reasonings.jsonl \
  --output_file output/llm_judge_results.json \
  --server_type openai
```

### LLM Judge (vLLM)

```bash
python3 eval/judge_reasoning/llm_judge_eval.py \
  --model_file logging/FB/model_predictions.json \
  --gt_file /path/to/your_ground_truth_reasonings.jsonl \
  --output_file output/llm_judge_results.json \
  --server_type vllm \
  --base_url http://localhost:8000/v1 \
  --model_name Qwen/Qwen3-8B
```
