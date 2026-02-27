"""
Configuration file for LLM-based reasoning evaluation.
Contains prompts and scoring rubrics for comparing model-generated reasoning with ground truth.
"""
LLM_JUDGE_PROMPT = """Compare the model-generated reasoning with the reference human reasoning for this hateful meme.

Reference: {reference_reasoning}
Model: {model_reasoning}
Model Prediction: {model_prediction}

Rate how well the model reasoning aligns with the reference on a scale of 0-10:
- 9-10: Excellent alignment, captures all key points
- 7-8: Good alignment, captures most key points  
- 5-6: Satisfactory alignment, captures some key points
- 3-4: Poor alignment, misses many key points
- 1-2: Very poor alignment, minimal understanding
- 0: Completely wrong or unrelated

If the model prediction is not hateful, which is incorrect, the highest score you may assign is 2.

Explanation: [1-2 sentences]
Score: [0-10]
"""


# OpenAI API configuration
OPENAI_CONFIG = {
    "model": "gpt-4o-mini-2024-07-18",
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 1.0,
}

# vLLM server configuration
VLLM_CONFIG = {
    "model": "Qwen/Qwen3-8B",  # Can be overridden
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 1.0,
}

# Default server configurations
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"

# Evaluation settings
EVALUATION_SETTINGS = {
    "batch_size": 30,  # Number of samples to evaluate in parallel
    "max_retries": 3,   # Maximum number of retries for failed API calls
    "retry_delay": 1,   # Delay between retries in seconds
    "use_short_prompt": False,  # Whether to use the shorter prompt
}
