"""
LLM-as-Judge Evaluation for Reasoning Comparison
Uses LLM to evaluate the quality of model-generated reasoning compared to ground truth.
Supports both OpenAI API and vLLM server endpoints.
"""

import json
import argparse
import os
import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
import re
from dataclasses import dataclass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not available. Install with: pip install openai")

try:
    from .config import (
        LLM_JUDGE_PROMPT,
        OPENAI_CONFIG,
        VLLM_CONFIG,
        DEFAULT_OPENAI_BASE_URL,
        DEFAULT_VLLM_BASE_URL,
        EVALUATION_SETTINGS,
    )
except ImportError:
    # Support direct script execution: `python eval/judge_reasoning/llm_judge_eval.py`
    from config import (  # type: ignore
        LLM_JUDGE_PROMPT,
        OPENAI_CONFIG,
        VLLM_CONFIG,
        DEFAULT_OPENAI_BASE_URL,
        DEFAULT_VLLM_BASE_URL,
        EVALUATION_SETTINGS,
    )


@dataclass 
class LLMResponse:
    """Response from LLM evaluation."""
    score: float
    explanation: str
    raw_response: str
    success: bool = True
    error_msg: str = ""


class LLMJudgeEvaluator:
    """Evaluator using LLM as a judge for reasoning comparison."""
    
    def __init__(self, server_type: str = "openai", base_url: str = None,
                 api_key: str = None, model_name: str = None,):
        """
        Initialize the LLM Judge evaluator.
        
        Args:
            server_type: Either "openai" or "vllm"
            base_url: Base URL for the API server
            api_key: API key (required for OpenAI, optional for vLLM)
            model_name: Model name to use

        """
        self.server_type = server_type.lower()
        missing = []
        if not NUMPY_AVAILABLE:
            missing.append("numpy")
        if not REQUESTS_AVAILABLE:
            missing.append("requests")
        if not TQDM_AVAILABLE:
            missing.append("tqdm")
        if missing:
            raise ImportError(
                "Missing dependencies for LLM-judge evaluation: "
                + ", ".join(missing)
                + ". Install with: pip install -r eval/judge_reasoning/requirements.txt"
            )

        if self.server_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required for OpenAI server. Install with: pip install openai")
            
            self.base_url = base_url or DEFAULT_OPENAI_BASE_URL
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key.")

            self.model_name = model_name if model_name else OPENAI_CONFIG["model"]
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            
        elif self.server_type == "vllm":
            self.base_url = base_url or DEFAULT_VLLM_BASE_URL
            self.api_key = api_key or "dummy_key"  # vLLM might not require real key
            self.model_name = model_name if model_name else VLLM_CONFIG["model"]
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            
        else:
            raise ValueError("server_type must be either 'openai' or 'vllm'")
        
        # Setup retry strategy for requests
        self.session = requests.Session()
        retry_strategy = Retry(
            total=EVALUATION_SETTINGS["max_retries"],
            backoff_factor=EVALUATION_SETTINGS["retry_delay"],
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        print(f"Initialized LLM Judge with {self.server_type} server")
        print(f"Base URL: {self.base_url}")
        print(f"Model: {self.model_name}")
    
    def load_model_predictions(
        self, model_file_path: str
    ) -> Tuple[Dict[str, str], Dict[str, Optional[int]], Dict[str, Optional[bool]]]:
        """Load reasoning predictions, label predictions, and correctness flags from JSON file."""
        with open(model_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions = {}
        text_predictions = {}
        correct = {}
        for item in data:
            image_id = item['image_id']
            reasoning = item.get('thinking', '').strip()
            predictions[image_id] = reasoning
            text_predictions[image_id] = item.get('text_prediction')
            correct[image_id] = item.get('correct')

        return predictions, text_predictions, correct

    def load_ground_truth(self, gt_file_path: str) -> Dict[str, str]:
        """Load ground truth reasoning from JSONL file."""
        ground_truth = {}
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                img_filename = item['img']
                image_id = img_filename.replace('.png', '')
                
                reasonings = item.get('reasonings', [])
                if reasonings:
                    reasoning = ' '.join(reasonings).strip()
                else:
                    reasoning = ''
                    
                ground_truth[image_id] = reasoning
        
        print(f"Loaded {len(ground_truth)} ground truth entries from {gt_file_path}")
        return ground_truth

    def create_evaluation_prompt(self, image_id: str, reference: str, candidate: str, prediction: str) -> str:
        """Create evaluation prompt for LLM judge."""
        prompt_template = LLM_JUDGE_PROMPT
        
        return prompt_template.format(
            image_id=image_id,
            reference_reasoning=reference,
            model_reasoning=candidate,
            model_prediction=prediction
            
        )
    
    def parse_llm_response(self, response_text: str) -> LLMResponse:
        """
        Parse LLM response to extract score and explanation.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            LLMResponse object with parsed score and explanation
        """
        try:
            # Look for score in various formats
            score_patterns = [
                r"Score:\s*([0-9]{1,2}(?:\.[0-9]+)?)",
                r"score:\s*([0-9]{1,2}(?:\.[0-9]+)?)",
                r"Score\s*=\s*([0-9]{1,2}(?:\.[0-9]+)?)",
                r"([0-9]{1,2}(?:\.[0-9]+)?)/10",
                r"Rating:\s*([0-9]{1,2}(?:\.[0-9]+)?)",
            ]
            
            score = None
            for pattern in score_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    break
            
            if score is None:
                # Try to find any number between 0-10
                numbers = re.findall(r'\b([0-9]{1,2}(?:\.[0-9]+)?)\b', response_text)
                valid_scores = [float(n) for n in numbers if 0 <= float(n) <= 10]
                if valid_scores:
                    score = valid_scores[0]
            
            if score is None or not (0 <= score <= 10):
                return LLMResponse(
                    score=0.0,
                    explanation="Failed to parse score",
                    raw_response=response_text,
                    success=False,
                    error_msg="Could not extract valid score (0-10) from response"
                )
            
            # Extract explanation
            explanation_patterns = [
                r"Explanation:\s*(.+?)(?:\n|$)",
                r"explanation:\s*(.+?)(?:\n|$)", 
                r"Justification:\s*(.+?)(?:\n|$)",
            ]
            
            explanation = ""
            for pattern in explanation_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    explanation = match.group(1).strip()
                    break
            
            if not explanation:
                # Use the full response as explanation if no specific pattern found
                explanation = response_text.strip()
            
            return LLMResponse(
                score=score,
                explanation=explanation,
                raw_response=response_text,
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                score=0.0,
                explanation="Error parsing response",
                raw_response=response_text,
                success=False,
                error_msg=str(e)
            )
    
    async def evaluate_single_pair_async(self, image_id: str, reference: str, 
                                       candidate: str, raw_prediction_binary: int) -> LLMResponse:
        """Evaluate a single reference-candidate pair asynchronously."""
        try:
            
            
            if raw_prediction_binary == 1:
                pred = "hateful"
            else:
                pred = "not hateful"
                
            prompt = self.create_evaluation_prompt(image_id, reference, candidate, pred)
            if self.server_type == "openai":
                config = OPENAI_CONFIG
            else:
                config = VLLM_CONFIG
            if "gpt-5" in self.model_name:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    top_p=config.get("top_p", 1.0)
                )
            
            response_text = response.choices[0].message.content
            return self.parse_llm_response(response_text)
            
        except Exception as e:
            return LLMResponse(
                score=0.0,
                explanation="API call failed",
                raw_response="",
                success=False,
                error_msg=str(e)
            )
    
    async def evaluate_batch_async(
        self, batch_data: List[Tuple[str, str, str, Optional[int]]]
    ) -> List[LLMResponse]:
        """Evaluate a batch of samples asynchronously."""
        tasks = []
        for image_id, reference, candidate, raw_prediction_binary in batch_data:
            task = self.evaluate_single_pair_async(image_id, reference, candidate, raw_prediction_binary)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(LLMResponse(
                    score=0.0,
                    explanation="Exception occurred",
                    raw_response="",
                    success=False,
                    error_msg=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def evaluate_all(self, 
                    model_predictions: Dict[str, str], 
                    ground_truth: Dict[str, str], 
                    text_predictions: Dict[str, int], 
                    correct: Dict[str, bool]) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Evaluate all prediction-reference pairs using LLM judge.
        
        Args:
            model_predictions: Dictionary of image_id -> model reasoning
            ground_truth: Dictionary of image_id -> ground truth reasoning
            
        Returns:
            Tuple of (detailed_results, average_scores)
        """
        # Find common image IDs
        common_ids = set(model_predictions.keys()) & set(ground_truth.keys())
        
        if not common_ids:
            raise ValueError("No common image IDs found between predictions and ground truth")
        
        common_ids = sorted(list(common_ids))
        print(f"Evaluating {len(common_ids)} common samples")
        
        # Prepare batch data
        all_data = []
        for image_id in common_ids:
            reference = ground_truth[image_id]
            candidate = model_predictions[image_id]
            raw_prediction_binary = text_predictions.get(image_id, None)
            all_data.append((image_id, reference, candidate, raw_prediction_binary))
        
        # Process in batches
        batch_size = EVALUATION_SETTINGS["batch_size"]
        # Set the batch size to 1000 for vllm requests
        if self.server_type == "vllm":
            batch_size = 1000
        all_responses = []
        
        # Add progress bar over batches
        num_batches = (len(all_data) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(all_data), batch_size), total=num_batches, desc="LLM Judge Batches", unit="batch"):
            batch_data = all_data[i:i + batch_size]

            # Run async evaluation for this batch
            batch_responses = asyncio.run(self.evaluate_batch_async(batch_data))
            all_responses.extend(batch_responses)

            # Small delay to avoid overwhelming the server (skip on last batch)
            if i + batch_size < len(all_data):
                time.sleep(0.05)
        
        # Create detailed results
        detailed_results = []
        scores = []
        successful_evaluations = 0

        for i, (image_id, reference, candidate, prediction) in enumerate(all_data):
            response = all_responses[i]
            
            detailed_results.append({
                'image_id': image_id,
                'reference': reference,
                'candidate': candidate,
                'text_prediction': prediction,
                'correct': correct.get(image_id, False),
                'llm_score': response.score,
                'explanation': response.explanation,
                'raw_response': response.raw_response,
                'success': response.success,
                'error_msg': response.error_msg
            })
            
            if response.success:
                scores.append(response.score)
                successful_evaluations += 1
        
        # Calculate statistics
        if scores:
            average_scores = {
                'llm_judge_score': np.mean(scores),
                'llm_judge_score_std': np.std(scores),
                'success_rate': successful_evaluations / len(all_responses),
                'total_samples': len(all_responses),
                'successful_evaluations': successful_evaluations
            }
        else:
            average_scores = {
                'llm_judge_score': 0.0,
                'llm_judge_score_std': 0.0,
                'success_rate': 0.0,
                'total_samples': len(all_responses),
                'successful_evaluations': 0
            }
        
        return detailed_results, average_scores
    
    def save_results(self, detailed_results: List[Dict], average_scores: Dict[str, float], 
                    output_path: str):
        """Save evaluation results to file."""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        
        results = {
            'summary': average_scores,
            'server_type': self.server_type,
            'model_name': self.model_name,
            'base_url': self.base_url,
            'detailed_results': detailed_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")


def main():
    """Main function to run LLM judge evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate reasoning with LLM as judge')
    parser.add_argument('--model_file', required=True,
                       help='Path to model predictions JSON file')
    parser.add_argument('--gt_file', required=True,
                       help='Path to ground truth JSONL file')
    parser.add_argument('--output_file', required=True,
                       help='Path to save evaluation results')
    parser.add_argument('--server_type', default='openai', choices=['openai', 'vllm'],
                       help='Type of server to use (openai or vllm)')
    parser.add_argument('--base_url', default=None,
                       help='Base URL for API server')
    parser.add_argument('--api_key', default=None,
                       help='API key (required for OpenAI)')
    parser.add_argument('--model_name', default=None,
                       help='Model name to use')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Model file not found: {args.model_file}")
    if not os.path.exists(args.gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {args.gt_file}")
    
    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(
        server_type=args.server_type,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
    )
    
    # Load data
    model_predictions, text_predictions, correct = evaluator.load_model_predictions(args.model_file)
    ground_truth = evaluator.load_ground_truth(args.gt_file)
    
    # Run evaluation
    detailed_results, average_scores = evaluator.evaluate_all(model_predictions, ground_truth, text_predictions, correct)
    
    # Print summary
    print("\n=== LLM Judge Evaluation Results ===")
    print(f"Average Score: {average_scores['llm_judge_score']:.4f} ± {average_scores['llm_judge_score_std']:.4f}")
    print(f"Success Rate: {average_scores['success_rate']:.2%}")
    print(f"Successful Evaluations: {average_scores['successful_evaluations']}/{average_scores['total_samples']}")
    
    # Save results
    evaluator.save_results(detailed_results, average_scores, args.output_file)


if __name__ == "__main__":
    main()
