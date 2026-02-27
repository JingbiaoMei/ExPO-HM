#!/usr/bin/env python3
"""
Fine-Grained Hateful Memes Multi-Class Classification Inference using vLLM backend.

This script performs fine-grained multi-class inference on hateful memes datasets using vLLM's offline 
inference engine for maximum throughput and speed compared to the HuggingFace sequential generation approach.

Features:
- Uses vLLM batch processing for faster inference
- Supports fine-grained datasets (FB-fine-grained-PC, FB-fine-grained-attack)
- Multi-class evaluation with macro/micro/weighted F1 metrics
- Evaluates on multiple data splits
- Logs results to wandb and CSV files
- Supports LoRA adapters
- Multimodal support for Qwen2.5-VL models

Usage:
    python eval/inference/qwen2_hm_inference_vllm_fine_grained.py --model_path "Qwen/Qwen2.5-VL-7B-Instruct" \
                                                                 --processor_path "Qwen/Qwen2.5-VL-7B-Instruct" \
                                                                 --data_split "dev_seen test_seen" \
                                                                 --dataset "FB-fine-grained-PC" \
                                                                 --batch_size 32
"""

# Set tokenizers parallelism before importing any HuggingFace libraries
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import re
import json
import gc
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb
import argparse
import functools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

from transformers import AutoTokenizer
from dataset import get_Dataloader
import csv
import math
import gc

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def exact_match_reward_fine_grained(raw_output, solution, all_classes, benign_class_name, **kwargs):
    """Reward function for fine-grained classification"""
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    
    reward = 0.0

    # Extract answer from solution if it has think/answer tags 
    sol_match = re.search(answer_tag_pattern, solution, re.DOTALL)
    ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()
    
    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL)
    student_answer = content_match.group(1).strip() if content_match else raw_output.strip()

    # Parse ground truth labels (assuming it's a string like "dehumanizing" or "pc_empty")
    if isinstance(ground_truth, str):
        gt_labels = [ground_truth]
    else:
        gt_labels = ground_truth

    # Extract predicted labels from student answer
    predicted_labels = []
    if "benign" in student_answer.lower():
        predicted_labels = [benign_class_name]
    else:
        found_labels = []
        answer_lower = student_answer.lower()
        for cls in all_classes:
            if cls.lower() in answer_lower:
                found_labels.append(cls)
        if len(found_labels) == 0:
            predicted_labels = [benign_class_name]
        else:
            predicted_labels = found_labels

    # Calculate reward based on exact match of label sets
    if set(gt_labels) == set(predicted_labels):
        reward = 1.0
    
    return reward


def write_to_csv_fine_grained(log_path, metrics):
    """Write fine-grained evaluation results to a CSV file."""
    with open(log_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["split", "accuracy", "micro_f1", "macro_f1", "weighted_f1", "reward", "invalid"])
        for split, values in metrics.items():
            values = [round(value, 4) for value in values]
            writer.writerow([split] + values)


def evaluate_predictions_fine_grained(y_true, y_pred, all_classes, rewards=None, invalid=None):
    """Calculate fine-grained evaluation metrics."""
    # Convert to binary format for multi-label evaluation
    mlb = MultiLabelBinarizer(classes=all_classes)
    binary_true = mlb.fit_transform(y_true)
    binary_pred = mlb.fit_transform(y_pred)
    
    metrics = []
    metrics.append(accuracy_score(binary_true, binary_pred))  # accuracy
    metrics.append(f1_score(binary_true, binary_pred, average='micro'))  # micro_f1
    metrics.append(f1_score(binary_true, binary_pred, average='macro'))  # macro_f1
    metrics.append(f1_score(binary_true, binary_pred, average='weighted'))  # weighted_f1
    
    if rewards is not None:
        metrics.append(np.mean(rewards))
    else:
        metrics.append(0.0)
        
    if invalid is not None:
        metrics.append(np.mean(invalid))
    else:
        metrics.append(0.0)
    
    # Also return the classification report
    report = classification_report(
        binary_true, 
        binary_pred, 
        target_names=all_classes,
        output_dict=True,
        zero_division=0
    )
    
    return metrics, report


class VLLMInferenceEngineFineGrained:
    """vLLM-based inference engine for fine-grained hateful meme classification."""
    
    def __init__(self, 
                 model_path: str = None,
                 base_model_path: str = None,
                 processor_path: str = None,
                 adapter_name_or_path: str = None,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 data_parallel_size: int = None,
                 max_model_len: int = 8192,
                 max_pixels: int = None,
                 dtype: str = "bfloat16",
                 trust_remote_code: bool = True,
                 **vllm_kwargs):
        """
        Initialize the vLLM inference engine for fine-grained classification.
        
        Args:
            model_path: Path to the model or model name
            base_model_path: Path to base model (for LoRA adapters)
            processor_path: Path to processor (for tokenizer)
            adapter_name_or_path: Path to LoRA adapter
            tensor_parallel_size: Number of GPUs for tensor parallelism
            pipeline_parallel_size: Number of GPUs for pipeline parallelism
            max_model_len: Maximum model context length
            max_pixels: Maximum pixels for image processing
            dtype: Model data type
            trust_remote_code: Whether to trust remote code
            **vllm_kwargs: Additional vLLM engine arguments
        """
        if not HAS_VLLM:
            raise ImportError("vLLM is required for this script. Please install it with: pip install vllm")
        
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.processor_path = processor_path or model_path
        self.adapter_name_or_path = adapter_name_or_path
        self.max_pixels = max_pixels
        
        # Default to using all available GPUs for DP if not specified, as DP results in higher throughput
        if data_parallel_size is None:
            data_parallel_size = torch.cuda.device_count()
        logger.info(f"Initializing vLLM engine for fine-grained classification...")
        logger.info(f"Model: {model_path}")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Adapter: {adapter_name_or_path}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"Pipeline parallel size: {pipeline_parallel_size}")
        logger.info(f"Max model length: {max_model_len}")
        
        # Determine which model to load
        if base_model_path and adapter_name_or_path:
            # Use base model for LoRA
            model_name_or_path = base_model_path
            enable_lora = True
        else:
            # Use the main model path
            model_name_or_path = model_path
            enable_lora = False
        print(model_name_or_path)
        
        # Initialize vLLM engine
        engine_args = {
            "model": model_name_or_path,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "disable_log_stats": True,
            "enable_lora": enable_lora,
            "limit_mm_per_prompt": {"image": 1, "video": 1, "audio": 1},  # Each request has only 1 image
            "max_lora_rank": 128,
            **vllm_kwargs
        }
        
        self.llm = LLM(**engine_args)
        
        # Initialize tokenizer for chat template processing
        logger.info("Loading tokenizer for chat template processing...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.processor_path, 
            trust_remote_code=trust_remote_code
        )
        
        # Initialize sampling parameters for greedy decoding (like the original)
        self.sampling_params = SamplingParams(
            repetition_penalty=1.0,
            temperature=0.0,  # Greedy decoding
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
            skip_special_tokens=True,
        )
        
        # Initialize LoRA request if adapter is provided
        self.lora_request = None
        if adapter_name_or_path:
            # Handle multiple LoRA adapters separated by ":"
            if ":" in adapter_name_or_path:
                # For now, use the first adapter - vLLM might not support multiple LoRAs simultaneously
                first_adapter = adapter_name_or_path.split(":")[0]
                logger.warning(f"Multiple LoRA adapters detected. Using only the first one: {first_adapter}")
                self.lora_request = LoRARequest("adapter_0", 1, first_adapter)
            else:
                self.lora_request = LoRARequest("adapter_0", 1, adapter_name_or_path)
        
        logger.info("vLLM engine initialized successfully")
    
    def _prepare_conversation_format(self, prompt_text: str, image: Image.Image) -> str:
        """
        Prepare conversation format for vLLM with Qwen2.5-VL using chat template.
        This matches exactly the working implementation in generate_dpo_data_vllm.py
        
        Args:
            prompt_text: The text prompt
            image: PIL Image object
            
        Returns:
            Formatted prompt string ready for vLLM
        """
        # Remove the <image> token from the prompt template since we'll handle it through messages
        text_prompt = prompt_text.replace("<image>", "").strip()
        
        if image is not None:
            # Create conversation messages with image and text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        else:
            # Text-only conversation
            messages = [
                {
                    "role": "user",
                    "content": text_prompt
                }
            ]
        
        # Apply chat template - this is crucial for vLLM multimodal support
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return formatted_prompt
    
    def _prepare_multimodal_data(self, image: Image.Image) -> dict:
        """
        Prepare multimodal data for vLLM input.
        This matches exactly the working implementation in generate_dpo_data_vllm.py
        """
        if image is None:
            return None
        
        # For vLLM with Qwen2.5-VL, we need to prepare the image in the expected format
        return {
            "image": image  # vLLM expects a single PIL image for Qwen2.5-VL
        }
    
    def load_image_from_dataloader(self, img_tensor):
        """
        Convert dataloader image tensor or path to PIL Image.
        """
        try:
            # Handle string paths (this is what we're getting from the dataset)
            if isinstance(img_tensor, str):
                if os.path.exists(img_tensor):
                    return Image.open(img_tensor).convert('RGB')
                else:
                    logging.error(f"Image path does not exist: {img_tensor}")
                    return None
            # Convert tensor image to PIL if necessary
            elif torch.is_tensor(img_tensor):
                # Handle different tensor formats
                if img_tensor.dim() == 3:  # CHW format
                    img_tensor = img_tensor.permute(1, 2, 0)  # Convert to HWC
                elif img_tensor.dim() == 4:  # BCHW format - take first image
                    img_tensor = img_tensor[0].permute(1, 2, 0)  # Convert to HWC
                
                img_np = img_tensor.cpu().numpy()
                
                # Normalize to 0-255 if needed
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:  # Normalized to [0,1]
                        img_np = (img_np * 255).astype(np.uint8)
                    else:  # Already in [0,255] range
                        img_np = img_np.astype(np.uint8)
                
                img = Image.fromarray(img_np)
                return img
            elif isinstance(img_tensor, Image.Image):
                return img_tensor
            else:
                logging.error(f"Unsupported image type: {type(img_tensor)}")
                return None
        except Exception as e:
            logging.error(f"Error converting image: {e}")
            return None
    
    def prepare_all_requests(self, dataloader, query: str):
        """
        Prepare all requests upfront for batch processing.
        Uses the exact same approach as the working generate_dpo_data_vllm.py script.
        
        Args:
            dataloader: DataLoader for the data
            query: Query template to use
            
        Returns:
            Tuple of (all_requests, request_metadata)
        """
        all_requests = []
        request_metadata = []
        
        question = query
        
        logging.info("Preparing all requests for batch processing...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Preparing requests")):
            # Since we're using batch_size=1 in DataLoader, each batch contains 1 element
            # Unpack the single-element batch
            images, texts, labels, image_ids = batch
            
            # Extract the single elements from the batch
            img_tensor = images[0] if isinstance(images, (list, tuple)) else images
            text = texts[0] if isinstance(texts, (list, tuple)) else texts
            img_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else image_ids
            label = labels[0] if isinstance(labels, (list, tuple)) else labels
            
            try:
                formatted_question = question.format(text=text)
                
                # Convert dataloader tensor to PIL Image using the new method
                image = self.load_image_from_dataloader(img_tensor)
                if image is None:
                    logging.error(f"Failed to convert image tensor for {img_id}")
                    continue
                
                # Use the working chat template approach from generate_dpo_data_vllm.py
                formatted_prompt = self._prepare_conversation_format(formatted_question, image)
                
                # Prepare multimodal data using the working approach
                multi_modal_data = self._prepare_multimodal_data(image)
                
                # Create vLLM input format exactly like the working script
                if multi_modal_data is not None:
                    # For multimodal inputs, use the formatted prompt with multimodal data
                    vllm_input = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": multi_modal_data
                    }
                else:
                    # For text-only inputs, just use the formatted prompt
                    vllm_input = {
                        "prompt": formatted_prompt
                    }
                
                all_requests.append(vllm_input)
                request_metadata.append({
                    "image_id": img_id,
                    "text": text,
                    "label": label if isinstance(label, str) else str(label.item() if hasattr(label, 'item') else label),
                    "batch_idx": batch_idx,
                    "idx_in_batch": 0  # Always 0 since batch_size=1
                })
                
            except Exception as e:
                logging.error(f"Error preparing request for image {img_id}: {e}")
                continue
        
        logging.info(f"Prepared {len(all_requests)} requests for vLLM batch processing")
        return all_requests, request_metadata
    
    def process_requests_in_batches(self, all_requests, request_metadata, batch_size, all_classes, benign_class_name):
        """
        Process all requests in batches using vLLM for fine-grained classification.
        Uses the exact same approach as the working generate_dpo_data_vllm.py script.
        
        Args:
            all_requests: List of vLLM input requests
            request_metadata: List of metadata for each request
            batch_size: Batch size for processing
            all_classes: List of all possible classes
            benign_class_name: Name of the benign/empty class
            
        Returns:
            Tuple of (batch_results, batch_pred_labels, batch_true_labels, batch_rewards, batch_ids, batch_invalid)
        """
        all_results = []
        all_pred_labels = []
        all_true_labels = []
        all_rewards = []
        all_ids = []
        all_invalid = []

        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(all_requests), batch_size), desc="Processing vLLM batches"):
            batch_requests = all_requests[i:i + batch_size]
            batch_metadata = request_metadata[i:i + batch_size]
            
            try:
                # Generate responses for this batch - this is where the true batching happens!
                batch_results = self.llm.generate(batch_requests, self.sampling_params, lora_request=self.lora_request)
 
                # Extract generated text and process responses
                for result, metadata in zip(batch_results, batch_metadata):
                    invalid = False
                    try:
                        response = result.outputs[0].text
                        
                        # Process the response
                        thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
                        thinking = thinking_match.group(1).strip() if thinking_match else ""
                        
                        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                        if not answer_match:
                            logger.warning(f"No answer tag found in response for image {metadata['image_id']}, setting to benign by default")
                            answer = benign_class_name
                            invalid = True
                        else:
                            answer = answer_match.group(1).strip()
                        
                        # Handle ground truth labels - properly parse from dataset format
                        raw_label = metadata['label']
                        if isinstance(raw_label, str):
                            # Handle string representations of lists like "['pc_empty']" or "['sex']"
                            if raw_label.startswith('[') and raw_label.endswith(']'):
                                try:
                                    # Safely evaluate the string representation of a list
                                    import ast
                                    gt_labels = ast.literal_eval(raw_label)
                                    if not isinstance(gt_labels, list):
                                        gt_labels = [gt_labels]
                                except (ValueError, SyntaxError):
                                    # If parsing fails, treat as single label
                                    gt_labels = [raw_label.strip("[]'\"")]
                            else:
                                # Regular string label
                                gt_labels = [raw_label]
                        elif isinstance(raw_label, list):
                            gt_labels = raw_label
                        else:
                            gt_labels = [str(raw_label)]
                        
                        # Extract predicted labels from response
                        predicted_labels = []
                        if "benign" in answer.lower() or "neutral" in answer.lower():
                            predicted_labels = [benign_class_name]
                        else:
                            found_labels = []
                            answer_lower = answer.lower()
                            for cls in all_classes:
                                if cls.lower() in answer_lower:
                                    found_labels.append(cls)
                            if len(found_labels) == 0:
                                predicted_labels = [benign_class_name]
                                logger.info(f"No specific labels found in answer for image {metadata['image_id']}, assigning benign")
                            else:
                                predicted_labels = found_labels
                        
                        # Calculate reward
                        reward = exact_match_reward_fine_grained(response, gt_labels[0], all_classes, benign_class_name, **{})
                        
                        result_dict = {
                            'image_id': metadata['image_id'],
                            'true_labels': gt_labels,
                            'predicted_labels': predicted_labels,
                            'raw_output': response,
                            'thinking': thinking,
                            'text_answer': answer,
                            'correct': set(gt_labels) == set(predicted_labels),
                            'reward': reward,
                            'invalid': invalid
                        }
                        
                        all_pred_labels.append(predicted_labels)
                        all_true_labels.append(gt_labels)
                        all_results.append(result_dict)
                        all_rewards.append(reward)
                        all_ids.append(metadata['image_id'])
                        all_invalid.append(invalid)

                    except Exception as e:
                        logging.error(f"Error processing response for image {metadata['image_id']}: {e}")
                        continue
                
                # Clear memory
                del batch_results
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

        return all_results, all_pred_labels, all_true_labels, all_rewards, all_ids, all_invalid


def process_split_vllm_fine_grained(inference_engine, dataloader, split_name, args, all_classes, benign_class_name):
    """
    Process a complete data split using vLLM with true batch processing for fine-grained classification.
    
    Args:
        inference_engine: VLLMInferenceEngineFineGrained instance
        dataloader: DataLoader for the split
        split_name: Name of the split being processed
        args: Command line arguments
        all_classes: List of all possible classes
        benign_class_name: Name of the benign/empty class
        
    Returns:
        Dictionary with results and metrics
    """
    logging.info(f"Processing {split_name} split using vLLM for fine-grained classification...")
    
    # Step 1: Prepare all requests upfront (like the working sampling script)
    all_requests, request_metadata = inference_engine.prepare_all_requests(dataloader, args.query)
    
    if not all_requests:
        logging.warning(f"{split_name} - No valid requests were prepared")
        raise ValueError("No valid requests to process")
    
    # Step 2: Process all requests in true batches (this is where the performance gain happens!)
    all_results, all_pred_labels, all_true_labels, all_rewards, all_ids, all_invalid = inference_engine.process_requests_in_batches(
        all_requests, request_metadata, args.batch_size, all_classes, benign_class_name
    )
    
    # Calculate metrics if we have predictions
    if all_pred_labels:
        metrics, report = evaluate_predictions_fine_grained(all_true_labels, all_pred_labels, all_classes, all_rewards, all_invalid)
        logging.info(f"{split_name} - Metrics: Acc={metrics[0]:.4f}, Micro F1={metrics[1]:.4f}, Macro F1={metrics[2]:.4f}, Weighted F1={metrics[3]:.4f}, Reward={metrics[4]:.4f}, Invalid={metrics[5]:.4f}")
    else:
        metrics = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        report = {}
        logging.warning(f"{split_name} - No valid predictions were collected")
    
    return {
        'metrics': metrics,
        'results': all_results,
        'ids': all_ids,
        'true_labels': all_true_labels,
        'pred_labels': all_pred_labels,
        'rewards': all_rewards,
        'invalid': all_invalid,
        'report': report,
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-Grained Hateful Memes Classification with vLLM')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint or model name')
    parser.add_argument('--base_model_path', type=str, default=None, help='Path to base model checkpoint (for LoRA)')
    parser.add_argument('--processor_path', type=str, required=True, help='Path to processor/tokenizer')
    parser.add_argument('--adapter_name_or_path', type=str, default=None, help='Path to LoRA adapter')
    
    # Data arguments
    parser.add_argument('--data_split', type=str, default='dev_seen test_seen', help='Evaluate on which dataset split')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (smaller batches recommended for multimodal vLLM)')
    parser.add_argument('--dataset', type=str, default="FB-fine-grained-PC", help='Dataset name (FB-fine-grained-PC or FB-fine-grained-attack)')
    
    # Logging arguments
    parser.add_argument('--log_name', type=str, default="", help='Log name')
    parser.add_argument('--group_name', type=str, default="inference_classifier_fg_vllm", help='Group name for wandb')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true', default=False, help='Run in debug mode (process limited batches)')
    parser.add_argument('--debug_batches', type=int, default=10, help='Number of batches to process in debug mode')
    
    # Query argument
    parser.add_argument('--query', type=str, 
                        default="Please identify the specific type of harmful content in this meme.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.\nFor the thinking process, consider both the image and any accompanying text. Analyze the content for different types of harmful speech including protected categories attacks, dehumanizing language, inciting violence, etc. Consider the context and intent. If the content is not harmful, classify it as benign.\nThe answer should specify the exact type of harmful content found, or 'benign' if no harmful content is detected. The output format should be as follows:\n<think> ... </think> <answer> ... </answer>\nPlease strictly follow the format.", 
                        help='Query to pass to the model')
    
    # vLLM-specific arguments
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--pipeline_parallel_size', type=int, default=1, help='Number of GPUs for pipeline parallelism')
    parser.add_argument('--data_parallel_size', type=int, default=None, help='Number of GPUs for data parallelism')
    parser.add_argument('--max_model_len', type=int, default=8192, help='Maximum model context length')
    parser.add_argument('--max_pixels', type=int, default=None, help='Maximum pixels for image processing')
    parser.add_argument('--dtype', type=str, default="bfloat16", help='Model data type')
    parser.add_argument('--trust_remote_code', action='store_true', help='Whether to trust remote code')
    
    args = parser.parse_args()
    
    # Setup debug mode
    if args.debug:
        logger.info(f"Debug mode enabled: Will process {args.debug_batches} batches per split")
        args.log_name += "_debug"
        args.group_name = "debug_fg_vllm"
    
    # Determine fine-grained classes and benign class name based on dataset
    if args.dataset.lower() == "fb-fine-grained-pc":
        # Protected categories fine-grained classification
        # Note: Using actual labels from the dataset - "sex" instead of "sexual_orientation"
        all_classes = ["pc_empty", "nationality", "religion", "race", "sex", "disability"]
        benign_class_name = "pc_empty"
    elif args.dataset.lower() == "fb-fine-grained-attack":
        # Attack types fine-grained classification 
        # Note: Using actual labels from the dataset
        all_classes = ["attack_empty", "dehumanizing", "inferiority", "mocking", "inciting_violence", "exclusion", "contempt", "slurs"]
        benign_class_name = "attack_empty"
    
    elif args.dataset.lower() == "mami-fine-grained-attack":
        # MAMI Attack types fine-grained classification 
        all_classes = ["attack_empty", "objectification", "shaming", "stereotype", "violence"]
        benign_class_name = "attack_empty"
    elif args.dataset.lower() == "pridemm-fine-grained-target":
        # PrideMM fine-grained target classification 
        all_classes = ["benign", "undirected", "individual", "community", "organization"]
        benign_class_name = "benign"
    elif args.dataset.lower() == "pridemm-fine-grained-stance":
        # PrideMM fine-grained stance classification 
        all_classes = ["neutral", "support", "oppose"]
        benign_class_name = "neutral"
    else:
        raise ValueError(f"Unsupported dataset for fine-grained evaluation: {args.dataset}")
    
    logger.info(f"Fine-grained classes: {all_classes}")
    logger.info(f"Benign class: {benign_class_name}")
    
    # Setup logging paths
    model_name = os.path.basename(args.model_path) if args.model_path else os.path.basename(args.base_model_path)
    log_path = f"./logging/{args.dataset}/{args.log_name}/"
    os.makedirs(log_path, exist_ok=True)
    log_path += args.log_name + ".csv"
    
    # Initialize wandb
    exp_name = f"{args.log_name}_{args.dataset}"
    tags = [args.dataset, model_name, "vllm", "fine-grained"]
    if args.debug:
        tags.append("debug")
        
    run = wandb.init(
        name=exp_name,
        project="RFT-Inference-FG-vLLM",
        config={
            "model": model_name,
            "dataset": args.dataset,
            "debug": args.debug,
            "backend": "vllm",
            "batch_size": args.batch_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "all_classes": all_classes,
            "benign_class": benign_class_name
        },
        group=args.group_name,
        tags=tags
    )
    
    # Initialize vLLM inference engine
    logger.info("Initializing vLLM inference engine for fine-grained classification...")
    
    # Handle LoRA adapters
    adapter_path = None
    if args.base_model_path and args.model_path != args.base_model_path:
        adapter_path = args.model_path
    elif args.adapter_name_or_path:
        adapter_path = args.adapter_name_or_path
    
    inference_engine = VLLMInferenceEngineFineGrained(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        processor_path=args.processor_path,
        adapter_name_or_path=adapter_path,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        max_model_len=args.max_model_len,
        max_pixels=args.max_pixels,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code
    )
    
    # Define splits for fine-grained datasets
    if "fb" in args.dataset.lower():
        splits = ["train", "dev_seen", "dev_unseen"]
    else:
        splits = ["train", "dev", "test"]

    # Get dataloaders
    logger.info("Loading dataloaders...")
    # Use batch_size=1 for DataLoader to avoid collation issues with varying sized elements
    # We'll do true batching in vLLM instead
    train, dev_seen, test_seen = get_Dataloader(
        preprocess=None,
        batch_size=1,  # Use batch_size=1 to avoid PyTorch collation issues
        train_batch_size=1,  # Use batch_size=1 to avoid PyTorch collation issues  
        num_workers=0,  # Use 0 workers to avoid multiprocessing issues with vLLM
        dataset=args.dataset,
    )
    loader_list = [train, dev_seen, test_seen]
    
    # Run inference on specified splits
    metrics_dict = {}
    all_results = {}
    
    for split in splits:
        if split not in args.data_split:
            continue
            
        logger.info(f"Running inference on {split} split")
        split_index = splits.index(split)
        dataloader = loader_list[split_index]
        
        # Process the split
        split_result = process_split_vllm_fine_grained(inference_engine, dataloader, split, args, all_classes, benign_class_name)
        
        # Store metrics and results
        metrics_dict[split] = split_result['metrics']
        all_results[split] = split_result['results']

        # Create logging table for wandb
        logging_columns = ["image_id", "gt_labels", "pred_labels", "raw_output", "thinking", "text_answer", "correct", "reward", "invalid"]
        logging_table = wandb.Table(columns=logging_columns)
        
        # Fill the table with results
        for result in split_result['results']:
            logging_table.add_data(
                result['image_id'],
                ", ".join(result['true_labels']),
                ", ".join(result['predicted_labels']),
                result['raw_output'],
                result['thinking'],
                result['text_answer'],
                result['correct'],
                result['reward'],
                result['invalid']
            )
        
        # Create metrics table for wandb
        metrics_columns = ["accuracy", "micro_f1", "macro_f1", "weighted_f1", "reward", "invalid"]
        metrics_table = wandb.Table(columns=metrics_columns)
        metrics_table.add_data(*split_result['metrics'])
        
        # Create class-wise metrics table
        class_metrics_columns = ["class", "precision", "recall", "f1", "support"]
        class_metrics_table = wandb.Table(columns=class_metrics_columns)
        
        report = split_result['report']
        if report:
            for cls in all_classes:
                if cls in report:
                    cls_report = report[cls]
                    class_metrics_table.add_data(
                        cls,
                        cls_report['precision'],
                        cls_report['recall'],
                        cls_report['f1-score'],
                        int(cls_report['support'])
                    )
        
        # Log tables to wandb
        wandb.log({
            f"prediction_table_{split}": logging_table,
            f"metrics_table_{split}": metrics_table,
            f"class_metrics_table_{split}": class_metrics_table
        })
        
        # Log metrics
        wandb.log({
            f"{split}/accuracy": split_result['metrics'][0],
            f"{split}/micro_f1": split_result['metrics'][1],
            f"{split}/macro_f1": split_result['metrics'][2],
            f"{split}/weighted_f1": split_result['metrics'][3],
            f"{split}/reward": split_result['metrics'][4],
            f"{split}/invalid": split_result['metrics'][5]
        })
        
        # Print results
        print(f"Model: {args.log_name}, Dataset: {args.dataset}, Backend: vLLM Fine-Grained")
        print(f"Metrics for {split} split: Accuracy: {split_result['metrics'][0]:.4f}, "
              f"Micro F1: {split_result['metrics'][1]:.4f}, Macro F1: {split_result['metrics'][2]:.4f}, "
              f"Weighted F1: {split_result['metrics'][3]:.4f}, Reward: {split_result['metrics'][4]:.4f}, "
              f"Invalid: {split_result['metrics'][5]:.4f}")
        
        # Print classification report
        if report:
            print(f"Classification Report for {split}:")
            for cls in all_classes:
                if cls in report:
                    cls_report = report[cls]
                    print(f"  {cls}: P={cls_report['precision']:.4f}, R={cls_report['recall']:.4f}, F1={cls_report['f1-score']:.4f}, Support={int(cls_report['support'])}")

    # Save results
    write_to_csv_fine_grained(log_path, metrics_dict)
    
    # Save detailed results for each split
    results_path = []
    for split, results in all_results.items():
        split_results_path = log_path.replace('.csv', f'_{split}_details.json')
        results_path.append(split_results_path)
        with open(split_results_path, 'w') as f:
            json.dump(results, f, indent=2)

    logger.info(f"Results saved to {log_path}")
    logger.info(f"Detailed results saved to {results_path}")
    
    run.finish()


if __name__ == "__main__":
    main()
