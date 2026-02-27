#!/usr/bin/env python3
"""
Conditional Decision Entropy (CDE) Algorithm for Reward Computation

This module provides a clean, abstract implementation of the Conditional Decision Entropy
algorithm for use in online RL training (e.g., GRPO). The CDE measures the uncertainty
in binary decisions given reasoning paths and can be used as a reward signal to encourage
both accuracy and appropriate confidence calibration.

Mathematical Formulation:
    CDE(x) = (1/K) * Σ_i H(d | r_i, x)
    
where:
    - x is the input example
    - r_i are K sampled reasoning paths  
    - H(d | r_i, x) is the binary entropy of decision d given reasoning path r_i
    - Binary entropy: H(d) = -[p(Yes)*log(p(Yes)) + p(No)*log(p(No))]

Key Properties:
    - Lower entropy indicates higher confidence in the decision
    - Higher entropy indicates uncertainty/poor calibration
    - Can be used as a reward to encourage well-calibrated confidence
    - Works with any binary classification task

Usage for Online RL:
    1. Generate multiple reasoning paths per example during training
    2. Extract decision logits/probabilities for each path
    3. Compute CDE using this module
    4. Use CDE as reward signal (e.g., negative CDE for confidence reward)
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class CDEComputer:
    """
    Conditional Decision Entropy computer for binary classification tasks.
    
    This class provides methods to compute CDE from logits or probabilities
    extracted from model generations during online RL training.
    """
    
    def __init__(
        self, 
        yes_tokens: Optional[List[str]] = None,
        no_tokens: Optional[List[str]] = None,
        epsilon: float = 1e-10,
        temperature: float = 1.0
    ):
        """
        Initialize the CDE computer.
        
        Args:
            yes_tokens: List of tokens representing positive class (default: common Yes variants)
            no_tokens: List of tokens representing negative class (default: common No variants)
            epsilon: Small value for numerical stability in log computations
            temperature: Temperature for softmax computation (default: 1.0)
        """
        # Default token mappings for common tokenizers
        self.yes_tokens = yes_tokens or [
            # Basic Yes tokens
            "ĠYes", "Yes", "yes", "Ġyes", "YES", "ĠYES", "▁Yes", "▁yes",
            # After closing tag variations
            ">Yes", ">yes", ">YES",
            # After punctuation variations  
            ",Yes", ",yes", ",YES",
            ":Yes", ":yes", ":YES", 
            ".Yes", ".yes", ".YES",
            # Boolean true variants
            ">true", ">True", ">TRUE", "true", "True", "TRUE", "Ġtrue", "Ġtrue", "ĠTrue", "ĠTRUE",
            ",true", ",True", ",TRUE",
            ":true", ":True", ":TRUE", 
            ".true", ".True", ".TRUE",
            # Additional variations with spaces and prefixes
            " Yes", " yes", " YES", " true", " True", " TRUE",
            "\nYes", "\nyes", "\nYES", "\ntrue", "\nTrue", "\nTRUE",
            # SentencePiece variants
            "▁true", "▁True", "▁TRUE"
        ]
        self.no_tokens = no_tokens or [
            # Basic No tokens
            "ĠNo", "No", "no", "Ġno", "NO", "ĠNO", "▁No", "▁no",
            # After closing tag variations
            ">No", ">no", ">NO",
            # After punctuation variations
            ",No", ",no", ",NO",
            ":No", ":no", ":NO",
            ".No", ".no", ".NO", 
            # Boolean false variants
            ">false", ">False", ">FALSE", "false", "False", "FALSE", "Ġfalse", "ĠFalse", "ĠFALSE",
            ",false", ",False", ",FALSE",
            ":false", ":False", ":FALSE",
            ".false", ".False", ".FALSE",
            # Additional variations with spaces and prefixes
            " No", " no", " NO", " false", " False", " FALSE",
            "\nNo", "\nno", "\nNO", "\nfalse", "\nFalse", "\nFALSE",
            # SentencePiece variants
            "▁false", "▁False", "▁FALSE"
        ]
        
        self.epsilon = epsilon
        self.temperature = temperature
    
    def compute_binary_entropy_from_logits(
        self, 
        yes_logit: float, 
        no_logit: float,
        return_probs: bool = False
    ) -> Union[float, Tuple[float, Tuple[float, float]]]:
        """
        Compute binary entropy from Yes/No logits.
        
        Args:
            yes_logit: Logit value for positive class
            no_logit: Logit value for negative class  
            return_probs: Whether to also return the probabilities
            
        Returns:
            Binary entropy value, optionally with (yes_prob, no_prob) tuple
        """
        # Apply temperature scaling and compute probabilities
        logits = torch.tensor([yes_logit, no_logit], dtype=torch.float32)
        probs = F.softmax(logits / self.temperature, dim=0)
        
        # Compute binary entropy: -[p(Yes)*log(p(Yes)) + p(No)*log(p(No))]
        entropy = -torch.sum(probs * torch.log(probs + self.epsilon))
        
        if return_probs:
            return entropy.item(), (probs[0].item(), probs[1].item())
        return entropy.item()
    
    def compute_binary_entropy_from_probs(
        self, 
        yes_prob: float, 
        no_prob: float
    ) -> float:
        """
        Compute binary entropy from Yes/No probabilities.
        
        Args:
            yes_prob: Probability for positive class
            no_prob: Probability for negative class
            
        Returns:
            Binary entropy value
        """
        # Normalize probabilities to ensure they sum to 1
        total_prob = yes_prob + no_prob
        if total_prob == 0:
            warnings.warn("Both probabilities are zero, returning maximum entropy")
            return math.log(2)  # Maximum entropy for binary case
        
        yes_prob_norm = yes_prob / total_prob
        no_prob_norm = no_prob / total_prob
        
        # Compute binary entropy with numerical stability
        entropy = 0.0
        if yes_prob_norm > self.epsilon:
            entropy -= yes_prob_norm * math.log(yes_prob_norm)
        if no_prob_norm > self.epsilon:
            entropy -= no_prob_norm * math.log(no_prob_norm)
            
        return entropy
    
    def extract_decision_logits_from_vocab_logits(
        self,
        vocab_logits: torch.Tensor,
        tokenizer,
        aggregate_method: str = "max"
    ) -> Optional[Tuple[float, float]]:
        """
        Extract Yes/No logits from full vocabulary logits.
        
        Args:
            vocab_logits: Tensor of shape [vocab_size] with logits for each token
            tokenizer: Tokenizer to convert token IDs to strings
            aggregate_method: How to aggregate multiple Yes/No tokens ("max", "sum", "mean")
            
        Returns:
            Tuple of (yes_logit, no_logit) or None if tokens not found
        """
        yes_logits = []
        no_logits = []
        
        # Find all Yes/No tokens in vocabulary
        for token_id, logit in enumerate(vocab_logits):
            try:
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
                
                if token_str in self.yes_tokens:
                    yes_logits.append(logit.item())
                elif token_str in self.no_tokens:
                    no_logits.append(logit.item())
            except (IndexError, TypeError):
                # Skip invalid token IDs
                continue
        
        if not yes_logits and not no_logits:
            return None
            
        # Aggregate logits using specified method
        if aggregate_method == "max":
            yes_logit = max(yes_logits) if yes_logits else float('-inf')
            no_logit = max(no_logits) if no_logits else float('-inf')
        elif aggregate_method == "sum":
            yes_logit = sum(yes_logits) if yes_logits else float('-inf')
            no_logit = sum(no_logits) if no_logits else float('-inf')
        elif aggregate_method == "mean":
            yes_logit = sum(yes_logits) / len(yes_logits) if yes_logits else float('-inf')
            no_logit = sum(no_logits) / len(no_logits) if no_logits else float('-inf')
        else:
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}")
        
        # Handle cases where only one class of tokens is found
        if yes_logit == float('-inf') and no_logit != float('-inf'):
            # Only No tokens found - assign low probability to Yes
            yes_logit = no_logit - 10.0
        elif no_logit == float('-inf') and yes_logit != float('-inf'):
            # Only Yes tokens found - assign low probability to No  
            no_logit = yes_logit - 10.0
        elif yes_logit == float('-inf') and no_logit == float('-inf'):
            # Neither found - cannot compute entropy
            return None
            
        return yes_logit, no_logit
    
    def compute_cde_from_reasoning_paths(
        self,
        reasoning_path_logits: List[Tuple[float, float]],
        valid_only: bool = True
    ) -> Dict[str, Union[float, int, List[float]]]:
        """
        Compute Conditional Decision Entropy from multiple reasoning paths.
        
        This is the main function for CDE computation during online RL training.
        
        Args:
            reasoning_path_logits: List of (yes_logit, no_logit) tuples, one per reasoning path
            valid_only: Whether to skip None/invalid entries
            
        Returns:
            Dictionary containing:
                - cde: Average conditional decision entropy
                - individual_entropies: List of entropy values for each path
                - valid_paths: Number of valid reasoning paths used
                - total_paths: Total number of reasoning paths provided
                - min_entropy: Minimum entropy across paths
                - max_entropy: Maximum entropy across paths
                - entropy_std: Standard deviation of entropies
        """
        individual_entropies = []
        valid_paths = 0
        
        for yes_logit, no_logit in reasoning_path_logits:
            if yes_logit is not None and no_logit is not None:
                entropy = self.compute_binary_entropy_from_logits(yes_logit, no_logit)
                individual_entropies.append(entropy)
                valid_paths += 1
            elif not valid_only:
                # Include invalid paths as maximum entropy if valid_only=False
                individual_entropies.append(math.log(2))  # Maximum binary entropy
        
        if not individual_entropies:
            return {
                "cde": None,
                "individual_entropies": [],
                "valid_paths": 0,
                "total_paths": len(reasoning_path_logits),
                "min_entropy": None,
                "max_entropy": None,
                "entropy_std": None
            }
        
        # Compute statistics
        cde = sum(individual_entropies) / len(individual_entropies)
        min_entropy = min(individual_entropies)
        max_entropy = max(individual_entropies)
        
        # Compute standard deviation
        if len(individual_entropies) > 1:
            mean_entropy = cde
            variance = sum((e - mean_entropy) ** 2 for e in individual_entropies) / len(individual_entropies)
            entropy_std = math.sqrt(variance)
        else:
            entropy_std = 0.0
        
        return {
            "cde": cde,
            "individual_entropies": individual_entropies,
            "valid_paths": valid_paths,
            "total_paths": len(reasoning_path_logits),
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "entropy_std": entropy_std
        }
    
    def compute_cde_batch(
        self,
        batch_reasoning_logits: List[List[Tuple[float, float]]],
        return_individual: bool = False
    ) -> Union[List[Dict[str, Union[float, int, List[float]]]], Dict[str, Union[float, List[float]]]]:
        """
        Compute CDE for a batch of examples, each with multiple reasoning paths.
        
        Args:
            batch_reasoning_logits: List of reasoning path logits for each example in batch
            return_individual: Whether to return individual example results or batch statistics
            
        Returns:
            If return_individual=True: List of CDE results for each example
            If return_individual=False: Batch-level statistics
        """
        batch_results = []
        
        for example_logits in batch_reasoning_logits:
            result = self.compute_cde_from_reasoning_paths(example_logits)
            batch_results.append(result)
        
        if return_individual:
            return batch_results
        
        # Compute batch-level statistics
        valid_cdes = [r["cde"] for r in batch_results if r["cde"] is not None]
        
        if not valid_cdes:
            return {
                "batch_cde_mean": None,
                "batch_cde_std": None,
                "valid_examples": 0,
                "total_examples": len(batch_reasoning_logits)
            }
        
        batch_cde_mean = sum(valid_cdes) / len(valid_cdes)
        if len(valid_cdes) > 1:
            batch_variance = sum((cde - batch_cde_mean) ** 2 for cde in valid_cdes) / len(valid_cdes)
            batch_cde_std = math.sqrt(batch_variance)
        else:
            batch_cde_std = 0.0
        
        return {
            "batch_cde_mean": batch_cde_mean,
            "batch_cde_std": batch_cde_std,
            "batch_cde_values": valid_cdes,
            "valid_examples": len(valid_cdes),
            "total_examples": len(batch_reasoning_logits)
        }


def convert_to_bits(entropy_nats: float) -> float:
    """Convert entropy from nats to bits."""
    return entropy_nats / math.log(2)


def convert_to_nats(entropy_bits: float) -> float:
    """Convert entropy from bits to nats."""  
    return entropy_bits * math.log(2)


# Example usage functions for common RL scenarios
def cde_as_confidence_reward(
    cde_value: float,
    correct_prediction: bool,
    confidence_weight: float = 1.0,
    accuracy_weight: float = 1.0
) -> float:
    """
    Convert CDE to a reward signal encouraging both accuracy and confidence.
    
    Args:
        cde_value: Computed CDE value (lower = more confident)
        correct_prediction: Whether the prediction was correct
        confidence_weight: Weight for confidence component (negative CDE)
        accuracy_weight: Weight for accuracy component
        
    Returns:
        Reward value (higher is better)
    """
    # Accuracy reward: +1 for correct, -1 for incorrect
    accuracy_reward = 1.0 if correct_prediction else -1.0
    
    # Confidence reward: negative CDE (lower entropy = higher reward)
    confidence_reward = -cde_value
    
    return accuracy_weight * accuracy_reward + confidence_weight * confidence_reward


def cde_as_calibration_penalty(
    cde_values: List[float],
    prediction_accuracies: List[bool],
    target_entropy: float = 0.5  # Target entropy for well-calibrated predictions
) -> float:
    """
    Use CDE to compute a calibration penalty for training.
    
    Well-calibrated models should have:
    - Low entropy for correct predictions (confident when right)
    - Higher entropy for incorrect predictions (uncertain when wrong)
    
    Args:
        cde_values: List of CDE values for a batch
        prediction_accuracies: List of whether each prediction was correct
        target_entropy: Target entropy for incorrect predictions
        
    Returns:
        Calibration penalty (lower is better)
    """
    penalty = 0.0
    count = 0
    
    for cde, is_correct in zip(cde_values, prediction_accuracies):
        if cde is not None:
            if is_correct:
                # Penalize high entropy for correct predictions
                penalty += cde
            else:
                # Penalize deviation from target entropy for incorrect predictions  
                penalty += abs(cde - target_entropy)
            count += 1
    
    return penalty / count if count > 0 else 0.0


# Example integration with popular RL frameworks
class CDERewardComputer:
    """
    Example reward computer for integration with RL training loops.
    
    This class shows how to integrate CDE computation into online RL training
    such as GRPO, PPO, or other policy gradient methods.
    """
    
    def __init__(self, cde_computer: CDEComputer, reward_config: Dict[str, float]):
        """
        Initialize reward computer.
        
        Args:
            cde_computer: Initialized CDEComputer instance
            reward_config: Dictionary with reward weights and parameters
        """
        self.cde_computer = cde_computer
        self.reward_config = reward_config
    
    def compute_rewards(
        self,
        batch_logits: List[List[Tuple[float, float]]],  # [batch_size, num_paths, 2]
        ground_truth: List[bool],  # [batch_size] 
        predictions: List[List[bool]]  # [batch_size, num_paths]
    ) -> List[float]:
        """
        Compute rewards for a batch of examples for RL training.
        
        Args:
            batch_logits: Decision logits for each example and reasoning path
            ground_truth: True labels for each example
            predictions: Predictions for each example and reasoning path
            
        Returns:
            List of reward values for each example
        """
        rewards = []
        
        cde_results = self.cde_computer.compute_cde_batch(batch_logits, return_individual=True)
        
        for i, (cde_result, gt, example_preds) in enumerate(zip(cde_results, ground_truth, predictions)):
            if cde_result["cde"] is None:
                # Fallback reward for invalid examples
                rewards.append(self.reward_config.get("invalid_example_reward", -1.0))
                continue
            
            # Compute accuracy for this example (majority vote or average)
            if self.reward_config.get("use_majority_vote", True):
                correct_preds = sum(pred == gt for pred in example_preds if pred is not None)
                is_correct = correct_preds > len(example_preds) / 2
            else:
                # Use average accuracy across all reasoning paths
                accuracies = [pred == gt for pred in example_preds if pred is not None]
                is_correct = sum(accuracies) / len(accuracies) > 0.5 if accuracies else False
            
            # Compute reward using CDE and accuracy
            reward = cde_as_confidence_reward(
                cde_result["cde"],
                is_correct,
                confidence_weight=self.reward_config.get("confidence_weight", 1.0),
                accuracy_weight=self.reward_config.get("accuracy_weight", 1.0)
            )
            
            rewards.append(reward)
        
        return rewards


if __name__ == "__main__":
    # Example usage demonstration
    print("Conditional Decision Entropy (CDE) Algorithm Demo")
    print("=" * 50)
    
    # Initialize CDE computer
    cde_computer = CDEComputer()
    
    # Example 1: Single reasoning path
    print("\n1. Single reasoning path:")
    yes_logit, no_logit = 2.0, 0.5  # High confidence in Yes
    entropy = cde_computer.compute_binary_entropy_from_logits(yes_logit, no_logit)
    print(f"   Logits: Yes={yes_logit}, No={no_logit}")
    print(f"   Binary Entropy: {entropy:.4f} nats ({convert_to_bits(entropy):.4f} bits)")
    
    # Example 2: Multiple reasoning paths (Monte Carlo CDE)
    print("\n2. Multiple reasoning paths (Monte Carlo CDE):")
    reasoning_paths = [
        (2.0, 0.5),   # Confident Yes
        (1.5, 0.8),   # Moderate Yes  
        (0.2, 2.1),   # Confident No
        (0.9, 1.1)    # Uncertain
    ]
    
    result = cde_computer.compute_cde_from_reasoning_paths(reasoning_paths)
    print(f"   Reasoning paths: {reasoning_paths}")
    print(f"   CDE: {result['cde']:.4f} nats ({convert_to_bits(result['cde']):.4f} bits)")
    print(f"   Individual entropies: {[f'{e:.4f}' for e in result['individual_entropies']]}")
    print(f"   Entropy std: {result['entropy_std']:.4f}")
    
    # Example 3: Batch processing
    print("\n3. Batch processing:")
    batch_data = [
        [(2.0, 0.5), (1.8, 0.7)],  # Example 1: 2 paths
        [(0.1, 2.5), (0.3, 2.2), (0.2, 2.4)],  # Example 2: 3 paths
        [(1.0, 1.0)]  # Example 3: 1 path (uncertain)
    ]
    
    batch_result = cde_computer.compute_cde_batch(batch_data)
    print(f"   Batch CDE mean: {batch_result['batch_cde_mean']:.4f} nats")
    print(f"   Batch CDE std: {batch_result['batch_cde_std']:.4f}")
    print(f"   Valid examples: {batch_result['valid_examples']}/{batch_result['total_examples']}")
    
    # Example 4: Reward computation
    print("\n4. Reward computation:")
    cde_value = 0.3
    correct = True
    reward = cde_as_confidence_reward(cde_value, correct, confidence_weight=0.5, accuracy_weight=1.0)
    print(f"   CDE: {cde_value}, Correct: {correct}")
    print(f"   Reward: {reward:.4f}")
    
    print(f"\nCDE algorithm ready for integration with online RL training!")