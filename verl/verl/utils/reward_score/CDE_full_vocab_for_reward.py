#!/usr/bin/env python3
"""
Full Vocabulary Conditional Decision Entropy (CDE) Algorithm for Reward Computation

This module provides a clean, abstract implementation of the Full Vocabulary Conditional 
Decision Entropy algorithm for use in online RL training (e.g., GRPO). Unlike the binary 
version, this computes entropy over the full vocabulary distribution at decision positions,
providing a more general measure of model uncertainty.

Mathematical Formulation:
    CDE(x) = (1/K) * Σ_i H(vocab | r_i, x)
    
where:
    - x is the input example
    - r_i are K sampled reasoning paths  
    - H(vocab | r_i, x) is the categorical entropy over vocabulary given reasoning path r_i
    - Categorical entropy: H(vocab) = -Σ_j p(token_j) * log(p(token_j))

Key Properties:
    - Lower entropy indicates higher confidence in the decision
    - Higher entropy indicates uncertainty/poor calibration
    - Works with any vocabulary size and tokenizer
    - More general than binary classification - captures full model uncertainty
    - Can be used as a reward to encourage well-calibrated confidence

Usage for Online RL:
    1. Generate multiple reasoning paths per example during training
    2. Extract full vocabulary logits/probabilities at decision positions
    3. Compute CDE using this module
    4. Use CDE as reward signal (e.g., negative CDE for confidence reward)
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class FullVocabCDEComputer:
    """
    Full Vocabulary Conditional Decision Entropy computer for general language modeling tasks.
    
    This class provides methods to compute CDE from full vocabulary logits/probabilities
    extracted from model generations during online RL training.
    """
    
    def __init__(
        self, 
        epsilon: float = 1e-10,
        temperature: float = 1.0,
        max_vocab_size: Optional[int] = None
    ):
        """
        Initialize the Full Vocabulary CDE computer.
        
        Args:
            epsilon: Small value for numerical stability in log computations
            temperature: Temperature for softmax computation (default: 1.0)
            max_vocab_size: Maximum vocabulary size to consider (for memory efficiency)
        """
        self.epsilon = epsilon
        self.temperature = temperature
        self.max_vocab_size = max_vocab_size
    
    def compute_vocab_entropy_from_logits(
        self, 
        vocab_logits: torch.Tensor,
        return_probs: bool = False,
        top_k: Optional[int] = None
    ) -> Union[float, Tuple[float, torch.Tensor]]:
        """
        Compute categorical entropy from full vocabulary logits.
        
        Args:
            vocab_logits: Tensor of shape [vocab_size] with logits for each token
            return_probs: Whether to also return the probability distribution
            top_k: If specified, only consider top-k tokens for entropy computation
            
        Returns:
            Categorical entropy value, optionally with probability tensor
        """
        if top_k is not None:
            # Only consider top-k tokens for efficiency
            top_k = min(top_k, vocab_logits.size(0))
            topk_logits, topk_indices = torch.topk(vocab_logits, top_k)
            # Create a sparse probability distribution
            probs = F.softmax(topk_logits / self.temperature, dim=0)
        else:
            # Use full vocabulary
            probs = F.softmax(vocab_logits / self.temperature, dim=0)
        
        # Compute categorical entropy: -Σ p(i) * log(p(i))
        entropy = -torch.sum(probs * torch.log(probs + self.epsilon))
        
        if return_probs:
            if top_k is not None:
                # Expand sparse probs back to full vocab size for consistency
                full_probs = torch.zeros_like(vocab_logits)
                full_probs[topk_indices] = probs
                return entropy.item(), full_probs
            return entropy.item(), probs
        return entropy.item()
    
    def compute_vocab_entropy_from_probs(
        self, 
        vocab_probs: torch.Tensor
    ) -> float:
        """
        Compute categorical entropy from vocabulary probabilities.
        
        Args:
            vocab_probs: Tensor of shape [vocab_size] with probabilities for each token
            
        Returns:
            Categorical entropy value
        """
        # Normalize probabilities to ensure they sum to 1
        vocab_probs_norm = vocab_probs / (torch.sum(vocab_probs) + self.epsilon)
        
        # Compute categorical entropy with numerical stability
        entropy = -torch.sum(vocab_probs_norm * torch.log(vocab_probs_norm + self.epsilon))
        return entropy.item()
    
    def compute_vocab_entropy_from_logprobs(
        self,
        vocab_logprobs: Dict[int, float],
        vocab_size: Optional[int] = None
    ) -> Tuple[float, Dict[str, Union[float, int]]]:
        """
        Compute vocabulary entropy from a dictionary of token logprobs.
        
        This is useful when working with top-k logprobs from vLLM or similar systems.
        
        Args:
            vocab_logprobs: Dictionary mapping token_id -> logprob
            vocab_size: Total vocabulary size (for proper normalization)
            
        Returns:
            Tuple of (entropy, info_dict) with detailed statistics
        """
        if not vocab_logprobs:
            return float('inf'), {"num_tokens": 0, "max_prob": 0.0, "min_prob": 0.0}
        
        # Convert logprobs to probabilities
        logprobs = torch.tensor(list(vocab_logprobs.values()), dtype=torch.float32)
        probs = torch.exp(logprobs)  # Convert logprob to prob
        
        # Normalize probabilities (they should already be normalized from softmax, but let's be safe)
        probs = probs / torch.sum(probs)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + self.epsilon))
        
        # Gather statistics
        max_prob = torch.max(probs).item()
        min_prob = torch.min(probs).item()
        num_tokens = len(vocab_logprobs)
        
        info = {
            "num_tokens": num_tokens,
            "max_prob": max_prob,
            "min_prob": min_prob,
            "prob_std": torch.std(probs).item(),
            "effective_vocab_size": torch.exp(entropy).item()  # Perplexity
        }
        
        return entropy.item(), info
    
    def extract_decision_entropy_from_generation_logprobs(
        self,
        generation_logprobs: List[Dict[int, float]],
        decision_position: int
    ) -> Optional[Tuple[float, Dict[str, Union[float, int]]]]:
        """
        Extract vocabulary entropy from generation logprobs at a specific decision position.
        
        Args:
            generation_logprobs: List of logprob dictionaries for each token position
            decision_position: Position where decision is made
            
        Returns:
            Tuple of (entropy, info_dict) or None if position invalid
        """
        if decision_position >= len(generation_logprobs):
            return None
        
        position_logprobs = generation_logprobs[decision_position]
        return self.compute_vocab_entropy_from_logprobs(position_logprobs)
    
    def compute_cde_from_reasoning_paths(
        self,
        reasoning_path_entropies: List[float],
        valid_only: bool = True
    ) -> Dict[str, Union[float, int, List[float]]]:
        """
        Compute Conditional Decision Entropy from multiple reasoning paths.
        
        This is the main function for CDE computation during online RL training.
        
        Args:
            reasoning_path_entropies: List of vocabulary entropy values, one per reasoning path
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
                - perplexity: Average perplexity (exp(entropy))
        """
        individual_entropies = []
        valid_paths = 0
        
        for entropy in reasoning_path_entropies:
            if entropy is not None and not math.isnan(entropy) and not math.isinf(entropy):
                individual_entropies.append(entropy)
                valid_paths += 1
            elif not valid_only:
                # Include invalid paths as maximum entropy if valid_only=False
                individual_entropies.append(10.0)  # High entropy value for invalid paths
        
        if not individual_entropies:
            return {
                "cde": None,
                "individual_entropies": [],
                "valid_paths": 0,
                "total_paths": len(reasoning_path_entropies),
                "min_entropy": None,
                "max_entropy": None,
                "entropy_std": None,
                "perplexity": None
            }
        
        # Compute statistics
        cde = sum(individual_entropies) / len(individual_entropies)
        min_entropy = min(individual_entropies)
        max_entropy = max(individual_entropies)
        perplexity = math.exp(cde)
        
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
            "total_paths": len(reasoning_path_entropies),
            "min_entropy": min_entropy,
            "max_entropy": max_entropy,
            "entropy_std": entropy_std,
            "perplexity": perplexity
        }
    
    def compute_cde_from_logprobs_batch(
        self,
        batch_reasoning_logprobs: List[List[List[Dict[int, float]]]],
        decision_positions: List[int],
        return_individual: bool = False
    ) -> Union[List[Dict[str, Union[float, int, List[float]]]], Dict[str, Union[float, List[float]]]]:
        """
        Compute CDE for a batch of examples from logprobs, each with multiple reasoning paths.
        
        Args:
            batch_reasoning_logprobs: List of [num_paths, num_tokens] logprobs for each example
            decision_positions: Decision position for each example in the batch
            return_individual: Whether to return individual example results or batch statistics
            
        Returns:
            If return_individual=True: List of CDE results for each example
            If return_individual=False: Batch-level statistics
        """
        batch_results = []
        
        for example_logprobs, decision_pos in zip(batch_reasoning_logprobs, decision_positions):
            # Extract entropy for each reasoning path
            path_entropies = []
            for path_logprobs in example_logprobs:
                entropy_result = self.extract_decision_entropy_from_generation_logprobs(
                    path_logprobs, decision_pos
                )
                if entropy_result is not None:
                    path_entropies.append(entropy_result[0])
                else:
                    path_entropies.append(None)
            
            # Compute CDE for this example
            result = self.compute_cde_from_reasoning_paths(path_entropies)
            batch_results.append(result)
        
        if return_individual:
            return batch_results
        
        # Compute batch-level statistics
        valid_cdes = [r["cde"] for r in batch_results if r["cde"] is not None]
        
        if not valid_cdes:
            return {
                "batch_cde_mean": None,
                "batch_cde_std": None,
                "batch_perplexity": None,
                "valid_examples": 0,
                "total_examples": len(batch_reasoning_logprobs)
            }
        
        batch_cde_mean = sum(valid_cdes) / len(valid_cdes)
        batch_perplexity = math.exp(batch_cde_mean)
        
        if len(valid_cdes) > 1:
            batch_variance = sum((cde - batch_cde_mean) ** 2 for cde in valid_cdes) / len(valid_cdes)
            batch_cde_std = math.sqrt(batch_variance)
        else:
            batch_cde_std = 0.0
        
        return {
            "batch_cde_mean": batch_cde_mean,
            "batch_cde_std": batch_cde_std,
            "batch_cde_values": valid_cdes,
            "batch_perplexity": batch_perplexity,
            "valid_examples": len(valid_cdes),
            "total_examples": len(batch_reasoning_logprobs)
        }
    
    def compute_cde_batch(
        self,
        batch_reasoning_entropies: List[List[float]],
        return_individual: bool = False
    ) -> Union[List[Dict[str, Union[float, int, List[float]]]], Dict[str, Union[float, List[float]]]]:
        """
        Compute CDE for a batch of examples, each with multiple reasoning path entropies.
        
        Args:
            batch_reasoning_entropies: List of entropy values for each example in batch
            return_individual: Whether to return individual example results or batch statistics
            
        Returns:
            If return_individual=True: List of CDE results for each example
            If return_individual=False: Batch-level statistics
        """
        batch_results = []
        
        for example_entropies in batch_reasoning_entropies:
            result = self.compute_cde_from_reasoning_paths(example_entropies)
            batch_results.append(result)
        
        if return_individual:
            return batch_results
        
        # Compute batch-level statistics
        valid_cdes = [r["cde"] for r in batch_results if r["cde"] is not None]
        
        if not valid_cdes:
            return {
                "batch_cde_mean": None,
                "batch_cde_std": None,
                "batch_perplexity": None,
                "valid_examples": 0,
                "total_examples": len(batch_reasoning_entropies)
            }
        
        batch_cde_mean = sum(valid_cdes) / len(valid_cdes)
        batch_perplexity = math.exp(batch_cde_mean)
        
        if len(valid_cdes) > 1:
            batch_variance = sum((cde - batch_cde_mean) ** 2 for cde in valid_cdes) / len(valid_cdes)
            batch_cde_std = math.sqrt(batch_variance)
        else:
            batch_cde_std = 0.0
        
        return {
            "batch_cde_mean": batch_cde_mean,
            "batch_cde_std": batch_cde_std,
            "batch_cde_values": valid_cdes,
            "batch_perplexity": batch_perplexity,
            "valid_examples": len(valid_cdes),
            "total_examples": len(batch_reasoning_entropies)
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
    accuracy_weight: float = 1.0,
    entropy_scale: float = 1.0
) -> float:
    """
    Convert Full Vocabulary CDE to a reward signal encouraging both accuracy and confidence.
    
    Args:
        cde_value: Computed CDE value (lower = more confident)
        correct_prediction: Whether the prediction was correct
        confidence_weight: Weight for confidence component (negative CDE)
        accuracy_weight: Weight for accuracy component
        entropy_scale: Scale factor for entropy values (since vocab entropy is typically higher than binary)
        
    Returns:
        Reward value (higher is better)
    """
    # Accuracy reward: +1 for correct, -1 for incorrect
    accuracy_reward = 1.0 if correct_prediction else -1.0
    
    # Confidence reward: negative scaled CDE (lower entropy = higher reward)
    confidence_reward = -cde_value / entropy_scale
    
    return accuracy_weight * accuracy_reward + confidence_weight * confidence_reward


def cde_as_perplexity_penalty(
    perplexity_value: float,
    correct_prediction: bool,
    target_perplexity: float = 10.0  # Target perplexity for well-calibrated predictions
) -> float:
    """
    Use perplexity (exp(CDE)) to compute a calibration penalty for training.
    
    Args:
        perplexity_value: Computed perplexity (exp(CDE))
        correct_prediction: Whether the prediction was correct
        target_perplexity: Target perplexity for well-calibrated models
        
    Returns:
        Calibration penalty (lower is better)
    """
    if correct_prediction:
        # For correct predictions, penalize high perplexity (low confidence)
        return max(0, perplexity_value - target_perplexity) / target_perplexity
    else:
        # For incorrect predictions, penalize low perplexity (overconfidence)
        return max(0, target_perplexity - perplexity_value) / target_perplexity


def adaptive_entropy_reward(
    cde_value: float,
    correct_prediction: bool,
    difficulty_estimate: float = 0.5,  # 0 = easy, 1 = hard
    base_reward: float = 1.0
) -> float:
    """
    Adaptive reward that considers task difficulty in entropy expectations.
    
    Args:
        cde_value: Computed CDE value
        correct_prediction: Whether prediction was correct
        difficulty_estimate: Estimated difficulty of the example (0-1)
        base_reward: Base reward value
        
    Returns:
        Adaptive reward value
    """
    # Expected entropy increases with difficulty
    expected_entropy = 2.0 + 3.0 * difficulty_estimate  # Range: 2-5 nats
    
    # Compute entropy deviation penalty
    entropy_deviation = abs(cde_value - expected_entropy) / expected_entropy
    
    # Base accuracy reward
    accuracy_reward = base_reward if correct_prediction else -base_reward
    
    # Confidence calibration bonus (reward good calibration)
    calibration_bonus = base_reward * (1.0 - entropy_deviation)
    
    return accuracy_reward + 0.5 * calibration_bonus


# Example integration with popular RL frameworks
class FullVocabCDERewardComputer:
    """
    Full vocabulary CDE reward computer for integration with RL training loops.
    
    This class shows how to integrate full vocabulary CDE computation into online RL training
    such as GRPO, PPO, or other policy gradient methods.
    """
    
    def __init__(self, cde_computer: FullVocabCDEComputer, reward_config: Dict[str, float]):
        """
        Initialize reward computer.
        
        Args:
            cde_computer: Initialized FullVocabCDEComputer instance
            reward_config: Dictionary with reward weights and parameters
        """
        self.cde_computer = cde_computer
        self.reward_config = reward_config
    
    def compute_rewards_from_entropies(
        self,
        batch_entropies: List[List[float]],  # [batch_size, num_paths]
        ground_truth: List[bool],  # [batch_size] 
        predictions: List[List[bool]]  # [batch_size, num_paths]
    ) -> List[float]:
        """
        Compute rewards for a batch of examples from precomputed entropies.
        
        Args:
            batch_entropies: Vocabulary entropies for each example and reasoning path
            ground_truth: True labels for each example
            predictions: Predictions for each example and reasoning path
            
        Returns:
            List of reward values for each example
        """
        rewards = []
        
        cde_results = self.cde_computer.compute_cde_batch(batch_entropies, return_individual=True)
        
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
                accuracy_weight=self.reward_config.get("accuracy_weight", 1.0),
                entropy_scale=self.reward_config.get("entropy_scale", 5.0)  # Scale for vocab entropy
            )
            
            rewards.append(reward)
        
        return rewards
    
    def compute_rewards_from_logprobs(
        self,
        batch_logprobs: List[List[List[Dict[int, float]]]],  # [batch_size, num_paths, num_tokens]
        decision_positions: List[int],  # [batch_size]
        ground_truth: List[bool],  # [batch_size] 
        predictions: List[List[bool]]  # [batch_size, num_paths]
    ) -> List[float]:
        """
        Compute rewards for a batch of examples from logprobs.
        
        Args:
            batch_logprobs: Logprob dictionaries for each example, path, and token position
            decision_positions: Decision position for each example
            ground_truth: True labels for each example
            predictions: Predictions for each example and reasoning path
            
        Returns:
            List of reward values for each example
        """
        # First, extract entropies from logprobs
        batch_entropies = []
        
        for example_logprobs, decision_pos in zip(batch_logprobs, decision_positions):
            example_entropies = []
            for path_logprobs in example_logprobs:
                entropy_result = self.cde_computer.extract_decision_entropy_from_generation_logprobs(
                    path_logprobs, decision_pos
                )
                if entropy_result is not None:
                    example_entropies.append(entropy_result[0])
                else:
                    example_entropies.append(None)
            batch_entropies.append(example_entropies)
        
        # Now compute rewards using extracted entropies
        return self.compute_rewards_from_entropies(batch_entropies, ground_truth, predictions)


if __name__ == "__main__":
    # Example usage demonstration
    print("Full Vocabulary Conditional Decision Entropy (CDE) Algorithm Demo")
    print("=" * 65)
    
    # Initialize CDE computer
    cde_computer = FullVocabCDEComputer()
    
    # Example 1: Single reasoning path with vocab logits
    print("\n1. Single reasoning path with vocabulary logits:")
    vocab_size = 1000
    vocab_logits = torch.randn(vocab_size)  # Random logits
    vocab_logits[42] = 5.0  # Make token 42 very likely
    
    entropy, probs = cde_computer.compute_vocab_entropy_from_logits(vocab_logits, return_probs=True)
    most_likely_token = torch.argmax(probs).item()
    max_prob = torch.max(probs).item()
    
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Most likely token: {most_likely_token} (prob: {max_prob:.4f})")
    print(f"   Vocabulary Entropy: {entropy:.4f} nats ({convert_to_bits(entropy):.4f} bits)")
    print(f"   Perplexity: {math.exp(entropy):.2f}")
    
    # Example 2: Using top-k logprobs (more realistic scenario)
    print("\n2. Using top-k logprobs (realistic vLLM scenario):")
    # Simulate vLLM-style top-k logprobs
    top_k_logprobs = {
        42: -0.1,   # Most likely token
        123: -1.2,  # Second most likely
        456: -2.1,  # Third most likely
        789: -2.8,  # Fourth most likely
        999: -3.5   # Fifth most likely
    }
    
    entropy, info = cde_computer.compute_vocab_entropy_from_logprobs(top_k_logprobs)
    print(f"   Top-k tokens: {list(top_k_logprobs.keys())}")
    print(f"   Logprobs: {list(top_k_logprobs.values())}")
    print(f"   Entropy: {entropy:.4f} nats ({convert_to_bits(entropy):.4f} bits)")
    print(f"   Effective vocab size: {info['effective_vocab_size']:.2f}")
    print(f"   Max probability: {info['max_prob']:.4f}")
    
    # Example 3: Multiple reasoning paths (Monte Carlo CDE)
    print("\n3. Multiple reasoning paths (Monte Carlo CDE):")
    reasoning_path_entropies = [
        2.1,  # Low entropy (confident)
        4.5,  # Medium entropy
        6.8,  # High entropy (uncertain)
        3.2   # Medium-low entropy
    ]
    
    result = cde_computer.compute_cde_from_reasoning_paths(reasoning_path_entropies)
    print(f"   Path entropies: {reasoning_path_entropies}")
    print(f"   CDE: {result['cde']:.4f} nats ({convert_to_bits(result['cde']):.4f} bits)")
    print(f"   Average perplexity: {result['perplexity']:.2f}")
    print(f"   Entropy range: [{result['min_entropy']:.2f}, {result['max_entropy']:.2f}]")
    print(f"   Entropy std: {result['entropy_std']:.4f}")
    
    # Example 4: Batch processing
    print("\n4. Batch processing:")
    batch_data = [
        [2.1, 2.3],           # Example 1: 2 confident paths
        [5.5, 6.1, 5.8],      # Example 2: 3 uncertain paths
        [1.5]                 # Example 3: 1 very confident path
    ]
    
    batch_result = cde_computer.compute_cde_batch(batch_data)
    print(f"   Batch CDE mean: {batch_result['batch_cde_mean']:.4f} nats")
    print(f"   Batch CDE std: {batch_result['batch_cde_std']:.4f}")
    print(f"   Batch perplexity: {batch_result['batch_perplexity']:.2f}")
    print(f"   Valid examples: {batch_result['valid_examples']}/{batch_result['total_examples']}")
    
    # Example 5: Reward computation
    print("\n5. Reward computation:")
    cde_value = 3.5
    correct = True
    reward = cde_as_confidence_reward(cde_value, correct, confidence_weight=0.5, accuracy_weight=1.0, entropy_scale=5.0)
    print(f"   CDE: {cde_value} nats, Correct: {correct}")
    print(f"   Reward: {reward:.4f}")
    
    # Example 6: Adaptive reward
    print("\n6. Adaptive reward (difficulty-aware):")
    difficulty = 0.8  # Hard example
    adaptive_reward = adaptive_entropy_reward(cde_value, correct, difficulty)
    print(f"   CDE: {cde_value} nats, Correct: {correct}, Difficulty: {difficulty}")
    print(f"   Adaptive reward: {adaptive_reward:.4f}")
    
    print(f"\nFull Vocabulary CDE algorithm ready for integration with online RL training!")