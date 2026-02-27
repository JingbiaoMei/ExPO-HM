import math
import re
from typing import Dict, List, Optional, Tuple, Union

from .CDE_full_vocab_for_reward import FullVocabCDEComputer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    Fine-grained accuracy reward function for meme classification.
    Handles multiple categories separated by commas (e.g., "inciting_violence, dehumanizing").
    Uses <answer> tags for extraction and provides partial rewards for matching categories.
    """
    try:
        # Convert to lowercase for comparison
        predict_str_lower = predict_str.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Pattern for extracting content from <answer> tags
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        
        # Extract answer from ground truth if it has answer tags
        sol_match = re.search(answer_tag_pattern, ground_truth_lower, re.DOTALL)
        extracted_ground_truth = sol_match.group(1).strip() if sol_match else ground_truth_lower.strip()
        
        # Extract answer from prediction if it has answer tags
        content_match = re.search(answer_tag_pattern, predict_str_lower, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else predict_str_lower.strip()
        
        # Transfer the solution to a list of items
        ground_truth_list = extracted_ground_truth.split(",") if "," in extracted_ground_truth else [extracted_ground_truth]
        
        # Similarly transfer the student answer to a list of items
        student_answer_list = student_answer.split(",") if "," in student_answer else [student_answer]
        
        # Normalize by removing spaces and underscores
        ground_truth_list = [item.replace(' ', '').replace('_', '').lower() for item in ground_truth_list]
        student_answer_list = [item.replace(' ', '').replace('_', '').lower() for item in student_answer_list]
        
        reward = 0.0
        
        # Two cases: if the lists are equal length, compare them directly
        if len(ground_truth_list) == len(student_answer_list):
            # If exact match for the item in the list, assign a partial reward
            for item in ground_truth_list:
                if item in student_answer_list:
                    reward += 1.0 / len(ground_truth_list)
        else:
            # If the length is not equal, take the reward's denominator 
            # as the length of the ground truth or student answer, whichever is larger
            denominator = max(len(ground_truth_list), len(student_answer_list))
            # If the item in the ground truth is in the student answer, assign a partial reward
            for item in ground_truth_list:
                if item in student_answer_list:
                    reward += 1.0 / denominator
        
        return reward
        
    except Exception:
        return 0.0


def extract_full_vocab_entropy_from_answer_tags(
    predict_str: str,
    ground_truth: str,
    top_k_logprobs: Optional[List[List[float]]] = None,
    top_k_token_ids: Optional[List[List[int]]] = None,
    tokenizer=None
) -> Optional[Tuple[float, Dict[str, Union[float, int]]]]:
    """
    Extract full vocabulary entropy from the answer portion of the generated text.
    
    This function identifies the <answer> tag location and computes entropy
    over the full vocabulary distribution at the corresponding position.
    
    Args:
        predict_str: Generated text containing <answer>category1, category2</answer>
        ground_truth: Ground truth answer (used for logging purposes)
        top_k_logprobs: List of top-k logprobs for each token position
        top_k_token_ids: List of top-k token IDs for each token position  
        tokenizer: Tokenizer to convert token IDs to strings
        
    Returns:
        Tuple of (entropy, info_dict) or None if extraction fails
    """
    if top_k_logprobs is None or top_k_token_ids is None or tokenizer is None:
        print("[FULL VOCAB DEBUG] Missing required parameters")
        return None
    
    try:
        # Initialize full vocab CDE computer
        full_cde_computer = FullVocabCDEComputer()
        
        # Find the answer tag pattern in the generated text
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_tag_pattern, predict_str.lower())
        
        if not answer_match:
            print(f"[FULL VOCAB DEBUG] No answer tag found in: {predict_str}")
            return None
        
        # Get the answer content (can be multiple categories)
        answer_content = answer_match.group(1).strip()
        #print(f"[FULL VOCAB DEBUG] Found answer: '{answer_content}'")
        
        # For fine-grained classification, we don't restrict to specific values
        # Just check that it's not empty
        if not answer_content:
            print(f"[FULL VOCAB DEBUG] Empty answer content")
            return None
            
        # Use similar position finding logic as the binary version
        answer_search_positions = []
        
        #print(f"[FULL VOCAB DEBUG] Searching for <answer> tag position in {len(top_k_logprobs)} token positions")
        
        try:
            # For each position, get the most likely token to reconstruct the sequence
            likely_tokens = []
            for pos_idx, (pos_logprobs, pos_token_ids) in enumerate(zip(top_k_logprobs, top_k_token_ids)):
                if len(pos_logprobs) > 0:
                    most_likely_token_id = pos_token_ids[0]
                    most_likely_token = tokenizer.convert_ids_to_tokens([most_likely_token_id])[0]
                    likely_tokens.append((pos_idx, most_likely_token))
            
            # Reconstruct approximate text and find <answer> tag
            reconstructed_parts = []
            position_map = {}  # reconstructed_char_index -> token_position
            
            char_idx = 0
            for pos_idx, token in likely_tokens:
                # Clean up token (remove special prefixes)
                clean_token = token.replace('Ġ', ' ').replace('▁', ' ')
                reconstructed_parts.append(clean_token)
                # Map character positions to token positions
                for i in range(len(clean_token)):
                    position_map[char_idx + i] = pos_idx
                char_idx += len(clean_token)
            
            reconstructed_text = ''.join(reconstructed_parts)
            #print(f"[FULL VOCAB DEBUG] Reconstructed text snippet: {reconstructed_text[:100]}...")
            
            # Find both opening and closing <answer> tags
            answer_start_match = re.search(r'<answer>', reconstructed_text.lower())
            answer_end_match = re.search(r'</answer>', reconstructed_text.lower())
            
            if answer_start_match and answer_end_match:
                # Find token positions corresponding to the content INSIDE <answer></answer>
                start_char = answer_start_match.end()  # After <answer>
                end_char = answer_end_match.start()    # Before </answer>
                
                # Get token positions between the tags
                start_pos = None
                end_pos = None
                
                for char_pos, token_pos in position_map.items():
                    if char_pos >= start_char and start_pos is None:
                        start_pos = token_pos
                    if char_pos >= end_char and end_pos is None:
                        end_pos = token_pos
                        break
                
                if start_pos is not None and end_pos is not None:
                    answer_search_positions = list(range(start_pos, end_pos))
                    #print(f"[FULL VOCAB DEBUG] Found <answer> content between token positions: {start_pos} to {end_pos-1}")
            
            # Fallback: if tag-based method failed, search near where we expect the answer
            if not answer_search_positions:
                #print(f"[FULL VOCAB DEBUG] Tag method failed, using fallback search")
                fallback_start = max(0, len(top_k_logprobs) - len(top_k_logprobs) // 5)
                answer_search_positions = list(range(fallback_start, len(top_k_logprobs)))
                #print(f"[FULL VOCAB DEBUG] Using fallback positions: {fallback_start} to {len(top_k_logprobs)}")
        
        except Exception as e:
            print(f"[FULL VOCAB DEBUG] Error finding <answer> position: {e}")
            fallback_start = max(0, len(top_k_logprobs) - len(top_k_logprobs) // 5)
            answer_search_positions = list(range(fallback_start, len(top_k_logprobs)))
            print(f"[FULL VOCAB DEBUG] Using fallback positions: {fallback_start} to {len(top_k_logprobs)}")
        
        # Now compute full vocabulary entropy at the identified positions
        best_entropy = None
        best_info = None
        
        # Use the first few positions to avoid excessive processing
        search_positions = answer_search_positions[:5]  # Just check first few positions
        #print(f"[FULL VOCAB DEBUG] Computing full vocab entropy at positions: {search_positions}")
        
        for pos_idx in search_positions:
            if pos_idx >= len(top_k_logprobs):
                continue
                
            pos_logprobs = top_k_logprobs[pos_idx]
            pos_token_ids = top_k_token_ids[pos_idx]
            
            if len(pos_logprobs) != len(pos_token_ids):
                continue
            
            # Create dictionary mapping token_id -> logprob
            vocab_logprobs = {}
            for token_id, logprob in zip(pos_token_ids, pos_logprobs):
                vocab_logprobs[token_id] = logprob
            
            # Compute full vocabulary entropy
            entropy, info = full_cde_computer.compute_vocab_entropy_from_logprobs(vocab_logprobs)
            
            #print(f"[FULL VOCAB DEBUG] Position {pos_idx}: entropy={entropy:.3f}, tokens={info['num_tokens']}, max_prob={info['max_prob']:.3f}")
            
            # Take the first valid entropy (could be enhanced to take best/average)
            if best_entropy is None and entropy != float('inf'):
                best_entropy = entropy
                best_info = info
                break
        
        #print(f"[FULL VOCAB DEBUG] Final entropy: {best_entropy}")
        if best_entropy is not None:
            return (best_entropy, best_info)
        else:
            return None
        
    except Exception as e:
        print(f"[FULL VOCAB DEBUG] Error extracting full vocab entropy: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_detailed_score(
    predict_str: str, 
    ground_truth: str, 
    format_score: float = 0.1,
    cde_weight: float = 0.2,
    # New piecewise CDE hyperparameters
    low_entropy_cutoff: float = 0.15,      # Threshold for "very confident" decisions
    high_entropy_cutoff_correct: float = 0.60,  # Above this, correct but mushy gets no reward
    high_entropy_cutoff_wrong: float = 0.85,    # Above this, wrong but uncertain gets full reward
    confidence_penalty_ratio: float = 0.0,      # Penalty for confidently wrong (0 = no penalty, 0.5 = half penalty)
    use_sigmoid_smoothing: bool = False,         # Whether to use smooth sigmoid transitions
    sigmoid_steepness: float = 8.0,              # Steepness of sigmoid transitions (higher = sharper)
    top_k_logprobs: Optional[List[List[float]]] = None,
    top_k_token_ids: Optional[List[List[int]]] = None,
    tokenizer=None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute detailed reward scores using full vocabulary CDE reward strategy for fine-grained classification.
    
    This function implements a sophisticated CDE reward that:
    - Uses fine-grained accuracy reward from meme_fg.py for multi-category classification
    - Uses full vocabulary entropy computation (no binary yes/no extraction)
    - Rewards confidence when the prediction is correct
    - Rewards uncertainty when the prediction is wrong  
    - Optionally penalizes overconfident wrong predictions
    - Supports smooth sigmoid transitions to avoid sharp discontinuities
    
    The reward uses normalized entropy h ∈ [0,1] where entropy is normalized by log(vocab_size).
    
    Piecewise reward structure:
    - For correct predictions (z>0): Full reward when h ≤ low_cutoff, linearly decreasing to 0 at high_cutoff_correct
    - For wrong predictions (z=0): Zero/penalty when h ≤ low_cutoff, linearly increasing to full reward at high_cutoff_wrong
    
    Args:
        predict_str: Generated text containing answer
        ground_truth: Ground truth answer
        format_score: Weight for format reward component (add up to 1 with acc reward)
        cde_weight: Weight (strength, max reward) for CDE reward component (w in the formulas)
        low_entropy_cutoff: Cutoff 'a' for confident decisions (default: 0.15)
        high_entropy_cutoff_correct: Cutoff 'b' for correct but mushy predictions (default: 0.60)  
        high_entropy_cutoff_wrong: Cutoff 'd' for wrong but uncertain predictions (default: 0.85)
        confidence_penalty_ratio: Penalty ratio ρ for confidently wrong predictions (default: 0.0)
        use_sigmoid_smoothing: Whether to use smooth sigmoid instead of piecewise linear (default: False)
        sigmoid_steepness: Steepness parameter k for sigmoid smoothing (default: 8.0)
        top_k_logprobs: Token logprobs for CDE computation
        top_k_token_ids: Token IDs for CDE computation
        tokenizer: Tokenizer for converting IDs to tokens
        
    Returns:
        Dictionary with detailed score breakdown including new CDE analysis
    """
    # Compute base rewards using fine-grained functions from meme_fg.py
    accuracy_reward = acc_reward(predict_str, ground_truth)
    format_reward_val = format_reward(predict_str)
    is_correct = accuracy_reward > 0.0  # For fine-grained, any partial match counts as "correct"
    
    # Compute detailed CDE analysis if data is available
    cde_reward = 0.0
    entropy = None
    normalized_entropy = None
    vocab_info = None
    has_cde_data = (top_k_logprobs is not None and 
                    top_k_token_ids is not None and 
                    tokenizer is not None)
    
    if has_cde_data:
        try:
            # Full vocabulary mode only: Extract full vocab entropy
            full_entropy_result = extract_full_vocab_entropy_from_answer_tags(
                predict_str, ground_truth, top_k_logprobs, top_k_token_ids, tokenizer
            )
            
            if full_entropy_result is not None:
                entropy, vocab_info = full_entropy_result
                
                # Normalize full vocab entropy to [0,1] range
                # For full vocab, we use a practical normalization based on effective vocab size
                # or use log of top-k size as approximation
                max_entropy_approx = math.log(vocab_info.get('num_tokens', 50))  # Use actual top-k size
                normalized_entropy = min(1.0, entropy / max_entropy_approx) if max_entropy_approx > 0 else 0.0
                
                #print(f"[CDE DEBUG] Full vocab entropy: {entropy:.3f}, normalized: {normalized_entropy:.3f}")
                
            else:
                print("[CDE DEBUG] No valid full vocab entropy extracted")
            
            # Compute piecewise CDE reward based on correctness and entropy
            if normalized_entropy is not None:
                if use_sigmoid_smoothing:
                    # Smooth sigmoid version
                    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-sigmoid_steepness * x))
                    
                    if is_correct:
                        # Reward confident correct predictions: sigmoid(k(a-h))
                        cde_reward = cde_weight * sigmoid(sigmoid_steepness * (low_entropy_cutoff - normalized_entropy))
                    else:
                        # Reward uncertain wrong predictions: sigmoid(k(h-a))
                        base_reward = cde_weight * sigmoid(sigmoid_steepness * (normalized_entropy - low_entropy_cutoff))
                        if confidence_penalty_ratio > 0 and normalized_entropy <= low_entropy_cutoff:
                            # Add penalty for confidently wrong
                            penalty = -confidence_penalty_ratio * cde_weight
                            cde_reward = penalty + base_reward
                        else:
                            cde_reward = base_reward
                else:
                    # Piecewise linear version (default)
                    if is_correct:
                        # Case 1: Correct prediction
                        if normalized_entropy <= low_entropy_cutoff:
                            # Very confident and correct: full reward
                            cde_reward = cde_weight
                        elif normalized_entropy < high_entropy_cutoff_correct:
                            # Linearly decreasing reward from low_cutoff to high_cutoff_correct
                            linear_factor = (high_entropy_cutoff_correct - normalized_entropy) / (high_entropy_cutoff_correct - low_entropy_cutoff)
                            cde_reward = cde_weight * linear_factor
                        else:
                            # Too uncertain even when correct: no reward
                            cde_reward = 0.0
                    else:
                        # Case 2: Wrong prediction  
                        if normalized_entropy <= low_entropy_cutoff:
                            # Confidently wrong: penalty or zero
                            cde_reward = -confidence_penalty_ratio * cde_weight if confidence_penalty_ratio > 0 else 0.0
                        elif normalized_entropy < high_entropy_cutoff_wrong:
                            # Linearly increasing reward from low_cutoff to high_cutoff_wrong
                            linear_factor = (normalized_entropy - low_entropy_cutoff) / (high_entropy_cutoff_wrong - low_entropy_cutoff)
                            base_reward = cde_weight * linear_factor
                            penalty = -confidence_penalty_ratio * cde_weight if confidence_penalty_ratio > 0 else 0.0
                            # Interpolate between penalty and base_reward
                            cde_reward = penalty + (base_reward - penalty) * linear_factor
                        else:
                            # Very uncertain when wrong: full reward (good epistemic behavior)
                            cde_reward = cde_weight
                            
            else:
                print("[CDE DEBUG] No valid entropy extracted, skipping CDE reward and assigning 0")
        except Exception as e:
            print(f"Error in detailed CDE computation: {e}")
    
    # Combine all components (CDE is additive to accuracy)
    accuracy_weight = 1.0 - format_score  # CDE is additive, doesn't reduce accuracy weight
    total_score = (
        accuracy_weight * accuracy_reward + 
        format_score * format_reward_val + 
        cde_reward
    )
    
    # Prepare detailed results for logging
    result = {
        "score": total_score,
        "accuracy_reward": accuracy_reward,
        "format_reward": format_reward_val,   
        "cde_reward": cde_reward,            
        "accuracy_weight": accuracy_weight,
        "format_weight": format_score,
        "cde_weight": cde_weight,
        "has_cde_data": has_cde_data,
        "is_correct": float(is_correct),
        # Full vocab entropy mode is always used for fine-grained classification
        "entropy_mode": "full",
        "low_entropy_cutoff": low_entropy_cutoff,
        "high_entropy_cutoff_correct": high_entropy_cutoff_correct,
        "high_entropy_cutoff_wrong": high_entropy_cutoff_wrong,
        "confidence_penalty_ratio": confidence_penalty_ratio,
        "use_sigmoid_smoothing": float(use_sigmoid_smoothing),
        "sigmoid_steepness": sigmoid_steepness,
    }
    
    # Add CDE-specific metrics for detailed analysis
    if entropy is not None and normalized_entropy is not None:
        result.update({
            "cde_entropy_raw": entropy,              # Raw entropy value in nats
            "cde_entropy_normalized": normalized_entropy,  # Normalized entropy in [0,1]
            "cde_confidence": 1.0 - normalized_entropy,    # Confidence as 1 - normalized_entropy
        })
    else:
        result.update({
            "cde_entropy_raw": 0.0,
            "cde_entropy_normalized": 0.0, 
            "cde_confidence": 0.0,
        })
        
    # Add full vocabulary specific metrics
    if vocab_info is not None:
        result.update({
            "full_vocab_num_tokens": vocab_info.get('num_tokens', 0),
            "full_vocab_max_prob": vocab_info.get('max_prob', 0.0),
            "full_vocab_min_prob": vocab_info.get('min_prob', 0.0),
            "full_vocab_effective_size": vocab_info.get('effective_vocab_size', 0.0),
        })
    else:
        # Add default values for missing full vocab metrics
        result.update({
            "full_vocab_num_tokens": 0,
            "full_vocab_max_prob": 0.0,
            "full_vocab_min_prob": 0.0,
            "full_vocab_effective_size": 0.0,
        })
    
    return result


def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.1, **kwargs) -> float:
    """
    Simple wrapper function for backward compatibility.
    Returns only the final score for basic usage.
    """
    result = compute_detailed_score(predict_str, ground_truth, format_score, **kwargs)
    return result["score"]
