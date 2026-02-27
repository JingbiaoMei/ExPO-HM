import math
import re
from typing import Dict, List, Optional, Tuple, Union

from mathruler.grader import extract_boxed_content, grade_answer

from .CDE_binary_for_reward import CDEComputer
from .CDE_full_vocab_for_reward import FullVocabCDEComputer


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    """
    Custom accuracy reward function based on exact match with answer tags.
    Expects answers to be either "yes" or "no" and uses <answer> tags for extraction.
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
        
        # Normalize by removing spaces and underscores
        extracted_ground_truth = extracted_ground_truth.replace(' ', '').replace('_', '')
        student_answer = student_answer.replace(' ', '').replace('_', '')
        
        # Check if student answer is empty
        if student_answer == "" or student_answer == " ":
            return 0.0

        # Check if answer is valid (only "yes" or "no" allowed)
        if not (student_answer == "yes" or student_answer == "no"):
            return 0.0
            
        # Compare the extracted answers (bidirectional containment check)
        if extracted_ground_truth in student_answer or student_answer in extracted_ground_truth:
            return 1.0
            
        return 0.0
        
    except Exception:
        return 0.0


def extract_decision_logits_from_answer_tags(
    predict_str: str,
    ground_truth: str,
    top_k_logprobs: Optional[List[List[float]]] = None,
    top_k_token_ids: Optional[List[List[int]]] = None,
    tokenizer=None
) -> Optional[Tuple[float, float]]:
    """
    Extract Yes/No decision logits from the answer portion of the generated text.
    
    This function identifies the <answer> tag location and extracts the logits
    for Yes/No tokens at the corresponding position in the generation.
    
    Args:
        predict_str: Generated text containing <answer>yes</answer> or <answer>no</answer>
        top_k_logprobs: List of top-k logprobs for each token position
        top_k_token_ids: List of top-k token IDs for each token position  
        tokenizer: Tokenizer to convert token IDs to strings
        
    Returns:
        Tuple of (yes_logit, no_logit) or None if extraction fails
    """
    if top_k_logprobs is None or top_k_token_ids is None or tokenizer is None:
        print("[LOGIT DEBUG] Missing required parameters")
        return None
    
    try:
        # Initialize CDE computer with standard Yes/No tokens
        cde_computer = CDEComputer()

        # Find the answer tag pattern in the generated text
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_tag_pattern, predict_str.lower())
        
        if not answer_match:
            print(f"[LOGIT DEBUG] No answer tag found in: {predict_str}")
            return None
        
        # Get the answer content (should be "yes" or "no")
        answer_content = answer_match.group(1).strip()
        #print(f"[LOGIT DEBUG] Found answer: '{answer_content}'")
        if answer_content not in ["yes", "no"]:
            print(f"[LOGIT DEBUG] Invalid answer content: '{answer_content}'")
            return None
            
        # First, we need to find the position of <answer> tag in the tokenized sequence
        # We'll reconstruct the text from tokens to find where <answer> appears
        answer_start_positions = []
        
        #print(f"[LOGIT DEBUG] Searching for <answer> tag position in {len(top_k_logprobs)} token positions")
        
        # Look for <answer> tag by reconstructing text and finding tag positions
        try:
            # For each position, get the most likely token to reconstruct the sequence
            likely_tokens = []
            for pos_idx, (pos_logprobs, pos_token_ids) in enumerate(zip(top_k_logprobs, top_k_token_ids)):
                if len(pos_logprobs) > 0:
                    # Take the most likely token (first in the list, highest logprob)
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
            #print(f"[LOGIT DEBUG] Reconstructed text snippet: {reconstructed_text[:100]}...")
            
            # Method 1: Find both opening and closing <answer> tags
            answer_start_match = re.search(r'<answer>', reconstructed_text.lower())
            answer_end_match = re.search(r'</answer>', reconstructed_text.lower())
            
            answer_search_positions = []
            
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
                    #print(f"[LOGIT DEBUG] Found <answer> content between token positions: {start_pos} to {end_pos-1}")
                    #print(f"[LOGIT DEBUG] Answer content positions: {answer_search_positions[:10]}...")
            
            # Method 2: If we found the answer text, try to locate it in the token sequence
            if not answer_search_positions and answer_content:
                print(f"[LOGIT DEBUG] Method 1 failed, trying to locate answer text '{answer_content}' in tokens")
                # Search for tokens that might represent the answer
                for pos_idx in range(max(0, len(top_k_logprobs) - 30), len(top_k_logprobs)):
                    try:
                        pos_token_ids = [top_k[pos_idx] for top_k in top_k_token_ids]
                        token_strings = tokenizer.convert_ids_to_tokens(pos_token_ids)
                        
                        # Check if any token contains our answer
                        for token_str in token_strings[:5]:  # Check top 5
                            if answer_content.lower() in token_str.lower() or token_str.lower() in answer_content.lower():
                                answer_search_positions.append(pos_idx)
                                print(f"[LOGIT DEBUG] Found potential answer token '{token_str}' at position {pos_idx}")
                                break
                    except:
                        continue
            
            # Fallback: if both methods failed, search near where we expect the answer
            if not answer_search_positions:
                print(f"[LOGIT DEBUG] Both methods failed, using fallback search")
                # Search in last 20% of sequence where answer is likely
                fallback_start = max(0, len(top_k_logprobs) - len(top_k_logprobs) // 5)
                answer_search_positions = list(range(fallback_start, len(top_k_logprobs)))
                print(f"[LOGIT DEBUG] Using fallback positions: {fallback_start} to {len(top_k_logprobs)}")
        
        except Exception as e:
            print(f"[LOGIT DEBUG] Error finding <answer> position: {e}")
            # Fallback: search in last 20% of sequence where answer is likely
            fallback_start = max(0, len(top_k_logprobs) - len(top_k_logprobs) // 5)
            answer_search_positions = list(range(fallback_start, len(top_k_logprobs)))
            print(f"[LOGIT DEBUG] Using fallback positions: {fallback_start} to {len(top_k_logprobs)}")
        
        # Now search for Yes/No tokens in the identified positions
        best_logits = None
        best_confidence = 0.0
        all_yes_tokens = {}  # position -> (token, logprob)
        all_no_tokens = {}   # position -> (token, logprob)
        
        # Use the positions we found from our dual extraction approach
        search_positions = answer_search_positions[:20]  # Limit to avoid excessive processing
        #print(f"[LOGIT DEBUG] Searching for Yes/No tokens in positions: {search_positions}")
        
        for pos_idx in search_positions:
            if pos_idx >= len(top_k_logprobs):
                continue
                
            pos_logprobs = top_k_logprobs[pos_idx]
            pos_token_ids = top_k_token_ids[pos_idx]
            
            if len(pos_logprobs) != len(pos_token_ids):
                continue
                
            # Convert token IDs to strings
            try:
                token_strings = tokenizer.convert_ids_to_tokens(pos_token_ids)
                #print(f"[LOGIT DEBUG] Position {pos_idx} token strings: {token_strings[:10]}")  # Show first 10 tokens
                
                # Show what we're looking for vs what we found
                #print(f"[LOGIT DEBUG] Position {pos_idx}: Looking for Yes tokens: {cde_computer.yes_tokens[:4]}...")
                #print(f"[LOGIT DEBUG] Position {pos_idx}: Looking for No tokens: {cde_computer.no_tokens[:4]}...")
                
            except:
                print(f"[LOGIT DEBUG] Error converting token IDs at position {pos_idx}")
                continue
            
            # Look for Yes/No tokens at this position
            yes_logit = float('-inf')
            no_logit = float('-inf')
            
            found_tokens = []
            for token_str, logprob in zip(token_strings, pos_logprobs):
                if token_str in cde_computer.yes_tokens:
                    yes_logit = max(yes_logit, logprob)
                    # Only store if this is the best token so far for this position
                    if pos_idx not in all_yes_tokens or logprob > all_yes_tokens[pos_idx][1]:
                        all_yes_tokens[pos_idx] = (token_str, logprob)
                    found_tokens.append(f"YES:{token_str}:{logprob:.3f}")
                elif token_str in cde_computer.no_tokens:
                    no_logit = max(no_logit, logprob)
                    # Only store if this is the best token so far for this position
                    if pos_idx not in all_no_tokens or logprob > all_no_tokens[pos_idx][1]:
                        all_no_tokens[pos_idx] = (token_str, logprob)
                    found_tokens.append(f"NO:{token_str}:{logprob:.3f}")
            
            if found_tokens:
                #print(f"[LOGIT DEBUG] Position {pos_idx}: {found_tokens}")
                # Show which tokens were stored as the best for this position
                if pos_idx in all_yes_tokens:
                    best_yes = all_yes_tokens[pos_idx]
                    #print(f"[LOGIT DEBUG] Position {pos_idx}: Best Yes token stored: {best_yes[0]}:{best_yes[1]:.3f}")
                if pos_idx in all_no_tokens:
                    best_no = all_no_tokens[pos_idx] 
                    #print(f"[LOGIT DEBUG] Position {pos_idx}: Best No token stored: {best_no[0]}:{best_no[1]:.3f}")
            else:
                #print(f"[LOGIT DEBUG] Position {pos_idx}: No Yes/No tokens found in top-k")
                
                # Check for any tokens that might be "no" or "yes" related but not in our exact list
                potential_tokens = []
                for token_str, logprob in zip(token_strings, pos_logprobs):
                    token_lower = token_str.lower().strip()
                    if 'no' in token_lower or 'yes' in token_lower:
                        potential_tokens.append(f"{token_str}:{logprob:.3f}")
                
                if potential_tokens:
                    print(f"[LOGIT DEBUG] Position {pos_idx}: Potential Yes/No-like tokens: {potential_tokens[:5]}")
                #else:
                    #print(f"[LOGIT DEBUG] Position {pos_idx}: No Yes/No-like tokens found either")
                    
            # Check if we found meaningful Yes/No logits AT THE SAME POSITION
            if yes_logit > float('-inf') and no_logit > float('-inf'):
                # Calculate confidence as the difference between max and min
                confidence = abs(yes_logit - no_logit)
                #print(f"[LOGIT DEBUG] Position {pos_idx}: yes={yes_logit:.3f}, no={no_logit:.3f}, conf={confidence:.3f}")
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_logits = (yes_logit, no_logit)
        
        # If we didn't find both at the same position, try to find them at different positions
        if best_logits is None:
            #print(f"[LOGIT DEBUG] No same-position match, trying cross-position matching")
            #print(f"[LOGIT DEBUG] Found Yes tokens at positions: {list(all_yes_tokens.keys())}")
            #print(f"[LOGIT DEBUG] Found No tokens at positions: {list(all_no_tokens.keys())}")
            
            # Strategy 1: Find the best Yes and best No across all positions
            if len(all_yes_tokens) > 0 and len(all_no_tokens) > 0:
                best_yes_logit = max(logprob for _, logprob in all_yes_tokens.values())
                best_no_logit = max(logprob for _, logprob in all_no_tokens.values())
                
                confidence = abs(best_yes_logit - best_no_logit)
                #print(f"[LOGIT DEBUG] Cross-position: yes={best_yes_logit:.3f}, no={best_no_logit:.3f}, conf={confidence:.3f}")
                best_logits = (best_yes_logit, best_no_logit)
                best_confidence = confidence
                
            # Strategy 2: If we only have one type, synthesize the missing one
            elif len(all_yes_tokens) > 0:
                best_yes_logit = max(logprob for _, logprob in all_yes_tokens.values())
                # Synthesize a reasonable no_logit (assume low probability for the opposite)
                synthetic_no_logit = best_yes_logit - 8.0  # Reasonable gap for confident decision
                confidence = abs(best_yes_logit - synthetic_no_logit)
                #print(f"[LOGIT DEBUG] Yes-only synthesis: yes={best_yes_logit:.3f}, no={synthetic_no_logit:.3f} (synthetic), conf={confidence:.3f}")
                best_logits = (best_yes_logit, synthetic_no_logit)
                best_confidence = confidence
                
            elif len(all_no_tokens) > 0:
                best_no_logit = max(logprob for _, logprob in all_no_tokens.values())
                # Synthesize a reasonable yes_logit (assume low probability for the opposite)
                synthetic_yes_logit = best_no_logit - 8.0  # Reasonable gap for confident decision
                confidence = abs(synthetic_yes_logit - best_no_logit)
                #print(f"[LOGIT DEBUG] No-only synthesis: yes={synthetic_yes_logit:.3f} (synthetic), no={best_no_logit:.3f}, conf={confidence:.3f}")
                best_logits = (synthetic_yes_logit, best_no_logit)
                best_confidence = confidence
        
        #print(f"[LOGIT DEBUG] Best logits: {best_logits}, confidence: {best_confidence:.3f}")
        return best_logits
        
    except Exception as e:
        print(f"[LOGIT DEBUG] Error extracting decision logits: {e}")
        import traceback
        traceback.print_exc()
        return None



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
        predict_str: Generated text containing <answer>yes</answer> or <answer>no</answer>
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
        
        # Get the answer content (should be "yes" or "no")
        answer_content = answer_match.group(1).strip()
        #print(f"[FULL VOCAB DEBUG] Found answer: '{answer_content}'")
        if answer_content not in ["yes", "no"]:
            print(f"[FULL VOCAB DEBUG] Invalid answer content: '{answer_content}'")
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


def compute_detailed_score_legacy(
    predict_str: str, 
    ground_truth: str, 
    format_score: float = 0.1,
    cde_weight: float = 0.1,
    top_k_logprobs: Optional[List[List[float]]] = None,
    top_k_token_ids: Optional[List[List[int]]] = None,
    tokenizer=None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    LEGACY FUNCTION - ARCHIVED FOR REFERENCE
    
    This is the original compute_detailed_score function that used simple entropy-based CDE.
    It has been replaced with a new piecewise CDE reward strategy that provides better
    training dynamics by rewarding confidence when correct and uncertainty when wrong.
    
    This function is kept for reference and backward compatibility but is no longer
    actively used in training.
    
    Original functionality:
    - Computed detailed reward scores with breakdown for analysis and logging
    - Used simple negative entropy scaling for CDE reward
    
    Returns:
        Dictionary with detailed score breakdown including CDE analysis
    """
    # Compute base rewards
    accuracy_reward = acc_reward(predict_str, ground_truth)
    format_reward_val = format_reward(predict_str)
    
    # Compute detailed CDE analysis if data is available
    cde_reward = 0.0
    entropy = None
    decision_logits = None
    has_cde_data = (top_k_logprobs is not None and 
                    top_k_token_ids is not None and 
                    tokenizer is not None)
    
    if has_cde_data:
        try:
            # Extract decision logits
            decision_logits = extract_decision_logits_from_answer_tags(
                predict_str, ground_truth, top_k_logprobs, top_k_token_ids, tokenizer
            )
            
            if decision_logits is not None:
                yes_logit, no_logit = decision_logits
                
                # Compute entropy
                cde_computer = CDEComputer()
                entropy = cde_computer.compute_binary_entropy_from_logits(yes_logit, no_logit)
                cde_reward = -cde_weight * entropy
            else:
                print("[CDE DEBUG] No valid decision logits extracted, skipping CDE reward and assigning 0")
        except Exception as e:
            print(f"Error in detailed CDE computation: {e}")
    
    # Combine all components
    accuracy_weight = 1.0 - format_score  # CDE is additive, doesn't reduce accuracy weight
    total_score = (
        accuracy_weight * accuracy_reward + 
        format_score * format_reward_val + 
        cde_reward
    )
    
    # Prepare detailed results for logging
    result = {
        "score": total_score,
        "accuracy_reward": accuracy_reward,  # Will be logged as val-aux/meme_cde/accuracy_reward/mean@N
        "format_reward": format_reward_val,   # Will be logged as val-aux/meme_cde/format_reward/mean@N
        "cde_reward": cde_reward,            # Will be logged as val-aux/meme_cde/cde_reward/mean@N
        "accuracy_weight": accuracy_weight,
        "format_weight": format_score,
        "cde_weight": cde_weight,
        "has_cde_data": has_cde_data
    }
    
    # Add CDE-specific metrics for detailed analysis (always include to maintain consistent lengths)
    if entropy is not None:
        result.update({
            "cde_entropy": entropy,              # Raw entropy value
            "cde_confidence": -entropy,          # Inverted entropy as confidence measure
        })
    else:
        # Add default values when entropy is not available to maintain consistent lengths
        result.update({
            "cde_entropy": 0.0,                 # Default entropy when no valid decision found
            "cde_confidence": 0.0,              # Default confidence when no valid decision found
        })
        
    if decision_logits is not None:
        yes_logit, no_logit = decision_logits
        result.update({
            "yes_logit": yes_logit,             # Raw yes token logit
            "no_logit": no_logit,               # Raw no token logit
            "logit_difference": abs(yes_logit - no_logit),  # Confidence measure
        })
    else:
        # Add default values when decision logits are not available to maintain consistent lengths
        result.update({
            "yes_logit": 0.0,                  # Default yes logit when no valid decision found
            "no_logit": 0.0,                   # Default no logit when no valid decision found
            "logit_difference": 0.0,           # Default logit difference when no valid decision found
        })
    
    return result


def compute_detailed_score(
    predict_str: str, 
    ground_truth: str, 
    format_score: float = 0.1,
    cde_weight: float = 0.2,
    # Entropy computation mode
    entropy_mode: str = "binary",                       # "binary" or "full" - type of entropy to compute
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
    Compute detailed reward scores using improved piecewise CDE reward strategy.
    
    This function implements a sophisticated CDE reward that:
    - Rewards confidence when the prediction is correct
    - Rewards uncertainty when the prediction is wrong  
    - Optionally penalizes overconfident wrong predictions
    - Supports smooth sigmoid transitions to avoid sharp discontinuities
    
    The reward uses normalized entropy h ∈ [0,1] where h = H(d|r,x) / ln(2) for binary classification.
    
    Piecewise reward structure:
    - For correct predictions (z=1): Full reward when h ≤ low_cutoff, linearly decreasing to 0 at high_cutoff_correct
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
    # Compute base rewards
    accuracy_reward = acc_reward(predict_str, ground_truth)
    format_reward_val = format_reward(predict_str)
    is_correct = accuracy_reward > 0.5  # Convert to boolean
    
    # Compute detailed CDE analysis if data is available
    cde_reward = 0.0
    entropy = None
    normalized_entropy = None
    decision_logits = None
    has_cde_data = (top_k_logprobs is not None and 
                    top_k_token_ids is not None and 
                    tokenizer is not None)
    
    if has_cde_data:
        try:
            if entropy_mode == "binary":
                # Binary mode: Extract yes/no decision logits
                decision_logits = extract_decision_logits_from_answer_tags(
                    predict_str, ground_truth, top_k_logprobs, top_k_token_ids, tokenizer
                )
                
                if decision_logits is not None:
                    yes_logit, no_logit = decision_logits
                    
                    # Compute entropy and normalize to [0,1] range
                    cde_computer = CDEComputer()
                    entropy = cde_computer.compute_binary_entropy_from_logits(yes_logit, no_logit)
                    # For binary classification, max entropy is ln(2) ≈ 0.6931
                    normalized_entropy = entropy / 0.6931  # Now in [0,1] range
                    
                else:
                    print("[CDE DEBUG] No valid decision logits extracted in binary mode")
                    
            elif entropy_mode == "full":
                # Full vocabulary mode: Extract full vocab entropy
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
                    
            else:
                raise ValueError(f"Invalid entropy_mode: '{entropy_mode}'. Must be 'binary' or 'full'.")
            
            # Compute piecewise CDE reward based on correctness and entropy (same for both modes)
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
                print("[CDE DEBUG] No valid decision logits extracted, skipping CDE reward and assigning 0")
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
        # New piecewise CDE hyperparameters for logging
        "entropy_mode": entropy_mode,
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
        
    # Add mode-specific metrics
    if entropy_mode == "binary" and 'decision_logits' in locals() and decision_logits is not None:
        yes_logit, no_logit = decision_logits
        result.update({
            "yes_logit": yes_logit,
            "no_logit": no_logit,
            "logit_difference": abs(yes_logit - no_logit),
        })
    elif entropy_mode == "full" and 'vocab_info' in locals() and vocab_info is not None:
        result.update({
            "full_vocab_num_tokens": vocab_info.get('num_tokens', 0),
            "full_vocab_max_prob": vocab_info.get('max_prob', 0.0),
            "full_vocab_min_prob": vocab_info.get('min_prob', 0.0),
            "full_vocab_effective_size": vocab_info.get('effective_vocab_size', 0.0),
        })
    else:
        # Add default values for missing mode-specific metrics
        if entropy_mode == "binary":
            result.update({
                "yes_logit": 0.0,
                "no_logit": 0.0,
                "logit_difference": 0.0,
            })
        else:  # full mode
            result.update({
                "full_vocab_num_tokens": 0,
                "full_vocab_max_prob": 0.0,
                "full_vocab_min_prob": 0.0,
                "full_vocab_effective_size": 0.0,
            })
    return result

