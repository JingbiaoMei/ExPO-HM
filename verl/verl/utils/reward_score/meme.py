import re

from mathruler.grader import extract_boxed_content, grade_answer


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


# Add place for additional kwargs like cde_weight, format_score
def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.1, **kwargs) -> dict:
    """
    Compute reward score with detailed logging for format and accuracy rewards.
    
    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer
        format_score: Weight for format reward component (default: 0.1)
        **kwargs: Additional parameters (for compatibility)
    
    Returns:
        dict: Detailed results including score breakdown and logging metrics
    """
    # Compute individual reward components
    format_reward_val = format_reward(predict_str)
    accuracy_reward_val = acc_reward(predict_str, ground_truth)
    
    # Compute weighted total score
    accuracy_weight = 1.0 - format_score
    total_score = accuracy_weight * accuracy_reward_val + format_score * format_reward_val
    
    # Create detailed result dictionary with logging
    result = {
        "score": total_score,
        "accuracy_reward": accuracy_reward_val,
        "format_reward": format_reward_val,
        "accuracy_weight": accuracy_weight,
        "format_weight": format_score,
        "is_correct": accuracy_reward_val,  # For binary tasks, accuracy reward indicates correctness
    }
    
    return result
