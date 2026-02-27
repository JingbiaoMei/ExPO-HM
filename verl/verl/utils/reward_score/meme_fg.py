import re
from typing import Dict, Union

from mathruler.grader import extract_boxed_content, grade_answer


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


def compute_score(
    predict_str: str, 
    ground_truth: str, 
    format_score: float = 0.1,
    **kwargs
) -> Dict[str, Union[float, str]]:
    """
    Compute detailed reward scores for fine-grained meme classification with logging.
    
    This is the baseline version (CDE-free) that provides detailed score breakdown
    for logging and analysis purposes.
    
    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth answer (can contain multiple categories)
        format_score: Weight for format reward component (default: 0.1)
        **kwargs: Additional parameters (for compatibility with CDE versions)
    
    Returns:
        Dictionary with detailed score breakdown including:
        - score: Final weighted score
        - accuracy_reward: Fine-grained accuracy component
        - format_reward: Format compliance component
        - accuracy_weight: Weight applied to accuracy component
        - format_weight: Weight applied to format component
        - is_correct: Boolean indicator for correctness (>0 for partial matches)
    """
    # Compute individual reward components
    accuracy_reward_val = acc_reward(predict_str, ground_truth)
    format_reward_val = format_reward(predict_str)
    is_correct = accuracy_reward_val > 0.0  # For fine-grained, any partial match counts as correct
    
    # Compute weighted total score
    accuracy_weight = 1.0 - format_score
    total_score = accuracy_weight * accuracy_reward_val + format_score * format_reward_val
    
    # Create detailed result dictionary with logging metrics
    result = {
        "score": total_score,
        "accuracy_reward": accuracy_reward_val,
        "format_reward": format_reward_val,
        "accuracy_weight": accuracy_weight,
        "format_weight": format_score,
        "is_correct": float(is_correct),  # Store as float for consistency with CDE versions
    }
    
    return result



