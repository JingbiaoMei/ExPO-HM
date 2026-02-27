# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the HatefulMemes dataset to parquet format
"""

import argparse
import json
import math
import os

import datasets
from PIL import Image
from tqdm import tqdm


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} and width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="./data/gt/")
    parser.add_argument("--dataset_name", default="FB")
    parser.add_argument("--split", default="train.jsonl test_seen.jsonl")
    parser.add_argument("--image_dir", default="./data/image")
    parser.add_argument("--output_dir", default="./data/verl/")
    parser.add_argument("--max_pixels", type=int, default=448 *
                        448, help="Maximum number of pixels for images")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--reward_type", type=str, default=None)
    args = parser.parse_args()
    if not "fine-grained" in args.dataset_name.lower():
        # fine grain need special path
        dataset_dir = os.path.join(args.base_dir, args.dataset_name)
        image_dir = os.path.join(args.image_dir, args.dataset_name)
    else:
        if "fb" in args.dataset_name.lower():
            dataset_dir = os.path.join(
                args.base_dir, "fine_grained_hateful_memes")
            image_dir = os.path.join(args.image_dir, "FB")
        elif "mami" in args.dataset_name.lower():
            dataset_dir = os.path.join(
                args.base_dir, "MAMI")
            image_dir = os.path.join(args.image_dir, "MAMI")
        elif "pridemm" in args.dataset_name.lower():
            dataset_dir = os.path.join(
                args.base_dir, "PrideMM")
            image_dir = os.path.join(args.image_dir, "PrideMM")
    # Check if both dataset_dir and image_dir exist
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' not found.")
    
    
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    image_dir = os.path.join(image_dir, "All")

    # Process splits and create datasets

    def process_split(split_name):
        instruction_following = (
            r"Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The answer should be Yes or No, if the meme is hateful than output Yes ; if the meme is Benign than output No. "
            r"The output format should be as follows:\n<think> ... </think> <answer>...</answer>\n. Please strictly follow the format."
        )
        split_path = os.path.join(dataset_dir, f"{split_name}")
        entries = []

        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found, skipping...")
            return None

        with open(split_path, 'r') as f:
            for idx, line in enumerate(tqdm(f, desc=f"Processing {split_name}",
                                            total=sum(1 for _ in open(split_path)))):
                data = json.loads(line)
                original_id = data.get("id")
                # Note data_source controlls the reward model used in verl
                data_source = "meme"
                # Override data_source if reward_type is specified
                if args.reward_type is not None:
                    data_source = args.reward_type

                # Remove 'img/' prefix if present
                if args.dataset_name == "FB" or args.dataset_name.lower() == "fb-fine-grained-pc" or args.dataset_name.lower() == "fb-fine-grained-attack":
                    img_filename = data.get("img", "")
                    img_filename = img_filename.replace("img/", "")

                elif args.dataset_name.lower() == "harmeme" or args.dataset_name.lower() == "harmc" or args.dataset_name.lower() == "harmp":
                    img_filename = original_id + ".png"
                elif args.dataset_name.lower() == "harmeme-fine-grained":
                    img_filename = original_id + ".png"
                elif args.dataset_name.lower() == "mami" or args.dataset_name.lower() == "mami-fine-grained-attack":
                    img_filename = original_id + ".jpg"
                elif args.dataset_name.lower() == "pridemm" or "pridemm-fine-grained" in args.dataset_name.lower():
                    img_filename = original_id + ".png"
                else:
                    raise ValueError(
                        f"Unsupported dataset name: {args.dataset_name}")
                img_path = os.path.join(image_dir, img_filename)

                try:
                    # Load image
                    image = Image.open(img_path).convert("RGB")

                    if args.max_pixels:
                        # Get original image dimensions
                        width, height = image.size

                        # Calculate resized dimensions using smart_resize
                        resized_height, resized_width = smart_resize(
                            height,
                            width,
                            factor=28,  # patch_size * merge_size equivalent
                            min_pixels=56 * 56,  # minimum pixels
                            max_pixels=args.max_pixels,
                        )

                        # Resize the image
                        image = image.resize(
                            (resized_width, resized_height), Image.LANCZOS)

                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    continue

                # Create question and answer
                if args.dataset_name == "FB":
                    question = "<image>Is it hateful?"
                elif args.dataset_name.lower() == "harmeme" or args.dataset_name.lower() == "harmc" or args.dataset_name.lower() == "harmp":
                    question = "<image>Is it harmful?"
                elif args.dataset_name.lower() == "mami":
                    question = "<image>Is it a misogynistic meme?"
                elif args.dataset_name.lower() == "pridemm":
                    question = "<image>Is it a hateful meme towards LGBTQ+?"
                # Fine-grained datasets
                elif "fine-grained" in args.dataset_name.lower():
                    data_source = "meme_fg"
                    # Override data_source if reward_type is specified
                    if args.reward_type is not None:
                        data_source = args.reward_type
                    instruction_following = (
                        r"Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. "
                        r"The output format should be as follows:\n<think> ... </think> <answer>...</answer>\n. Please strictly follow the format."
                    )
                    if args.dataset_name.lower() == "fb-fine-grained-attack":
                        question = """<image>Does this meme use any of the following types of attack against a group?
                        Dehumanizing, Inferiority, Inciting violence, Mocking, Contempt, Slurs, Exclusion
                        Please respond with one or more of the attack types if applicable. 
                        If the meme does not contain any hateful content, respond with Benign.
                        """

                    elif args.dataset_name.lower() == "fb-fine-grained-pc":
                        question = """<image>Does this meme express hate or negative bias toward any of the following protected groups?
                        Religion, Race, Sex, Nationality, Disability
                        Please respond with one or more of the protected categories if applicable.
                        If the meme does not contain any hateful content, respond with Benign.
                        """
                    elif args.dataset_name.lower() == "mami-fine-grained-attack":
                        question = """<image>Does this meme use any of the following types of attack against a group?
                        objectification, shaming, stereotype, violence 
                        Please respond with one or more of the attack types if applicable.     
                        If the meme does not contain any hateful content, respond with Benign.
                        """
                    elif args.dataset_name.lower() == "pridemm-fine-grained-stance":
                        question = """<image>Based on the content and context of this meme, what is the stance towards LGBTQ+ individuals or communities?
                        Choose from the following options:
                        neutral, support, oppose
                        What is the stance?
                        """
                    elif args.dataset_name.lower() == "pridemm-fine-grained-target":
                        question = """<image>Based on the content and context of this meme, Does this meme express hate or negative bias toward any of the following targets?
                        Choose from the following options:
                        undirected individual community organization
                        If the meme does not contain any hateful content, respond with Benign.
                        What is the target type?
                        """
                    else:
                        raise ValueError(
                            f"Unsupported fine-grained dataset name: {args.dataset_name}")

                else:
                    raise ValueError(
                        f"Unsupported dataset name: {args.dataset_name}")
                prompt = question + " " + instruction_following

                # Get the label and convert to answer format
                if args.dataset_name == "FB":
                    label = data.get("label", 0)

                    answer = "Yes" if int(label) == 1 else "No"
                elif args.dataset_name.lower() == "harmeme" or args.dataset_name.lower() == "harmc" or args.dataset_name.lower() == "harmp":
                    label = data.get("labels")
                    if not label:
                        print(
                            f"Warning: No labels found for {img_filename}, skipping...")
                        continue
                    if "not harmful" in label:
                        answer = "No"
                    elif "somewhat harmful" in label or "very harmful" in label:
                        answer = "Yes"
                    else:
                        print(
                            f"Warning: Unexpected labels {label} for {img_filename}, skipping...")
                        continue
                elif args.dataset_name.lower() == "pridemm":
                    label = data.get("label", 0)

                    answer = "Yes" if int(label) == 1 else "No"
                elif args.dataset_name.lower() == "mami":
                    label = data.get("label", 0)

                    answer = "Yes" if int(label) == 1 else "No"
                elif "fine-grained" in args.dataset_name.lower() or "fg" in args.dataset_name.lower():
                    label = "dummy"
                    if args.dataset_name.lower() == "fb-fine-grained-attack":

                        gold_attack_lists = data["gold_attack"]

                        if len(gold_attack_lists) == 1:
                            if gold_attack_lists[0] == "attack_empty":
                                answer = "Benign"
                            else:
                                answer = gold_attack_lists[0]
                        else:
                            # Join the attack types with commas
                            answer = ", ".join(gold_attack_lists)
                    elif args.dataset_name.lower() == "fb-fine-grained-pc":

                        gold_protected_lists = data["gold_pc"]

                        if len(gold_protected_lists) == 1:
                            if gold_protected_lists[0] == "pc_empty":
                                answer = "Benign"
                            else:
                                answer = gold_protected_lists[0]
                        else:
                            # Join the protected categories with commas
                            answer = ", ".join(gold_protected_lists)
                    elif args.dataset_name.lower() == "mami-fine-grained-attack":
                        
                        gold_attack_lists = data["gold_attack"]

                        if len(gold_attack_lists) == 1:
                            
                            if gold_attack_lists[0] == "attack_empty":
                                answer = "Benign"
                           
                        else:
                            # Join the attack types with commas
                            answer = ", ".join(gold_attack_lists)
                    elif args.dataset_name.lower() == "pridemm-fine-grained-stance":
                        gold_stance_lists = data["gold_stance"]
                        if len(gold_stance_lists) == 1:
                            answer = gold_stance_lists[0]
                        else:
                            # Join the stance types with commas
                            print("Warning: Multiple stances found, joining with commas. This should not happen. check data!")
                            answer = ", ".join(gold_stance_lists)
                    elif args.dataset_name.lower() == "pridemm-fine-grained-target":
                        gold_target_lists = data["gold_target"]
                        if len(gold_target_lists) == 1:
                            if gold_target_lists[0].lower() == "benign":
                                answer = "Benign"
                            else:
                                answer = gold_target_lists[0]
                        else:
                            # Join the target types with commas
                            print("Warning: Multiple targets found, joining with commas. This should not happen. check data!")
                            answer = ", ".join(gold_target_lists)
                entry = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "images": [image],
                    "ability": "meme_understanding",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "split": split_name,
                        "index": idx,
                        "answer": answer,
                        "question": question,
                        "original_label": label,
                    },
                }
                entries.append(entry)

        return entries

    # Process train and test splits
    splits = args.split.split()
    train_entries = []
    test_entries = []

    for split in splits:
        entries = process_split(split)

        if entries:
            if "train" in split:
                train_entries.extend(entries)
            else:
                test_entries.extend(entries)

    # Create HuggingFace datasets
    if train_entries:
        train_dataset = datasets.Dataset.from_list(train_entries)
        print(f"Created train dataset with {len(train_dataset)} entries")

    if test_entries:
        test_dataset = datasets.Dataset.from_list(test_entries)
        print(f"Created test dataset with {len(test_dataset)} entries")

    # Save datasets to parquet
    os.makedirs(output_dir, exist_ok=True)

    if train_entries:
        train_output_path = os.path.join(output_dir, "train.parquet")
        train_dataset.to_parquet(train_output_path)
        print(f"Saved train dataset to {train_output_path}")

    if test_entries:
        test_output_path = os.path.join(output_dir, "test.parquet")
        test_dataset.to_parquet(test_output_path)
        print(f"Saved test dataset to {test_output_path}")
