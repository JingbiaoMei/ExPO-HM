import argparse
import os

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge FSDP checkpoints")
    parser.add_argument("--local_dir", type=str, required=True, help="Local directory containing FSDP checkpoints")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory to save the merged model")

    args = parser.parse_args()

    # Ensure target directory exists
    os.makedirs(args.target_dir, exist_ok=True)

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=args.local_dir,
        target_dir=args.target_dir,
        hf_model_config_path=os.path.join(args.local_dir, "huggingface"),
    )
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
