# Reproducibility Notes (Open-Source)

This document defines which files/prompts are considered canonical for reproducing paper results.

Note: this repo uses newer versions of [verl](https://github.com/volcengine/verl) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) than were used during the initial development, so reproduced results may not exactly match the paper numbers.

## Scope

Included datasets in this release:

- FB
- MAMI
- PrideMM


## Terminology Mapping (Paper vs Code)

This section clarifies naming differences only. These are aliases, not different methods unless explicitly noted.

Notes:

- In this repository, `FB` is shorthand for the Facebook Hateful Memes dataset split/files used by the GRPO/SFT pipelines.
- Paper wording may say “fine-grained” while scripts may use `fg`; these refer to the same task family.

### 1) SFT Policy-Manual Data

SFT reads datasets from `sft/lmfactory_data/dataset_info.sft_release.json`.

For fine-grained SFT, the datasets are the `*_choice` JSON files:

- `data/gt/FB/sharegpt_FB_train_pc_choice.json`
- `data/gt/FB/sharegpt_FB_train_attack_choice.json`
- `data/gt/MAMI/sharegpt_MAMI_train_attack_choice.json`
- `data/gt/PrideMM/sharegpt_PrideMM_train_stance_choice.json`
- `data/gt/PrideMM/sharegpt_PrideMM_train_target_choice.json`

These reflect the policy-manual prompt style used for SFT training. The GRPO data generation scripts read from the original `train.jsonl` files, not these `*_choice` files, so the GRPO prompts do not use the policy-manual style. 

### 2) GRPO Data

GRPO data are generated in conversion code, not loaded from SFT sharegpt JSON.

Prompt sources:

- `scripts/grpo/data_prep/hatefulmemes.py`

GRPO Training scripts consume generated parquet files under `data/verl/`.

### 3) GRPO Data Generation Mapping

All commands are run from repo root.

| Script | Raw input files | Generated output |
|---|---|---|
| `scripts/grpo/data_prep/generate_hatefulmemes.sh` | `data/gt/FB/{train.jsonl,test_seen.jsonl}`, `data/gt/MAMI/{train.jsonl,test.jsonl}`, `data/gt/PrideMM/{train.jsonl,test.jsonl}` | `data/verl/FB/{train,test}.parquet`, `data/verl/MAMI/{train,test}.parquet`, `data/verl/PrideMM/{train,test}.parquet` |
| `scripts/grpo/data_prep/generate_cde_all.sh` | `data/gt/FB/train.jsonl`, `data/gt/MAMI/train.jsonl`, `data/gt/PrideMM/train.jsonl`, `data/gt/fine_grained_hateful_memes/{train.json,dev_seen.json}` | `data/verl/CDE/FB/train.parquet`, `data/verl/CDE/MAMI/train.parquet`, `data/verl/CDE/PrideMM/train.parquet`, `data/verl/CDE/fb-fine-grained-pc/{train,test}.parquet`, `data/verl/CDE/fb-fine-grained-attack/{train,test}.parquet`, merged train set `data/verl/CDE/fb-fine-grained-combined/train.parquet` |

The merged data can be used for the Curriculum Learning stage of GRPO.


### 4) Checkpoint Merging (SFT and GRPO)

#### SFT (LoRA adapter -> merged HF checkpoint for GRPO init)

If you want to initialize GRPO from SFT, point `actor_rollout_ref.model.path` to a **merged HF checkpoint directory**.

If you trained SFT with LoRA (e.g., `scripts/sft/qwen25vl/qwen25vl_7b_lora.yaml`), you can merge/export the LoRA adapter into a full model via LLaMA-Factory:

```bash
llamafactory-cli export scripts/sft/qwen25vl/qwen25vl_7b_merge_lora.yaml
```

This produces:

- LoRA adapter: `checkpoints/fb/qwen2_5vl_7b/sft/`
- Merged HF model: `checkpoints/fb/qwen2_5vl_7b/sft_merged/` (use as GRPO `actor_rollout_ref.model.path`, e.g. via `MODEL_PATH` in `scripts/grpo/train/run_qwen2_5_vl-7b_cde_paper.sh`)

#### GRPO (FSDP actor shards -> HF merged directory)

GRPO training (verl + FSDP) saves sharded checkpoints like:

- `./checkpoints/<project>/<experiment>/global_step_*/actor/`

For vLLM/HF-style inference, merge an `actor/` directory into a single directory:

```bash
# Merge latest checkpoint under CKPT_ROOT
CKPT_ROOT=./checkpoints/<project>/<experiment> \
  bash scripts/grpo/tools/merge_actor_checkpoint.sh

# Or merge a specific global_step
bash scripts/grpo/tools/merge_actor_checkpoint.sh ./checkpoints/<project>/<experiment> 180
```

This writes `./checkpoints/<project>/<experiment>/checkpoint-<step>-merged/`, which is what `scripts/grpo/eval/*.sh` expects by default.
