# Repository Layout

This page describes where code, scripts, and evaluation utilities live.

## Top-Level Directories

- `sft/`: SFT code and LLaMA-Factory integration.
- `scripts/grpo/`: GRPO train/data-prep/merge/eval scripts.
- `scripts/sft/`: SFT shell entrypoints.
- `eval/`: inference and reasoning-evaluation code.
- `data/`: source metadata, images, and generated parquet.
- `docs/`: reproducibility and layout documentation.
- `paper/`: paper assets and release-related materials.
- `verl/`: upstream `verl` snapshot used during migration.

## GRPO Lane

- Data conversion: `scripts/grpo/data_prep/`
- Runtime training scripts: `scripts/grpo/train/`
- Merge helpers: `scripts/grpo/tools/merge_actor_checkpoint.sh`, `scripts/grpo/tools/`

Main GRPO run scripts:

- `scripts/grpo/train/run_qwen2_5_vl-7b-baseline.sh` (baseline)
- `scripts/grpo/train/run_qwen2_5_vl-7b_cde_paper.sh` (paper-aligned CDE)

## SFT Lane

- SFT scripts: `scripts/sft/`
- Dataset registry: `sft/lmfactory_data/dataset_info.sft_release.json`

## Evaluation

- Inference utilities: `eval/inference/`
- LLM judge evaluation: `eval/judge_reasoning/`
- GRPO eval entry scripts: `scripts/grpo/eval/`

## Design Conventions

- Run scripts from repo root.
- Keep `scripts/` as entrypoints and `src`/module directories for implementation logic.
- Keep generated artifacts under `output/` and generated parquet under `data/verl/`.
