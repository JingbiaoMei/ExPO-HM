# GRPO Data Preparation

Run all commands from repo root (`expo-hm/`).

## Important

- GRPO prompts are generated in conversion code (`hatefulmemes.py`).
- GRPO does not read SFT `sharegpt_*.json` directly.
- Input metadata is read from `data/gt/` and images from `data/image/`.
- Output parquet is written to `data/verl/` (or `data/verl/CDE/` for CDE scripts).

## Core Generation Scripts

- Base binary datasets: `bash scripts/grpo/data_prep/generate_hatefulmemes.sh`
- Combined CDE (binary + fine-grained): `bash scripts/grpo/data_prep/generate_cde_all.sh`

## Path Defaults

All scripts accept env override and default to:

- `BASE_DIR=./data/gt`
- `IMAGE_DIR=./data/image`
- `OUTPUT_DIR=./data/verl` or `OUTPUT_DIR=./data/verl/CDE`

Example:

```bash
BASE_DIR=./data/gt IMAGE_DIR=./data/image OUTPUT_DIR=./data/verl \
  bash scripts/grpo/data_prep/generate_hatefulmemes.sh
```

## Output Layout Examples

- Base FB: `data/verl/FB/train.parquet`, `data/verl/FB/test.parquet`
- CDE FB train set: `data/verl/CDE/FB/train.parquet`
- FB fine-grained CDE merged train set: `data/verl/CDE/fb-fine-grained-combined/train.parquet`

## Reference

For full source-to-output mapping and reproducibility notes, see:

- `docs/reproducibility.md`
