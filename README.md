# ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection

This is the official repository for **ExPO-HM (ICLR 2026)**.

Resources:
- Paper (OpenReview): https://openreview.net/forum?id=bEejbORUI5
- Reproducibility notes: `docs/reproducibility.md`
- Environment setup: `docs/environment_setup.md`

---

## Updates

- [2026] ExPO-HM accepted at ICLR 2026.
- [2026] Open-source release with GRPO and SFT training pipelines.


## Table of Contents

- [ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection](#expo-hm-learning-to-explain-then-detect-for-hateful-meme-detection)
  - [Updates](#updates)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Environment Setup](#environment-setup)
  - [Quick Start](#quick-start)
    - [1) Prepare GRPO Data](#1-prepare-grpo-data)
    - [2) Run GRPO Training](#2-run-grpo-training)
    - [3) Evaluate](#3-evaluate)
  - [Project Structure](#project-structure)
  - [Reproducibility](#reproducibility)
  - [Citation](#citation)

---

## Overview

ExPO-HM provides two training lanes:

- `scripts/grpo/`: GRPO/CDE scripts (data prep, training, merge, eval) on top of `verl`
- `sft/`: supervised fine-tuning via `LLaMA-Factory`

## Environment Setup

Use two separate conda environments:

- `verl` for GRPO (`verl`)
- `expohm-sft` for SFT (`LLaMA-Factory`)

Setup guide:

- `docs/environment_setup.md`

## Quick Start

Run all commands from repo root.

### 1) Prepare GRPO Data

```bash
bash scripts/grpo/data_prep/generate_hatefulmemes.sh
bash scripts/grpo/data_prep/generate_cde_all.sh
```

### 2) Run GRPO Training

```bash
bash scripts/grpo/train/run_qwen2_5_vl-7b-baseline.sh
bash scripts/grpo/train/run_qwen2_5_vl-7b_cde_paper.sh
```

### 3) Evaluate

```bash
bash scripts/grpo/eval/fb_inference_grpo.sh
python3 eval/judge_reasoning/llm_judge_eval.py --help
```

## Project Structure

- `scripts/grpo/`: GRPO data prep, training, merge, and eval entrypoints
- `scripts/grpo/train/`: GRPO training entrypoints
- `scripts/sft/`: SFT entry scripts
- `data/gt/`: source metadata
- `data/image/`: source images
- `data/verl/`: generated parquet for GRPO
- `eval/`: inference and reasoning judge code
- `docs/`: setup, layout, and reproducibility documents

For detailed layout and data conventions:

- `docs/repo_layout.md`
- `docs/data_layout.md`
- `docs/README.md`

## Reproducibility

Reproducibility mapping is documented in:

- `docs/reproducibility.md`


## Citation

If this repository helps your research, please cite:

```bibtex
@inproceedings{
EXPOHM2026Mei,
title={Ex{PO}-{HM}: Learning to Explain-then-Detect for Hateful Meme Detection},
author={Jingbiao Mei and Mingsheng Sun and Jinghong Chen and Pengda Qin and Yuhong Li and Da Chen and Bill Byrne},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=bEejbORUI5}
}
```
