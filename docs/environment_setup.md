# Environment Setup

Use **two separate conda environments**: one for GRPO and one for SFT.

## 1) GRPO Environment (`verl`)

Follow the official `verl` installation procedure.

```bash
conda create -n verl python==3.12 -y
conda activate verl

cd verl

# If you need Megatron:
bash scripts/install_vllm_sglang_mcore.sh

# If you only need FSDP:
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install --no-deps -e .
cd ..
```

For ExPO-HM reasoning judge evaluation in the same env:

```bash
pip install -r eval/judge_reasoning/requirements.txt
```

Post-installation checks (`verl` docs):

- `torch` and torch series
- `vLLM`
- `SGLang`
- `pyarrow`
- `tensordict`
- `nvidia-cudnn-cu12` (for Megatron backend)

## 2) SFT Environment (`expohm-sft`)

`LLaMA-Factory` in this repo requires Python `>=3.11`. We recommend `3.11`.

```bash
conda create -n expohm-sft python=3.11 -y
conda activate expohm-sft

pip install -U pip setuptools wheel
pip install -e ./sft/LLaMA-Factory
pip install -r sft/LLaMA-Factory/requirements/metrics.txt
```

Optional (if you use DeepSpeed training paths):

```bash
pip install -r sft/LLaMA-Factory/requirements/deepspeed.txt
```

Quick check:

```bash
llamafactory-cli --help
```

## Usage Rule

- Run GRPO scripts in `verl`.
- Run SFT scripts in `expohm-sft`.
- Do not mix GRPO and SFT dependencies in one env.
