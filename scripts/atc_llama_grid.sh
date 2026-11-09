#!/bin/bash
# One SLURM job that runs a whole grid sequentially (the old atc_llama_clean workflow).
# Submit from the repo root:
#   sbatch -t 3-00:00:00 --mem=50G --nodes=1 --gpus-per-node=2 -A sbrunswi --partition=a100-40gb scripts/atc_llama_grid.sh
# Finished runs are skipped and interrupted ones resume from their last step checkpoint, so if the
# 3-day limit is hit, submit the same command again.  For parallel jobs (recommended) use instead:
#   scope run --models llama31_8b --methods all --seeds 42 123 456 --launcher slurm
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

conda activate scope                        # ← your environment (install once: pip install -e ".[gpu]")
export HF_HOME=/scratch/gilbreth/$USER/hf_cache
export TORCH_HOME=/scratch/gilbreth/$USER/torch_cache
export HF_DATASETS_CACHE=/scratch/gilbreth/$USER/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HF_TOKEN / WANDB_API_KEY: from ~/.bashrc, or `huggingface-cli login` and `wandb login` once.

#scope run --models llama31_8b --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic

scope run --models qwen3_8b --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic

scope run --models gpt2_large --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic

scope run --models gemma2_9b_it --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic

scope run --models mistral_7b_instruct --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic


/bin/hostname
