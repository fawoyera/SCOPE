#!/bin/bash
#SBATCH --job-name=scope-grid
#SBATCH --output=slurm-%x-%j.out
# One SLURM job that runs a whole grid sequentially (the old atc_llama_clean workflow).
# Submit from the repo root:
#   sbatch -t 3-00:00:00 --mem=50G --nodes=1 --gpus-per-node=2 -A sbrunswi --partition=a100-40gb scripts/atc_llama_grid.sh
# Log: slurm-scope-grid-<jobid>.out in the repo root; per-run logs in results/atc/<model>/<method>/s<seed>/run.log
# Finished runs are skipped and interrupted ones resume from their last step checkpoint, so if the
# 3-day limit is hit, submit the same command again.  For parallel jobs (recommended) use instead:
#   scope run --models llama31_8b --methods all --seeds 42 123 456 --launcher slurm
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# ── environment: activate the one you installed SCOPE into (pip install -e ".[gpu]") ──────────
module load anaconda 2>/dev/null || true
conda activate scope                       # ← uncomment / adapt
PY=${PYTHON:-python}                         # or: PYTHON=/path/to/env/bin/python sbatch ...

export HF_HOME=/scratch/gilbreth/$USER/hf_cache
export TORCH_HOME=/scratch/gilbreth/$USER/torch_cache
export HF_DATASETS_CACHE=/scratch/gilbreth/$USER/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HF_TOKEN / WANDB_API_KEY: from ~/.bashrc, or `huggingface-cli login` and `wandb login` once.

# ── diagnostics (show up at the top of the log) ─────────────────────────────────────────────
echo "host=$(hostname)  job=$SLURM_JOB_ID  dir=$(pwd)  date=$(date)"
echo "python=$(command -v $PY)  $($PY --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi failed"
$PY -c "import scope, torch, transformers; print('scope', scope.__version__, '| torch', torch.__version__, \
'| cuda', torch.cuda.is_available(), torch.cuda.device_count(), '| transformers', transformers.__version__)" \
  || { echo "ERROR: SCOPE is not importable with $PY — activate the right environment above"; exit 1; }
[ -f data/atc_pairs_enriched.jsonl ] || { echo "ERROR: data/atc_pairs_enriched.jsonl not found in $(pwd)"; exit 1; }
[ -n "$HF_TOKEN" ] || [ -f "$HF_HOME/token" ] || [ -f ~/.cache/huggingface/token ] \
  || echo "WARNING: no Hugging Face token found — gated models (Llama) will fail to download"

#$PY -m scope.cli run --models llama31_8b --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic

$PY -m scope.cli run --models qwen3_8b --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic

$PY -m scope.cli run --models gpt2_large --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic

$PY -m scope.cli run --models gemma2_9b_it --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic

$PY -m scope.cli run --models mistral_7b_instruct --methods all --seeds 42 123 456 --launcher local --all-seeds-for-deterministic
