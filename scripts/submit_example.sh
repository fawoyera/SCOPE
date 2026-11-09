#!/bin/bash
# End-to-end example on the cluster (run from the repo root on a login node).
set -e
export WANDB_PROJECT=scope            # optional; or set logging.wandb.project
# 0) once per domain: domain-adapted BERTScore encoder (needs a GPU or ~1h CPU)
#    sbatch --gpus-per-node=1 --time=02:00:00 --wrap "scope bert-mlm --domain atc"

# 1) the full grid: 3 models × 6 methods × 3 seeds, chained with afterok dependencies
scope run --domain atc --models llama31_8b qwen3_8b gpt2_large --methods all \
          --seeds 42 123 456 --launcher slurm

# 2) when the jobs are done
scope table --domain atc                          # tables/results_atc.tex (+ .md/.csv)
scope stats --domain atc --reference scope        # tables/significance_atc.tex, method_ci_atc.csv
scope plot  --domain atc --out plots/atc          # training dashboards, Pareto, seed curves
