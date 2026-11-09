#!/bin/bash
# Install flash-attn once per environment (run on a GPU node with the CUDA toolkit loaded).
# SCOPE uses it automatically when present (attn_implementation: auto), otherwise SDPA.
set -e
pip install wheel ninja packaging
pip install "flash-attn>=2.1" --no-build-isolation
python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
