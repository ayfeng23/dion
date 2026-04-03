#!/bin/bash
set -euo pipefail
# Cluster entry point: download data if needed, then launch training.
# Called by cjob with USE_TORCHRUN=0 so we control torchrun ourselves.
#
# Usage: cluster_entry.sh <config> <gpus> [extra train.py flags...]

CONFIG="$1"; shift
GPUS="$1"; shift
# Remaining args are extra flags for train.py (e.g. --no_wandb true)

echo "[cluster_entry] Downloading FineWeb data..."
pip install huggingface-hub 2>/dev/null
python -c "
import os, sys
from huggingface_hub import hf_hub_download
local_dir = 'data/fineweb10B'
os.makedirs(local_dir, exist_ok=True)
def get(fname):
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename=fname, repo_type='dataset', local_dir=local_dir)
get('fineweb_val_%06d.bin' % 0)
num_chunks = 30
for i in range(1, num_chunks + 1):
    get('fineweb_train_%06d.bin' % i)
print(f'Downloaded {num_chunks} train chunks + 1 val chunk to {local_dir}')
"

echo "[cluster_entry] Launching torchrun with $GPUS GPUs, config=$CONFIG"
exec torchrun --standalone --nproc_per_node="$GPUS" train.py --config "$CONFIG" "$@"
