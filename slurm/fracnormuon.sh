#!/bin/bash
#SBATCH --job-name=normuon_experiments
#SBATCH --output=logs/fracnormuon_2_%a.out 
#SBATCH --error=logs/fracnormuon_2_%a.err
#SBATCH --time=17:00:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --array=0                 # Creates 3 sub-jobs
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
module load Python/3.10.8-GCCcore-12.2.0

export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

CONFIG="configs/fracnormuon_350m.yaml"

CONFIG_TMP=$(mktemp)

WARMUP=0.0
ORTHO_FRACTION=0.125
PARTIAL_WARMUP=true

sed -E \
  -e "s/^warmup_ratio:[[:space:]]*[0-9.]+/warmup_ratio: ${WARMUP}/" \
  -e "s/^ortho_fraction:[[:space:]]*[0-9.]+/ortho_fraction: ${ORTHO_FRACTION}/" \
  -e "s/^partial_warmup:[[:space:]]*(true|false)/partial_warmup: ${PARTIAL_WARMUP}/" \
  "$CONFIG" > "$CONFIG_TMP"

torchrun --standalone --nproc_per_node=1 train.py --config "$CONFIG_TMP"
rm -f "$CONFIG_TMP"
