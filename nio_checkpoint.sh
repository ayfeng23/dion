#!/bin/bash
#SBATCH --job-name=branch_nio
#SBATCH --output=logs/h200_branch_nio_%j.out
#SBATCH --error=logs/h200_branch_nio_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
module load Python/3.10.8-GCCcore-12.2.0

export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

# ============================================================
# Edit only this block
# ============================================================
CONFIG="configs/muon_160m.yaml"                 # or configs/normuon_160m.yaml
CHECKPOINT_DIR="checkpoints/muon_160m_0.04"     # folder that contains step_XXXXXXX/ or latest/
RESUME_STEP=2400                                # checkpoint step to load
NEW_OPTIMIZER="niomuon"                         # niomuon or nionormuon
LR=0.04                                         # new LR after resume
RESET_FACTOR=0.0                                # only used by NIO optimizers
WANDB_JOB_NAME="branch"                         # optional tag
# ============================================================

RESUME_NAME=$(python - <<PY
s=int("${RESUME_STEP}")
print(f"step_{s:04d}")
PY
)

echo "==============================================="
echo "CONFIG         = ${CONFIG}"
echo "CHECKPOINT_DIR = ${CHECKPOINT_DIR}"
echo "RESUME_NAME    = ${RESUME_NAME}"
echo "NEW_OPTIMIZER  = ${NEW_OPTIMIZER}"
echo "LR             = ${LR}"
echo "RESET_FACTOR   = ${RESET_FACTOR}"
echo "WANDB_JOB_NAME = ${WANDB_JOB_NAME}"
echo "==============================================="

torchrun --standalone --nproc_per_node=1 train.py \
  --config "${CONFIG}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --resume_name "${RESUME_NAME}" \
  --optimizer_switch_mode "auto"
  --optimizer "${NEW_OPTIMIZER}" \
  --lr "${LR}" \
  --reset_factor "${RESET_FACTOR}" \
  --wandb_job_name "${WANDB_JOB_NAME}"