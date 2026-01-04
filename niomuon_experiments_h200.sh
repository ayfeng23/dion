#!/bin/bash
#SBATCH --job-name=niomuon_grid
#SBATCH --output=logs/h200_niomuon_grid_%a.out
#SBATCH --error=logs/h200_niomuon_grid_%a.err
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --array=0-5

export PATH="$HOME/.local/bin:$PATH"

# If your venv relies on system python modules, load module BEFORE activating venv.
module load Python/3.10.8-GCCcore-12.2.0
source .venv/bin/activate

# -------- Grid --------
LRS=(0.04 0.02)
RESET_FACTORS=(0 0.5 0.9)

# Map SLURM_ARRAY_TASK_ID -> (lr_idx, rf_idx)
NUM_RF=${#RESET_FACTORS[@]}
LR_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_RF ))
RF_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_RF ))

MY_LR=${LRS[$LR_IDX]}
MY_RF=${RESET_FACTORS[$RF_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: lr=${MY_LR}, reset_factor=${MY_RF}"

# -------- W&B --------
# Better: export this in your shell or a secrets file, not hardcoded in the script.
export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b
export WANDB_RUN_GROUP="Test Optimizer"
export WANDB_NAME="niomuon_lr${MY_LR}_rf${MY_RF}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"

mkdir -p logs configs/tmp

TMP_CONFIG="configs/tmp/niomuon_lr${MY_LR}_rf${MY_RF}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

# Start from your base config
cp configs/niomuon_160m.yaml "$TMP_CONFIG"

# -------- Edit config --------
# 1) set algorithm to niomuon (assumes a line like: algorithm: something)
sed -i "s/^algorithm:.*/algorithm: niomuon/" "$TMP_CONFIG"

# 2) set lr (assumes a line like: lr: 0.0004 etc.)
sed -i "s/^lr:.*/lr: ${MY_LR}/" "$TMP_CONFIG"

sed -i "s/^mu:.*/mu: 1.0/" "$TMP_CONFIG"

# 3) set reset_factor (assumes a line like: reset_factor: 0.0 etc.)
sed -i "s/^reset_factor:.*/reset_factor: ${MY_RF}/" "$TMP_CONFIG"

echo "Using config: $TMP_CONFIG"
echo "---- Config diff (key lines) ----"
grep -E "^(algorithm|lr|reset_factor):" "$TMP_CONFIG" || true
echo "---------------------------------"

torchrun --standalone --nproc_per_node=1 train.py --config "$TMP_CONFIG"

rm -f "$TMP_CONFIG"
