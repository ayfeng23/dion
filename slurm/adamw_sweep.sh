#!/bin/bash
#SBATCH --job-name=adamw_lr_sweep
#SBATCH --output=logs/h200_lr_%a.out 
#SBATCH --error=logs/h200_lr_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --array=0-1

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
module load Python/3.10.8-GCCcore-12.2.0

export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

# ---- LR grid (log-scale, sensible) ----
LRS=(2e-3 3e-3)

LR=${LRS[$SLURM_ARRAY_TASK_ID]}

mkdir -p configs/tmp
TMP_CONFIG="configs/tmp/lr_${LR}_${SLURM_ARRAY_JOB_ID}.yaml"

# Replace learning_rate in YAML
sed -E "s/^lr:[[:space:]]*[0-9.eE+-]+/lr: ${LR}/" \
    configs/adamw_160m.yaml > "$TMP_CONFIG"

echo "Task ID $SLURM_ARRAY_TASK_ID starting with learning_rate: $LR"

export WANDB_RUN_NAME="adamw_lr_${LR}"
export WANDB_GROUP="adamw_lr_sweep"

torchrun --standalone --nproc_per_node=1 train.py --config "$TMP_CONFIG"

rm "$TMP_CONFIG"
