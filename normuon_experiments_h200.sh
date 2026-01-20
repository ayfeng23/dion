#!/bin/bash
#SBATCH --job-name=normuon_experiments
#SBATCH --output=logs/h200_fraction_%a.out 
#SBATCH --error=logs/h200_fraction_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=scavenge_gpu
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --array=0                 # Creates 3 sub-jobs
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
module load Python/3.10.8-GCCcore-12.2.0

# FRACTIONS=(0.5 0.75 0.25)
export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

CONFIG="configs/fracnormuon_160m.yaml"

CONFIG_TMP=$(mktemp)

WARMUP=0.05
ORTHO_FRACTION=0.75
PARTIAL_WARMUP=true

sed -E \
  -e "s/^warmup_ratio:[[:space:]]*[0-9.]+/warmup_ratio: ${WARMUP}/" \
  -e "s/^ortho_fraction:[[:space:]]*[0-9.]+/ortho_fraction: ${ORTHO_FRACTION}/" \
  -e "s/^partial_warmup:[[:space:]]*(true|false)/partial_warmup: ${PARTIAL_WARMUP}/" \
  "$CONFIG" > "$CONFIG_TMP"

# sed -E "s/^warmup_ratio:[[:space:]]*[0-9.]+/warmup_ratio: ${WARMUP}/" \
#     "$CONFIG" > "$CONFIG_TMP"
torchrun --standalone --nproc_per_node=1 train.py --config "$CONFIG_TMP"
rm -f "$CONFIG_TMP"

# MY_FRACTION=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

# mkdir -p configs/tmp
# TMP_CONFIG="configs/tmp/frac_${MY_FRACTION}_${SLURM_ARRAY_JOB_ID}.yaml"

# # Use sed to replace the ortho_fraction value
# sed "s/ortho_fraction:.*/ortho_fraction: $MY_FRACTION/" configs/fracnormuon_160m.yaml > $TMP_CONFIG

# echo "Task ID $SLURM_ARRAY_TASK_ID starting with ortho_fraction: $MY_FRACTION"

# torchrun --standalone --nproc_per_node=2 train.py --config $TMP_CONFIG

# rm $TMP_CONFIG