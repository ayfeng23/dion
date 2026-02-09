#!/bin/bash
#SBATCH --job-name=normuon_experiments
#SBATCH --output=logs/h200_fraction_%a.out 
#SBATCH --error=logs/h200_fraction_%a.err
#SBATCH --time=2:05:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --array=0-4                 # Creates 3 sub-jobs
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
module load Python/3.10.8-GCCcore-12.2.0

FRACTIONS=(0.125 0.0625 0.03125 0.02 0.01)
export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

MY_FRACTION=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

mkdir -p configs/tmp
TMP_CONFIG="configs/tmp/frac_${MY_FRACTION}_${SLURM_ARRAY_JOB_ID}.yaml"

# Use sed to replace the ortho_fraction value
sed "s/ortho_fraction:.*/ortho_fraction: $MY_FRACTION/" configs/fracnormuon_160m.yaml > $TMP_CONFIG

echo "Task ID $SLURM_ARRAY_TASK_ID starting with ortho_fraction: $MY_FRACTION"

torchrun --standalone --nproc_per_node=1 train.py --config $TMP_CONFIG

rm $TMP_CONFIG