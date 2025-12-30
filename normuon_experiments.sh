#!/bin/bash
#SBATCH --job-name=normuon_experiments
#SBATCH --output=logs/fraction_%a.out 
#SBATCH --error=logs/fraction_%a.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --array=0-2                    # Creates 3 sub-jobs
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

FRACTIONS=(1.0 0.75 0.5)
export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

MY_FRACTION=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}

mkdir -p configs/tmp
TMP_CONFIG="configs/tmp/frac_${MY_FRACTION}_${SLURM_ARRAY_JOB_ID}.yaml"

# Use sed to replace the rank_fraction value
sed "s/rank_fraction:.*/rank_fraction: $MY_FRACTION/" configs/fracnormuon_160m.yaml > $TMP_CONFIG

echo "Task ID $SLURM_ARRAY_TASK_ID starting with rank_fraction: $MY_FRACTION"

torchrun --standalone --nproc_per_node=1 train.py --config $TMP_CONFIG

rm $TMP_CONFIG