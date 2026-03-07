#!/bin/bash
#SBATCH --job-name=normuon_experiments
#SBATCH --output=logs/fraction_%a.out 
#SBATCH --error=logs/fraction_%a.err
#SBATCH --time=4:05:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --array=0-11   # 12 combinations

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
module load Python/3.10.8-GCCcore-12.2.0

export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

FRACTIONS=(0.5 0.125)
MUS=(0.95 0.975 1.0)
EF_SUBSETS=(0.95 0.975)

# 6 combinations per fraction
F_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REM=$(( SLURM_ARRAY_TASK_ID % 6 ))

MU_IDX=$(( REM / 2 ))
EF_IDX=$(( REM % 2 ))

MY_FRACTION=${FRACTIONS[$F_IDX]}
MY_MU=${MUS[$MU_IDX]}
MY_EF=${EF_SUBSETS[$EF_IDX]}

mkdir -p configs/tmp
TMP_CONFIG="configs/tmp/frac_${MY_FRACTION}_mu_${MY_MU}_ef_${MY_EF}_${SLURM_ARRAY_JOB_ID}.yaml"

sed -e "s/ortho_fraction:.*/ortho_fraction: $MY_FRACTION/" \
    -e "s/^mu:.*/mu: $MY_MU/" \
    -e "s/^ef_subset:.*/ef_subset: $MY_EF/" \
    configs/fracnormuon_160m.yaml > $TMP_CONFIG

echo "Task $SLURM_ARRAY_TASK_ID running with:"
echo "  ortho_fraction = $MY_FRACTION"
echo "  mu             = $MY_MU"
echo "  ef_subset      = $MY_EF"

torchrun --standalone --nproc_per_node=1 train.py --config $TMP_CONFIG

rm $TMP_CONFIG