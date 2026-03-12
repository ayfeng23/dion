#!/bin/bash
#SBATCH --job-name=normuon_experiments
#SBATCH --output=logs/fraction_%a.out
#SBATCH --error=logs/fraction_%a.err
#SBATCH --time=4:05:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --array=0-3   # 2 fractions x 2 paired (mu, ef_subset) settings = 4 jobs

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
module load Python/3.10.8-GCCcore-12.2.0

export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

FRACTIONS=(0.5 0.125)

# Paired settings only:
# pair 0: mu=0.95,  ef_subset=0.975
# pair 1: mu=0.975, ef_subset=0.95
MUS=(0.95 0.975)
EF_SUBSETS=(0.975 0.95)

# 2 combinations per fraction
F_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
PAIR_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

MY_FRACTION=${FRACTIONS[$F_IDX]}
MY_MU=${MUS[$PAIR_IDX]}
MY_EF=${EF_SUBSETS[$PAIR_IDX]}

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