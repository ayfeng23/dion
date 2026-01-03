#!/bin/bash
#SBATCH --job-name=normuon_front
#SBATCH --output=logs/h200_normuon_front.out 
#SBATCH --error=logs/h200_normuon_front.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
module load Python/3.10.8-GCCcore-12.2.0

export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

CONFIG="configs/normuon_front_350m.yaml"

torchrun --standalone --nproc_per_node=1 train.py --config $CONFIG