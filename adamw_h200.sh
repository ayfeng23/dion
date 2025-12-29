#!/bin/bash
#SBATCH --job-name=adamw
#SBATCH --output=logs/h200_adamw.out 
#SBATCH --error=logs/h200_adamw.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_h200
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

export WANDB_API_KEY=6847fa93f84b5335cd0ba5f438e6ba60fbe5b76b

CONFIG="configs/adamw_160m.yaml"

torchrun --standalone --nproc_per_node=1 train.py --config $CONFIG