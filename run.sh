#!/bin/sh
#SBATCH --account=iris
#SBATCH --partition=iris --qos=normal
#SBATCH --exclude=iris4,iris5,iris6
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name="mg4"
#SBATCH --output=sbatch_logs/train_bc-%j.out
#SBATCH --mail-user=jingyuny@stanford.edu
#SBATCH --mail-type=ALL


torchrun --nproc_per_node 1 train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
