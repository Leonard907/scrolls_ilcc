#!/bin/bash
#SBATCH --output=/home/%u/slogs/sl-%x-%A-%a.out
#SBATCH --error=/home/%u/slogs/sl-%x-%A-%a.out
#SBATCH --job-name=qmsum_256-bart
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0

output_path=/home/%u/scrolls_ilcc/baselines/experiments/output

export XDG_CACHE_HOME=/home/%u/scrolls_ilcc/baselines/experiments/data/qmsum_256-bart

python ../scripts/execute.py ../scripts/finetune.py qmsum_256-bart_data

python ../scripts/execute.py ../scripts/finetune.py qmsum_256-bart \
  --output_dir=${output_path}