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

output_path=experiments/output/qmsum_1024-bart

export XDG_CACHE_HOME=experiments/data/qmsum_1024-bart

python scripts/execute.py scripts/commands/finetune.py qmsum_1024-bart_data

python scripts/execute.py scripts/commands/finetune.py qmsum_1024-bart \
  --output_dir=${output_path}

