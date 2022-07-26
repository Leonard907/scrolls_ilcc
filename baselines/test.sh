#!/bin/bash
#SBATCH --output=/home/%u/longt5_gov_tr_log/max_target_len_1024.out
#SBATCH --error=/home/%u/longt5_gov_tr_log/max_target_len_1024.out
#SBATCH --job-name=gov_longt5-local_8192
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --array=0

export XDG_CACHE_HOME=/home/$USER/rds/hpc-work/scrolls/scrolls_ilcc/scrolls_data/gov_report
python scripts/execute.py scripts/commands/finetune.py gov_report_longt5-local
