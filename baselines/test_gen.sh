#!/bin/bash
#SBATCH --output=/home/%u/longt5_gov_gen_log/knn_v1.4_8192_epoch_10_test.out
#SBATCH --error=/home/%u/longt5_gov_gen_log/knn_v1.4_8192_epoch_10_test.out
#SBATCH --job-name=gov_longt5-local
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --array=0

export XDG_CACHE_HOME=/home/$USER/rds/hpc-work/scrolls/scrolls_ilcc/scrolls_data/gov_report
python scripts/execute.py scripts/commands/generate.py gov_report_longt5-local_test --checkpoint_path outputs/google-long-t5-tglobal-base_8192_32_0.001_8192_scrolls_gov_report_sector-tough-378

# 8192 epoch 10 outputs/google-long-t5-tglobal-base_8192_32_0.001_8192_scrolls_gov_report_speed-deep-324