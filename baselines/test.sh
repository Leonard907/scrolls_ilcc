#!/bin/bash
#SBATCH --output=/home/%u/longt5_gov_tr_log/knn_mem_v1.6_32128_epoch_4.out
#SBATCH --error=/home/%u/longt5_gov_tr_log/knn_mem_v1.6_32128_epoch_4.out
#SBATCH --job-name=gov_longt5-tglobal-local_8192
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --array=0

export XDG_CACHE_HOME=~/scrolls_data/gov_report
python scripts/execute.py scripts/commands/finetune.py gov_report_longt5-local --resume_from_checkpoint outputs/saved_models/longt5_8192_inner_product_epoch_5_mem_32K
# google checkpoints
# 32128 epoch 4 inner product google-long-t5-tglobal-base_8192_32_0.001_8192_scrolls_gov_report_outcome-cry-1
# hpc checkpoints
# 32128 epoch 4 --resume_from_checkpoint outputs/google-long-t5-tglobal-base_8192_32_0.001_8192_scrolls_gov_report_language-truck-416 
# 32128 epoch 4 --resume_from_checkpoint outputs/google-long-t5-tglobal-base_8192_32_0.001_8192_scrolls_gov_report_sector-tough-378
# 8192 epoch 10 --resume_from_checkpoint outputs/google-long-t5-tglobal-base_8192_32_0.001_8192_scrolls_gov_report_speed-deep-324
