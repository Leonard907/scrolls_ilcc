#!/bin/bash
#SBATCH --output=/home/%u/rds/hpc-work/scrolls/slogs/sl-%x-%A-%a.out
#SBATCH --error=/home/%u/rds/hpc-work/scrolls/slogs/error-sl-%x-%A-%a.out
#SBATCH --job-name=qmsum_longt5-tglobal-base
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0

module load python/3.8 cuda/11.1 cudnn/8.0_cuda-11.1
jupyter lab --no-browser --ip=127.0.0.1 --port=8081