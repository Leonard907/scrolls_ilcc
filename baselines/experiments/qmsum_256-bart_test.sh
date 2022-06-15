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

model=qmsum_led-4096
output_path=experiments/output/${model}

export XDG_CACHE_HOME=experiments/data/${model}

python scripts/execute.py scripts/commands/finetune.py ${model}_data

python scripts/execute.py scripts/commands/finetune.py ${model} \
  --output_dir=${output_path}

python scripts/execute.py scripts/commands/generate.py ${model}_test --checkpoint_path ${output_path} --output_dir ${output_path}

python ~/scrolls_ilcc/evaluator/prepare_submission.py \
--gov_report_file ~/scrolls_ilcc/baselines/experiments/mock_gov_report.json \
--summ_screen_file /home/s1970716/scrolls_ilcc/baselines/outputs/facebook-bart-base_256_32_0.0001_4096_scrolls_qmsum_director-habit-5_scrolls_summ_screen_fd_bathroom-candidate-19/generated_predictions.json \
--qmsum_file ${output_path}/generated_predictions.json \
--narrative_qa_file ~/scrolls_ilcc/baselines/experiments/mock_nqa.json \
--qasper_file /home/s1970716/scrolls_ilcc/baselines/outputs/facebook-bart-base_256_32_0.0001_4096_scrolls_qmsum_director-habit-5_scrolls_qasper_light-traffic-23/generated_predictions.json \
--quality_file /home/s1970716/scrolls_ilcc/baselines/outputs/facebook-bart-base_256_32_0.0001_4096_scrolls_qmsum_director-habit-5_scrolls_quality_player-degree-24/generated_predictions.json \
--contract_nli_file /home/s1970716/scrolls_ilcc/baselines/outputs/facebook-bart-base_256_32_0.0001_4096_scrolls_qmsum_director-habit-5_scrolls_contract_nli_age-tough-25/generated_predictions.json \
--output_dir ${output_path}
