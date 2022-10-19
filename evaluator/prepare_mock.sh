python prepare_submission.py \
--gov_report_file ../baselines/outputs/google-long-t5-tglobal-base_8192_32_0.001_8192_scrolls_gov_report_efficiency-stay-285_scrolls_gov_report_brush-forever-287/generated_predictions_val.json \
--summ_screen_file ../baselines/experiments/mock_sfd.json \
--qmsum_file ../../temp/generated_predictions.json \
--narrative_qa_file ../baselines/experiments/mock_nqa.json \
--qasper_file ../baselines/experiments/mock_qasper.json \
--quality_file ../baselines/experiments/mock_qual.json \
--contract_nli_file ../baselines/experiments/mock_cnli.json \
--output_dir .