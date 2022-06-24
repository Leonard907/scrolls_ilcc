import os
from scripts.utils import prep_command
from scripts.commands.consts import *


def get_command(id_):
    os.environ["DEBUG"] = os.environ.get("DEBUG", "false")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # fmt: off

    commands_dict = {}

    tokens_bsz = 1024
    num_gpus = 2
    accum_steps = 64
    folder_suffix_params = ["max_source_length", "gradient_accumulation_steps", "learning_rate", "train_max_tokens"]
    folder_suffix = "$".join(folder_suffix_params)
    generate_in_eval = False

    gg_longt5_local_base_args = [
        f"--model_name_or_path google/long-t5-tglobal-base",
        f"--max_source_length 1024",
        f"--max_target_length {GG_LONGT5_MAX_LEN}",
        f"--fp16 {GG_LONGT5_FP16}",
        f"--train_max_tokens {tokens_bsz}",
        f"--gradient_accumulation_steps {accum_steps}",
        f"--attention_window {GG_LONGT5_ATTENTION_WINDOW}",
        f"--per_device_eval_batch_size {GG_LONGT5_per_device_eval_batch_size}",
        f"--folder_suffix {folder_suffix}",
        "--source_prefix \"summarize:  \""
    ]

    fb_bart_256_args = [
        f"--model_name_or_path facebook/bart-base",
        f"--max_source_length 256",
        f"--max_target_length {FB_BART_MAX_LEN}",
        f"--fp16 {FB_BART_FP16}",
        f"--train_max_tokens {tokens_bsz}",
        f"--gradient_accumulation_steps {accum_steps}",
        f"--per_device_eval_batch_size {FB_BART_per_device_eval_batch_size}",
        f"--folder_suffix {folder_suffix}",
    ]

    fb_bart_512_args = [
        f"--model_name_or_path facebook/bart-base",
        f"--max_source_length 512",
        f"--max_target_length {FB_BART_MAX_LEN}",
        f"--fp16 {FB_BART_FP16}",
        f"--train_max_tokens {tokens_bsz}",
        f"--gradient_accumulation_steps {accum_steps}",
        f"--per_device_eval_batch_size {FB_BART_per_device_eval_batch_size}",
        f"--folder_suffix {folder_suffix}",
    ]

    fb_bart_1024_args = [
        f"--model_name_or_path facebook/bart-base",
        f"--max_source_length {FB_BART_MAX_LEN}",
        f"--max_target_length {FB_BART_MAX_LEN}",
        f"--fp16 {FB_BART_FP16}",
        f"--train_max_tokens {tokens_bsz}",
        f"--gradient_accumulation_steps {accum_steps}",
        f"--per_device_eval_batch_size {FB_BART_per_device_eval_batch_size}",
        f"--folder_suffix {folder_suffix}",
    ]

    allenai_led_args = [
        "--model_name_or_path allenai/led-base-16384",
        f"--attention_window {ALLEN_AI_ATTENTION_WINDOW}",
        f"--max_target_length {ALLEN_AI_MAX_TARGET_LEN}",
        f"--fp16 {ALLEN_AI_FP16}",
        f"--train_max_tokens {tokens_bsz}",
        f"--gradient_accumulation_steps {accum_steps}",
        f"--per_device_eval_batch_size {ALLEN_AI_per_device_eval_batch_size}",
    ]

    all_learning_rates = {
        "qasper": {
            "256-bart": 5e-5,
            "512-bart": 5e-5,
            "1024-bart": 5e-5,
            "led-1024": 2e-5,
            "led-4096": 5e-5,
            "led-16384": 2e-5,
            "longt5-local": 1e-3,
        },
        "narrative_qa": {
            "256-bart": 1e-4,
            "512-bart": 5e-5,
            "1024-bart": 5e-5,
            "led-1024": 2e-5,
            "led-4096": 1e-5,
            "led-16384": 1e-5,
            "longt5-local": 1e-3,
        },
        "gov_report": {
            "256-bart": 5e-4,
            "512-bart": 5e-4,
            "1024-bart": 2e-4,
            "led-1024": 1e-4,
            "led-4096": 1e-4,
            "led-16384": 5e-5,
            "longt5-local": 1e-3,
        },
        "summ_screen_fd": {
            "256-bart": 2e-4,
            "512-bart": 2e-4,
            "1024-bart": 1e-4,
            "led-1024": 5e-5,
            "led-4096": 2e-5,
            "led-16384": 1e-5,
            "longt5-local": 1e-3,
        },
        "qmsum": {
            "256-bart": 1e-4,
            "512-bart": 5e-5,
            "1024-bart": 5e-5,
            "led-1024": 5e-5,
            "led-4096": 1e-5,
            "led-16384": 1e-5,
            "longt5-local": 1e-3,
        },
        "contract_nli": {
            "256-bart": 1e-4,
            "512-bart": 1e-4,
            "1024-bart": 5e-5,
            "led-1024": 5e-5,
            "led-4096": 2e-5,
            "led-16384": 1e-5,
            "longt5-local": 1e-3,
        },
        "quality": {
            "256-bart": 5e-5,
            "512-bart": 2e-5,
            "1024-bart": 1e-5,
            "led-1024": 2e-5,
            "led-4096": 1e-5,
            "led-16384": 1e-5,
            "longt5-local": 1e-3,
        },
    }

    distributed_str = f"-m torch.distributed.run --nproc_per_node={num_gpus}" if num_gpus > 1 else ""

    for dataset in ["qasper", "narrative_qa", "gov_report", "summ_screen_fd", "qmsum", "contract_nli", "quality"]:

        base_args = [
            f"python {distributed_str} src/run.py configs/train.json",
            f"--m configs/datasets/{dataset}.json",
            "--adam_epsilon 1e-6",
            "--adam_beta1 0.9",
            "--adam_beta2 0.98",
            "--weight_decay 0.001",
            "--logging_steps 10",
            "--gradient_checkpointing true",
            "--save_total_limit 2",
            "--preprocessing_num_workers 1",
            "--group_by_length true",
            "--do_eval True",
            "--load_best_model_at_end True",
            # "--lr_scheduler linear",
            "--warmup_ratio 0.0",
        ] + (
            ["--m configs/no_metrics.json", "--predict_with_generate False", "--prediction_loss_only True"]
            if not generate_in_eval
            else []
        )
        if dataset == "narrative_qa":
            base_args.append("--trim_very_long_strings")

        dataset_learning_rates = all_learning_rates[dataset]
        commands_dict[f"{dataset}_256-bart"] = base_args + fb_bart_256_args +[ f"--learning_rate {dataset_learning_rates['256-bart']}", f"--lr_scheduler {FB_BART_LR_SCHEDULER}"]
        commands_dict[f"{dataset}_512-bart"] = base_args + fb_bart_512_args + [f"--learning_rate {dataset_learning_rates['512-bart']}", f"--lr_scheduler {FB_BART_LR_SCHEDULER}"]
        commands_dict[f"{dataset}_1024-bart"] = base_args + fb_bart_1024_args + [f"--learning_rate {dataset_learning_rates['1024-bart']}", f"--lr_scheduler {FB_BART_LR_SCHEDULER}"]
        commands_dict[f"{dataset}_led-1024"] = base_args + allenai_led_args + [f"--learning_rate {dataset_learning_rates['led-1024']}", "--global_attention_first_token True", f"--folder_suffix global_attention_first_token${folder_suffix}", f"--max_source_length 1024", f"--lr_scheduler {ALLEN_AI_LR_SCHEDULER}"]
        commands_dict[f"{dataset}_led-4096"] = base_args + allenai_led_args + [f"--learning_rate {dataset_learning_rates['led-4096']}", "--global_attention_first_token True", f"--folder_suffix global_attention_first_token${folder_suffix}", f"--max_source_length 4096", f"--lr_scheduler {ALLEN_AI_LR_SCHEDULER}"]
        commands_dict[f"{dataset}_led-16384"] = base_args + allenai_led_args +\
                                                                    [f"--learning_rate {dataset_learning_rates['led-16384']}",
                                                                    "--global_attention_first_token True",
                                                                    f"--folder_suffix global_attention_first_token${folder_suffix}" ,
                                                                    "--max_source_length 16384",
                                                                    f"--lr_scheduler {ALLEN_AI_LR_SCHEDULER}"
        ]
        commands_dict[f"{dataset}_longt5-local"] = base_args + gg_longt5_local_base_args + [f"--learning_rate {dataset_learning_rates['longt5-local']}", f"--lr_scheduler {GG_LONGT5_LR_SCHEDULER}"]

        prepro_args = base_args[:]
        prepro_args[0] = prepro_args[0].replace(distributed_str, "")

        commands_dict[f"{dataset}_256-bart_data"] = prepro_args + fb_bart_256_args + ["--preprocess_only", "--learning_rate 1e-3"]
        commands_dict[f"{dataset}_512-bart_data"] = prepro_args + fb_bart_512_args + ["--preprocess_only", "--learning_rate 1e-3"]
        commands_dict[f"{dataset}_1024-bart_data"] = prepro_args + fb_bart_1024_args + ["--preprocess_only", "--learning_rate 1e-3"]
        commands_dict[f"{dataset}_led-1024_data"] = prepro_args + allenai_led_args + ["--preprocess_only", "--learning_rate 1e-3", "--global_attention_first_token True", f"--folder_suffix global_attention_first_token${folder_suffix}", "--max_source_length 1024"]
        commands_dict[f"{dataset}_led-4096_data"] = prepro_args + allenai_led_args + ["--preprocess_only", "--learning_rate 1e-3", "--global_attention_first_token True", f"--folder_suffix global_attention_first_token${folder_suffix}", "--max_source_length 4096"]
        commands_dict[f"{dataset}_led-16384_data"] = prepro_args + allenai_led_args + ["--preprocess_only", "--learning_rate 1e-3", "--global_attention_first_token True", f"--folder_suffix global_attention_first_token${folder_suffix}", "--max_source_length 16384"]
        commands_dict[f"{dataset}_longt5-local_data"] = prepro_args + gg_longt5_local_base_args + ["--preprocess_only", f"--learning_rate {dataset_learning_rates['longt5-local']}"]


    command_parts = commands_dict[id_]
    # fmt: on

    return prep_command(command_parts)
