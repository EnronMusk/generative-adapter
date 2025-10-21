import logging
from transformers import HfArgumentParser
from transformers.integrations import is_deepspeed_zero3_enabled
from fastlora.utils import ( 
    DefaultDataCollator,
    FileLogger,
    makedirs,
    format_numel_str,
    Metric,
)
from fastlora.args import ModelArgs, TrainingArgs
from fastlora.trainer import FastLoraTrainer
from fastlora.data import Data
from datetime import datetime
from fastlora import get_model_and_tokenizer

logger = logging.getLogger(__name__)


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    import os
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def main():
    parser = HfArgumentParser([ModelArgs, TrainingArgs])
    model_args, training_args = parser.parse_args_into_dataclasses()
    reset_wandb_env()
    # if training_args.run_name is None:
    #     training_args.run_name = f"{model_args.model_name_or_path.split('/')[-1]}.{'-'.join([x.split('/')[-2] for x in model_args.train_data])}.{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    # else:
    #     training_args.run_name = training_args.run_name + f".{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    # print(f"************ {training_args.report_to} ************")
    # print(f"************ {training_args.run_name} ************")

    model, tokenizer = get_model_and_tokenizer(model_args, device="cuda", evaluation_mode=False)
    print(model)

    # if model_args.enable_ultragist and training_args.only_train_ultragist:
    #     for name, param in model.named_parameters():
    #         if "fastlora" not in name:
    #             param.requires_grad_(False)
    
    for name, param in model.named_parameters():
        if "fastlora" not in name:
            param.requires_grad_(False)

    # if training_args.lora_tune:
    #     from peft import (
    #         LoraConfig,
    #         get_peft_model,
    #     )
    #     # copied from LongLoRA
    #     config = LoraConfig(
    #         r=training_args.lora_rank,
    #         lora_alpha=training_args.lora_alpha,
    #         target_modules=training_args.lora_targets,
    #         modules_to_save=training_args.lora_extra_params,
    #         lora_dropout=training_args.lora_dropout,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    #     model = get_peft_model(model, config)

    logger.info(f"Trainable Model params: {format_numel_str(sum(p.numel() for p in model.parameters() if p.requires_grad))}")
    with training_args.main_process_first():
        train_dataset = Data.prepare_train_data(
            model_args.train_data, 
            tokenizer=tokenizer,
            max_length=model_args.max_length,
            min_length=training_args.min_length,
            window_size=model_args.fastlora_window,
            chat_template=model_args.chat_template,
            enable_reconstruct=(model_args.fastlora_training_attention_mask and model_args.fastlora_training_attention_mask.startswith("abcdabcd")),
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
            max_train_samples=model_args.max_train_samples,
        )

    with training_args.main_process_first():
        if is_deepspeed_zero3_enabled() and training_args.eval_method != "perplexity":
            logger.warning(f"In deepspeed zero3, evaluation with generation is may lead to hang because of the unequal number of forward passes across different devices.")
        eval_dataset = Data.prepare_eval_data(
            model_args.eval_data, 
            tokenizer=tokenizer,
            max_length=training_args.eval_max_length,
            min_length=training_args.eval_min_length,
            window_size=model_args.fastlora_window,
            chat_template=model_args.chat_template,
            max_eval_num=model_args.max_eval_samples,
            seed=training_args.seed,
            cache_dir=model_args.dataset_cache_dir,
        )
    
    if (training_args.report_to == "wandb" or "wandb" in training_args.report_to) and training_args.wandb_watch_log_freq is not None:
        if training_args.local_rank == -1 or training_args.local_rank == 0:
            logger.info("Enabling wandb.watch()...")
            import wandb, os
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                name=training_args.run_name,
            )
            wandb.watch(
                model,
                log="all",
                log_freq=training_args.wandb_watch_log_freq,
                log_graph=True,
            )

    trainer = FastLoraTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(tokenizer),
        file_logger=FileLogger(makedirs(training_args.log_path)),
        compute_metrics=Metric.get_metric_fn(
            metrics=training_args.metrics,
            save_path=Metric.get_save_path(
                model_args.eval_data,
                training_args.output_dir
            ) if model_args.eval_data is not None else None
        ),
    )
    if train_dataset is not None:
        trainer.train()
    elif eval_dataset is not None:
        trainer.evaluate()

if __name__ == "__main__":
    main()
