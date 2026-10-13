"""Map resolved settings to HF arguments without initializing CUDA."""


def training_argument_values(config, output):
    training = config["training"]
    logging = config["logging"]
    strategy = training.get("save_strategy", "steps")
    return {
        "output_dir": str(output),
        "deepspeed": training.get("deepspeed"),
        "seed": config["seed"],
        "data_seed": config.get("data_seed", config["seed"]),
        "max_steps": training["max_optimizer_steps"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "warmup_ratio": training["warmup_ratio"],
        "lr_scheduler_type": training["scheduler"],
        "per_device_train_batch_size": training["completion_micro_batch_per_device"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "bf16": config["model"]["precision"] == "bf16",
        "gradient_checkpointing": config["model"]["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": bool(training.get("deepspeed"))},
        "optim": "adamw_torch",
        "report_to": ["tensorboard"] if logging["tensorboard"] else [],
        "logging_steps": training.get("logging_steps", 1),
        "logging_dir": logging.get("logging_dir"),
        "run_name": logging.get("run_name"),
        # HF sees one synthetic dataset epoch. Our callback uses real data epochs.
        "save_strategy": "no" if strategy == "epoch" else strategy,
        "save_steps": training["save_steps"],
        "save_total_limit": training["save_total_limit"],
        "save_only_model": training.get("save_only_model", False),
        "remove_unused_columns": False,
        "label_names": [],
        "eval_strategy": "no",
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "disable_tqdm": False,
        "ddp_timeout": 3600,
        "torch_empty_cache_steps": training.get("empty_cache_steps"),
    }
