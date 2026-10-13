"""Translate explicit Invert4TVG-style options into the internal configuration."""

import argparse
import os
from pathlib import Path

from vtg.config import BRANCHES, load_config


def boolean(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Use true or false")


def add_boolean(parser, name):
    parser.add_argument(name, type=boolean, nargs="?", const=True, default=None)


def training_parser():
    parser = argparse.ArgumentParser(
        description="Eight-GPU VTG training: explicit command-line options override JSON defaults."
    )
    parser.add_argument(
        "--config", help="Optional experiment JSON (default: configs/e5_8xa100.json)"
    )
    parser.add_argument("--paths", help="Optional local JSON overrides")
    parser.add_argument("--branch", choices=tuple(BRANCHES))
    parser.add_argument("--output_dir", "--output", dest="output")
    parser.add_argument("--resume_from_checkpoint", "--resume", dest="resume")
    parser.add_argument("--stop_after", "--stop-after", type=int)
    parser.add_argument("--coverage_size", "--coverage-size", type=int)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print settings without reading data or loading torch",
    )
    parser.add_argument("--prepare_data", choices=("auto", "never"), default="never")
    parser.add_argument("--prepare_only", action="store_true")

    paths = parser.add_argument_group("Model and data")
    for name in (
        "model_name_or_path",
        "parent_checkpoint",
        "train_data_path",
        "video_folder",
        "artifact_root",
        "spacy_model",
    ):
        paths.add_argument(f"--{name}")
    paths.add_argument("--dataset_name", choices=("timer1_train_2k5", "timer1", "charades"))

    training = parser.add_argument_group("Optimizer and rollout")
    for name in (
        "world_size",
        "num_generations",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_prompt_length",
        "max_completion_length",
        "num_train_epochs",
        "max_steps",
        "seed",
        "data_seed",
        "schedule_seed",
    ):
        training.add_argument(f"--{name}", type=int)
    for name in ("learning_rate", "weight_decay", "warmup_ratio", "temperature", "top_p", "beta"):
        training.add_argument(f"--{name}", type=float)
    training.add_argument(
        "--lr_scheduler_type", choices=("constant", "constant_with_warmup", "linear", "cosine")
    )
    training.add_argument("--attn_implementation", choices=("sdpa", "flash_attention_2", "eager"))
    training.add_argument("--torch_dtype", choices=("bfloat16", "float32"))
    for name in ("bf16", "gradient_checkpointing", "fix_vit"):
        add_boolean(training, f"--{name}")

    logging = parser.add_argument_group("Logging and checkpoints")
    for name in ("logging_steps", "save_steps", "save_total_limit"):
        logging.add_argument(f"--{name}", type=int)
    for name in ("logging_dir", "run_name"):
        logging.add_argument(f"--{name}")
    logging.add_argument("--report_to", choices=("tensorboard", "none"))
    logging.add_argument("--save_strategy", choices=("steps", "epoch", "no"))
    add_boolean(logging, "--save_only_model")

    video = parser.add_argument_group("Dynamic video budget")
    for name in (
        "max_frames",
        "min_frames",
        "max_pixels",
        "min_pixels",
        "total_pixels",
        "visual_token_cap",
    ):
        video.add_argument(f"--{name}", type=int)
    video.add_argument("--fps", type=float)

    preference = parser.add_argument_group("VTG method")
    for name in ("lambda_pair", "beta_dpo", "min_iou", "min_gap"):
        preference.add_argument(f"--{name}", type=float)
    preference.add_argument("--matched_calibration")
    preference.add_argument("--reward_funcs", nargs="+", choices=("iou", "format"))
    preference.add_argument("--prompt_type", choices=("event_core_v1",))

    unsupported = parser.add_argument_group(
        "Compatibility checks: unsupported options fail explicitly"
    )
    training.add_argument("--deepspeed", help="ZeRO-3 JSON path, or none for DDP")
    unsupported.add_argument("--eval_data_path")
    for name in ("slide_window", "is_curriculum_learning", "is_early_stopping"):
        add_boolean(unsupported, f"--{name}")
    return parser


def resolve_arguments(args, parser):
    for name in (
        "eval_data_path",
        "slide_window",
        "is_curriculum_learning",
        "is_early_stopping",
    ):
        if getattr(args, name):
            parser.error(
                f"--{name} is not implemented by this trainer. The option will not be silently ignored."
            )
    if args.max_steps is not None and args.num_train_epochs is not None:
        parser.error("Choose --max_steps or --num_train_epochs, not both")
    if args.bf16 is not None and args.torch_dtype is not None:
        if args.bf16 != (args.torch_dtype == "bfloat16"):
            parser.error("--bf16 and --torch_dtype disagree")
    branch = args.branch or "E5"
    if args.config:
        path = Path(args.config)
    else:
        config_names = {
            "E5": "e5",
            "N0_E5_cont": "n0",
            "N1_IoU_pair": "n1",
            "N2_Event_pair": "n2",
            "N3_Swap": "n3",
            "N4_Event_swap": "n4",
            "A_matched": "a_matched",
            "A_verb_gate": "a_verb_gate",
            "A_sem_add": "a_sem_add",
        }
        path = (
            Path(__file__).resolve().parents[3] / "configs" / f"{config_names[branch]}_8xa100.json"
        )
        if not path.is_file():
            parser.error("Supply --config when using an installed wheel outside the repository")
    overrides = {}

    def assign(section, key, value):
        if value is not None:
            overrides.setdefault(section, {})[key] = value

    if args.branch is not None:
        overrides["branch"] = args.branch
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.data_seed is not None:
        overrides["data_seed"] = args.data_seed
        assign("continuation", "schedule_seed", args.data_seed)
    for argument, key in {
        "model_name_or_path": "model_path",
        "train_data_path": "train_annotations",
        "video_folder": "video_root",
        "artifact_root": "artifact_root",
        "spacy_model": "spacy_model",
    }.items():
        assign("paths", key, getattr(args, argument))
    mapping = {
        "num_generations": "num_generations",
        "per_device_train_batch_size": "completion_micro_batch_per_device",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "max_prompt_length": "max_prompt_tokens",
        "max_completion_length": "max_completion_tokens",
        "num_train_epochs": "num_train_epochs",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "warmup_ratio": "warmup_ratio",
        "lr_scheduler_type": "scheduler",
        "temperature": "temperature",
        "top_p": "top_p",
        "beta": "kl_beta",
        "attn_implementation": "attn_implementation",
        "logging_steps": "logging_steps",
        "save_strategy": "save_strategy",
        "save_steps": "save_steps",
        "save_total_limit": "save_total_limit",
        "save_only_model": "save_only_model",
    }
    for argument, key in mapping.items():
        assign("training", key, getattr(args, argument))
    if args.deepspeed is not None:
        from vtg.training.backend import read_deepspeed_config

        overrides.setdefault("training", {})["deepspeed"] = (
            None if args.deepspeed.lower() == "none" else read_deepspeed_config(args.deepspeed)
        )
    world = args.world_size
    if "WORLD_SIZE" in os.environ:
        actual = int(os.environ["WORLD_SIZE"])
        if world is not None and world != actual:
            parser.error("--world_size differs from torchrun WORLD_SIZE")
        world = actual
    assign("training", "world_size", world)
    assign("training", "unique_prompts_per_update", args.gradient_accumulation_steps)
    if args.max_steps is not None:
        assign("training", "max_optimizer_steps", args.max_steps)
        overrides["training"]["num_train_epochs"] = None
    assign("data", "dataset", args.dataset_name)
    for name in ("run_name", "logging_dir"):
        assign("logging", name, getattr(args, name))
    if args.report_to is not None:
        assign("logging", "tensorboard", args.report_to == "tensorboard")
    for argument, key in {
        "max_frames": "max_frames",
        "min_frames": "min_frames",
        "fps": "fps",
        "max_pixels": "max_pixels_per_frame",
        "min_pixels": "min_pixels_per_frame",
        "total_pixels": "total_pixels",
        "visual_token_cap": "visual_token_cap",
    }.items():
        assign("video", key, getattr(args, argument))
    for name in (
        "parent_checkpoint",
        "schedule_seed",
        "lambda_pair",
        "beta_dpo",
        "min_iou",
        "min_gap",
        "matched_calibration",
    ):
        assign("continuation", name, getattr(args, name))
    assign("model", "gradient_checkpointing", args.gradient_checkpointing)
    if args.fix_vit is not None:
        assign("model", "freeze_vision", args.fix_vit)
        assign("model", "freeze_visual_projector", args.fix_vit)
    precision = args.torch_dtype
    if args.bf16 is not None:
        precision = "bfloat16" if args.bf16 else "float32"
    if precision is not None:
        assign("model", "precision", "bf16" if precision == "bfloat16" else "fp32")
    assign("reward", "grounding_functions", args.reward_funcs)
    return load_config(path, args.paths, overrides)
