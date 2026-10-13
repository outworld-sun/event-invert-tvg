"""Composable JSON configuration with explicit paths and validated budgets."""

import copy
import json
import math
import os
from pathlib import Path

BRANCHES = {
    "E5": (False, "none", False),
    "N0_E5_cont": (False, "none", False),
    "N1_IoU_pair": (True, "none", False),
    "N2_Event_pair": (True, "joint", False),
    "N3_Swap": (False, "none", True),
    "N4_Event_swap": (True, "joint", True),
    "A_matched": (True, "none", False),
    "A_verb_gate": (True, "predicate", False),
    "A_sem_add": (False, "none", False),
}


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_config(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError("Circular configuration inheritance")
    config = json.loads(path.read_text())
    backend = config.get("training", {}).get("deepspeed")
    if isinstance(backend, str):
        from vtg.training.backend import read_deepspeed_config

        config["training"]["deepspeed"] = read_deepspeed_config(path.parent / backend)
    parent = config.pop("extends", None)
    if parent:
        config = merge(read_config(path.parent / parent, (*stack, path)), config)
    return config


def load_config(path, paths=None, overrides=None):
    config = read_config(path)
    if paths:
        config = merge(config, read_config(paths))
    config = merge(config, overrides or {})

    # Relative resource paths are relative to the caller's working directory.
    # Model IDs and spaCy package names remain valid string identifiers.
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            expanded = os.path.expanduser(os.path.expandvars(value))
            if "${" in expanded:
                raise ValueError(f"Unresolved environment variable: {value}")
            return expanded
        return value

    config = expand(config)
    branch = config["branch"]
    if branch not in BRANCHES:
        raise ValueError(f"Unsupported branch: {branch}")
    pair, gate, swap = BRANCHES[branch]
    config["continuation"].update(
        branch=branch, pair_enabled=pair, candidate_gate=gate, swap_enabled=swap
    )
    validate(config)
    return config


def validate(config):
    training = config["training"]
    video = config["video"]
    from vtg.preference.protocol import protocol

    protocol(config)
    if not config["model"]["freeze_vision"] or not config["model"]["freeze_visual_projector"]:
        raise ValueError("This trainer requires a frozen vision encoder and projector")
    if config["model"]["lora"]["scope"] != "language_model_only":
        raise ValueError("Only language-side LoRA is supported")
    if config["model"]["precision"] not in ("bf16", "fp32"):
        raise ValueError("Supported precisions are bf16 and fp32")
    if not isinstance(config["model"]["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be a boolean")
    from vtg.training.backend import validate_deepspeed_config

    validate_deepspeed_config(training.get("deepspeed"))
    strategy = training.get("save_strategy", "steps")
    if strategy not in ("steps", "epoch", "no"):
        raise ValueError("save_strategy must be steps, epoch, or no")
    epochs = training.get("num_train_epochs")
    if epochs is not None and (
        not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1
    ):
        raise ValueError("num_train_epochs must be a positive integer or null")
    if strategy == "epoch" and epochs is None:
        raise ValueError("save_strategy=epoch requires num_train_epochs, not max_steps")
    for key in ("logging_steps", "save_steps", "save_total_limit"):
        value = training.get(key, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    functions = config["reward"].get("grounding_functions", ["iou", "format"])
    if not functions or len(set(functions)) != len(functions) or set(functions) - {"iou", "format"}:
        raise ValueError(
            "grounding_functions supports iou and format; upstream iou_v2 is not an alias"
        )
    for key in (
        "world_size",
        "completion_micro_batch_per_device",
        "num_generations",
        "gradient_accumulation_steps",
        "max_optimizer_steps",
    ):
        value = training[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if (
        training["world_size"] * training["completion_micro_batch_per_device"]
        != training["num_generations"]
    ):
        raise ValueError("One global group requires G = world_size × local completions")
    if training["num_generations"] < 2:
        raise ValueError("GRPO requires at least two candidates")
    if training["gradient_accumulation_steps"] != training["unique_prompts_per_update"]:
        raise ValueError("Each accumulation step consumes one independent prompt")
    for key in ("min_frames", "max_frames"):
        if video[key] < 2 or video[key] % 2:
            raise ValueError(f"video.{key} must be positive and even")
    if video["min_frames"] > video["max_frames"] or video["fps"] <= 0:
        raise ValueError("Invalid frame range or FPS")
    if video["allocation"] not in ("dynamic", "fixed"):
        raise ValueError("Pixel allocation must be dynamic or fixed")
    if not 784 <= video["min_pixels_per_frame"] <= video["max_pixels_per_frame"]:
        raise ValueError("Invalid per-frame pixel bounds")
    if min(video["total_pixels"], video["visual_token_cap"]) <= 0:
        raise ValueError("Pixel and token budgets must be positive")
    for key in ("learning_rate", "temperature"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if config["model"]["lora"]["dropout"] != 0:
        raise ValueError("Single-iteration on-policy likelihood requires zero LoRA dropout")
    for key in ("alpha_swap", "beta_dpo", "lambda_pair", "min_iou", "min_gap", "weight_gap"):
        value = config["continuation"][key]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid preference parameter {key}")
    if min(config["continuation"][k] for k in ("beta_dpo", "min_gap", "weight_gap")) <= 0:
        raise ValueError("Preference beta and gap scales must be positive")
