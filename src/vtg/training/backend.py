"""Backend configuration stays importable without torch or DeepSpeed installed."""

import copy
import json
from pathlib import Path


def read_deepspeed_config(path):
    config = json.loads(Path(path).expanduser().read_text())
    zero = config.get("zero_optimization", {})
    # Invert4TVG uses the other spelling. Accelerate 1.7 reads this alias.
    if "gather_16bit_weights_on_model_save" in zero:
        value = zero.pop("gather_16bit_weights_on_model_save")
        if zero.get("stage3_gather_16bit_weights_on_model_save", value) != value:
            raise ValueError("Conflicting ZeRO-3 gather-on-save settings")
        zero["stage3_gather_16bit_weights_on_model_save"] = value
    validate_deepspeed_config(config)
    return config


def validate_deepspeed_config(config):
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError("training.deepspeed must be a JSON object or null")
    zero = config.get("zero_optimization", {})
    if zero.get("stage") != 3:
        raise ValueError("The DeepSpeed backend requires ZeRO stage 3")
    if not zero.get("stage3_gather_16bit_weights_on_model_save"):
        raise ValueError("ZeRO-3 requires gathered adapter export on save")
    if config.get("optimizer", {}).get("type") != "AdamW":
        raise ValueError("Use the Invert4TVG AdamW optimizer configuration")


def backend_report(config):
    ds = config["training"].get("deepspeed")
    return {
        "backend": "deepspeed_zero3" if ds else "ddp",
        "deepspeed_config": copy.deepcopy(ds),
    }
