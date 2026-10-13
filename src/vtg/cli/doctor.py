"""Show runtime versions, visible GPUs, configuration, and missing resources."""

import argparse
import importlib.metadata
import json
from pathlib import Path

from vtg.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths")
    args = parser.parse_args()
    config = load_config(args.config, args.paths)
    import torch
    import spacy

    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "accelerate", "peft", "av", "deepspeed")
    }
    devices = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "memory_gib": torch.cuda.get_device_properties(index).total_memory / 2**30,
        }
        for index in range(torch.cuda.device_count())
    ]
    checks = {
        key: Path(config["paths"][key]).exists()
        for key in ("model_path", "video_root", "train_annotations", "artifact_root")
    }
    model = config["paths"]["spacy_model"]
    checks["spacy_model"] = Path(model).exists() or spacy.util.is_package(model)
    report = {
        "backend": "deepspeed_zero3" if config["training"].get("deepspeed") else "ddp",
        "versions": versions,
        "cuda": torch.version.cuda,
        "devices": devices,
        "paths_exist": checks,
        "configured_world_size": config["training"]["world_size"],
        "num_generations": config["training"]["num_generations"],
        "video": config["video"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(devices) < config["training"]["world_size"] or not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
