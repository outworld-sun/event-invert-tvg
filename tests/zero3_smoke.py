"""Explicit GPU integration witness, separate from the offline unittest suite.

Uses a tiny random Qwen saved locally, real autoregressive generation, real
ZeRO-3/CPUAdam, and controlled rewards/pairs. This is not a scientific experiment.
Launch each phase in a fresh torchrun process, including checkpoint resume.
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from test_model_integration import tiny_model, tiny_processor
from test_protocol import record
from vtg.config import load_config
from vtg.utils.io import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(123)
    base = tiny_model()
    base.generation_config.suppress_tokens = [5, 6, 7, 8]
    base.save_pretrained(root / "base")
    tiny_processor().save_pretrained(root / "base")
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["model.layers.0.self_attn.q_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.uniform_(-0.1, 0.1)
    model.save_pretrained(root / "parent")
    (root / "videos").mkdir()
    (root / "videos/fixture.mp4").write_bytes(b"video-loader-fixture")
    write_jsonl(root / "data/fit.jsonl", [record()])
    write_jsonl(root / "data/aux-fit.jsonl", [record()])


def execute(options):
    from vtg.tasks.schemas import RewardBreakdown
    from vtg.training.runner import run

    root = options.root.resolve()
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.set_num_threads(1)
    config = load_config(
        ROOT / ("configs/e5_8xa100.json" if options.branch == "E5" else "configs/n1_8xa100.json"),
        overrides={
            "paths": {
                "model_path": str(root / "base"),
                "artifact_root": str(root / "data"),
                "video_root": str(root / "videos"),
            },
            "model": {"lora": {"rank": 2, "alpha": 4, "targets": ["q_proj"]}},
            "continuation": {"parent_checkpoint": str(root / "parent")},
            "training": {
                "world_size": world,
                "num_generations": world,
                "num_train_epochs": None,
                "max_optimizer_steps": 2,
                "unique_prompts_per_update": 2,
                "gradient_accumulation_steps": 2,
                "max_completion_tokens": 3,
                "learning_rate": 0.01,
                "warmup_ratio": 0,
                "save_steps": 1,
                "attn_implementation": "eager",
                "save_only_model": options.model_only,
            },
            "logging": {"tensorboard": False},
        },
    )
    frames = torch.zeros(4, 3, 56, 56, dtype=torch.uint8)
    meta = {
        "effective_fps": 2.0,
        "visual_tokens": 8,
        "video_sample_hash": "fixture",
        "shape": list(frames.shape),
        "pts": [39.0, 39.5, 40.0, 40.5],
        "repeated_frames": 0,
    }

    def selection(trainer, sample, item, completions):
        # Real frozen generation with rank-asymmetric call counts and cache hits.
        verdict = trainer.verifier.verify(sample, [2, 6])
        if rank == 0:
            trainer.verifier.verify(sample, [1, 7])
            trainer.verifier.verify(sample, [2, 6])
        pair = {"winner": 0, "loser": 1, "weight": 0.5}
        return pair, {
            "global_step": trainer.state.global_step,
            "group_id": item["group_id"],
            "sample_id": sample["sample_id"],
            "video_id": sample["video_id"],
            "pair": pair,
            "gt_admitted": True,
            "candidates": [],
            "gt_judge": verdict,
        }

    def score(*args, **kwargs):
        return RewardBreakdown(total=float(rank), validity=True)

    with (
        patch("vtg.tasks.reference_parser.CoreParser", return_value=None),
        patch("vtg.video.decoder.VideoAdapter.load", return_value=(frames, meta)),
        patch("vtg.training.rollout.score_completion", side_effect=score),
    ):
        if options.branch == "E5":
            run(
                config,
                options.output,
                resume=options.resume,
                stop_after=options.stop_after,
                coverage_size=options.coverage_size,
            )
        else:
            with patch("vtg.training.rollout.select_preference", side_effect=selection):
                run(
                    config,
                    options.output,
                    resume=options.resume,
                    stop_after=options.stop_after,
                    coverage_size=options.coverage_size,
                )
    if rank == 0:
        status = json.loads((options.output / "status.json").read_text())
        assert status["backend"] == "deepspeed_zero3"
        backend = json.loads((options.output / "backend.json").read_text())
        assert backend["optimizer"] == "DeepSpeedCPUAdam"
        assert backend["stage"] == 3
        assert backend["partitioned_parameters"] > 0
        assert status["frozen_C0_unchanged"]
        assert status["rank_parameter_max_error"] == 0
        if options.coverage_size:
            assert status["status"] == "coverage_complete"
            print("ZERO3_COVERAGE_OK", flush=True)
            return
        checkpoint = options.output / f"checkpoint-{status['global_step']}"
        assert (checkpoint / "adapter_model.safetensors").is_file()
        assert (checkpoint / "latest").exists() != options.model_only
        rows = read_jsonl(options.output / "losses.rank0.jsonl")
        assert all(torch.isfinite(torch.tensor(row["grpo_loss"])) for row in rows)
        if options.branch != "E5":
            assert any(row["pair_loss"] > 0 for row in rows)
        print("ZERO3_SMOKE_OK", json.dumps(status), flush=True)


def compare(root):
    from safetensors.torch import load_file

    resumed = load_file(root / "resumed/model/adapter_model.safetensors")
    uninterrupted = load_file(root / "uninterrupted/model/adapter_model.safetensors")
    parent = load_file(root / "parent/adapter_model.safetensors")
    for name in resumed:
        torch.testing.assert_close(resumed[name], uninterrupted[name], rtol=0, atol=0)
    assert any(
        not torch.equal(value, parent[name].to(value.dtype)) for name, value in resumed.items()
    )
    print("ZERO3_RESUME_EXACT_MATCH_AND_ACTOR_UPDATED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixture", "run", "compare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--branch", choices=("E5", "N1"), default="N1")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--coverage-size", type=int)
    options = parser.parse_args()
    if options.mode == "fixture":
        fixture(options.root)
    elif options.mode == "compare":
        compare(options.root)
    else:
        execute(options)


if __name__ == "__main__":
    main()
