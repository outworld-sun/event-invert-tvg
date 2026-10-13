"""Resume identities prevent mixing data, visual budgets, or candidate layouts."""

import json
from datetime import datetime
from pathlib import Path

from transformers import TrainerCallback

from vtg.utils.io import read_jsonl, save_json, write_jsonl


def validate_resume(checkpoint, output, identity, config):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent != Path(output).resolve():
        raise ValueError("Resume must belong to this output directory")
    state = json.loads((checkpoint / "schedule_state.json").read_text())
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    if config["training"].get("save_only_model", False):
        raise ValueError("Model-only checkpoints cannot resume optimizer/RNG state")
    step = trainer["global_step"]
    expected = step * config["training"]["gradient_accumulation_steps"]
    if state["identity"] != identity or state["next_schedule_index"] != expected:
        raise ValueError("Resume code/config/data/weights/layout identity differs")
    required = ["adapter_model.safetensors"]
    if config["training"].get("deepspeed"):
        latest = checkpoint / "latest"
        tag = latest.read_text().strip()
        if tag != f"global_step{step}":
            raise ValueError("DeepSpeed checkpoint tag differs from Trainer global_step")
        shards = checkpoint / tag
        world = config["training"]["world_size"]
        precision_prefix = "bf16_" if config["model"]["precision"] == "bf16" else ""
        for rank in range(world):
            for filename in (
                f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt",
                f"{precision_prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
            ):
                if not (shards / filename).is_file():
                    raise FileNotFoundError(shards / filename)
    else:
        required.extend(["optimizer.pt", "scheduler.pt"])
    world = config["training"]["world_size"]
    required += (
        [f"rng_state_{rank}.pth" for rank in range(world)] if world > 1 else ["rng_state.pth"]
    )
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    return step


def reconcile_logs(output, step):
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(output)
    for pattern in ("rollouts.rank*.jsonl", "losses.rank*.jsonl", "pair_diagnostics.jsonl"):
        for path in root.glob(pattern):
            rows = read_jsonl(path)
            retained = [r for r in rows if r["global_step"] < step]
            discarded = [r for r in rows if r["global_step"] >= step]
            if discarded:
                write_jsonl(root / "discarded_attempts" / f"{tag}_{path.name}", discarded)
                write_jsonl(path, retained)


class RunCheckpoint(TrainerCallback):
    def __init__(self, identity, accumulation, stop_after=None, epoch_save_steps=()):
        self.identity = identity
        self.accumulation = accumulation
        self.stop_after = stop_after
        self.epoch_save_steps = set(epoch_save_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.epoch_save_steps:
            control.should_save = True
        if self.stop_after is not None and state.global_step >= self.stop_after:
            control.should_save = True
            control.should_training_stop = True

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            save_json(
                Path(args.output_dir) / f"checkpoint-{state.global_step}" / "schedule_state.json",
                {
                    "identity": self.identity,
                    "next_schedule_index": state.global_step * self.accumulation,
                },
            )
