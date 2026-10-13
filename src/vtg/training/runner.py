"""One entry point for E5 training and all implemented continuation branches."""

import copy
import json
import time
from pathlib import Path

from vtg.training.schedule import build_schedule, epoch_budget
from vtg.utils.io import (
    digest,
    implementation_identity,
    read_jsonl,
    save_json,
    weight_identity,
    write_jsonl,
)


def training_inputs(config):
    root = Path(config["paths"]["artifact_root"])
    grounding = read_jsonl(root / "fit.jsonl")
    auxiliary = read_jsonl(root / "aux-fit.jsonl")
    if len({r["sample_id"] for r in grounding}) != len(grounding):
        raise ValueError("Duplicate training sample IDs")
    if not {r["sample_id"] for r in auxiliary} <= {r["sample_id"] for r in grounding}:
        raise ValueError("Auxiliary rows must come from the training manifest")
    if any(r.get("split") != "fit" for r in grounding + auxiliary):
        raise ValueError("Training accepts fit rows only, never test/dev data")
    config = copy.deepcopy(config)
    training = config["training"]
    prompts = training["unique_prompts_per_update"]
    epochs = training.get("num_train_epochs")
    if epochs is not None:
        budget = epoch_budget(len(grounding), epochs, prompts)
        training["max_optimizer_steps"] = budget["steps"]
    budget = {"steps": training["max_optimizer_steps"], "prompts_per_update": prompts}
    if epochs is not None:
        budget["epochs"] = epochs
    schedule = build_schedule(
        grounding,
        auxiliary,
        config["experiment_id"],
        budget,
        config["continuation"]["schedule_seed"],
    )
    return config, grounding, auxiliary, schedule


def preflight(config):
    config, grounding, auxiliary, schedule = training_inputs(config)
    from vtg.training.schedule import epoch_checkpoint_steps

    if not Path(config["paths"]["video_root"]).is_dir():
        raise FileNotFoundError(config["paths"]["video_root"])
    if config["branch"] != "E5":
        parent = Path(config["continuation"]["parent_checkpoint"])
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (parent / filename).is_file():
                raise FileNotFoundError(parent / filename)
    if config["branch"] == "A_matched":
        calibration = config["continuation"]["matched_calibration"]
        if not calibration or not Path(calibration).is_file():
            raise ValueError(
                "A_matched requires an existing training-only matched_calibration file"
            )
    from vtg.training.backend import backend_report

    return {
        **backend_report(config),
        "branch": config["branch"],
        "training_queries": len(grounding),
        "auxiliary_queries": len(auxiliary),
        "schedule_groups": len(schedule),
        "optimizer_steps": config["training"]["max_optimizer_steps"],
        "world_size": config["training"]["world_size"],
        "G": config["training"]["num_generations"],
        "video": config["video"],
        "manifest_hash": digest([grounding, auxiliary]),
        "schedule_hash": digest(schedule),
        "gpu_execution_verified": False,
        "save_strategy": config["training"].get("save_strategy", "steps"),
        "epoch_checkpoint_steps": (
            epoch_checkpoint_steps(schedule, config["training"]["gradient_accumulation_steps"])
            if config["training"].get("save_strategy", "steps") == "epoch"
            else []
        ),
    }


def run(config, output, resume=None, stop_after=None, coverage_size=None):
    import torch
    from transformers import TrainingArguments, set_seed
    from tqdm.auto import tqdm

    from vtg.models.loading import load_training_model
    from vtg.models.processing import load_processor
    from vtg.preference.diagnostics import coverage_report
    from vtg.tasks.reference_parser import CoreParser
    from vtg.training.checkpoints import RunCheckpoint, reconcile_logs, validate_resume
    from vtg.training.trainer import VTGTrainer
    from vtg.utils.distributed import barrier, broadcast, collective_call, gather_objects

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA. Use --dry-run or the CPU unit tests locally.")
    config, grounding, auxiliary, schedule = training_inputs(config)
    training = config["training"]
    total_steps = training["max_optimizer_steps"]
    if stop_after is not None and not 1 <= stop_after <= total_steps:
        raise ValueError("stop-after must be within the configured update budget")
    output = Path(output).resolve()
    parent = (
        Path(config["continuation"]["parent_checkpoint"]).resolve()
        if config["branch"] != "E5"
        else None
    )
    if parent and (output == parent or parent in output.parents or output in parent.parents):
        raise ValueError("Use a separate directory from the immutable parent checkpoint")
    from vtg.training.options import training_argument_values
    from vtg.training.schedule import epoch_checkpoint_steps

    args = TrainingArguments(**training_argument_values(config, output))
    _ = args.device
    if args.world_size != training["world_size"]:
        raise ValueError(f"Expected {training['world_size']} processes, got {args.world_size}")
    set_seed(config["seed"])

    def setup():
        if args.process_index != 0:
            return None
        preflight(config)
        if output.exists() and any(output.iterdir()) and not resume:
            raise ValueError(
                "Output directory is not empty. Choose a new output or explicitly resume."
            )
        parent_hash = weight_identity(parent or config["paths"]["model_path"])
        identity = {
            "config_hash": digest(config),
            "schedule_hash": digest(schedule),
            "manifest_hash": digest([grounding, auxiliary]),
            "source_hash": implementation_identity(),
            "parent_hash": parent_hash,
            "base_hash": weight_identity(config["paths"]["model_path"]) if parent else parent_hash,
            "calibration_hash": (
                digest(json.loads(Path(config["continuation"]["matched_calibration"]).read_text()))
                if config["branch"] == "A_matched"
                else None
            ),
        }
        step = validate_resume(resume, output, identity, config) if resume else 0
        if stop_after is not None and stop_after <= step:
            raise ValueError("stop-after must exceed the resumed step")
        if coverage_size and resume:
            raise ValueError("Coverage uses a fresh frozen-policy run, not a training resume")
        output.mkdir(parents=True, exist_ok=True)
        if resume:
            reconcile_logs(output, step)
        save_json(output / "run_identity.json", identity)
        save_json(output / "resolved_config.json", config)
        write_jsonl(output / "task_schedule.jsonl", schedule)
        save_json(output / "status.json", {"status": "initializing", "step": step})
        return {"identity": identity, "step": step, "parent_hash": parent_hash}

    setup_result = broadcast(collective_call(setup))
    model, guard = collective_call(lambda: load_training_model(config))
    processor = collective_call(lambda: load_processor(config))
    parser = collective_call(lambda: CoreParser(config["paths"]["spacy_model"]))
    callbacks = [
        RunCheckpoint(
            setup_result["identity"],
            args.gradient_accumulation_steps,
            stop_after,
            (
                epoch_checkpoint_steps(schedule, args.gradient_accumulation_steps)
                if training.get("save_strategy", "steps") == "epoch"
                else ()
            ),
        )
    ]
    # The dataset supplies HF's batch accounting. The immutable schedule owns prompts.
    dataset = [{"index": index} for index in range(len(schedule) * training["num_generations"])]
    trainer = VTGTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        config=config,
        schedule=schedule,
        records={r["sample_id"]: r for r in grounding},
        parser=parser,
        guard=guard,
        parent_hash=setup_result["parent_hash"],
        callbacks=callbacks,
    )
    if trainer.matched and resume and trainer.is_world_process_zero():
        for row in read_jsonl(output / "pair_diagnostics.jsonl"):
            if row["pair"]:
                trainer.matched.counts[row["geometry_bin"]] += 1
    # Model initialization is identical; stochastic candidates must differ across ranks.
    # HF restores each rank's saved RNG when resuming.
    set_seed(config["seed"] + args.process_index)
    started = time.monotonic()
    from vtg.training.zero import adapter_snapshot, check_parameters

    frozen = adapter_snapshot(model)
    try:
        if coverage_size is not None:
            if config["branch"] == "E5" or not 1 <= coverage_size <= len(grounding):
                raise ValueError(
                    "Coverage requires a continuation branch and a valid training-only sample size"
                )
            import random

            chosen = sorted(grounding, key=lambda r: r["sample_id"])
            random.Random(config["continuation"]["schedule_seed"]).shuffle(chosen)
            trainer.schedule = [
                {
                    "group_id": f"coverage-{i}",
                    "schedule_index": i,
                    "sample_id": r["sample_id"],
                    "task": "TVG",
                    "hidden_field": None,
                }
                for i, r in enumerate(chosen[:coverage_size])
            ]
            trainer.prepare_coverage_model()
            for index in tqdm(
                range(coverage_size),
                desc="Frozen-policy pair coverage",
                disable=args.process_index != 0,
            ):
                trainer.state.global_step = index // args.gradient_accumulation_steps
                batch = trainer._prepare_inputs({})
                del batch
            barrier()
            if args.process_index == 0:
                report = coverage_report(
                    read_jsonl(output / "pair_diagnostics.jsonl"),
                    len(grounding),
                    len(grounding),
                    "full_query_v2",
                )
                report.update(
                    identity=setup_result["identity"],
                    candidate_group_size=training["num_generations"],
                )
                save_json(output / "coverage.json", report)
            status = "coverage_complete"
        else:
            metrics = trainer.train(resume_from_checkpoint=str(resume) if resume else None).metrics
            trainer.save_model(str(output / "model"))
            if args.process_index == 0:
                save_json(output / "train_metrics.json", metrics)
            status = "trained" if trainer.state.global_step == total_steps else "paused"
        parameter_error = check_parameters(model, frozen, args.device)
        costs = gather_objects(
            {
                "rank": args.process_index,
                "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "judge": trainer.verifier.cost,
            }
        )
        if args.process_index == 0:
            save_json(
                output / "status.json",
                {
                    "status": status,
                    "backend": "deepspeed_zero3" if trainer.is_deepspeed_enabled else "ddp",
                    "global_step": trainer.state.global_step,
                    "frozen_C0_unchanged": True,
                    "rank_parameter_max_error": float(parameter_error),
                    "ranks": costs,
                },
            )
    except Exception as exc:
        save_json(
            output / f"failure.rank{args.process_index}.json",
            {"error": repr(exc), "step": trainer.state.global_step},
        )
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
