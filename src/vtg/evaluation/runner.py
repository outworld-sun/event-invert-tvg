"""Resumable benchmark inference with disjoint GPU shards and exact denominators."""

import json
import os
import time
from pathlib import Path

from vtg.evaluation.metrics import evaluate_grounding
from vtg.models.processing import encode_video, load_processor
from vtg.tasks.builder import build_task_instance
from vtg.utils.distributed import barrier, broadcast, collective_call
from vtg.utils.io import (
    append_jsonl,
    digest,
    implementation_identity,
    read_jsonl,
    save_json,
    weight_identity,
    write_jsonl,
)
from vtg.video.decoder import VideoAdapter


def merge_predictions(directory, annotations):
    rows = [
        row
        for path in sorted(Path(directory).glob("predictions.rank*.jsonl"))
        for row in read_jsonl(path)
    ]
    evaluate_grounding(rows, annotations)
    by_id = {r["sample_id"]: r for r in rows}
    return [by_id[r["sample_id"]] for r in annotations]


def run(config, checkpoint, split, output, base=False, limit=None):
    import torch
    from tqdm.auto import tqdm
    from transformers import Qwen2_5_VLForConditionalGeneration

    if bool(checkpoint) == bool(base):
        raise ValueError("Choose exactly one of --checkpoint or --base")
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires CUDA")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    output = Path(output)
    try:
        spec = config["benchmarks"][split]
        annotations = read_jsonl(spec["manifest"])
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            annotations = annotations[:limit]
        if not annotations or len({r["sample_id"] for r in annotations}) != len(annotations):
            raise ValueError("Benchmark must be nonempty and have unique sample IDs")

        def setup():
            if rank != 0:
                return None
            identity = {
                "checkpoint_hash": weight_identity(checkpoint or config["paths"]["model_path"]),
                "base_model": str(Path(config["paths"]["model_path"]).resolve()),
                "base_hash": weight_identity(config["paths"]["model_path"]),
                "video": config["video"],
                "manifest_hash": digest(annotations),
                "split": split,
                "source_hash": implementation_identity(),
                "max_completion_tokens": config["training"]["max_completion_tokens"],
                "attn_implementation": config["training"]["attn_implementation"],
                "partial_benchmark": limit is not None,
            }
            output.mkdir(parents=True, exist_ok=True)
            path = output / "evaluation_identity.json"
            if path.exists() and json.loads(path.read_text()) != identity:
                raise ValueError(
                    "Evaluation output belongs to different weights/data/config. Use a new directory."
                )
            if not path.exists() and any(output.iterdir()):
                raise ValueError("Nonempty evaluation output has no verified identity")
            save_json(path, identity)
            save_json(output / "resolved_config.json", config)
            return identity

        identity = broadcast(collective_call(setup))
        previous = [
            row
            for path in sorted(output.glob("predictions.rank*.jsonl"))
            for row in read_jsonl(path)
        ]
        done = {r["sample_id"] for r in previous}
        if len(done) != len(previous) or not done <= {r["sample_id"] for r in annotations}:
            raise ValueError("Resumed shards contain duplicate or unexpected prediction IDs")
        pending = [
            r
            for index, r in enumerate(annotations)
            if index % world == rank and r["sample_id"] not in done
        ]
        started = time.monotonic()
        if pending:
            processor = load_processor(config)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config["paths"]["model_path"],
                torch_dtype=torch.bfloat16,
                attn_implementation=config["training"]["attn_implementation"],
            )
            if checkpoint:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, checkpoint)
            model.to(torch.device("cuda", local_rank)).eval()
            video = VideoAdapter(
                spec["video_root"], config["video"], processor.image_processor.to_dict()
            )
            progress = tqdm(
                pending, desc=f"{split} GPU {rank + 1}/{world}", position=rank, dynamic_ncols=True
            )
            for record in progress:
                completion = ""
                meta = {}
                error = record.get("input_error")
                if not error:
                    instance = build_task_instance(record, "TVG", config=config)
                    try:
                        frames, meta = video.load(instance.video_request)
                    except (ValueError, OSError, EOFError) as exc:
                        error = str(exc)
                    if not error:
                        inputs = encode_video(
                            processor, instance.prompt_text, frames, meta, config
                        ).to(model.device)
                        with torch.inference_mode():
                            prediction = model.generate(
                                **inputs,
                                max_new_tokens=config["training"]["max_completion_tokens"],
                                do_sample=False,
                                use_cache=True,
                                temperature=None,
                                top_p=None,
                            )
                        length = inputs["input_ids"].shape[1]
                        completion = processor.batch_decode(
                            prediction[:, length:], skip_special_tokens=True
                        )[0]
                        meta.update(
                            prompt_tokens=length, generated_tokens=prediction.shape[1] - length
                        )
                        del inputs, prediction, frames
                append_jsonl(
                    output / f"predictions.rank{rank}.jsonl",
                    {
                        "sample_id": record["sample_id"],
                        "video_id": record["video_id"],
                        "completion": completion,
                        "input_error": error,
                        "video": meta,
                    },
                )
        barrier()
        if rank == 0:
            predictions = merge_predictions(output, annotations)
            metrics = evaluate_grounding(predictions, annotations)
            metrics.update(
                split=split,
                checkpoint=str(checkpoint),
                world_size=world,
                partial_benchmark=limit is not None,
                evaluation_identity=digest(identity),
                segment_elapsed_seconds=time.monotonic() - started,
                input_error_count=sum(bool(r.get("input_error")) for r in predictions),
            )
            metrics["by_dataset"] = {}
            for name in sorted({r.get("dataset_name", split) for r in annotations}):
                subset = [r for r in annotations if r.get("dataset_name", split) == name]
                ids = {r["sample_id"] for r in subset}
                metrics["by_dataset"][name] = evaluate_grounding(
                    [p for p in predictions if p["sample_id"] in ids], subset
                )
            write_jsonl(output / "predictions.jsonl", predictions)
            save_json(output / "metrics.json", metrics)
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
