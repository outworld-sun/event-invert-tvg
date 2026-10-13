"""Frozen full-query matching (or explicit legacy SER) on actual video crops."""

import copy
import json
import time
from pathlib import Path
from vtg.tasks.builder import build_task_instance
from vtg.tasks.schemas import TaskInstance
from vtg.rewards.scoring import score_completion
from vtg.utils.io import digest, save_json
from vtg.video.decoder import DECODER_VERSION
from vtg.preference.pairs import absolute_crop
from vtg.preference.protocol import FULL_QUERY, protocol, query_prompt, parse_query_verdict


def judge_instance(record, local_interval, config):
    if protocol(config) == FULL_QUERY:
        request = {
            "video_relpath": record["video_relpath"],
            "video_id": record["video_id"],
            "interval": absolute_crop(record, local_interval),
            "sampling": copy.deepcopy(config.get("video", {})),
        }
        identity = digest([record["sample_id"], FULL_QUERY, local_interval, record["query"]])
        return TaskInstance(
            identity,
            identity,
            "QUERY_MATCH",
            record["sample_id"],
            query_prompt(record["query"]),
            request,
            {},
            [],
            [],
            template_version=FULL_QUERY,
            sampling_seed=config.get("seed", 3407),
        )
    instance = build_task_instance(record, "SER", config=config)
    instance.video_request["interval"] = absolute_crop(record, local_interval)
    return instance


class CandidateVerifier:

    def __init__(self, model, processor, video, parser, config, adapter_hash, cache_dir):
        self.model, self.processor, self.video, self.parser = (model, processor, video, parser)
        self.config, self.adapter_hash = (config, adapter_hash)
        self.cache_dir = Path(cache_dir)
        self.cost = {
            "judge_calls": 0,
            "cache_hits": 0,
            "judge_seconds": 0.0,
            "visual_tokens": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "decode_failures": 0,
        }

    def verify(self, record, interval):
        import torch

        inst = judge_instance(record, interval, self.config)
        try:
            frames, meta = self.video.load(inst.video_request)
        except (ValueError, OSError, EOFError) as error:
            self.cost["decode_failures"] += 1
            return {"decoded": False, "parseable": False, "joint": False, "error": str(error)}
        from vtg.models.processing import encode_video

        batch = encode_video(self.processor, inst.prompt_text, frames, meta, self.config)
        batch = batch.to(next(self.model.parameters()).device)
        import hashlib

        tensor_hashes = {}
        for key, value in batch.items():
            if hasattr(value, "detach"):
                tensor_hashes[key] = hashlib.sha256(
                    value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            else:
                tensor_hashes[key] = digest(value)
        teacher_input = digest([tensor_hashes, inst.prompt_text, self.adapter_hash])
        source = Path(self.video.root) / record["video_relpath"]
        stat = source.stat()
        identity = [
            record.get("source"),
            record["video_id"],
            str(source),
            stat.st_size,
            stat.st_mtime_ns,
            record.get("media_offset", 0),
            record["duration_sec"],
            inst.video_request["interval"],
            meta["pts"],
            self.config["video"],
            self.video.processor_config,
            DECODER_VERSION,
            inst.supported_fields,
            inst.template_version,
            self.adapter_hash,
            teacher_input,
        ]
        key = digest(identity)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity_hash"] != key:
                raise ValueError("Judge cache identity mismatch")
            completion = saved["completion"]
            self.cost["cache_hits"] += 1
        else:
            start = time.monotonic()
            with torch.no_grad():
                output = self.model.generate(
                    **batch,
                    do_sample=False,
                    synced_gpus=False,
                    use_cache=True,
                    max_new_tokens=self.config["training"]["max_completion_tokens"],
                    temperature=None,
                    top_p=None,
                )
            length = batch["input_ids"].shape[1]
            completion = self.processor.batch_decode(output[:, length:], skip_special_tokens=True)[
                0
            ]
            self.cost["judge_calls"] += 1
            self.cost["judge_seconds"] += time.monotonic() - start
            self.cost["visual_tokens"] += meta["visual_tokens"]
            self.cost["prompt_tokens"] += length
            self.cost["generated_tokens"] += output.shape[1] - length
            save_json(
                path,
                {
                    "identity_hash": key,
                    "completion": completion,
                    "teacher_input_hash": teacher_input,
                    "frame_pts": meta["pts"],
                },
            )
            del output
        if protocol(self.config) == FULL_QUERY:
            verdict = parse_query_verdict(completion)
        else:
            result = score_completion(inst, completion, self.config["reward"], self.parser)
            verdict = {
                "parseable": result.validity,
                "joint": bool(result.field_scores) and all(result.field_scores.values()),
                "predicate_match": bool(result.field_scores.get("predicate", 0)),
                "semantic": result.r_semantic,
                "candidate": result.candidate,
                "failure_code": result.failure_code,
            }
        del batch, frames
        return {
            "decoded": True,
            **verdict,
            "completion": completion,
            "verification_protocol": protocol(self.config),
            "teacher_input_hash": teacher_input,
            "cache_key": key,
            "frame_pts": meta["pts"],
            "repeated_frames": meta["repeated_frames"],
        }
