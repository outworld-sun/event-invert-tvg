"""Generate a global candidate group before any actor backward graph is open."""

import copy
import time

import torch

from vtg.models.processing import encode_video
from vtg.preference.pairs import swap_penalty
from vtg.preference.selection import select_preference
from vtg.rewards.scoring import score_completion
from vtg.tasks.builder import build_task_instance
from vtg.training.losses import completion_logps, completion_mask
from vtg.training.schedule import compute_group_advantages
from vtg.training.zero import rollout_model
from vtg.utils.distributed import collective_call, gather_objects
from vtg.utils.io import append_jsonl, digest


def prepare_rollout(trainer):
    started = time.monotonic()
    config = trainer.config
    accumulation = trainer.args.gradient_accumulation_steps
    index = trainer.state.global_step * accumulation + trainer.micro_cursor % accumulation
    trainer.micro_cursor += 1
    item = trainer.schedule[index]
    record = copy.deepcopy(trainer.records[item["sample_id"]])
    if config["experiment_id"] == "A_action_patient":
        record["semantics"]["fields"].pop("agent", None)
        record["semantics"]["supported_fields"] = [
            f for f in record["semantics"]["supported_fields"] if f != "agent"
        ]
    instance = build_task_instance(
        record, item["task"], item["hidden_field"], {**config, "group_id": item["group_id"]}
    )

    def encode():
        frames, meta = trainer.video.load(instance.video_request)
        batch = encode_video(
            trainer.processing_class,
            instance.prompt_text,
            frames,
            meta,
            config,
            copies=trainer.args.per_device_train_batch_size,
        )
        return batch.to(trainer.args.device), meta

    batch, meta = collective_call(encode)
    witnesses = gather_objects(
        (item["group_id"], digest(instance.prompt_text), meta["video_sample_hash"])
    )
    if len(set(witnesses)) != 1:
        raise ValueError("Ranks must see the same prompt and decoded frames")
    with rollout_model(trainer) as actor:

        def generate():
            actor.eval()
            with torch.no_grad():
                output = actor.generate(
                    **batch,
                    do_sample=True,
                    temperature=config["training"]["temperature"],
                    top_p=config["training"]["top_p"],
                    max_new_tokens=config["training"]["max_completion_tokens"],
                    use_cache=True,
                    synced_gpus=False,
                    pad_token_id=trainer.processing_class.tokenizer.pad_token_id,
                )
            return output[:, batch["input_ids"].shape[1] :]

        completion_ids = collective_call(generate)
        mask = completion_mask(completion_ids, trainer.processing_class.tokenizer.eos_token_id)
        completions = trainer.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        def score():
            results = []
            for text in completions:
                result = score_completion(instance, text, config["reward"], trainer.parser)
                penalty = swap_penalty(
                    instance,
                    result,
                    config["continuation"]["swap_enabled"],
                    config["continuation"]["alpha_swap"],
                )
                result.r_semantic -= penalty
                result.total -= penalty
                results.append({**result.to_dict(), "completion": text, "swap_penalty": penalty})
            return results

        details = collective_call(score)
        totals = [
            value for shard in gather_objects([r["total"] for r in details]) for value in shard
        ]
        with trainer.guard.reference() as frozen:
            reference = collective_call(
                lambda: completion_logps(frozen, batch, completion_ids, mask).detach()
            )
            pair, diagnostic = select_preference(trainer, record, item, completions)
        actor.train()
    original_totals = totals[:]
    if config["branch"] == "A_sem_add" and diagnostic["gt_admitted"]:
        totals = [
            (
                reward
                + config["continuation"]["semantic_eta"] * candidate["judge"].get("semantic", 0.0)
                if candidate["valid"] and candidate["iou"] >= config["continuation"]["min_iou"]
                else reward
            )
            for reward, candidate in zip(totals, diagnostic["candidates"])
        ]
    group_size = config["training"]["num_generations"]
    advantages = compute_group_advantages(
        [item["group_id"]] * group_size, totals, group_size, config["training"]["advantage_epsilon"]
    )
    offset = trainer.args.process_index * len(details)
    diagnostic.update(
        original_rewards=original_totals,
        rewards=totals,
        advantages=advantages,
        num_generations=group_size,
        prepare_seconds=time.monotonic() - started,
    )
    if trainer.is_world_process_zero():
        append_jsonl(trainer.output / "pair_diagnostics.jsonl", diagnostic)
    for local_index, detail in enumerate(details):
        append_jsonl(
            trainer.output / f"rollouts.rank{trainer.args.process_index}.jsonl",
            {
                **detail,
                "global_step": trainer.state.global_step,
                "group_id": item["group_id"],
                "sample_id": record["sample_id"],
                "task": item["task"],
                "rollout_index": offset + local_index,
                "rank": trainer.args.process_index,
                "advantage": advantages[offset + local_index],
                "video": meta,
            },
        )
    trainer.last_rollout = {
        "reward": sum(totals) / len(totals),
        "pair_rate": float(pair is not None),
        "visual_tokens": meta["visual_tokens"],
        "frames": meta["shape"][0],
        "zero_variance": float(max(totals) == min(totals)),
    }
    return {
        "batch": batch,
        "completion_ids": completion_ids,
        "completion_mask": mask,
        "reference_logps": reference,
        "pair": pair,
        "group_id": item["group_id"],
        "advantages": torch.tensor(
            advantages[offset : offset + len(details)], device=trainer.args.device
        ),
    }
