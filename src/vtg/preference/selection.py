"""Select one real on-policy pair after the frozen GT and candidate gates."""

from vtg.preference.pairs import (
    candidate,
    enumerate_pairs,
    no_pair_reason,
    prompt_eligible,
    reference_eligible,
    select_pair,
)
from vtg.preference.protocol import protocol
from vtg.preference.diagnostics import geometry_bin
from vtg.utils.distributed import broadcast, collective_call, gather_objects


def select_preference(trainer, record, item, completions):
    config = trainer.config
    preference = config["continuation"]
    enabled = item["task"] == "TVG" and (
        preference["pair_enabled"] or config["branch"] == "A_sem_add"
    )
    gt = {}
    admitted = False
    candidates = []
    pair = None
    counts = {}
    if enabled:

        def judge_gt():
            if trainer.is_world_process_zero() and reference_eligible(
                record, trainer.reference_ids, protocol(config)
            ):
                return trainer.verifier.verify(record, record["gt_sec"])
            return None

        gt = broadcast(collective_call(judge_gt)) or {}
        admitted = prompt_eligible(record, trainer.reference_ids, gt, protocol(config))

        def judge_candidates():
            rows = []
            offset = trainer.args.process_index * len(completions)
            for index, text in enumerate(completions):
                row = candidate(record, text, offset + index)
                if admitted and row["valid"]:
                    row["judge"] = trainer.verifier.verify(record, row["interval"])
                rows.append(row)
            return rows

        local = collective_call(judge_candidates)
        candidates = [row for shard in gather_objects(local) for row in shard]
        pairs, counts = enumerate_pairs(candidates, preference, admitted)

        def choose():
            if not trainer.is_world_process_zero() or not preference["pair_enabled"]:
                return None
            selected = select_pair(pairs, config["seed"], item["group_id"])
            if selected and trainer.matched:
                key = geometry_bin(selected, record, candidates)
                if not trainer.matched.accept(key, item["group_id"]):
                    selected = None
            return selected

        pair = broadcast(collective_call(choose))
    diagnostic = {
        "split": "train",
        "group_id": item["group_id"],
        "global_step": trainer.state.global_step,
        "schedule_index": item["schedule_index"],
        "sample_id": record["sample_id"],
        "video_id": record["video_id"],
        "task": item["task"],
        "gt_admitted": admitted,
        "gt_judge": gt,
        "verification_protocol": protocol(config),
        "counts": counts,
        "candidates": candidates,
        "pair": pair,
        "no_pair_reason": (
            None
            if pair
            else (
                no_pair_reason(admitted, gt, candidates, counts, preference["candidate_gate"])
                if enabled
                else "judge_not_enabled"
            )
        ),
        "geometry_bin": geometry_bin(pair, record, candidates) if pair else None,
    }
    return pair, diagnostic
