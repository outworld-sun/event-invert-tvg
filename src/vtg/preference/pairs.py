"""Pure gates over a real global candidate group, without synthetic outputs."""

import itertools
import random
from vtg.utils.io import digest
from vtg.rewards.scoring import parse_interval, tiou


def legal_gt(record):
    import math

    s, e = record["gt_sec"]
    t = record["duration_sec"]
    return all((math.isfinite(v) for v in (s, e, t))) and 0 <= s < e <= t


def absolute_crop(record, interval):
    s, e = interval
    if not 0 <= s < e <= record["duration_sec"]:
        raise ValueError("Never repair a real candidate interval")
    offset = record.get("media_offset", 0.0)
    return [offset + s, offset + e]


def reference_eligible(record, reference_ids, verification_protocol="legacy_ser_v1"):
    from vtg.preference.protocol import FULL_QUERY, LEGACY_SER

    if verification_protocol not in (FULL_QUERY, LEGACY_SER):
        raise ValueError("Unknown verification protocol")
    if verification_protocol == FULL_QUERY:
        return (
            legal_gt(record)
            and isinstance(record.get("query"), str)
            and bool(record["query"].strip())
        )
    return (
        record["sample_id"] in reference_ids
        and legal_gt(record)
        and (len(record["semantics"]["supported_fields"]) >= 2)
    )


def prompt_eligible(record, reference_ids, gt_judge, verification_protocol="legacy_ser_v1"):
    return (
        reference_eligible(record, reference_ids, verification_protocol)
        and gt_judge.get("decoded", False)
        and gt_judge.get("parseable", False)
        and gt_judge.get("joint", False)
    )


def candidate(record, completion, index, judge=None):
    interval = parse_interval(completion, record["duration_sec"])
    return {
        "index": index,
        "interval": interval,
        "iou": tiou(interval, record["gt_sec"]),
        "valid": interval is not None,
        "judge": judge or {},
        "sample_id": record["sample_id"],
        "source": "actual_rollout",
    }


def enumerate_pairs(candidates, cfg, admitted=True):
    if len(candidates) < 2 or sorted(c["index"] for c in candidates) != list(
        range(len(candidates))
    ):
        raise ValueError("Pair selection requires one complete, consecutively indexed global group")
    if len({c["sample_id"] for c in candidates}) != 1:
        raise ValueError("Mixed prompts in a preference group")
    if any((c.get("source") != "actual_rollout" for c in candidates)):
        raise ValueError("Synthetic candidates are not on-policy rollouts")
    counts = {
        "valid_candidates": 0,
        "joint_candidates": 0,
        "iou_candidates": 0,
        "eligible_candidates": 0,
        "gap_pairs": 0,
        "distinct_input_pairs": 0,
    }
    if not admitted:
        return ([], counts)
    eligible = []
    for c in candidates:
        j = c["judge"]
        if not c["valid"] or not j.get("decoded"):
            continue
        counts["valid_candidates"] += 1
        counts["joint_candidates"] += bool(j.get("parseable") and j.get("joint"))
        if c["iou"] < cfg["min_iou"]:
            continue
        counts["iou_candidates"] += 1
        gate = cfg["candidate_gate"]
        if gate == "joint" and (not (j.get("parseable") and j.get("joint"))):
            continue
        if gate == "predicate" and (not (j.get("parseable") and j.get("predicate_match"))):
            continue
        eligible.append(c)
    counts["eligible_candidates"] = len(eligible)
    pairs = []
    for a, b in itertools.combinations(eligible, 2):
        w, l = (a, b) if a["iou"] > b["iou"] else (b, a)
        gap = w["iou"] - l["iou"]
        if gap + 1e-12 < cfg["min_gap"] or w["interval"] == l["interval"]:
            continue
        counts["gap_pairs"] += 1
        if w["judge"]["teacher_input_hash"] == l["judge"]["teacher_input_hash"]:
            continue
        counts["distinct_input_pairs"] += 1
        pairs.append(
            {
                "winner": w["index"],
                "loser": l["index"],
                "gap": gap,
                "weight": min(gap / cfg["weight_gap"], 1.0),
                "winner_iou": w["iou"],
                "loser_iou": l["iou"],
            }
        )
    return (pairs, counts)


def no_pair_reason(admitted, gt, candidates, counts, gate):
    if not admitted:
        if not gt:
            return "reference_ineligible"
        if not gt.get("decoded"):
            return "gt_decode_failure"
        if not gt.get("parseable"):
            return "gt_verdict_unparseable"
        return (
            "gt_match_uncertain"
            if gt.get("candidate", {}).get("match") == "uncertain"
            else "gt_match_failed"
        )
    if sum((c["valid"] for c in candidates)) < 2:
        return "fewer_than_two_legal_intervals"
    if counts.get("valid_candidates", 0) < 2:
        return "fewer_than_two_decodable_candidates"
    if gate == "joint" and counts.get("joint_candidates", 0) < 2:
        return "fewer_than_two_semantic_matches"
    if counts.get("eligible_candidates", 0) < 2:
        return "fewer_than_two_candidates_passing_gate_and_iou"
    if not counts.get("gap_pairs"):
        return "insufficient_iou_gap"
    if not counts.get("distinct_input_pairs"):
        return "identical_teacher_inputs"
    return "pair_disabled_or_matched_control_rejected"


def select_pair(pairs, seed, group_id):
    return random.Random(digest([seed, group_id, "pair"])).choice(pairs) if pairs else None


def swap_penalty(instance, breakdown, enabled, alpha):
    if not enabled or instance.task != "SER" or (not breakdown.validity):
        return 0.0
    fields = instance.target
    if not {"agent", "patient"} <= set(fields):
        return 0.0
    a, p = (fields["agent"]["value"], fields["patient"]["value"])
    if not a or not p or a == p:
        return 0.0
    if not fields["agent"].get("spans") or not fields["patient"].get("spans"):
        return 0.0
    return alpha * float(
        breakdown.candidate.get("agent") == p and breakdown.candidate.get("patient") == a
    )


def boundary_probes(record):
    """Diagnostic only. Synthetic proposals never enter enumerate_pairs."""
    s, e = record["gt_sec"]
    length = e - s
    t = record["duration_sec"]
    if not legal_gt(record):
        return []
    rows = []
    for fraction in (0.1, 0.2, 0.4):
        d = fraction * length
        for a, b in (
            (s + d, e),
            (s, e - d),
            (s + d, e + d),
            (s - d, e - d),
            (s + d, e - d),
            (s - d, e + d),
        ):
            interval = [max(0.0, a), min(t, b)]
            if interval[0] < interval[1] and interval not in [x["interval"] for x in rows]:
                rows.append(
                    {
                        "interval": interval,
                        "iou": tiou(interval, [s, e]),
                        "source": "synthetic_diagnostic_only",
                        "fraction": fraction,
                    }
                )
    return rows[:: max(1, len(rows) // 8)][:8]
