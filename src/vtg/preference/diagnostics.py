"""Coverage denominators, per-bin matched control and cost evidence."""

from collections import Counter
import json
import random
from pathlib import Path
from vtg.utils.io import digest, save_json


def geometry_bin(pair, record, candidates):
    length = record["gt_sec"][1] - record["gt_sec"][0]

    def bucket(value, edges):
        return sum((value >= e for e in edges))

    ratios = [
        (candidates[pair[k]]["interval"][1] - candidates[pair[k]]["interval"][0]) / length
        for k in ("winner", "loser")
    ]
    return str(
        (
            len(record["semantics"]["supported_fields"]),
            bucket(pair["winner_iou"], (0.5, 0.7, 0.9)),
            bucket(pair["loser_iou"], (0.5, 0.7, 0.9)),
            bucket(pair["gap"], (0.2, 0.3)),
            *(bucket(x, (0.5, 1.0, 1.5, 2.0)) for x in ratios),
        )
    )


def calibrate_matched(ordinary, event, output, target_questions=2444):
    import math

    def counts(rows):
        if any((r.get("split") != "train" for r in rows)):
            raise ValueError("Calibration accepts only explicitly marked training diagnostics")
        ids = [r["group_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate calibration groups")
        return Counter((r["geometry_bin"] for r in rows if r.get("pair")))

    a, b = (counts(ordinary), counts(event))
    if {r["group_id"] for r in ordinary} != {r["group_id"] for r in event}:
        raise ValueError("Calibrate on the same fixed training questions")
    identities = {
        (r.get("verification_protocol", "legacy_ser_v1"), r["group_id"], r["sample_id"])
        for r in ordinary
    }
    if identities != {
        (r.get("verification_protocol", "legacy_ser_v1"), r["group_id"], r["sample_id"])
        for r in event
    }:
        raise ValueError(
            "Matched calibration requires the same verification protocol and sample IDs"
        )
    protocols = {r.get("verification_protocol", "legacy_ser_v1") for r in ordinary}
    if len(protocols) != 1:
        raise ValueError("Matched calibration requires exactly one verification protocol")
    if not ordinary or target_questions <= 0:
        raise ValueError("Nonempty calibration and positive target exposure required")
    layouts = {row.get("num_generations", 4) for row in ordinary + event}
    if len(layouts) != 1:
        raise ValueError("Matched calibration requires the same candidate group size")
    if len({r["sample_id"] for r in ordinary}) != len(ordinary):
        raise ValueError("Calibration must use unique reference questions")
    exposure = target_questions / len(ordinary)
    result = {
        "version": 1,
        "frozen": True,
        "source": "training_precalibration",
        "verification_protocol": next(iter(protocols)),
        "calibration_questions": len(ordinary),
        "target_reference_questions": target_questions,
        "quotas": {
            k: {
                "target": math.ceil(b[k] * exposure),
                "calibration_target": b[k],
                "available": a[k],
                "keep_probability": min(b[k] / a[k], 1.0) if a[k] else 0.0,
            }
            for k in sorted(a.keys() | b.keys())
        },
    }
    result["num_generations"] = next(iter(layouts))
    result["calibration_hash"] = digest(result)
    save_json(output, result)
    return result


class MatchedControl:

    def __init__(self, path, seed, verification_protocol=None):
        self.spec = json.loads(Path(path).read_text())
        if not self.spec.get("frozen") or self.spec.get("source") != "training_precalibration":
            raise ValueError("A_matched requires frozen training-only calibration")
        if (
            verification_protocol is not None
            and self.spec.get("verification_protocol") != verification_protocol
        ):
            raise ValueError("Matched calibration verification protocol differs; recalibrate")
        self.seed, self.counts = (seed, Counter())

    def accept(self, key, group_id):
        q = self.spec["quotas"].get(key, {"target": 0, "keep_probability": 0})
        if self.counts[key] >= q["target"]:
            return False
        keep = (
            random.Random(digest([self.seed, group_id, "matched"])).random() < q["keep_probability"]
        )
        self.counts[key] += int(keep)
        return keep


def coverage_report(
    rows, total_tvg=2444, reference_count=None, verification_protocol="full_query_v2"
):
    from vtg.preference.protocol import FULL_QUERY, LEGACY_SER

    if verification_protocol not in (FULL_QUERY, LEGACY_SER):
        raise ValueError("Unknown verification protocol")
    if reference_count is None:
        reference_count = total_tvg if verification_protocol == FULL_QUERY else 387
    if not 0 < len(rows) <= reference_count <= total_tvg:
        raise ValueError("Coverage counts must satisfy 0 < sampled <= reference pool <= TVG")
    if verification_protocol == FULL_QUERY and reference_count != total_tvg:
        raise ValueError("Full-query coverage must use all TVG samples as its population")
    if any(
        (
            r.get("verification_protocol", verification_protocol) != verification_protocol
            for r in rows
        )
    ):
        raise ValueError("Mixed verification protocols in coverage rows")
    ids = [r["group_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Coverage must count global questions once, not once per rank")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Fixed coverage sample must contain unique questions")
    n = len(rows)
    paired = [r for r in rows if r.get("pair")]
    coverage = {
        "sampled_questions": n,
        "reference_count": reference_count,
        "verification_protocol": verification_protocol,
        "tvg_count": total_tvg,
        "reference_rate": reference_count / total_tvg,
        "gt_joint_pass": sum((r.get("gt_admitted", False) for r in rows)),
        "two_candidate_joint_pass": sum(
            (r.get("counts", {}).get("joint_candidates", 0) >= 2 for r in rows)
        ),
        "iou_gap_pass": sum((r.get("counts", {}).get("gap_pairs", 0) > 0 for r in rows)),
        "distinct_teacher_input_pass": sum(
            (r.get("counts", {}).get("distinct_input_pairs", 0) > 0 for r in rows)
        ),
        "pair_count": len(paired),
        "paired_unique_queries": len({r["sample_id"] for r in paired}),
        "paired_unique_videos": len({r["video_id"] for r in paired}),
        "conditional_pair_rate": len(paired) / n if n else 0,
        "scope": (
            "fixed all-TVG subset; not success-selected"
            if verification_protocol == FULL_QUERY
            else "fixed legacy reference-pool subset; not success-selected"
        ),
        "no_pair_reasons": dict(
            Counter((r.get("no_pair_reason", "unspecified") for r in rows if not r.get("pair")))
        ),
    }
    rate = coverage["reference_rate"] * coverage["conditional_pair_rate"]
    coverage["projected_all_tvg_pair_rate"] = rate
    coverage["projected_pilot_pairs"] = 960 * rate
    coverage["projected_confirmation_pairs"] = total_tvg * rate
    coverage["engineering_ready"] = (
        coverage["projected_pilot_pairs"] >= 10 and coverage["paired_unique_videos"] >= 3
    )
    coverage["readiness_rule"] = (
        "Conservative preregistered engineering minimum: >=10 projected pilot pairs and >=3 observed videos; not statistical power."
    )
    return coverage
