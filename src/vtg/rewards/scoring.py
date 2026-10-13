"""Scoring for temporal video grounding."""

import math
import re
from vtg.tasks.schemas import RewardBreakdown
from vtg.tasks.candidate_parser import answer_body, parse_candidate
from vtg.tasks.normalize import normalize_value
from vtg.rewards.verbs import score_verb


def parse_interval(text, duration):
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    match = re.search(
        "<answer>\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))\\s+to\\s+([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))\\s*</answer>",
        text,
    )
    if not match:
        return None
    s, e = map(float, match.groups())
    return [s, e] if all(map(math.isfinite, (s, e, duration))) and 0 <= s < e <= duration else None


def tiou(pred, gt):
    if pred is None:
        return 0.0
    return max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0])) / (
        max(pred[1], gt[1]) - min(pred[0], gt[0])
    )


def score_completion(instance, completion_text, reward_config, parser=None):
    r = RewardBreakdown()
    task = instance.task
    if task == "TVG":
        gt = instance.target["gt_sec"]
        duration = instance.target["duration_sec"]
        if not all(map(math.isfinite, (*gt, duration))) or not 0 <= gt[0] < gt[1] <= duration:
            raise ValueError("Broken grounding reference")
        pred = parse_interval(completion_text, instance.target["duration_sec"])
        r.r_format = float(
            bool(
                re.fullmatch(
                    "\\s*<think>.*?</think>\\s*<answer>[^<>]+</answer>\\s*", completion_text, re.S
                )
            )
        )
        r.validity = pred is not None
        r.candidate = {"interval": pred}
        r.r_semantic = tiou(pred, instance.target["gt_sec"])
        functions = reward_config.get("grounding_functions", ["iou", "format"])
        r.total = (
            (r.r_format if "format" in functions else 0.0)
            + (r.r_semantic if "iou" in functions else 0.0)
            if r.validity
            else 0.0
        )
        r.failure_code = "" if r.validity else "invalid_interval"
        return r
    body = answer_body(completion_text)
    r.r_format = float(body is not None)
    if body is None:
        r.failure_code = "invalid_answer_format"
        return r
    if not instance.supported_fields or "predicate" not in instance.target:
        raise ValueError("Broken auxiliary reference")
    if task in ("VC", "AR", "VD"):
        r.r_semantic = score_verb(task, body, instance.target, parser)
        r.validity = True
        r.total = 1 + r.r_semantic
        return r
    if task == "SCC":
        f = instance.hidden_fields[0]
        candidate = {f: normalize_value(f, body, parser.nlp)}
        failure = ""
        fields = [f]
    else:
        candidate, failure = parse_candidate(task, body, parser)
        fields = instance.supported_fields
    r.candidate = candidate
    r.validity = not failure
    r.failure_code = failure
    if failure:
        r.total = 1.0
        return r
    if reward_config.get("unbound", False) and task != "SCC":
        values = {v for v in candidate.values() if v is not None}
        r.field_scores = {f: float(instance.target[f]["value"] in values) for f in fields}
    else:
        r.field_scores = {f: float(candidate.get(f) == instance.target[f]["value"]) for f in fields}
    c = sum(r.field_scores.values()) / len(fields)
    joint = float(all(r.field_scores.values()))
    m = int(len(fields) >= 2)
    lam = reward_config.get("joint_lambda", 1.0) if task != "SCC" else 0.0
    r.joint_score = joint if m else 0.0
    r.r_semantic = (c + lam * m * joint) / (1 + lam * m)
    r.total = 1 + r.r_semantic
    if not all(r.field_scores.values()):
        r.failure_code = "field_mismatch_or_missing"
    return r
