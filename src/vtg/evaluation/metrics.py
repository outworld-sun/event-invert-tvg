import json
import math
import random
from collections import defaultdict
from pathlib import Path
from vtg.rewards.scoring import parse_interval, tiou, score_completion
from vtg.utils.io import read_jsonl, save_json, digest
from vtg.tasks.builder import build_task_instance


def evaluate_grounding(predictions, annotations, metric_config=None):
    ids = [r["sample_id"] for r in annotations]
    pids = [r["sample_id"] for r in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation IDs")
    if len(pids) != len(set(pids)) or set(pids) != set(ids):
        raise ValueError("Missing, duplicate or extra prediction IDs")
    pred = {r["sample_id"]: r for r in predictions}
    scores = []
    invalid = 0
    for r in annotations:
        interval = parse_interval(pred[r["sample_id"]]["completion"], r["duration_sec"])
        invalid += interval is None
        scores.append(tiou(interval, r["gt_sec"]))
    if not scores:
        raise ValueError("Empty evaluation")
    return {
        **{
            f"R1_0{int(th * 10)}": sum((x >= th for x in scores)) / len(scores)
            for th in (0.3, 0.5, 0.7)
        },
        "mIoU": sum(scores) / len(scores),
        "invalid_interval_rate": invalid / len(scores),
        "missing_prediction_count": 0,
        "n": len(scores),
    }


def paired_bootstrap(pred_a, pred_b, annotations, n=2000, seed=3407):
    evaluate_grounding(pred_a, annotations)
    evaluate_grounding(pred_b, annotations)
    a = {r["sample_id"]: r for r in pred_a}
    b = {r["sample_id"]: r for r in pred_b}
    groups = defaultdict(list)
    for r in annotations:
        ia = tiou(parse_interval(a[r["sample_id"]]["completion"], r["duration_sec"]), r["gt_sec"])
        ib = tiou(parse_interval(b[r["sample_id"]]["completion"], r["duration_sec"]), r["gt_sec"])
        groups[r["video_id"]].append(
            [float(ib >= th) - float(ia >= th) for th in (0.3, 0.5, 0.7)] + [ib - ia]
        )
    videos = sorted(groups)
    rng = random.Random(seed)
    draws = [[] for _ in range(4)]
    for _ in range(n):
        rows = [v for vid in rng.choices(videos, k=len(videos)) for v in groups[vid]]
        for k in range(4):
            draws[k].append(sum((r[k] for r in rows)) / len(rows))

    def interval(x):
        x = sorted(x)
        return [x[int(0.025 * (len(x) - 1))], x[int(0.975 * (len(x) - 1))]]

    return {
        k: {"ci95": interval(d), "bootstrap_mean": sum(d) / len(d)}
        for k, d in zip(("R1_03", "R1_05", "R1_07", "mIoU"), draws)
    }
