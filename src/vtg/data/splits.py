"""Splits for temporal video grounding."""

import hashlib
import math
import random


def video_hash(video_id, seed=3407):
    return hashlib.sha256(f"{seed}:{video_id}".encode()).hexdigest()


def split_videos(rows, seed=3407, fraction=0.1):
    videos = sorted({r["video_id"] for r in rows}, key=lambda v: video_hash(v, seed))
    dev = set(videos[: math.ceil(fraction * len(videos))])
    return (
        [dict(r, split="fit") for r in rows if r["video_id"] not in dev],
        [dict(r, split="dev") for r in rows if r["video_id"] in dev],
    )


def pilot_subset(fit, maximum=2048):
    rows = sorted(fit, key=lambda r: r["sample_id"])
    random.Random(3408).shuffle(rows)
    return rows[:maximum]


def dev_subset(dev, maximum=256):
    return sorted(dev, key=lambda r: (video_hash(r["video_id"]), r["sample_id"]))[:maximum]


def validate_splits(fit, dev, test):
    pools = [{r["video_id"] for r in x} for x in (fit, dev, test)]
    for i in range(3):
        for j in range(i):
            if pools[i] & pools[j]:
                raise ValueError("Video leakage between splits")
    ids = [r["sample_id"] for x in (fit, dev, test) for r in x]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
