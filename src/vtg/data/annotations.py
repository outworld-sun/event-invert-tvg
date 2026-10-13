"""Read Time-R1 list JSON or Invert4TVG-style Charades video dictionaries."""

import json
from pathlib import Path

from vtg.utils.io import digest


def read_training_annotations(path, dataset):
    raw = json.loads(Path(path).read_text())
    rows = []
    if dataset in ("timer1", "timer1_train_2k5"):
        if not isinstance(raw, list):
            raise ValueError("Time-R1 annotations must be a JSON list")
        for index, item in enumerate(raw):
            qid = item.get("qid", "")
            parts = qid.split("|")
            video = Path(item["video"])
            relative = str(Path(video.parent.name) / video.name)
            rows.append(
                {
                    "schema_version": "event_core_v1",
                    "sample_id": digest(
                        ["timer1_train_2k5", qid, index, item["timestamp"], item["sentence"]]
                    ),
                    "source": "timer1_train",
                    "source_line": index,
                    "video_id": parts[2] if len(parts) >= 3 else video.stem,
                    "video_relpath": relative,
                    "query": item["sentence"].strip(),
                    "gt_sec": item["timestamp"],
                    "duration_sec": item["duration"],
                    "source_video_start": item.get("video_start"),
                    "source_video_end": item.get("video_end"),
                }
            )
    elif dataset == "charades":
        if not isinstance(raw, dict):
            raise ValueError("Charades annotations must be a video-ID dictionary")
        for video_id, item in raw.items():
            if len(item["timestamps"]) != len(item["sentences"]):
                raise ValueError(f"Mismatched timestamps and sentences for {video_id}")
            for query_index, (interval, query) in enumerate(
                zip(item["timestamps"], item["sentences"])
            ):
                rows.append(
                    {
                        "schema_version": "event_core_v1",
                        "sample_id": digest(
                            ["charades_train", video_id, query_index, interval, query]
                        ),
                        "source": "charades_train",
                        "source_line": len(rows),
                        "video_id": video_id,
                        "video_relpath": (
                            video_id if video_id.endswith(".mp4") else video_id + ".mp4"
                        ),
                        "query": query.strip(),
                        "gt_sec": interval,
                        "duration_sec": item["duration"],
                        "source_video_start": None,
                        "source_video_end": None,
                    }
                )
    else:
        raise ValueError(f"Unsupported training dataset: {dataset}")
    return rows
