"""Independent Charades-STA and TVGBench evaluation manifests; no test parsing for training."""

import json
import math
import re
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm
from vtg.utils.io import digest, read_jsonl, write_jsonl, save_json


def benchmark_rows(name, path):
    raw = json.loads(Path(path).read_text())
    rows = []
    if name == "charades":
        for video, item in raw.items():
            if len(item["sentences"]) != len(item["timestamps"]):
                raise ValueError("Charades query/GT count mismatch")
            for query, gt in zip(item["sentences"], item["timestamps"]):
                rows.append(
                    {
                        "video_id": video,
                        "video_relpath": video + ".mp4",
                        "query": query.strip(),
                        "gt_sec": gt,
                        "duration_sec": float(item["duration"]),
                        "dataset_name": "charades",
                        "source_video_start": None,
                        "source_video_end": None,
                    }
                )
    elif name == "tvgbench":
        number = "[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)"
        for item in raw:
            m = re.fullmatch("\\s*(" + number + ")\\s*-\\s*(" + number + ")\\s*", item["answer"])
            if not m:
                raise ValueError("Unrecognized TVGBench GT: " + repr(item["answer"]))
            video = Path(item["path"])
            rows.append(
                {
                    "video_id": video.stem,
                    "video_relpath": video.parent.name + "/" + video.name,
                    "query": item["question"].strip(),
                    "gt_sec": list(map(float, m.groups())),
                    "duration_sec": float(item["duration"]),
                    "dataset_name": item.get(
                        "dataset_name", item.get("bins", {}).get("dataset", "unspecified")
                    ),
                    "source_video_start": item.get("start"),
                    "source_video_end": item.get("end"),
                }
            )
    else:
        raise ValueError("Unknown benchmark " + name)
    for i, row in enumerate(rows):
        s, e = row["gt_sec"]
        d = row["duration_sec"]
        if not all(map(math.isfinite, (s, e, d))) or not 0 <= s <= e or d <= 0:
            raise ValueError("Malformed benchmark reference")
        row.update(
            schema_version="event_core_v1",
            source=name + "_test",
            split="test",
            source_line=i,
            sample_id=digest([name, i, row]),
            annotation_duration_sec=d,
            media_offset=0.0,
            gt_zero_duration=s == e,
            semantics={"fields": {}, "supported_fields": [], "eligible_tasks": []},
        )
    return rows


def benchmark_media(row, physical, name):
    """Retain the published GT, and resolve only the actual video input window."""
    row = dict(row)
    d = row["annotation_duration_sec"]
    row["physical_duration_sec"] = physical
    start = row["source_video_start"]
    end = row["source_video_end"]
    if name == "charades" or (start is None and end is None):
        row["duration_sec"] = physical
        row["input_window_protocol"] = "entire physical video"
    else:
        tolerance = max(0.25, 0.005 * d)
        if abs(physical - d) <= tolerance:
            row["duration_sec"] = physical
            row["input_window_protocol"] = "already local media"
        elif (
            start is not None
            and end is not None
            and (0 <= start < physical)
            and (abs(end - start - d) <= tolerance)
        ):
            row["media_offset"] = float(start)
            row["duration_sec"] = min(end, physical) - start
            row["input_window_protocol"] = (
                "annotated source window intersected with available media"
            )
            row["requested_window_truncated"] = end > physical
        else:
            row["input_error"] = "ambiguous_media_coordinates"
    row["gt_exceeds_input_duration"] = row["gt_sec"][1] > row["duration_sec"]
    return row


def prepare_benchmarks(c):
    import av
    from concurrent.futures import ThreadPoolExecutor

    training = read_jsonl(Path(c["paths"]["artifact_root"]) / "fit.jsonl")
    train_ids = {r["video_id"] for r in training}
    report = {}
    sets = {}
    for name, spec in c["benchmarks"].items():
        rows = benchmark_rows(name, spec["annotations"])
        videos = sorted({r["video_relpath"] for r in rows})

        def inspect(rel):
            try:
                with av.open(str(Path(spec["video_root"]) / rel)) as container:
                    stream = container.streams.video[0]
                    duration = (
                        float(stream.duration * stream.time_base)
                        if stream.duration
                        else (
                            float(container.duration / av.time_base) if container.duration else None
                        )
                    )
                if duration is None or not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Missing finite duration")
                return (rel, {"duration": duration})
            except Exception as e:
                return (rel, {"error": str(e)})

        with ThreadPoolExecutor(max_workers=8) as pool:
            media = dict(tqdm(pool.map(inspect, videos), total=len(videos), desc=f"Scan {name}"))
        final = []
        for row in rows:
            info = media[row["video_relpath"]]
            if "error" in info:
                row = {**row, "input_error": info["error"]}
            else:
                row = benchmark_media(row, info["duration"], name)
            final.append(row)
        overlap = sorted(train_ids & {r["video_id"] for r in final})
        sets[name] = {r["video_id"] for r in final}
        write_jsonl(spec["manifest"], final)
        report[name] = {
            "annotations": spec["annotations"],
            "count": len(final),
            "videos": len(videos),
            "datasets": dict(Counter((r["dataset_name"] for r in final))),
            "input_errors": dict(
                Counter((r["input_error"] for r in final if r.get("input_error")))
            ),
            "gt_exceeds_input_duration": sum(
                (r.get("gt_exceeds_input_duration", False) for r in final)
            ),
            "zero_duration_gt": sum((r["gt_zero_duration"] for r in final)),
            "nonzero_media_offsets": sum((r["media_offset"] != 0 for r in final)),
            "training_video_id_overlap": overlap,
            "manifest_hash": digest(final),
            "denominator_policy": "All published queries retained, including point GT (zero temporal IoU). Flagged media failures produce an empty prediction and zero IoU; original GT is never clipped.",
        }
        report[name]["truncated_input_windows"] = sum(
            (r.get("requested_window_truncated", False) for r in final)
        )
    report["cross_benchmark_video_overlap"] = len(
        sets.get("charades", set()) & sets.get("tvgbench", set())
    )
    report["overlap_check"] = (
        "Exact source video IDs, not perceptual duplicate detection; Charades and TVGBench results reported separately."
    )
    save_json(Path(c["paths"]["artifact_root"]) / "benchmark_report.json", report)
    return report
