"""Prepare for temporal video grounding."""

import math
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from vtg.utils.io import digest, save_json, write_jsonl
from vtg.tasks.reference_parser import CoreParser
from vtg.data.splits import split_videos, pilot_subset, dev_subset, validate_splits
from vtg.video.decoder import scan_video, select_pts


def read_sta(path, source):
    rows = []
    for line_id, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        prefix, query = line.split("##", 1)
        video, s, e = prefix.split()
        s, e = (float(s), float(e))
        rows.append(
            {
                "schema_version": "event_core_v1",
                "sample_id": digest([source, video, query, line_id, s, e]),
                "source": source,
                "source_line": line_id,
                "video_id": video,
                "video_relpath": video + ".mp4",
                "query": query.strip(),
                "gt_sec": [s, e],
            }
        )
    return rows


def prepare(config):
    paths = config["paths"]
    out = Path(paths["artifact_root"])
    out.mkdir(parents=True, exist_ok=True)
    import json
    from vtg.data.annotations import read_training_annotations

    train = read_training_annotations(paths["train_annotations"], config["data"]["dataset"])
    test = []
    _, raw_dev = (
        split_videos(train, config["data"]["split_seed"], config["data"]["dev_video_fraction"])
        if not config["data"].get("use_all_training_annotations", False)
        else (train, [])
    )
    dev_ids = {r["video_id"] for r in raw_dev}
    allrows = train + test
    videos = sorted({r["video_relpath"] for r in allrows})
    scan_path = out / "video_scan.json"
    scans = {}
    if scan_path.exists():
        import json

        scans = json.loads(scan_path.read_text())

    def probe(rel):
        path = Path(paths["video_root"]) / rel
        try:
            return (rel, {**scan_video(path), "source_identity": source_identity(rel)})
        except Exception as e:
            return (rel, {"error": str(e)})

    def source_identity(rel):
        path = Path(paths["video_root"]) / rel
        try:
            stat = path.stat()
            return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
        except OSError:
            return None

    todo = [
        v
        for v in videos
        if v not in scans
        or scans[v].get("error")
        or scans[v].get("source_identity") != source_identity(v)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (v, result) in enumerate(
            tqdm(pool.map(probe, todo), total=len(todo), desc="Scan training videos")
        ):
            scans[v] = result
            if i % 100 == 0:
                save_json(scan_path, scans)
    save_json(scan_path, scans)
    parser = CoreParser(paths["spacy_model"])
    bad = []
    aux_bad = []
    valid = []
    for row in tqdm(allrows, desc="Parse training annotations"):
        info = scans[row["video_relpath"]]
        s, e = row["gt_sec"]
        duration = float(row["duration_sec"])
        physical = info.get("duration_sec", 0.0)
        reason = info.get("error")
        if not reason and (
            not (all(map(math.isfinite, (s, e, duration))) and 0 <= s < e <= duration)
        ):
            reason = "invalid_ground_truth"
        offset = 0.0
        if not reason:
            tolerance = max(0.25, 0.005 * duration)
            start = row["source_video_start"]
            end = row["source_video_end"]
            if abs(physical - duration) <= tolerance:
                pass
            elif (
                start is not None
                and end is not None
                and (start >= 0)
                and (abs(end - start - duration) <= tolerance)
                and (end <= physical + tolerance)
            ):
                offset = float(start)
            else:
                reason = "ambiguous_media_coordinates"
        if reason:
            bad.append({**row, "reason": reason})
            continue
        row["media_offset"] = offset
        row["physical_duration_sec"] = physical
        row["coordinate_protocol"] = (
            "timestamp local to supplied annotated video window; decoder adds media_offset"
        )
        row["duration_sec"] = duration
        sem = (
            parser.parse(row["query"])
            if row["source"].endswith("train")
            else {
                "fields": {},
                "supported_fields": [],
                "eligible_tasks": [],
                "failure_code": "test_not_parsed",
                "event_count": 0,
            }
        )
        if sem["eligible_tasks"]:
            try:
                select_pts(info["pts"], [v + offset for v in row["gt_sec"]], config["video"])
            except ValueError:
                sem["eligible_tasks"] = []
                sem["failure_code"] = "no_gt_frames"
        row["semantics"] = sem
        valid.append(row)
        if not sem["eligible_tasks"] and row["source"].endswith("train"):
            aux_bad.append({"sample_id": row["sample_id"], "reason": sem["failure_code"]})
    train = [r for r in valid if r["source"].endswith("train")]
    test = [dict(r, split="test") for r in valid if r["source"].endswith("test")]
    fit = [dict(r, split="fit") for r in train if r["video_id"] not in dev_ids]
    dev = [dict(r, split="dev") for r in train if r["video_id"] in dev_ids]
    validate_splits(fit, dev, test)
    pilot = pilot_subset(fit, config["data"]["pilot_max_queries"])
    aux = [r for r in fit if r["semantics"]["eligible_tasks"]]
    ep = [r for r in pilot if r["semantics"]["eligible_tasks"]]
    manifests = {
        "fit": fit,
        "dev": dev,
        "test": test,
        "pilot": pilot,
        "dev-mini": dev_subset(dev),
        "aux-fit": aux,
        "aux-pilot": ep,
        "aux-dev": dev_subset([r for r in dev if r["semantics"]["eligible_tasks"]]),
        "quarantine": bad,
        "aux-exclusions": aux_bad,
    }
    for name, rows in manifests.items():
        write_jsonl(out / (name + ".jsonl"), rows)
    report = {
        "counts": {k: len(v) for k, v in manifests.items()},
        "aux_coverage": len(aux) / len(fit) if fit else 0,
        "aux_pilot_videos": len({r["video_id"] for r in ep}),
        "field_counts": dict(Counter((len(r["semantics"]["supported_fields"]) for r in aux))),
        "exclusion_reasons": dict(Counter((r["reason"] for r in aux_bad))),
        "field_distributions": {
            f: dict(
                Counter(
                    (
                        r["semantics"]["fields"][f]["value"]
                        for r in aux
                        if f in r["semantics"]["fields"]
                    )
                )
            )
            for f in ("agent", "predicate", "patient")
        },
        "manifest_hashes": {k: digest(v) for k, v in manifests.items()},
        "coverage_gate": len(ep) >= 128 and len({r["video_id"] for r in ep}) >= 32,
    }
    save_json(out / "data_report.json", report)
    return report
