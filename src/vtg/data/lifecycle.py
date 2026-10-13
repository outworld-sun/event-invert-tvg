"""Serialize data preparation before distributed model initialization."""

import fcntl
import json
from pathlib import Path

from vtg.utils.io import file_hash, implementation_identity, save_json


def ensure_training_data(config):
    """All local ranks share a lock and reuse only matching prepared manifests."""
    root = Path(config["paths"]["artifact_root"])
    root.mkdir(parents=True, exist_ok=True)
    signature_path = root / "training_source.json"
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        annotations = Path(config["paths"]["train_annotations"])
        source = {
            "annotations": str(annotations.resolve()),
            "annotation_hash": file_hash(annotations),
            "video_root": str(Path(config["paths"]["video_root"]).resolve()),
            "spacy_model": config["paths"]["spacy_model"],
            "data": config["data"],
            "video": config["video"],
            "source_hash": implementation_identity(),
        }
        required = [root / name for name in ("fit.jsonl", "aux-fit.jsonl", "data_report.json")]
        if signature_path.exists() and all(path.is_file() for path in required):
            previous = json.loads(signature_path.read_text())
            hashes = {path.name: file_hash(path) for path in required}
            if previous == {"source": source, "manifests": hashes}:
                return
        from vtg.data.prepare import prepare

        print(f"Preparing training manifests in {root}", flush=True)
        prepare(config)
        save_json(
            signature_path,
            {"source": source, "manifests": {path.name: file_hash(path) for path in required}},
        )
