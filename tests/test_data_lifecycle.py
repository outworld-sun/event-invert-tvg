"""Preparation is reused only when annotations, configuration, and outputs agree."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtg.config import load_config
from vtg.data.lifecycle import ensure_training_data

ROOT = Path(__file__).resolve().parents[1]


class DataLifecycleTests(unittest.TestCase):
    def test_auto_prepare_and_invalidation(self):
        config = load_config(ROOT / "configs/e5_8xa100.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "train.json"
            annotation.write_text("[]")
            artifact = root / "artifacts"
            config["paths"].update(train_annotations=str(annotation), artifact_root=str(artifact))

            def prepare(_):
                for name in ("fit.jsonl", "aux-fit.jsonl"):
                    (artifact / name).write_text("")
                (artifact / "data_report.json").write_text("{}")

            with patch("vtg.data.prepare.prepare", side_effect=prepare) as mocked:
                ensure_training_data(config)
                ensure_training_data(config)
                self.assertEqual(mocked.call_count, 1)
                annotation.write_text('[{"changed": true}]')
                ensure_training_data(config)
                self.assertEqual(mocked.call_count, 2)
                (artifact / "fit.jsonl").write_text("corrupt")
                ensure_training_data(config)
                self.assertEqual(mocked.call_count, 3)
                changed = copy.deepcopy(config)
                changed["video"]["max_frames"] = 128
                ensure_training_data(changed)
                self.assertEqual(mocked.call_count, 4)
            self.assertTrue((artifact / "training_source.json").is_file())


if __name__ == "__main__":
    unittest.main()
