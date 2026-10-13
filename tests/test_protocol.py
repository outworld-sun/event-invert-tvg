"""CPU checks for scientific invariants and portable configuration."""

import io
import json
import math
import tempfile
import tokenize
import unittest
from collections import Counter
from pathlib import Path

from vtg.config import load_config
from vtg.evaluation.metrics import evaluate_grounding
from vtg.evaluation.runner import merge_predictions
from vtg.preference.pairs import candidate, enumerate_pairs, select_pair
from vtg.preference.protocol import parse_query_verdict
from vtg.tasks.builder import build_task_instance, serialize_model_input
from vtg.training.schedule import build_schedule, compute_group_advantages, epoch_budget
from vtg.utils.io import write_jsonl
from vtg.video.budget import frame_pixel_budget, resize_shape, visual_tokens
from vtg.video.decoder import select_pts

ROOT = Path(__file__).resolve().parents[1]


def record(index=0):
    return {
        "sample_id": str(index),
        "video_id": str(index),
        "video_relpath": "fixture.mp4",
        "duration_sec": 10,
        "query": "A woman carries a cup.",
        "gt_sec": [2, 6],
        "media_offset": 39,
        "split": "fit",
        "semantics": {
            "supported_fields": ["agent", "predicate", "patient"],
            "eligible_tasks": ["SCC", "SER", "ED"],
            "fields": {
                "agent": {"value": "woman", "spans": [[2, 7]]},
                "predicate": {"value": "carry", "spans": [[8, 15]]},
                "patient": {"value": "cup", "spans": [[18, 21]]},
            },
        },
    }


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/n1_8xa100.json")

    def test_all_branch_configs(self):
        for path in (ROOT / "configs").glob("*_8xa100.json"):
            config = load_config(path)
            self.assertEqual(config["training"]["num_generations"], 8)
            self.assertEqual(config["video"]["max_frames"], 768)

    def test_dynamic_budget_and_alignment(self):
        video = self.config["video"]
        for count, expected in ((8, 602112), (64, 87808), (128, 43904), (256, 21952), (768, 13171)):
            self.assertEqual(frame_pixel_budget(count, video), expected)
            for height, width in ((1080, 1920), (720, 1280), (1920, 1080), (336, 336)):
                resized = resize_shape(height, width, video["min_pixels_per_frame"], expected)
                self.assertTrue(all(side % 28 == 0 for side in resized))
                self.assertLessEqual(visual_tokens(count, *resized), video["visual_token_cap"])
        self.assertLess(frame_pixel_budget(768, video), frame_pixel_budget(128, video))

    def test_full_interval_sampling_and_no_outside_crop(self):
        pts = [index / 10 for index in range(10000)]
        selected = select_pts(pts, [100, 900], self.config["video"])
        self.assertEqual(len(selected["pts"]), 768)
        self.assertGreater(selected["pts"][-1], 899)
        self.assertTrue(all(100 <= value < 900 for value in selected["pts"]))
        short = select_pts([0.0, 0.1], [0.0, 0.2], self.config["video"])
        self.assertEqual(len(short["pts"]), 4)
        self.assertEqual(short["repeated_frames"], 2)
        with self.assertRaises(ValueError):
            select_pts(pts, [1001, 1002], self.config["video"])

    def test_g8_advantages_and_epoch_schedule(self):
        values = list(range(8))
        advantages = compute_group_advantages(["g"] * 8, values, 8)
        for value, actual in zip(values, advantages):
            self.assertAlmostEqual(actual, (value - 3.5) / (math.sqrt(5.25) + 1e-4))
        pool = [record(index) for index in range(7)]
        budget = epoch_budget(7, 5)
        schedule = build_schedule(pool, pool[:3], "E5_tri_joint", {**budget, "epochs": 5}, 3407)
        self.assertEqual(
            Counter(r["sample_id"] for r in schedule if r["task"] == "TVG"),
            Counter({str(i): 5 for i in range(7)}),
        )
        self.assertEqual(epoch_budget(2444, 1)["steps"], 764)

    def test_g8_pair_gates_and_no_synthetic_candidates(self):
        rows = []
        for index in range(8):
            text = f"<answer>2 to {3 + index / 2}</answer>"
            rows.append(
                candidate(
                    record(),
                    text,
                    index,
                    {
                        "decoded": True,
                        "parseable": True,
                        "joint": index % 2 == 0,
                        "teacher_input_hash": str(index),
                    },
                )
            )
        config = self.config["continuation"]
        pairs, _ = enumerate_pairs(rows, config)
        self.assertTrue(pairs)
        self.assertEqual(select_pair(pairs, 3407, "g"), select_pair(pairs, 3407, "g"))
        gated, _ = enumerate_pairs(rows, {**config, "candidate_gate": "joint"})
        self.assertTrue(all(p["winner"] % 2 == 0 and p["loser"] % 2 == 0 for p in gated))
        rows[0]["source"] = "synthetic"
        with self.assertRaises(ValueError):
            enumerate_pairs(rows, config)

    def test_targets_are_not_serialized_and_offsets_are_decoder_only(self):
        instance = build_task_instance(record(), "SER", config=self.config)
        before = serialize_model_input(instance)
        self.assertEqual(instance.video_request["interval"], [41, 45])
        instance.target["patient"]["value"] = "secret"
        self.assertEqual(before, serialize_model_input(instance))
        self.assertNotIn("secret", before["prompt_text"])

    def test_metrics_keep_failures_and_merge_requires_all_ids(self):
        annotations = [record(0), record(1)]
        predictions = [
            {"sample_id": "0", "completion": "<answer>2 to 6</answer>"},
            {"sample_id": "1", "completion": ""},
        ]
        metrics = evaluate_grounding(predictions, annotations)
        self.assertEqual(metrics["mIoU"], 0.5)
        self.assertEqual(metrics["invalid_interval_rate"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            write_jsonl(Path(directory) / "predictions.rank0.jsonl", predictions[:1])
            with self.assertRaises(ValueError):
                merge_predictions(directory, annotations)
            write_jsonl(Path(directory) / "predictions.rank1.jsonl", predictions[1:])
            self.assertEqual(merge_predictions(directory, annotations), predictions)

    def test_strict_verdicts(self):
        self.assertTrue(
            parse_query_verdict('<answer>{"match":"yes","action_match":"yes"}</answer>')["joint"]
        )
        self.assertFalse(
            parse_query_verdict(
                '<answer>{"match":"yes","match":"no","action_match":"yes"}</answer>'
            )["parseable"]
        )

    def test_no_python_statement_semicolons(self):
        for path in (ROOT / "src").rglob("*.py"):
            for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
                self.assertFalse(token.type == tokenize.OP and token.string == ";", str(path))


if __name__ == "__main__":
    unittest.main()
