"""Exercise the real shell, option precedence, and training-argument wiring."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vtg.cli.arguments import resolve_arguments, training_parser
from vtg.data.annotations import read_training_annotations
from vtg.training.options import training_argument_values
from vtg.training.schedule import build_schedule, epoch_budget, epoch_checkpoint_steps
from test_protocol import record

ROOT = Path(__file__).resolve().parents[1]


def resolve(options, environment=None):
    with patch.dict(os.environ, environment or {}, clear=True):
        parser = training_parser()
        return resolve_arguments(parser.parse_args(options), parser)


class ArgumentTests(unittest.TestCase):
    def test_explicit_eight_gpu_parameters_reach_hf(self):
        config = resolve(
            [
                "--branch",
                "N1_IoU_pair",
                "--model_name_or_path",
                "/models/model with spaces",
                "--train_data_path",
                "/data/train.json",
                "--video_folder",
                "/data/videos",
                "--parent_checkpoint",
                "/models/C0",
                "--num_generations",
                "8",
                "--per_device_train_batch_size",
                "1",
                "--gradient_accumulation_steps",
                "2",
                "--learning_rate",
                "0.000003",
                "--num_train_epochs",
                "3",
                "--max_prompt_length",
                "8192",
                "--max_completion_length",
                "256",
                "--logging_steps",
                "5",
                "--logging_dir",
                "/logs/test",
                "--run_name",
                "trial",
                "--report_to",
                "none",
                "--save_strategy",
                "epoch",
                "--save_only_model",
                "false",
                "--bf16",
                "--torch_dtype",
                "bfloat16",
                "--gradient_checkpointing",
                "false",
                "--data_seed",
                "42",
                "--reward_funcs",
                "iou",
                "format",
            ],
            {"WORLD_SIZE": "8"},
        )
        self.assertEqual(config["paths"]["model_path"], "/models/model with spaces")
        self.assertEqual(config["training"]["unique_prompts_per_update"], 2)
        self.assertEqual(config["continuation"]["schedule_seed"], 42)
        values = training_argument_values(config, "/output")
        self.assertEqual(values["learning_rate"], 3e-6)
        self.assertEqual(values["logging_steps"], 5)
        self.assertEqual(values["logging_dir"], "/logs/test")
        self.assertEqual(values["run_name"], "trial")
        self.assertFalse(values["gradient_checkpointing"])
        self.assertTrue(values["bf16"])
        self.assertEqual(values["report_to"], [])
        self.assertEqual(values["save_strategy"], "no")
        self.assertFalse(values["save_only_model"])

    def test_json_then_paths_then_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "paths.json"
            local.write_text(
                json.dumps(
                    {"training": {"learning_rate": 0.002}, "paths": {"video_root": "/local/videos"}}
                )
            )
            config = resolve(
                [
                    "--config",
                    str(ROOT / "configs/n2_8xa100.json"),
                    "--paths",
                    str(local),
                    "--learning_rate",
                    "0.001",
                ]
            )
        self.assertEqual(config["branch"], "N2_Event_pair")
        self.assertEqual(config["paths"]["video_root"], "/local/videos")
        self.assertEqual(config["training"]["learning_rate"], 0.001)

    def test_max_steps_fp32_and_model_only(self):
        config = resolve(
            [
                "--max_steps",
                "9",
                "--save_strategy",
                "steps",
                "--bf16",
                "false",
                "--save_only_model",
                "true",
            ]
        )
        values = training_argument_values(config, "/out")
        self.assertIsNone(config["training"]["num_train_epochs"])
        self.assertEqual(values["max_steps"], 9)
        self.assertFalse(values["bf16"])
        self.assertTrue(values["save_only_model"])
        self.assertEqual(config["model"]["precision"], "fp32")

    def test_unsupported_flags_are_not_silent(self):
        for options in (
            ["--is_early_stopping", "true"],
            ["--is_curriculum_learning", "true"],
            ["--reward_funcs", "iou_v2", "format"],
            ["--num_train_epochs", "2", "--max_steps", "5"],
            ["--bf16", "--torch_dtype", "float32"],
            ["--world_size", "1"],
        ):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    resolve(options, {"WORLD_SIZE": "8"})
        config = resolve(
            [
                "--slide_window",
                "false",
                "--is_early_stopping",
                "false",
                "--is_curriculum_learning",
                "false",
            ]
        )
        self.assertEqual(config["branch"], "E5")

    def test_real_shell_overrides_and_quoted_paths(self):
        environment = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHON_BIN": sys.executable,
            "BRANCH": "N1_IoU_pair",
            "LOAD_MODEL_PATH": "/models/path with spaces",
            "GPU_LIST": "0,1,2,3,4,5,6,7",
            "NPROC_PER_NODE": "8",
        }
        environment.pop("WORLD_SIZE", None)
        environment.pop("RANK", None)
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/train.sh"),
                "--learning_rate",
                "0.000007",
                "--print_config",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        config = json.loads(result.stdout)
        self.assertEqual(config["paths"]["model_path"], "/models/path with spaces")
        self.assertEqual(config["training"]["learning_rate"], 7e-6)
        self.assertEqual(config["training"]["world_size"], 8)
        self.assertEqual(config["training"]["save_strategy"], "epoch")

    def test_epoch_boundaries_follow_data_epochs(self):
        pool = [record(index) for index in range(7)]
        budget = epoch_budget(len(pool), 3, 4)
        schedule = build_schedule(pool, pool[:2], "E5_tri_joint", {**budget, "epochs": 3}, 3407)
        steps = epoch_checkpoint_steps(schedule, 4)
        self.assertEqual(steps[-1], budget["steps"])
        self.assertEqual(len(steps), 3)
        for epoch, step in enumerate(steps, 1):
            count = sum(row["data_epoch"] <= epoch for row in schedule)
            self.assertGreaterEqual(step * 4, count)
            self.assertLess((step - 1) * 4, count)

    def test_annotation_readers_for_timer1_and_charades(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "video": "/source/charades/A.mp4",
                            "timestamp": [1, 2],
                            "sentence": "A person sits.",
                            "duration": 5,
                        }
                    ]
                )
            )
            timer1 = read_training_annotations(path, "timer1")
            self.assertEqual(timer1[0]["video_relpath"], "charades/A.mp4")
            path.write_text(
                json.dumps(
                    {
                        "A": {
                            "timestamps": [[1, 2]],
                            "sentences": ["A person sits."],
                            "duration": 5,
                            "Missing": ["A person."],
                            "verb": ["sit"],
                        }
                    }
                )
            )
            charades = read_training_annotations(path, "charades")
            self.assertEqual(charades[0]["video_relpath"], "A.mp4")
            self.assertEqual(charades[0]["source"], "charades_train")
            self.assertEqual(charades[0]["gt_sec"], [1, 2])


if __name__ == "__main__":
    unittest.main()
