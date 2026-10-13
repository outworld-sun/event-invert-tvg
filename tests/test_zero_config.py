"""ZeRO configuration and portable command-line behavior without GPUs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from test_arguments import resolve
from vtg.training.backend import read_deepspeed_config, validate_deepspeed_config
from vtg.training.options import training_argument_values

ROOT = Path(__file__).resolve().parents[1]


class ZeroConfigurationTests(unittest.TestCase):
    def test_default_enables_official_zero3_offload(self):
        config = resolve([])
        ds = training_argument_values(config, "/output")["deepspeed"]
        zero = ds["zero_optimization"]
        self.assertEqual(zero["stage"], 3)
        self.assertEqual(zero["offload_optimizer"]["device"], "cpu")
        self.assertEqual(zero["offload_param"]["device"], "cpu")
        self.assertEqual(ds["optimizer"]["type"], "AdamW")
        self.assertEqual(ds["gradient_accumulation_steps"], "auto")
        self.assertTrue(zero["stage3_gather_16bit_weights_on_model_save"])

    def test_official_alias_and_command_line_override(self):
        ds = read_deepspeed_config(ROOT / "scripts/zero3_offload.json")
        zero = ds["zero_optimization"]
        zero["gather_16bit_weights_on_model_save"] = zero.pop(
            "stage3_gather_16bit_weights_on_model_save"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero config.json"
            path.write_text(json.dumps(ds))
            config = resolve(["--deepspeed", str(path)])
        actual = config["training"]["deepspeed"]["zero_optimization"]
        self.assertTrue(actual["stage3_gather_16bit_weights_on_model_save"])
        self.assertNotIn("gather_16bit_weights_on_model_save", actual)
        self.assertIsNone(resolve(["--deepspeed", "none"])["training"]["deepspeed"])

    def test_nonzero3_or_incomplete_export_is_rejected(self):
        ds = read_deepspeed_config(ROOT / "scripts/zero3_offload.json")
        for setting, value in (("stage", 2), ("stage3_gather_16bit_weights_on_model_save", False)):
            changed = copy.deepcopy(ds)
            changed["zero_optimization"][setting] = value
            with self.assertRaises(ValueError):
                validate_deepspeed_config(changed)


if __name__ == "__main__":
    unittest.main()
