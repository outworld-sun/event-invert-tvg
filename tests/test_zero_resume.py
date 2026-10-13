"""Check complete DeepSpeed shard layouts before handing restoration to HF."""

import tempfile
import unittest
from pathlib import Path

from vtg.config import load_config
from vtg.training.checkpoints import validate_resume
from vtg.utils.io import save_json

ROOT = Path(__file__).resolve().parents[1]


class ZeroResumeTests(unittest.TestCase):
    def test_sharded_scheduler_and_missing_rank(self):
        config = load_config(ROOT / "configs/n1_8xa100.json")
        config["training"]["world_size"] = 2
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-1"
            save_json(checkpoint / "trainer_state.json", {"global_step": 1})
            save_json(
                checkpoint / "schedule_state.json",
                {"identity": {"fixture": 1}, "next_schedule_index": 4},
            )
            (checkpoint / "latest").write_text("global_step1")
            (checkpoint / "adapter_model.safetensors").touch()
            shards = checkpoint / "global_step1"
            shards.mkdir()
            for rank in range(2):
                (checkpoint / f"rng_state_{rank}.pth").touch()
                (shards / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt").touch()
                (shards / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").touch()
            self.assertEqual(validate_resume(checkpoint, directory, {"fixture": 1}, config), 1)
            config["model"]["precision"] = "fp32"
            for rank in range(2):
                (shards / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").rename(
                    shards / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                )
            self.assertEqual(validate_resume(checkpoint, directory, {"fixture": 1}, config), 1)
            (shards / "zero_pp_rank_1_mp_rank_00_optim_states.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_resume(checkpoint, directory, {"fixture": 1}, config)
