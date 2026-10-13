"""Offline tests of real Qwen video processing, adapter switching, and backward."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from vtg.config import load_config
from vtg.models.adapters import load_adapters
from vtg.models.processing import encode_video
from vtg.training.losses import completion_logps
from vtg.video.decoder import VideoAdapter

ROOT = Path(__file__).resolve().parents[1]


def tiny_processor():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor

    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[EOS]": 2,
        "user": 3,
        "assistant": 4,
        "<|vision_start|>": 5,
        "<|vision_end|>": 6,
        "<|video_pad|>": 7,
        "<|image_pad|>": 8,
        "event": 9,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=[
            "<|vision_start|>",
            "<|vision_end|>",
            "<|video_pad|>",
            "<|image_pad|>",
        ],
    )
    template = "{% for message in messages %}user {% for item in message['content'] %}{% if item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>{% else %}event {% endif %}{% endfor %}{% endfor %} assistant"
    return Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer, chat_template=template
    )


def tiny_model():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_VLConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        image_token_id=8,
        video_token_id=7,
        vision_start_token_id=5,
        vision_end_token_id=6,
        eos_token_id=2,
        pad_token_id=1,
        attention_dropout=0.0,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
            "window_size": 112,
        },
    )
    config._attn_implementation = "eager"
    return Qwen2_5_VLForConditionalGeneration(config)


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.config = load_config(ROOT / "configs/n1_8xa100.json")

    def test_processor_keeps_long_video_geometry(self):
        processor = tiny_processor()
        frames = torch.zeros(768, 3, 84, 140, dtype=torch.uint8)
        meta = {"effective_fps": 2.0, "visual_tokens": 768 // 2 * 3 * 5}
        batch = encode_video(processor, "event", frames, meta, self.config)
        self.assertEqual(batch["video_grid_thw"].tolist(), [[384, 6, 10]])
        self.assertEqual(batch["input_ids"].eq(7).sum().item(), 5760)

    def test_real_qwen_autoregressive_generation(self):
        model = tiny_model().eval()
        frames = torch.zeros(4, 3, 56, 56, dtype=torch.uint8)
        batch = encode_video(
            tiny_processor(),
            "event",
            frames,
            {"effective_fps": 2.0, "visual_tokens": 8},
            self.config,
        )
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=3, do_sample=False, use_cache=True)
        self.assertGreater(generated.shape[1], batch["input_ids"].shape[1])
        torch.testing.assert_close(generated[:, : batch["input_ids"].shape[1]], batch["input_ids"])

    def test_real_decoder_crop_and_processor(self):
        import av

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=8)
                stream.width = 224
                stream.height = 112
                stream.pix_fmt = "yuv420p"
                for index in range(16):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((112, 224, 3), index * 10, dtype=np.uint8), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            adapter = VideoAdapter(directory, self.config["video"])
            frames, meta = adapter.load({"video_relpath": "fixture.mp4", "interval": [0.5, 1.5]})
            self.assertTrue(all(0.5 <= pts < 1.5 for pts in meta["pts"]))
            self.assertEqual(frames.shape[0], 4)
            batch = encode_video(tiny_processor(), "event", frames, meta, self.config)
            self.assertEqual(
                int(batch["video_grid_thw"].prod(-1).sum()) // 4, meta["visual_tokens"]
            )

    def test_frozen_adapter_and_real_qwen_backward(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

        with tempfile.TemporaryDirectory() as directory:
            initial = tiny_model()
            original = get_peft_model(
                initial,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    target_modules=["model.layers.0.self_attn.q_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            with torch.no_grad():
                for name, parameter in original.named_parameters():
                    if "lora_" in name:
                        parameter.uniform_(-0.01, 0.01)
            original.save_pretrained(directory)
            model, guard = load_adapters(tiny_model(), directory)
            expected = {
                key: value.clone()
                for key, value in get_peft_model_state_dict(model, adapter_name="frozen_C0").items()
            }
            processor = tiny_processor()
            frames = torch.zeros(4, 3, 56, 56, dtype=torch.uint8)
            batch = encode_video(
                processor, "event", frames, {"effective_fps": 2.0, "visual_tokens": 8}, self.config
            )
            completion = torch.tensor([[9, 2]])
            mask = torch.ones_like(completion)
            with guard.reference() as frozen:
                reference = completion_logps(frozen, batch, completion, mask)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.train()
            actor = completion_logps(model, batch, completion, mask)
            torch.testing.assert_close(reference, actor)
            (-actor.sum()).backward()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=0.01
            )
            guard.assert_optimizer(optimizer)
            optimizer.step()
            for key, value in get_peft_model_state_dict(model, adapter_name="frozen_C0").items():
                torch.testing.assert_close(expected[key], value, rtol=0, atol=0)

    def test_trainer_checkpoint_resume_with_real_qwen(self):
        from peft import LoraConfig, get_peft_model
        from transformers import TrainingArguments
        from test_protocol import record
        from vtg.tasks.schemas import RewardBreakdown
        from vtg.training.checkpoints import RunCheckpoint
        from vtg.training.trainer import VTGTrainer

        config = self.config
        config["branch"] = "N0_E5_cont"
        config["continuation"].update(branch="N0_E5_cont", pair_enabled=False)
        config["training"].update(
            world_size=1, num_generations=2, completion_micro_batch_per_device=2
        )
        processor = tiny_processor()
        frames = torch.zeros(4, 3, 56, 56, dtype=torch.uint8)
        meta = {
            "effective_fps": 2.0,
            "visual_tokens": 8,
            "video_sample_hash": "fixture",
            "shape": list(frames.shape),
        }
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            output = Path(directory) / "run"
            output.mkdir()
            original = get_peft_model(
                tiny_model(),
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    target_modules=["model.layers.0.self_attn.q_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            original.save_pretrained(parent)
            arguments = TrainingArguments(
                output_dir=str(output),
                use_cpu=True,
                max_steps=2,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                learning_rate=0.01,
                report_to=[],
                remove_unused_columns=False,
                save_steps=1,
                logging_steps=1,
                disable_tqdm=False,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            schedule = [
                {
                    "group_id": f"g{i}",
                    "schedule_index": i,
                    "sample_id": "0",
                    "task": "TVG",
                    "hidden_field": None,
                }
                for i in range(4)
            ]

            def make_trainer(stop):
                torch.manual_seed(123)
                model, guard = load_adapters(tiny_model(), parent)

                def generate(input_ids, **kwargs):
                    tail = torch.tensor([[9, 2], [10, 2]], device=input_ids.device)
                    return torch.cat([input_ids, tail], dim=1)

                model.generate = generate
                return VTGTrainer(
                    model=model,
                    args=arguments,
                    train_dataset=[{"index": i} for i in range(8)],
                    processing_class=processor,
                    config=config,
                    schedule=schedule,
                    records={"0": record()},
                    parser=None,
                    guard=guard,
                    parent_hash="fixture",
                    callbacks=[RunCheckpoint({"fixture": True}, 2, stop)],
                )

            def score(instance, completion, config, parser):
                return RewardBreakdown(total=float(completion.strip() == "event"), validity=True)

            with (
                patch("vtg.video.decoder.VideoAdapter.load", return_value=(frames, meta)),
                patch("vtg.training.rollout.score_completion", side_effect=score),
            ):
                trainer = make_trainer(1)
                trainer.train()
                self.assertEqual(trainer.state.global_step, 1)
                checkpoint = output / "checkpoint-1"
                self.assertTrue((checkpoint / "optimizer.pt").exists())
                self.assertFalse((checkpoint / "frozen_C0").exists())
                trainer = make_trainer(None)
                trainer.train(resume_from_checkpoint=str(checkpoint))
                self.assertEqual(trainer.state.global_step, 2)
                from vtg.utils.io import read_jsonl

                groups = [row["group_id"] for row in read_jsonl(output / "pair_diagnostics.jsonl")]
                self.assertEqual(groups, ["g0", "g1", "g2", "g3"])
                from peft import get_peft_model_state_dict

                resumed = {
                    key: value.clone()
                    for key, value in get_peft_model_state_dict(trainer.model).items()
                }
                arguments.output_dir = str(Path(directory) / "uninterrupted")
                Path(arguments.output_dir).mkdir()
                uninterrupted = make_trainer(None)
                uninterrupted.train()
                for key, value in get_peft_model_state_dict(uninterrupted.model).items():
                    torch.testing.assert_close(resumed[key], value, rtol=0, atol=0)

    def test_data_epoch_checkpoint_and_model_only_mode(self):
        from peft import LoraConfig, get_peft_model
        from transformers import TrainingArguments
        from test_protocol import record
        from vtg.tasks.schemas import RewardBreakdown
        from vtg.training.checkpoints import RunCheckpoint
        from vtg.training.trainer import VTGTrainer

        config = self.config
        config["branch"] = "N0_E5_cont"
        config["continuation"].update(branch="N0_E5_cont", pair_enabled=False)
        config["training"].update(world_size=1, num_generations=2)
        processor = tiny_processor()
        frames = torch.zeros(4, 3, 56, 56, dtype=torch.uint8)
        meta = {
            "effective_fps": 2.0,
            "visual_tokens": 8,
            "video_sample_hash": "fixture",
            "shape": list(frames.shape),
        }
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            output = Path(directory) / "run"
            output.mkdir()
            original = get_peft_model(
                tiny_model(),
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    target_modules=["model.layers.0.self_attn.q_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
            original.save_pretrained(parent)
            model, guard = load_adapters(tiny_model(), parent)

            def generate(input_ids, **kwargs):
                tail = torch.tensor([[9, 2], [10, 2]], device=input_ids.device)
                return torch.cat([input_ids, tail], dim=1)

            model.generate = generate
            args = TrainingArguments(
                output_dir=str(output),
                use_cpu=True,
                max_steps=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                save_strategy="no",
                save_only_model=True,
                report_to=[],
                remove_unused_columns=False,
                disable_tqdm=True,
            )
            schedule = [
                {
                    "group_id": f"g{i}",
                    "schedule_index": i,
                    "sample_id": "0",
                    "task": "TVG",
                    "hidden_field": None,
                }
                for i in range(3)
            ]
            trainer = VTGTrainer(
                model=model,
                args=args,
                train_dataset=[{"index": i} for i in range(6)],
                processing_class=processor,
                config=config,
                schedule=schedule,
                records={"0": record()},
                parser=None,
                guard=guard,
                parent_hash="fixture",
                callbacks=[RunCheckpoint({"fixture": True}, 1, epoch_save_steps=[2, 3])],
            )
            with (
                patch("vtg.video.decoder.VideoAdapter.load", return_value=(frames, meta)),
                patch(
                    "vtg.training.rollout.score_completion",
                    return_value=RewardBreakdown(total=1.0, validity=True),
                ),
            ):
                trainer.train()
            self.assertFalse((output / "checkpoint-1").exists())
            for step in (2, 3):
                checkpoint = output / f"checkpoint-{step}"
                self.assertTrue((checkpoint / "adapter_model.safetensors").exists())
                self.assertTrue((checkpoint / "schedule_state.json").exists())
                self.assertFalse((checkpoint / "optimizer.pt").exists())


if __name__ == "__main__":
    unittest.main()
