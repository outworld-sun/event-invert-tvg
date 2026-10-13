"""HF owns DDP/ZeRO-3, progress, RNG restoration, and gradient accumulation."""

from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainerCallback

from vtg.preference.diagnostics import MatchedControl
from vtg.preference.loss import completion_logp_sum, distributed_pair_loss
from vtg.preference.protocol import protocol
from vtg.preference.verifier import CandidateVerifier
from vtg.training.losses import completion_logps, grpo_loss
from vtg.training.rollout import prepare_rollout
from vtg.utils.io import append_jsonl
from vtg.video.decoder import VideoAdapter


class BackendWitness(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        from vtg.training.zero import record_backend

        record_backend(self.trainer)


class VTGTrainer(Trainer):
    def __init__(self, *args, config, schedule, records, parser, guard, parent_hash, **kwargs):
        self.config = config
        self.schedule = schedule
        self.records = records
        self.parser = parser
        self.guard = guard
        self.micro_cursor = 0
        self.last_rollout = {}
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        if self.accelerator.gradient_accumulation_steps != 1:
            raise ValueError("HF must own accumulation scaling exactly once")
        if self.is_fsdp_enabled:
            raise ValueError("Choose ZeRO-3 or DDP, not FSDP")
        self.output = Path(self.args.output_dir)
        self.add_callback(BackendWitness(self))
        self.video = VideoAdapter(
            config["paths"]["video_root"],
            config["video"],
            self.processing_class.image_processor.to_dict(),
        )
        self.reference_ids = set(records)
        self.verifier = CandidateVerifier(
            guard.model,
            self.processing_class,
            self.video,
            parser,
            config,
            parent_hash,
            self.output / f"judge_cache.rank{self.args.process_index}",
        )
        calibration = config["continuation"]["matched_calibration"]
        self.matched = (
            MatchedControl(calibration, config["seed"], protocol(config))
            if config["branch"] == "A_matched"
            else None
        )
        if (
            self.matched
            and self.matched.spec.get("num_generations") != config["training"]["num_generations"]
        ):
            raise ValueError("Matched calibration candidate group size differs from this run")

    def prepare_coverage_model(self):
        """Initialize the sharded engine even when coverage skips Trainer.train."""
        if self.is_deepspeed_enabled:
            from transformers.integrations.deepspeed import deepspeed_init

            self.guard.lock()
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=self.args.max_steps
            )
            self.guard.assert_optimizer(self.optimizer)
            self.model_wrapped, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
            self.deepspeed = self.model_wrapped
        from vtg.training.zero import record_backend

        record_backend(self)

    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)

    def _prepare_inputs(self, inputs):
        return prepare_rollout(self)

    def create_optimizer(self):
        self.guard.lock()
        result = super().create_optimizer()
        self.guard.assert_optimizer(self.optimizer)
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.guard.backward_open = True
        actor = completion_logps(
            model, inputs["batch"], inputs["completion_ids"], inputs["completion_mask"]
        )
        training = self.config["training"]
        preference = self.config["continuation"]
        loss, kl = grpo_loss(
            actor,
            inputs["reference_logps"],
            inputs["completion_mask"],
            inputs["advantages"],
            training["kl_beta"],
            training["policy_clip_epsilon"],
        )
        relative = completion_logp_sum(actor - inputs["reference_logps"], inputs["completion_mask"])
        pair_loss, metrics = distributed_pair_loss(relative, inputs["pair"], preference["beta_dpo"])
        total = loss + preference["lambda_pair"] * pair_loss
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training loss")
        self.last_rollout.update(kl=float(kl), pair_loss=metrics["pair_loss"])
        append_jsonl(
            self.output / f"losses.rank{self.args.process_index}.jsonl",
            {
                **metrics,
                "kl": float(kl),
                "grpo_loss": float(loss.detach()),
                "global_step": self.state.global_step,
                "group_id": inputs["group_id"],
            },
        )
        return (total, {}) if return_outputs else total

    def training_step(self, *args, **kwargs):
        try:
            return super().training_step(*args, **kwargs)
        finally:
            self.guard.backward_open = False
            self.guard.lock()

    def log(self, logs, start_time=None):
        super().log({**logs, **self.last_rollout}, start_time)

    def _save(self, output_dir=None, state_dict=None):
        from transformers.trainer import TRAINING_ARGS_NAME

        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.accelerator.unwrap_model(self.model).save_pretrained(
            destination,
            state_dict=state_dict,
            selected_adapters=["default"],
            safe_serialization=True,
        )
        self.processing_class.save_pretrained(destination)
        torch.save(self.args, destination / TRAINING_ARGS_NAME)
        self.guard.lock()

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        # Load actor weights only. Frozen C0 is rebuilt from the immutable parent.
        state = load_peft_weights(str(resume_from_checkpoint), device="cpu")
        set_peft_model_state_dict(self.guard.model, state, adapter_name="default")
        self.guard.activate_actor()
