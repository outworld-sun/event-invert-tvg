"""Language-only LoRA and a memory-sharing frozen reference policy."""

from contextlib import contextmanager

from vtg.models.adapters import FrozenAdapters, load_adapters


class BaseReference(FrozenAdapters):
    """E5 starts from base weights, so its reference disables the actor adapter."""

    @contextmanager
    def reference(self):
        import torch

        if self.backward_open:
            raise RuntimeError("Cannot change reference during actor backward")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self.model.disable_adapter():
                yield self.model
        finally:
            self.activate_actor()
            self.model.train(was_training)


def load_training_model(config):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["paths"]["model_path"],
        torch_dtype=torch.bfloat16 if config["model"]["precision"] == "bf16" else torch.float32,
        attn_implementation=config["training"]["attn_implementation"],
    )
    model.config.use_cache = False
    if config["branch"] == "E5":
        lora = config["model"]["lora"]
        targets = [
            name
            for name, _ in model.named_modules()
            if name.startswith("model.layers.") and name.rsplit(".", 1)[-1] in lora["targets"]
        ]
        if not targets:
            raise ValueError("No language-only attention modules found for LoRA")
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora["rank"],
                lora_alpha=lora["alpha"],
                lora_dropout=0.0,
                target_modules=targets,
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        guard = BaseReference(model)
    else:
        model, guard = load_adapters(model, config["continuation"]["parent_checkpoint"])
    if config["training"].get("deepspeed") and config["model"]["gradient_checkpointing"]:
        # Reentrant checkpointing requires a gradient-carrying embedding output.
        model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model, guard
