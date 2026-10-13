"""One base model, two adapters; no switch during actor forward/backward."""

from contextlib import contextmanager


class FrozenAdapters:
    actor = "default"
    frozen = "frozen_C0"

    def __init__(self, model):
        self.model = model
        self.backward_open = False
        self.actor_ids = {id(p) for n, p in model.named_parameters() if self._is_actor(n)}
        if not self.actor_ids:
            raise ValueError("Actor language LoRA parameters not found")
        self.activate_actor()

    def _is_actor(self, name):
        return "lora_" in name and ".default." in name and ("visual" not in name)

    def lock(self):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(self._is_actor(name))
        current = {id(p) for p in self.model.parameters() if p.requires_grad}
        if current != self.actor_ids:
            raise RuntimeError("Trainable parameter IDs changed")

    def activate_actor(self):
        self.model.set_adapter(self.actor)
        self.lock()

    def assert_optimizer(self, optimizer):
        if hasattr(optimizer, "param_groups"):
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
        else:
            # HF passes DummyOptim to DeepSpeed, which builds CPUAdam itself.
            parameters = optimizer.params
        ids = {id(parameter) for parameter in parameters}
        if ids != self.actor_ids:
            raise RuntimeError("Optimizer must contain exactly actor language LoRA parameters")

    @contextmanager
    def reference(self):
        import torch

        if self.backward_open:
            raise RuntimeError("Cannot switch adapters until actor backward has completed")
        was_training = self.model.training
        self.model.set_adapter(self.frozen)
        self.lock()
        self.model.eval()
        try:
            with torch.no_grad():
                yield self.model
        finally:
            self.activate_actor()
            self.model.train(was_training)


def load_adapters(base, checkpoint):
    import copy
    from peft import PeftModel, get_peft_model_state_dict
    import torch

    model = PeftModel.from_pretrained(
        base, str(checkpoint), adapter_name="default", is_trainable=True
    )
    frozen_config = copy.deepcopy(model.peft_config["default"])
    frozen_config.inference_mode = True
    model.add_adapter("frozen_C0", frozen_config)
    for name, parameter in model.named_parameters():
        if ".frozen_C0." in name:
            parameter.data = parameter.data.float()
    model.load_adapter(str(checkpoint), adapter_name="frozen_C0", is_trainable=False)
    from vtg.training.zero import gathered_parameters

    with gathered_parameters(p for n, p in model.named_parameters() if "lora_" in n):
        actor = get_peft_model_state_dict(model, adapter_name="default")
        frozen = get_peft_model_state_dict(model, adapter_name="frozen_C0")
        if actor.keys() != frozen.keys() or any(
            not torch.equal(actor[key], frozen[key]) for key in actor
        ):
            raise RuntimeError("Actor/frozen initialization must exactly equal C0")
    guard = FrozenAdapters(model)
    return (model, guard)
