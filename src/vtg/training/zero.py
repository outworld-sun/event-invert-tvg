"""Collective parameter access for ZeRO-3, with a no-op DDP path.

Every rank enters these contexts in the same order. Once inference weights are
gathered and hooks suspended, ranks may generate different lengths and execute
different numbers of cached candidate judgments without collective deadlocks.
"""

from contextlib import contextmanager


@contextmanager
def gathered_parameters(parameters):
    parameters = [parameter for parameter in parameters if hasattr(parameter, "ds_id")]
    if not parameters:
        yield
        return
    from deepspeed import zero

    with zero.GatheredParameters(parameters):
        yield


@contextmanager
def rollout_model(trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    if not trainer.is_deepspeed_enabled:
        yield model
        return
    engine = trainer.model_wrapped
    if not hasattr(engine, "optimizer"):
        raise RuntimeError("Initialize the ZeRO-3 engine before preparing rollouts")
    offload = engine.optimizer.parameter_offload
    with gathered_parameters(model.parameters()):
        for parameter in offload.module.parameters():
            if hasattr(parameter, "ds_active_sub_modules"):
                parameter.ds_active_sub_modules.clear()
        for hook in offload.forward_hooks + offload.backward_hooks:
            hook.remove()
        offload.forward_hooks = []
        offload.backward_hooks = []
        try:
            yield model
        finally:
            offload._register_deepspeed_module(offload.module)


def adapter_snapshot(model, adapter="frozen_C0"):
    selected = [(name, value) for name, value in model.named_parameters() if f".{adapter}." in name]
    with gathered_parameters(value for _, value in selected):
        return {name: value.detach().cpu().clone() for name, value in selected}


def record_backend(trainer):
    from vtg.utils.io import save_json

    report = {"backend": "ddp", "world_size": trainer.args.world_size}
    if trainer.is_deepspeed_enabled:
        engine = trainer.model_wrapped
        stage = engine.zero_optimization_stage()
        if stage != 3:
            raise RuntimeError("Expected an active ZeRO-3 engine")
        optimizer = engine.optimizer
        parameters = [parameter for group in optimizer.fp16_groups for parameter in group]
        if {id(parameter) for parameter in parameters} != trainer.guard.actor_ids:
            raise RuntimeError("ZeRO optimizer contains parameters outside actor LoRA")
        if engine.gradient_accumulation_steps() != trainer.args.gradient_accumulation_steps:
            raise RuntimeError("DeepSpeed gradient accumulation differs from the task schedule")
        report.update(
            backend="deepspeed_zero3",
            stage=stage,
            engine=type(engine).__name__,
            optimizer=type(optimizer.optimizer).__name__,
            gradient_accumulation_steps=engine.gradient_accumulation_steps(),
            partitioned_parameters=sum(
                hasattr(parameter, "ds_id") for parameter in trainer.model.parameters()
            ),
            resolved_deepspeed=trainer.accelerator.state.deepspeed_plugin.deepspeed_config,
        )
    if trainer.is_world_process_zero():
        save_json(trainer.output / "backend.json", report)


def check_parameters(model, frozen, device):
    import torch
    import torch.distributed as dist

    current = adapter_snapshot(model)
    if current.keys() != frozen.keys() or any(
        not torch.equal(current[name], frozen[name].to(current[name].dtype)) for name in frozen
    ):
        raise RuntimeError("Frozen C0 adapter changed")
    error = torch.zeros((), device=device)
    actor = [parameter for parameter in model.parameters() if parameter.requires_grad]
    with gathered_parameters(actor):
        if dist.is_initialized() and dist.get_world_size() > 1:
            for parameter in actor:
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                error = torch.maximum(error, (reference - parameter).abs().max().float())
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
    if not torch.isfinite(error) or float(error) > 1e-6:
        raise RuntimeError("Actor parameters diverged across ranks")
    return float(error)
