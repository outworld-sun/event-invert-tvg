"""Collectives that also work for single-process smoke tests."""


def gather_objects(value):
    import torch.distributed as dist

    if not dist.is_initialized():
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def broadcast(value):
    import torch.distributed as dist

    values = [value]
    if dist.is_initialized():
        dist.broadcast_object_list(values, src=0)
    return values[0]


def barrier():
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def collective_call(function):
    """Propagate local preparation failures before the next model collective."""
    result = None
    error = None
    try:
        result = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = gather_objects(error)
    if any(errors):
        raise RuntimeError(f"Distributed preparation failed: {errors}")
    return result
