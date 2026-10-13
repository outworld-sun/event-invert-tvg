"""Detached remote values, local autograd reinsertion, explicit DDP correction."""


def completion_logp_sum(logps, mask):
    if logps.shape != mask.shape:
        raise ValueError("Completion logps and mask must have identical shapes")
    return (logps.float() * mask).sum(-1)


def local_pair_loss(local_a, global_values, pair, rank, world_size, beta=0.1):
    import torch

    zero = local_a.sum() * 0.0
    if len(global_values) != local_a.numel() * world_size:
        raise ValueError("Global candidate layout mismatch")
    if pair is None:
        return (zero, zero.detach())
    start = rank * local_a.numel()

    def value(index):
        if not 0 <= index < len(global_values):
            raise ValueError("Pair index outside the global candidate group")
        return (
            local_a[index - start]
            if start <= index < start + local_a.numel()
            else global_values[index].detach()
        )

    z = value(pair["winner"]) - value(pair["loser"]) + zero
    loss = -pair["weight"] * torch.nn.functional.logsigmoid(beta * z)
    return (loss * world_size, loss.detach())


def distributed_pair_loss(local_a, pair, beta=0.1):
    import torch

    dist = torch.distributed
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    gathered = [torch.empty_like(local_a) for _ in range(world)]
    if world > 1:
        dist.all_gather(gathered, local_a.detach().contiguous())
    else:
        gathered = [local_a.detach()]
    values = torch.cat(gathered)
    loss, logged = local_pair_loss(local_a, values, pair, rank, world, beta)
    z = None if pair is None else float((values[pair["winner"]] - values[pair["loser"]]).item())
    return (loss, {"pair_loss": float(logged.item()), "z": z})
