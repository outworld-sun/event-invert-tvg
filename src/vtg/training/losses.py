"""Completion-only GRPO and likelihoods with EOS included in the mask."""

import torch


def completion_mask(ids, eos_token_id):
    eos = ids.eq(eos_token_id)
    # Keep the first EOS but discard every later padding/repeated EOS position.
    return (eos.cumsum(dim=1) - eos.long()).eq(0).long()


def completion_logps(model, batch, completion_ids, mask):
    prompt_length = batch["input_ids"].shape[1]
    kwargs = dict(batch)
    kwargs["input_ids"] = torch.cat([batch["input_ids"], completion_ids], dim=1)
    kwargs["attention_mask"] = torch.cat([batch["attention_mask"], mask], dim=1)
    output = model(**kwargs, use_cache=False)
    logits = output.logits[:, prompt_length - 1 : -1]
    # Convert only completion logits to FP32, never the full long-video prompt.
    selected = logits.float().gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    normalizer = torch.logsumexp(logits.float(), dim=-1)
    return selected - normalizer


def grpo_loss(actor_logps, reference_logps, mask, advantages, beta, epsilon):
    """Single rollout iteration: detached current policy is the old policy."""
    ratio = (actor_logps - actor_logps.detach()).exp()
    unclipped = ratio * advantages[:, None]
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantages[:, None]
    difference = reference_logps - actor_logps
    kl = difference.exp() - difference - 1
    per_token = -torch.minimum(unclipped, clipped) + beta * kl
    loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    return loss, ((kl.detach() * mask).sum() / mask.sum().clamp(min=1))
