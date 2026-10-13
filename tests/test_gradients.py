"""Real eight-process CPU DDP validates preference scaling and accumulation."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import torch

from vtg.preference.loss import distributed_pair_loss
from vtg.training.losses import completion_mask, grpo_loss


def ddp_worker(rank, rendezvous, destination, world):
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=world
    )
    model = torch.nn.Linear(3, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64))
    wrapped = DistributedDataParallel(model)
    design = torch.arange(world * 3, dtype=torch.float64).reshape(world, 3) / 7 - 1
    for micro in range(4):
        with wrapped.no_sync() if micro < 3 else nullcontext():
            local = wrapped(design[rank : rank + 1] * (micro + 1)).flatten()
            pair = None if micro == 3 else {"winner": micro, "loser": world - 1, "weight": 0.7}
            loss, _ = distributed_pair_loss(local, pair)
            (loss / 4).backward()
    if rank == 0:
        torch.save(model.weight.grad, destination)
    torch.distributed.destroy_process_group()


class GradientTests(unittest.TestCase):
    def test_real_eight_rank_ddp(self):
        import torch.multiprocessing as mp

        world = 8
        with tempfile.TemporaryDirectory() as directory:
            destination = str(Path(directory) / "gradient.pt")
            mp.spawn(
                ddp_worker,
                args=(str(Path(directory) / "rendezvous"), destination, world),
                nprocs=world,
                join=True,
            )
            actual = torch.load(destination, weights_only=True)
        theta = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64, requires_grad=True)
        design = torch.arange(world * 3, dtype=torch.float64).reshape(world, 3) / 7 - 1
        for micro in range(3):
            values = (design * (micro + 1) @ theta.T).flatten()
            loss = -0.7 * torch.nn.functional.logsigmoid(0.1 * (values[micro] - values[-1]))
            (loss / 4).backward()
        torch.testing.assert_close(actual, theta.grad, rtol=1e-10, atol=1e-10)

    def test_eos_mask_and_grpo_matches_explicit_formula(self):
        self.assertEqual(
            completion_mask(torch.tensor([[3, 2, 2], [3, 4, 5]]), 2).tolist(),
            [[1, 1, 0], [1, 1, 1]],
        )
        actor = torch.tensor([[-0.3, -0.7, -0.8], [-0.2, -0.9, -0.4]], requires_grad=True)
        reference = actor.detach() - 0.1
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        advantages = torch.tensor([1.0, -1.0])
        loss, _ = grpo_loss(actor, reference, mask, advantages, 0.04, 0.2)
        difference = reference - actor
        expected = (
            (
                -(actor - actor.detach()).exp() * advantages[:, None]
                + 0.04 * (difference.exp() - difference - 1)
            )
            * mask
        ).sum(-1) / mask.sum(-1)
        torch.testing.assert_close(loss, expected.mean())
        loss.backward()
        self.assertEqual(float(actor.grad[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
