"""Schedule for temporal video grounding."""

import math
import random
from collections import defaultdict
from vtg.utils.io import digest

TASKS = {
    "E1_tvg": [],
    "E2_invert_matched": ["VC", "AR", "VD"],
    "E3_scc_only": ["SCC"],
    "E4_tri_local": ["SCC", "SER", "ED"],
    "E5_tri_joint": ["SCC", "SER", "ED"],
    "E6_unbound_local": ["SCC", "SER", "ED"],
    "A_no_SCC": ["SER", "ED"],
    "A_no_SER": ["SCC", "ED"],
    "A_no_ED": ["SCC", "SER"],
    "A_action_patient": ["SCC", "SER", "ED"],
}


def epoch_budget(pool_size, epochs, prompts_per_update=4):
    """Exact grounding passes, with auxiliary groups filling the final full update."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1 or (pool_size < 1):
        raise ValueError("Epochs and grounding pool size must be positive integers")
    grounding = pool_size * epochs
    steps = (grounding * 5 + 4 * prompts_per_update - 1) // (4 * prompts_per_update)
    return {
        "steps": steps,
        "grounding_groups": grounding,
        "auxiliary_groups": steps * prompts_per_update - grounding,
    }


def build_schedule(grounding_pool, common_aux_pool, experiment, budget, seed):
    if experiment not in TASKS:
        raise ValueError(experiment)
    n = budget["steps"] * budget.get("prompts_per_update", 4)
    rng = random.Random(seed)
    modes = [True] * round(0.2 * n) + [False] * (n - round(0.2 * n))
    rng.shuffle(modes)
    gs = sorted(grounding_pool, key=lambda r: r["sample_id"])
    aux = sorted(common_aux_pool, key=lambda r: r["sample_id"])
    if not gs or (TASKS[experiment] and (not aux)):
        raise ValueError("Empty task pool")
    epochs = budget.get("epochs")
    epoch_numbers = None
    if epochs is not None:
        if not TASKS[experiment]:
            raise ValueError("Exact epoch packing currently requires an auxiliary-task experiment")
        if len({r["sample_id"] for r in gs}) != len(gs):
            raise ValueError("Duplicate grounding sample IDs")
        counts = epoch_budget(len(gs), epochs, budget.get("prompts_per_update", 4))
        if budget["steps"] != counts["steps"]:
            raise ValueError("Epoch/update budget mismatch")
        modes = []
        epoch_numbers = []
        for epoch in range(epochs):
            aux_count = (epoch + 1) * counts["auxiliary_groups"] // epochs - epoch * counts[
                "auxiliary_groups"
            ] // epochs
            block = [False] * len(gs) + [True] * aux_count
            rng.shuffle(block)
            modes.extend(block)
            epoch_numbers.extend([epoch + 1] * len(block))

    def stream(pool, count, seed_offset):
        rr = random.Random(seed + seed_offset)
        out = []
        while len(out) < count:
            cycle = pool[:]
            rr.shuffle(cycle)
            out.extend(cycle)
        return out[:count]

    ground = stream(gs, n, 11)
    auxiliary = stream(aux, sum(modes), 12) if aux else []
    ts = TASKS[experiment]
    mapped = [ts[i % len(ts)] for i in range(sum(modes))] if ts else []
    random.Random(seed + 13).shuffle(mapped)
    out = []
    a = 0
    g = 0
    for i, isaux in enumerate(modes):
        if isaux and ts:
            row = auxiliary[a]
            task = mapped[a]
            a += 1
        else:
            row = ground[g]
            task = "TVG"
            g += 1
        fields = row["semantics"]["supported_fields"]
        if experiment == "A_action_patient":
            fields = [f for f in fields if f != "agent"]
        hidden = random.Random(f"{seed}:{i}:mask").choice(fields) if task == "SCC" else None
        out.append(
            {
                "group_id": f"s{seed}-g{i:08d}",
                "schedule_index": i,
                "sample_id": row["sample_id"],
                "task": task,
                "hidden_field": hidden,
                "seed": seed,
            }
        )
        if epoch_numbers is not None:
            out[-1]["data_epoch"] = epoch_numbers[i]
    return out


def compute_group_advantages(group_ids, rewards, expected_group_size=4, epsilon=0.0001):
    groups = defaultdict(list)
    for i, (g, r) in enumerate(zip(group_ids, rewards)):
        if not math.isfinite(r):
            raise ValueError("Nonfinite reward")
        groups[g].append(i)
    if len(group_ids) != len(rewards):
        raise ValueError("Length mismatch")
    out = [0.0] * len(rewards)
    for indices in groups.values():
        if len(indices) != expected_group_size:
            raise ValueError("Incorrect group size")
        mean = sum((rewards[i] for i in indices)) / len(indices)
        std = math.sqrt(sum(((rewards[i] - mean) ** 2 for i in indices)) / len(indices))
        for i in indices:
            out[i] = (rewards[i] - mean) / (std + epsilon)
    return out


def epoch_checkpoint_steps(schedule, accumulation):
    """Save at the first completed optimizer update after each data epoch."""
    if not schedule or any("data_epoch" not in row for row in schedule):
        raise ValueError("Epoch checkpoints require an epoch-based task schedule")
    boundaries = []
    for index, row in enumerate(schedule):
        last = index + 1 == len(schedule)
        if last or schedule[index + 1]["data_epoch"] != row["data_epoch"]:
            boundaries.append(math.ceil((index + 1) / accumulation))
    return sorted(set(boundaries))
