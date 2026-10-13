"""Builder for temporal video grounding."""

import copy
from vtg.tasks.schemas import TaskInstance
from vtg.utils.io import digest

TVG_PROMPT = 'To accurately pinpoint the event "{query}" in the video, determine the precise time period of the event.\nOutput your thought process within <think> </think> tags. Then provide start and end times in seconds as <answer>start time to end time</answer>. Video duration: {duration:.2f} seconds.'
ED_PROMPT = "Describe one main event in the video in one sentence.\nState who does what and the involved object when visible.\nReturn the sentence in <answer>...</answer>."


def mask_field(query, spans, field):
    chars = list(query)
    for a, b in sorted(spans, reverse=True):
        if not 0 <= a < b <= len(query):
            raise ValueError("Invalid evidence span")
        chars[a:b] = list("[" + field + "]")
    import re

    return re.sub("(\\[" + field + "\\])(?:\\s*\\[" + field + "\\])+", "\\1", "".join(chars))


def build_task_instance(record, task, hidden_field=None, config=None):
    config = config or {}
    sem = record["semantics"]
    fields = sem["supported_fields"]
    target = copy.deepcopy(sem["fields"])
    if task != "TVG" and (not sem["eligible_tasks"]):
        raise ValueError("Ineligible auxiliary reference")
    hidden = []
    if task == "TVG":
        prompt = TVG_PROMPT.format(query=record["query"], duration=record["duration_sec"])
        target = {"gt_sec": record["gt_sec"], "duration_sec": record["duration_sec"]}
    elif task in ("SCC", "VC"):
        hidden_field = "predicate" if task == "VC" else hidden_field
        if hidden_field not in fields:
            raise ValueError("Hidden field has no evidence")
        hidden = [hidden_field]
        spans = (
            sem["fields"][hidden_field]["spans"][:1]
            if task == "VC"
            else sem["fields"][hidden_field]["spans"]
        )
        masked = mask_field(record["query"], spans, hidden_field)
        prompt = f'Use the video to fill the [{hidden_field}] slot in: "{masked}".\nReturn one answer in <answer>...</answer>.'
    elif task == "SER":
        prompt = (
            "Identify one main event in the video.\nReturn these fields: "
            + ", ".join(fields)
            + ".\nUse one JSON object inside <answer>...</answer>.\nGive one definite value per field, not a list of alternatives."
        )
    elif task in ("ED", "VD"):
        prompt = ED_PROMPT
    elif task == "AR":
        prompt = "Identify one action verb in the video. Return one verb in <answer>...</answer>."
    else:
        raise ValueError(task)
    offset = record.get("media_offset", 0.0)
    local_interval = [0.0, record["duration_sec"]] if task == "TVG" else record["gt_sec"]
    video = {
        "video_relpath": record["video_relpath"],
        "video_id": record["video_id"],
        "interval": [v + offset for v in local_interval],
        "sampling": copy.deepcopy(config.get("video", {})),
    }
    group = config.get("group_id", digest([record["sample_id"], task, hidden]))
    return TaskInstance(
        group,
        group,
        task,
        record["sample_id"],
        prompt,
        video,
        target,
        fields if task != "TVG" else [],
        hidden,
        sampling_seed=config.get("seed", 3407),
    )


def serialize_model_input(instance):
    return {"prompt_text": instance.prompt_text, "modalities": ["video"]}


def serialize_reward_metadata(instance):
    return {
        "task": instance.task,
        "target": copy.deepcopy(instance.target),
        "supported_fields": instance.supported_fields[:],
        "hidden_fields": instance.hidden_fields[:],
    }
