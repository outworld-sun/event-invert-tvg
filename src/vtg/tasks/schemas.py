"""Schemas for temporal video grounding."""

from dataclasses import dataclass, field, asdict

SCHEMA_VERSION = "event_core_v1"
FIELDS = ("agent", "predicate", "patient")


@dataclass
class ReferenceRecord:
    fields: dict = field(default_factory=dict)
    event_count: int = 0
    failure_code: str = ""
    parser_version: str = "core_rules_v1"
    alias_version: str = "aliases_v1"
    schema_version: str = SCHEMA_VERSION

    @property
    def supported_fields(self):
        return [k for k in FIELDS if k in self.fields]

    @property
    def eligible(self):
        return self.event_count == 1 and "predicate" in self.fields and (not self.failure_code)

    def to_dict(self):
        return {
            **asdict(self),
            "supported_fields": self.supported_fields,
            "eligible_tasks": ["SCC", "SER", "ED"] if self.eligible else [],
        }


@dataclass
class TaskInstance:
    instance_id: str
    group_id: str
    task: str
    sample_id: str
    prompt_text: str
    video_request: dict
    target: dict
    supported_fields: list
    hidden_fields: list
    template_version: str = "core_prompts_v1"
    sampling_seed: int = 3407
    reward_spec_version: str = "core_rewards_v1"

    def to_dict(self):
        return asdict(self)


@dataclass
class RewardBreakdown:
    r_format: float = 0.0
    r_semantic: float = 0.0
    total: float = 0.0
    validity: bool = False
    field_scores: dict = field(default_factory=dict)
    joint_score: float = 0.0
    failure_code: str = ""
    candidate: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
