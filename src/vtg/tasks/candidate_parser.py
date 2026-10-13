"""Candidate parser for temporal video grounding."""

import json
import re
from vtg.tasks.normalize import normalize_value


def answer_body(text):
    match = re.fullmatch("\\s*<answer>(.*?)</answer>\\s*", text, re.S)
    if not match or "<answer>" in match[1] or "</answer>" in match[1]:
        return None
    return match[1].strip()


def parse_description(answer_text, parser_config):
    return parser_config.parse(answer_text, reference=False)


def parse_candidate(task, body, parser):
    if task == "ED":
        rec = parse_description(body, parser)
        if rec["failure_code"] or rec["event_count"] != 1:
            return ({}, rec["failure_code"] or "not_single_event")
        fields = {k: v["value"] for k, v in rec["fields"].items()}
        if rec["fields"]["predicate"].get("polarity") == "negative":
            fields["predicate"] = None
        return (fields, "")
    if task == "SER":

        def unique_pairs(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise ValueError("duplicate_key")
                d[k] = v
            return d

        try:
            obj = json.loads(body, object_pairs_hook=unique_pairs)
        except (ValueError, TypeError):
            return ({}, "invalid_json")
        if not isinstance(obj, dict):
            return ({}, "invalid_json_type")
        return (
            {
                k: normalize_value(k, v, parser.nlp)
                for k, v in obj.items()
                if k in ("agent", "predicate", "patient")
            },
            "",
        )
    raise ValueError(task)
