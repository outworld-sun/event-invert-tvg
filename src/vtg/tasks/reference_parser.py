"""Reference parser for temporal video grounding."""

import copy
from vtg.tasks.schemas import ReferenceRecord
from vtg.tasks.normalize import noun_value, ALIASES
from vtg.utils.io import digest


class CoreParser:

    def __init__(self, model, parser_version="core_rules_v1", alias_version="aliases_v1"):
        import spacy

        self.nlp = spacy.load(model)
        self.parser_version = parser_version
        self.alias_version = alias_version
        self.cache = {}

    def parse(self, text, reference=True):
        key = digest(
            [text, self.parser_version, self.alias_version, "event_core_v1", ALIASES, reference]
        )
        if key not in self.cache:
            self.cache[key] = self._parse(text, reference).to_dict()
        return copy.deepcopy(self.cache[key])

    def _parse(self, text, reference):
        out = ReferenceRecord(parser_version=self.parser_version, alias_version=self.alias_version)
        doc = self.nlp(text)
        verbs = [t for t in doc if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass")]
        out.event_count = len(verbs)
        if len(verbs) != 1:
            out.failure_code = "not_single_event"
            return out
        verb = verbs[0]
        if not reference and (
            not any((t.dep_ in ("nsubj", "nsubjpass") for t in doc))
            or (
                sum((t.text == "," for t in doc)) >= 2
                and (not any((t.pos_ in ("DET", "AUX") for t in doc)))
            )
        ):
            out.failure_code = "not_event_sentence"
            return out
        if verb.dep_ not in ("ROOT", "root"):
            out.failure_code = "nested_or_uncertain"
            return out
        if any((t.lower_ in ("maybe", "perhaps", "possibly") for t in doc)) or (
            reference and any((t.lower_ == "or" or t.dep_ == "conj" for t in doc))
        ):
            out.failure_code = "enumeration_or_uncertain"
            return out
        if reference and any(
            (t.pos_ == "NUM" or t.lower_ in ("several", "many", "both") for t in doc)
        ):
            out.failure_code = "quantity_or_set"
            return out
        negative = any((t.dep_ == "neg" or t.lower_ in ("not", "never") for t in doc))
        if negative and reference:
            out.failure_code = "negative_reference"
            return out
        parts = [verb] + [t for t in verb.children if t.dep_ == "prt"]
        out.fields["predicate"] = {
            "value": "_".join([verb.lemma_.lower()] + [t.lower_ for t in parts[1:]]),
            "spans": [[t.idx, t.idx + len(t)] for t in parts],
            "polarity": "negative" if negative else "positive",
        }
        passive = any((t.dep_ in ("nsubjpass", "auxpass") for t in verb.children))
        for token in verb.children:
            role = None
            if token.dep_ == "nsubj" and (not passive):
                role = "agent"
            elif token.dep_ in ("dobj", "obj", "nsubjpass"):
                role = "patient"
            if role:
                self._noun(out, role, token)
            if passive and token.lower_ == "by" and (token.dep_ in ("agent", "prep")):
                for obj in token.children:
                    if obj.dep_ == "pobj":
                        self._noun(out, "agent", obj)
        if reference:
            for role in ("agent", "patient"):
                if role not in out.fields:
                    continue
                spans = {tuple(x) for x in out.fields[role]["spans"]}
                for token in doc:
                    if token.pos_ not in ("NOUN", "PROPN"):
                        continue
                    words = [t for t in token.lefts if t.dep_ == "compound"] + [token]
                    if (
                        noun_value(" ".join((t.text for t in words)), self.nlp)
                        == out.fields[role]["value"]
                    ):
                        spans.update(((t.idx, t.idx + len(t)) for t in words))
                out.fields[role]["spans"] = [list(x) for x in sorted(spans)]
        if not any((t.dep_ in ("nsubj", "nsubjpass") for t in doc)) and verb.tag_ not in (
            "VBG",
            "VB",
        ):
            out.failure_code = "no_event_clause"
        return out

    def _noun(self, out, role, token):
        if token.pos_ not in ("NOUN", "PROPN") or any(
            (t.dep_ == "conj" or t.lower_ == "or" for t in token.subtree)
        ):
            return
        words = [t for t in token.lefts if t.dep_ == "compound"] + [token]
        val = noun_value(" ".join((t.text for t in words)), self.nlp)
        if not val:
            return
        out.fields[role] = {"value": val, "spans": [[t.idx, t.idx + len(t)] for t in words]}


def parse_reference(query, parser_config):
    return parser_config.parse(query, reference=True)
