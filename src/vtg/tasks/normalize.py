"""Normalize for temporal video grounding."""

import re

ALIASES = {
    "cellphone": "cell_phone",
    "mobile phone": "cell_phone",
    "cell phone": "cell_phone",
    "television": "tv",
}
COLORS = {
    "red",
    "blue",
    "green",
    "black",
    "white",
    "yellow",
    "orange",
    "brown",
    "pink",
    "purple",
    "gray",
    "grey",
}


def ambiguous(text):
    return not isinstance(text, str) or bool(
        re.search("\\b(or|and|unknown|none|maybe|possibly)\\b|[/;|]", text, re.I)
    )


def noun_value(text, nlp):
    if ambiguous(text):
        return None
    doc = nlp(text.lower().replace("_", " "))
    nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]
    if not nouns:
        return None
    head = nouns[-1]
    components = [
        t.lemma_.lower()
        for t in head.lefts
        if t.dep_ == "compound" and t.pos_ in ("NOUN", "PROPN") and (t.lower_ not in COLORS)
    ] + [head.lemma_.lower()]
    raw = " ".join(components)
    return ALIASES.get(raw, raw.replace(" ", "_"))


def predicate_value(text, nlp):
    if ambiguous(text):
        return None
    normalized = text.lower().replace("_", " ")
    if re.search("\\b(not|never|no)\\b|n't", normalized):
        return None
    doc = nlp("They " + normalized + ".")
    verbs = [t for t in doc if t.pos_ == "VERB"]
    if len(verbs) != 1:
        return None
    verb = verbs[0]
    particles = [t.lower_ for t in verb.children if t.dep_ == "prt"]
    return "_".join([verb.lemma_.lower()] + particles)


def normalize_value(field, text, nlp):
    return predicate_value(text, nlp) if field == "predicate" else noun_value(text, nlp)
