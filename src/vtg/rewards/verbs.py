"""Matched VC/AR/VD lexical-verb scoring, separate from core role rewards."""


def verbs(text, parser):
    return {
        t.lemma_.lower()
        for t in parser.nlp(text.replace("_", " "))
        if t.pos_ == "VERB" and t.dep_ not in ("aux", "auxpass")
    }


def score_verb(task, body, target, parser):
    expected = {target["predicate"]["value"].split("_")[0]}
    predicted = verbs(body, parser)
    if task in ("VC", "AR"):
        predicted = verbs("They " + body.replace("_", " ") + ".", parser)
    if task == "VC":
        return float(predicted == expected)
    if task == "AR":
        return float(len(predicted) == 1 and predicted <= expected)
    if task == "VD":
        return len(expected & predicted) / len(expected)
    raise ValueError(task)
