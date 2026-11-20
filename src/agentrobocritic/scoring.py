"""Strict output parsing and the benchmark metrics."""
import json
from collections import Counter

from .data import LABEL_KEYS

MODES = tuple(f"M{i}" for i in range(1, 8))
SUBTYPES = tuple(f"S{i}" for i in range(1, 15))
BINARY = ("achieved", "not_achieved")
OUTCOMES = BINARY + ("partial_or_unconfirmed", "unverifiable", "not_applicable")


def unique_keys(items):
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate JSON key")
    return result


def parse(text, *, binary):
    try:
        value = json.loads(text, object_pairs_hook=unique_keys)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(value, dict) or set(value) != {*LABEL_KEYS, "failure_reason"}:
        return None, "wrong_keys"
    outcome, mode, subtype = (value[k] for k in LABEL_KEYS)
    reason = value["failure_reason"]
    if outcome not in (BINARY if binary else OUTCOMES):
        return None, "invalid_outcome"
    if mode is not None and mode not in MODES:
        return None, "invalid_mechanism"
    if subtype is not None and subtype not in SUBTYPES:
        return None, "invalid_subtype"
    if outcome == "not_achieved":
        if not isinstance(reason, str) or not reason.strip() or (mode == "M1") != (subtype is not None):
            return None, "inconsistent_failure"
    elif (mode, subtype, reason) != (None, None, None):
        return None, "failure_fields_on_nonfailure"
    return value, None


def validate_label(label):
    if not isinstance(label, dict) or set(label) != set(LABEL_KEYS):
        raise ValueError("Scoring needs a three-field human label for each action")
    outcome, mode, subtype = (label[k] for k in LABEL_KEYS)
    if outcome not in BINARY or (outcome == "not_achieved" and mode not in MODES
            or outcome == "achieved" and mode is not None
            or (subtype not in SUBTYPES if mode == "M1" else subtype is not None)):
        raise ValueError("Invalid human label")


def agreement(pairs):
    n = len(pairs)
    correct = sum(a == b for a, b in pairs)
    gold = Counter(a for a, b in pairs)
    pred = Counter(b for a, b in pairs)
    chance = sum(count * pred[key] for key, count in gold.items())
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None,
            "kappa": (n * correct - chance) / (n * n - chance) if n * n != chance else None}


def score(references, predictions):
    indexed = {}
    for p in predictions:
        if p["id"] in indexed:
            raise ValueError(f"Repeated prediction ID {p['id']}")
        indexed[p["id"]] = p
    outcomes, modes = [], []
    joint = invalid = 0
    for row in references:
        sid, label = row["id"], row["human_label"]
        validate_label(label)
        if sid not in indexed:
            raise ValueError(f"Missing prediction for {sid}")
        record = indexed.pop(sid)
        prediction = record["prediction"]
        valid = record["error"] is None and prediction is not None
        if valid:
            # A prediction that passed the four-field output parser may be stored
            # without its explanation, which is not scored.
            if not isinstance(prediction, dict) or not set(LABEL_KEYS) <= prediction.keys():
                raise ValueError(f"Missing normalized prediction fields for {sid}")
            oc, mechanism, sub = (prediction[k] for k in LABEL_KEYS)
            if (oc not in OUTCOMES or mechanism not in (None, *MODES) or sub not in (None, *SUBTYPES)
                    or oc != "not_achieved" and mechanism is not None
                    or (mechanism == "M1") != (sub is not None)):
                raise ValueError(f"Malformed normalized prediction for {sid}")
        invalid += not valid
        outcome = prediction["intent_success"] if valid else "__invalid__"
        mode = prediction["failure_mechanism"] if valid else None
        outcomes.append((label["intent_success"], outcome))
        if label["intent_success"] == "not_achieved":
            modes.append((label["failure_mechanism"], mode if outcome == "not_achieved" else None))
        joint += valid and outcome == label["intent_success"] and mode == label["failure_mechanism"]
    if indexed or not outcomes:
        raise ValueError("Predictions and references must cover the same nonempty cohort")
    return {"outcome": agreement(outcomes), "failure_mode": agreement(modes),
            "overall": {"n": len(outcomes), "correct": joint, "accuracy": joint / len(outcomes)},
            "invalid_predictions": invalid}
