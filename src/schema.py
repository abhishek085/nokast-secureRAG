"""
Record schema for Nokast-secureRAG synthetic data.

Single source of truth for the (Query Q, Context C) -> label contract that
data generation, the judge, training, and evaluation all read/write.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# Three labels per white paper section 4.1.
LABELS = ("safe", "suspicious", "malicious-instruction")

# Attack taxonomy per section 4.4 + Phase 2 plan.
ATTACK_TYPES = ("benign", "borderline", "ipi", "poisoning", "context-hijack", "pii-exfil")

# Which label each attack_type is allowed to carry. The teacher proposes the
# label; we reject combinations that are definitionally impossible.
ALLOWED_LABELS: Dict[str, tuple] = {
    "benign": ("safe",),
    "borderline": ("suspicious", "safe"),
    "ipi": ("malicious-instruction", "suspicious"),
    "poisoning": ("malicious-instruction", "suspicious"),
    "context-hijack": ("malicious-instruction", "suspicious"),
    "pii-exfil": ("malicious-instruction", "suspicious"),
}

SPLITS = ("train", "val", "test")

# JSON Schema handed to vLLM `guided_json` so every generation is structurally
# valid. Only the model-authored fields are constrained here; id/flip_pair_id/
# split are assigned by the pipeline afterwards.
GENERATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 5},
        "context": {"type": "string", "minLength": 10},
        "label": {"type": "string", "enum": list(LABELS)},
        "reasoning": {"type": "string", "minLength": 10},
    },
    "required": ["query", "context", "label", "reasoning"],
    "additionalProperties": False,
}


@dataclass
class Record:
    query: str
    context: str
    label: str
    attack_type: str
    reasoning: str
    id: str = ""
    flip_pair_id: Optional[int] = None
    split: Optional[str] = None
    source: str = "teacher"  # teacher | judge | seed | human

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SchemaError(ValueError):
    pass


def validate(rec: Dict[str, Any], require_split: bool = False) -> List[str]:
    """Return a list of problems with `rec`; empty list means valid."""
    errs: List[str] = []

    for f in ("query", "context", "label", "attack_type", "reasoning"):
        if not rec.get(f) or not str(rec[f]).strip():
            errs.append(f"missing/empty field: {f}")

    label = rec.get("label")
    if label and label not in LABELS:
        errs.append(f"bad label: {label!r} not in {LABELS}")

    atype = rec.get("attack_type")
    if atype and atype not in ATTACK_TYPES:
        errs.append(f"bad attack_type: {atype!r} not in {ATTACK_TYPES}")

    # Cross-field consistency: a benign carrier can never be malicious, etc.
    if label in LABELS and atype in ATTACK_TYPES:
        allowed = ALLOWED_LABELS.get(atype, LABELS)
        if label not in allowed:
            errs.append(f"label {label!r} not allowed for attack_type {atype!r} (allowed: {allowed})")

    split = rec.get("split")
    if require_split and split not in SPLITS:
        errs.append(f"bad/missing split: {split!r}")
    elif split is not None and split not in SPLITS:
        errs.append(f"bad split: {split!r}")

    fpid = rec.get("flip_pair_id")
    if fpid is not None and not isinstance(fpid, int):
        errs.append(f"flip_pair_id must be int or null, got {type(fpid).__name__}")

    return errs


def is_valid(rec: Dict[str, Any], require_split: bool = False) -> bool:
    return len(validate(rec, require_split=require_split)) == 0
