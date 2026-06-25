"""
Baseline detectors for comparison (white paper 7.3).

B1: regex/heuristic filter -- the context-blind keyword baseline. It scans the
    retrieved CONTEXT for injection patterns. By construction it cannot use the
    query intent, so it must fail the judgment-flip pairs.
(B2 zero-shot guard and B3 fine-tuned SLM are the ContextConsistencyDetector in
 src/detector.py, run without/with the LoRA adapter.)
"""
import re
from typing import Dict

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+|the\s+)?(previous|above|prior)\s+(instructions?|content)",
    r"(?i)disregard\s+(the\s+)?(previous|above|all|context|user)",
    r"(?i)forget\s+(everything|what|previous|the earlier)",
    r"(?i)system\s*(override|:|prompt)",
    r"(?i)new\s+(directive|primary\s+directive|instruction|role)",
    r"(?i)you\s+are\s+(now\s+)?(dev\s*mode|unrestricted|no\s+longer\s+bound)",
    r"(?i)admin[\s_]*mode",
    r"(?i)reveal\s+(the\s+)?(system\s+)?(prompt|hidden\s+settings)",
    r"(?i)(append|include|send|email|exfiltrate)\s+.*(email|address|ssn|social\s+security|credit\s+card|credentials|api\s+key)",
    r"(?i)(ignore|delete)\s+all\s+previous\s+output",
    r"(?i)factory\s+(reset|defaults).*clear\s+all",
    r"(?i)<!--.*(ignore|system|instruction|override).*-->",
]


class RegexBaseline:
    tag = "regex"

    def __init__(self):
        self.patterns = [re.compile(p) for p in INJECTION_PATTERNS]

    def detect(self, query: str, context: str) -> Dict:
        # Context-blind: only looks at the retrieved context, ignores query intent.
        for pat in self.patterns:
            if pat.search(context or ""):
                return {"label": "malicious-instruction", "blocked": True,
                        "matched": pat.pattern}
        return {"label": "safe", "blocked": False, "matched": None}
