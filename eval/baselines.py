"""
Baseline detectors for comparison (white paper 7.3).

B1: regex/heuristic filter -- the context-blind keyword baseline. It scans the
    retrieved CONTEXT for injection patterns. By construction it cannot use the
    query intent, so it must fail the judgment-flip pairs.
(B2 zero-shot guard and B3 fine-tuned SLM are the ContextConsistencyDetector in
 src/detector.py, run without/with the LoRA adapter.)
"""
import re
from typing import Dict, List

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


class ProtectAIBaseline:
    """ProtectAI's deberta-v3-base-prompt-injection-v2 -- a public, widely-used
    prompt-injection classifier (binary SAFE/INJECTION). Like the regex filter,
    it is a single-text classifier with no notion of a paired (query, context)
    input, so it is run context-blind on C alone -- the same limitation the
    paper's central claim targets, but a much stronger content-level model than
    a keyword filter."""
    tag = "protectai"

    def __init__(self, model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
                 device: str = "cuda"):
        from transformers import pipeline
        self.clf = pipeline("text-classification", model=model_name,
                             tokenizer=model_name, truncation=True, max_length=512,
                             device=0 if device == "cuda" else -1)

    def detect(self, query: str, context: str) -> Dict:
        out = self.clf(context or "")[0]
        label = "malicious-instruction" if out["label"] == "INJECTION" else "safe"
        return {"label": label, "blocked": label != "safe", "score": out["score"]}

    def detect_batch(self, rows, batch_size: int = 32) -> List[Dict]:
        texts = [r["context"] or "" for r in rows]
        outs = self.clf(texts, batch_size=batch_size)
        results = []
        for out in outs:
            label = "malicious-instruction" if out["label"] == "INJECTION" else "safe"
            results.append({"label": label, "blocked": label != "safe", "score": out["score"]})
        return results


class PromptGuardBaseline:
    """Meta's Prompt-Guard-86M -- a small, public (gated) 3-class classifier
    (BENIGN/INJECTION/JAILBREAK) purpose-built for prompt-injection detection.
    Like B4/B5 it takes a single text input, so it is run context-blind on $C$
    alone; INJECTION and JAILBREAK both map to malicious-instruction."""
    tag = "promptguard"

    def __init__(self, model_name: str = "meta-llama/Prompt-Guard-86M", device: str = "cuda"):
        from transformers import pipeline
        self.clf = pipeline("text-classification", model=model_name,
                             tokenizer=model_name, truncation=True, max_length=512,
                             device=0 if device == "cuda" else -1)

    def _map(self, out) -> str:
        return "safe" if out["label"] == "BENIGN" else "malicious-instruction"

    def detect(self, query: str, context: str) -> Dict:
        out = self.clf(context or "")[0]
        label = self._map(out)
        return {"label": label, "blocked": label != "safe", "score": out["score"]}

    def detect_batch(self, rows, batch_size: int = 32) -> List[Dict]:
        texts = [r["context"] or "" for r in rows]
        outs = self.clf(texts, batch_size=batch_size)
        results = []
        for out in outs:
            label = self._map(out)
            results.append({"label": label, "blocked": label != "safe", "score": out["score"]})
        return results


class LlamaGuardBaseline:
    """Llama Guard 3 1B~\\cite{llamaguard} -- Meta's general-purpose LLM
    input/output safeguard, gated on Hugging Face. Like B4, it takes a single
    text turn with no notion of a paired (query, context) input (this is the
    "Reasons over (Q,C)? No" row in the paper's positioning Table 1), so it is
    run context-blind on C alone, wrapped as a 'user' turn via its own chat
    template and safety taxonomy."""
    tag = "llamaguard"

    def __init__(self, model_name: str = "meta-llama/Llama-Guard-3-1B", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()

    def _label(self, context: str) -> str:
        import torch
        chat = [{"role": "user", "content": context or ""}]
        enc = self.tok.apply_chat_template(chat, return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=20,
                                       pad_token_id=self.tok.eos_token_id)
        input_len = enc["input_ids"].shape[-1]
        gen = self.tok.decode(out[0][input_len:], skip_special_tokens=True).strip().lower()
        return "malicious-instruction" if gen.startswith("unsafe") else "safe"

    def detect(self, query: str, context: str) -> Dict:
        label = self._label(context)
        return {"label": label, "blocked": label != "safe"}

    def detect_batch(self, rows, batch_size: int = 1) -> List[Dict]:
        # Llama Guard's chat template + generate() is run one at a time (no
        # batched-generation path needed here; the benchmark is small).
        return [self.detect(r.get("query", ""), r["context"]) for r in rows]
