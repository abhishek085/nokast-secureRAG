"""
Context-consistency detector for Nokast-secureRAG (inference).

The semantic firewall: given a user query Q and retrieved context C, classify the
pair as safe / suspicious / malicious-instruction. Wraps the base Qwen2.5-0.5B
(zero-shot baseline) or the LoRA-fine-tuned student (with adapter), using the same
reasoning-first prompt the model was trained on.

Runs inside the vllm container (CUDA torch). For the regex baseline see
eval/baselines.py.
"""
import re
import time
from typing import Dict, List, Optional

from src.train import SYSTEM, build_text  # reuse the exact training prompt

LABELS = ("safe", "suspicious", "malicious-instruction")


def parse_label(text: str) -> str:
    """Extract a label from the model's generated text (reasoning-first)."""
    m = re.search(r"Label:\s*(malicious-instruction|suspicious|safe)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    low = text.lower()
    if "malicious" in low:
        return "malicious-instruction"
    if "suspicious" in low:
        return "suspicious"
    return "safe"


class ContextConsistencyDetector:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-0.5B",
                 adapter_path: Optional[str] = None, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"  # for batched generation
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # fold adapter for faster inference
        model.eval()
        self.model = model
        self.tag = "tuned" if adapter_path else "base-zeroshot"

    def _prompt(self, query: str, context: str) -> str:
        return build_text({"query": query, "context": context,
                           "label": "", "reasoning": ""})["prompt"]

    def detect(self, query: str, context: str, max_new_tokens: int = 96) -> Dict:
        import torch
        prompt = self._prompt(query, context)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        latency_ms = (time.time() - t0) * 1000
        gen = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        label = parse_label(gen)
        return {"label": label, "reasoning": gen.strip(), "latency_ms": latency_ms,
                "blocked": label != "safe"}

    def detect_batch(self, pairs: List[Dict], batch_size: int = 32,
                     max_new_tokens: int = 96) -> List[Dict]:
        import torch
        results: List[Dict] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            prompts = [self._prompt(p["query"], p["context"]) for p in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            t0 = time.time()
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                          do_sample=False, pad_token_id=self.tok.pad_token_id)
            dt_ms = (time.time() - t0) * 1000
            per_call = dt_ms / len(chunk)
            for j, p in enumerate(chunk):
                gen = self.tok.decode(out[j][enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
                label = parse_label(gen)
                results.append({"label": label, "reasoning": gen.strip(),
                                "latency_ms": per_call, "blocked": label != "safe"})
        return results
