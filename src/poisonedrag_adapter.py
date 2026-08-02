"""
Adapter: PoisonedRAG (Zou et al., USENIX Security 2025, arXiv:2402.07867) targeted
knowledge-poisoning attacks -> our (query, context, label) schema, for an
external-benchmark generalization check (paper Limitations, "single-domain
validation" / "external-benchmark generalization" future-work item).

Uses PoisonedRAG's precomputed `adv_targeted_results/{nq,hotpotqa,msmarco}.json`
(no BEIR corpus download needed): each entry is a real user question plus up to
5 LLM-generated adversarial passages engineered to make a RAG system retrieve
the passage and answer with the attacker's target (incorrect) answer. These
passages are fluent, factually-fabricated misinformation with no imperative
"ignore previous instructions" language -- a different attack surface than the
IPI/context-hijack/pii-exfil families our training data covers.

We take the FIRST adv_text per question (one per question, matching the
original 300-pair evaluation) across all three source datasets: 100 questions
x 3 datasets = 300 (query, poisoned-context) pairs, all ground-truth
malicious (there is no benign/negative split in this benchmark -- it is a
targeted-attack-only dataset), so this yields a recall-only check, not
FPR/accuracy.
"""
import json
import os
from typing import Dict, List

DATASETS = ("nq", "hotpotqa", "msmarco")


def load_poisonedrag(data_dir: str = "data/external/poisonedrag") -> List[Dict]:
    rows: List[Dict] = []
    for ds in DATASETS:
        path = os.path.join(data_dir, f"{ds}.json")
        with open(path) as f:
            entries = json.load(f)
        for qid, entry in entries.items():
            adv_texts = entry.get("adv_texts") or []
            if not adv_texts:
                continue
            rows.append({
                "id": f"poisonedrag-{ds}-{qid}",
                "query": entry["question"],
                "context": adv_texts[0],
                "label": "malicious-instruction",  # ground truth: targeted poisoning
                "attack_type": "poisoning",
                "reasoning": "",
                "flip_pair_id": None,
                "split": "test",
                "source": f"poisonedrag-{ds}",
            })
    return rows


if __name__ == "__main__":
    rows = load_poisonedrag()
    print(f"loaded {len(rows)} PoisonedRAG (query, poisoned-context) pairs")
    print(json.dumps(rows[0], indent=2))
