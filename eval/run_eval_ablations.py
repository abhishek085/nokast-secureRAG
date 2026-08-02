"""
Input ablations (reviewer request): does the tuned detector's performance come
from genuine query-context consistency reasoning, or from superficial patterns
in C alone? Compares B3 (tuned SLM) under four input conditions on the same
659-row held-out agreed test set used for Table 3:

  full        - (Q, C) as trained (reference row, from results/eval_report.json)
  q-only      - context masked (""), query kept: isolates what Q alone predicts
  c-only      - query masked (""), context kept: isolates what C alone predicts
  mismatched  - context replaced by another row's context (fixed-seed derangement,
                no row keeps its own context); ground truth = the DONOR row's
                original label, since for non-flip records the label is driven
                by C's content. Tests whether the model's verdict follows the
                context's own nature regardless of which query it is paired
                with (a scaled-up, whole-test-set complement to the 120
                hand-authored flip pairs in Table 4).

Runs inside the vllm container: bash scripts/eval_ablations.sh
"""
import os
import sys
import json
import random
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.run_eval import load_split, security_metrics
from eval.baselines import RegexBaseline

SEED = 20260624


def derangement(n: int, seed: int) -> List[int]:
    """Random permutation of range(n) with no fixed points."""
    rng = random.Random(seed)
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def make_variant(rows: List[Dict], mode: str) -> List[Dict]:
    if mode == "q-only":
        return [{**r, "context": ""} for r in rows]
    if mode == "c-only":
        return [{**r, "query": ""} for r in rows]
    if mode == "mismatched":
        perm = derangement(len(rows), SEED)
        out = []
        for i, r in enumerate(rows):
            donor = rows[perm[i]]
            out.append({**r, "context": donor["context"], "label": donor["label"]})
        return out
    raise ValueError(mode)


def run_regex_variant(rows):
    b = RegexBaseline()
    preds = [b.detect(r["query"], r["context"])["label"] for r in rows]
    return security_metrics(rows, preds)


def run_model_variant(rows, adapter_path, base_model, batch_size):
    from src.detector import ContextConsistencyDetector
    det = ContextConsistencyDetector(base_model=base_model, adapter_path=adapter_path)
    res = det.detect_batch(rows, batch_size=batch_size)
    preds = [r["label"] for r in res]
    out = security_metrics(rows, preds)
    import torch, gc
    del det
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--base-model", default=os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B"))
    ap.add_argument("--adapter", default="models/adapters")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    rows = load_split(args.data, args.split)
    print(f"ablation eval on {len(rows)} rows from {args.data}:{args.split}")

    report = {"n": len(rows), "variants": {}}

    print("\n[regex] full (sanity: regex ignores Q, so full==c-only)")
    report["variants"]["regex-full"] = run_regex_variant(rows)

    for mode in ("q-only", "c-only", "mismatched"):
        variant_rows = make_variant(rows, mode)
        print(f"\n[B3 tuned] variant={mode} ...")
        m = run_model_variant(variant_rows, args.adapter, args.base_model, args.batch_size)
        report["variants"][f"B3-tuned-{mode}"] = m
        print(f"  recall={m['detection_recall']} fpr={m['fpr']} acc3={m['accuracy_3class']}")

    os.makedirs("results", exist_ok=True)
    with open("results/eval_report_ablations.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nreport -> results/eval_report_ablations.json")


if __name__ == "__main__":
    main()
