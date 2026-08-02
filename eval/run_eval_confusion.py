"""
Confusion matrix + per-class breakdown + Wilson confidence intervals for B3 on
the held-out agreed test set (reviewer request: per-class numbers for the
small `suspicious` class, and CIs on the headline metrics).

Runs inside the vllm container: bash scripts/eval_confusion.sh
"""
import os
import sys
import json
import math
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.run_eval import load_split
from src.schema import LABELS


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (round(max(0.0, lo), 4), round(min(1.0, hi), 4))


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
    from src.detector import ContextConsistencyDetector
    det = ContextConsistencyDetector(base_model=args.base_model, adapter_path=args.adapter)
    res = det.detect_batch(rows, batch_size=args.batch_size)
    preds = [r["label"] for r in res]

    # 3x3 confusion matrix: rows = ground truth, cols = prediction
    cm = {gt: {p: 0 for p in LABELS} for gt in LABELS}
    for r, p in zip(rows, preds):
        cm[r["label"]][p] += 1

    # Per-class precision/recall/F1 + Wilson CI on recall
    per_class = {}
    for lbl in LABELS:
        gt_n = sum(cm[lbl].values())
        tp = cm[lbl][lbl]
        pred_n = sum(cm[g][lbl] for g in LABELS)
        recall = tp / gt_n if gt_n else None
        precision = tp / pred_n if pred_n else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) > 0 else None)
        ci = wilson_ci(tp, gt_n) if gt_n else (None, None)
        per_class[lbl] = {
            "support": gt_n, "tp": tp,
            "recall": round(recall, 4) if recall is not None else None,
            "recall_ci95": ci,
            "precision": round(precision, 4) if precision is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    # Headline security metrics with Wilson CI (malicious-instruction recall, safe FPR)
    n_attack = sum(cm["malicious-instruction"].values())
    # "blocked" = not predicted safe, matches security_metrics() in run_eval.py
    tp_blocked = n_attack - cm["malicious-instruction"]["safe"]
    n_benign = sum(cm["safe"].values())
    fp_blocked = n_benign - cm["safe"]["safe"]

    headline = {
        "recall": round(tp_blocked / n_attack, 4) if n_attack else None,
        "recall_ci95": wilson_ci(tp_blocked, n_attack),
        "fpr": round(fp_blocked / n_benign, 4) if n_benign else None,
        "fpr_ci95": wilson_ci(fp_blocked, n_benign),
    }

    report = {
        "n": len(rows), "confusion_matrix": cm, "per_class": per_class,
        "headline_with_ci": headline,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/eval_report_confusion.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Confusion matrix (rows=ground truth, cols=predicted):")
    print(f"{'':>24}" + "".join(f"{l:>24}" for l in LABELS))
    for gt in LABELS:
        print(f"{gt:>24}" + "".join(f"{cm[gt][p]:>24}" for p in LABELS))
    print("\nPer-class:")
    for lbl, m in per_class.items():
        print(f"  {lbl:>22}: support={m['support']:>4} recall={m['recall']} "
              f"(95% CI {m['recall_ci95']}) precision={m['precision']} f1={m['f1']}")
    print(f"\nHeadline recall={headline['recall']} (CI {headline['recall_ci95']}), "
          f"FPR={headline['fpr']} (CI {headline['fpr_ci95']})")
    print("\nreport -> results/eval_report_confusion.json")


if __name__ == "__main__":
    main()
