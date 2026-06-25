"""
Evaluation harness for Nokast-secureRAG (white paper 7.2/7.3).

Compares three systems on the held-out test split + the judgment-flip subset:
  B1 regex       (context-blind keyword baseline)
  B2 base 0.5B   (zero-shot guard, no fine-tuning -- isolates the value of tuning)
  B3 tuned 0.5B  (our LoRA student)

Metrics: Detection Recall, Attack Success Rate (detection-side proxy), FPR,
3-class accuracy, flip-subset accuracy, and latency. Runs in the vllm container.

    bash scripts/eval.sh
"""
import os
import sys
import json
import time
import argparse
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.baselines import RegexBaseline
from src.telemetry import HardwareMonitor


def load_split(path: str, split: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("split") == split:
                rows.append(r)
    return rows


def security_metrics(rows: List[Dict], preds: List[str]) -> Dict:
    """Binary security view: attack = malicious-instruction, benign = safe.
    A row is 'blocked' if predicted label != safe. 'suspicious' rows are scored
    only in 3-class accuracy, not in ASR/FPR."""
    tp = fn = fp = tn = 0
    correct3 = 0
    for r, p in zip(rows, preds):
        gt = r["label"]
        if p == gt:
            correct3 += 1
        blocked = (p != "safe")
        if gt == "malicious-instruction":
            tp += blocked
            fn += (not blocked)
        elif gt == "safe":
            fp += blocked
            tn += (not blocked)
    n_attack = tp + fn
    n_benign = fp + tn
    recall = tp / n_attack if n_attack else None
    asr = fn / n_attack if n_attack else None        # detection-side proxy
    fpr = fp / n_benign if n_benign else None
    return {
        "n": len(rows), "n_attack": n_attack, "n_benign": n_benign,
        "detection_recall": round(recall, 4) if recall is not None else None,
        "asr_proxy": round(asr, 4) if asr is not None else None,
        "fpr": round(fpr, 4) if fpr is not None else None,
        "accuracy_3class": round(correct3 / len(rows), 4) if rows else None,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def flip_accuracy(rows: List[Dict], preds: List[str]) -> Dict:
    """Accuracy on flip records (exact label) + per-pair both-correct rate."""
    from collections import defaultdict
    flips = [(r, p) for r, p in zip(rows, preds) if r.get("flip_pair_id") is not None]
    if not flips:
        return {"flip_records": 0}
    exact = sum(1 for r, p in flips if p == r["label"])
    pairs = defaultdict(dict)
    for r, p in flips:
        pairs[r["flip_pair_id"]][r["label"]] = (p == r["label"])
    both = sum(1 for d in pairs.values() if all(d.values()) and len(d) == 2)
    return {
        "flip_records": len(flips),
        "flip_record_accuracy": round(exact / len(flips), 4),
        "flip_pairs": len(pairs),
        "flip_pair_both_correct_rate": round(both / len(pairs), 4),
    }


def run_regex(rows: List[Dict]) -> Dict:
    b = RegexBaseline()
    t0 = time.time()
    preds = [b.detect(r["query"], r["context"])["label"] for r in rows]
    latency_ms = (time.time() - t0) / len(rows) * 1000
    out = {"system": "B1-regex", "latency_ms_mean": round(latency_ms, 3)}
    out.update(security_metrics(rows, preds))
    out.update(flip_accuracy(rows, preds))
    return out


def run_model(rows: List[Dict], adapter_path, base_model, tag, batch_size) -> Dict:
    from src.detector import ContextConsistencyDetector
    det = ContextConsistencyDetector(base_model=base_model, adapter_path=adapter_path)
    res = det.detect_batch(rows, batch_size=batch_size)
    preds = [r["label"] for r in res]
    lat = sorted(r["latency_ms"] for r in res)
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    out = {"system": tag, "latency_ms_p50": round(p50, 2),
           "latency_ms_p95": round(p95, 2),
           "pred_dist": dict(Counter(preds))}
    out.update(security_metrics(rows, preds))
    out.update(flip_accuracy(rows, preds))
    # free GPU before the next model
    import torch, gc
    del det
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--base-model", default=os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B"))
    ap.add_argument("--adapter", default="models/adapters")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--systems", default="regex,base,tuned")
    args = ap.parse_args()

    rows = load_split(args.data, args.split)
    print(f"eval split='{args.split}' rows={len(rows)} "
          f"labels={dict(Counter(r['label'] for r in rows))} "
          f"flips={sum(1 for r in rows if r.get('flip_pair_id') is not None)}")

    mon = HardwareMonitor("eval", interval=1.0)
    mon.start()
    results = []
    want = args.systems.split(",")
    if "regex" in want:
        print("\n[B1] regex baseline..."); results.append(run_regex(rows))
    if "base" in want:
        print("[B2] base 0.5B zero-shot...")
        results.append(run_model(rows, None, args.base_model, "B2-base-zeroshot", args.batch_size))
    if "tuned" in want:
        print("[B3] tuned 0.5B (LoRA)...")
        results.append(run_model(rows, args.adapter, args.base_model, "B3-tuned-slm", args.batch_size))
    mon.stop()

    report = {"split": args.split, "n": len(rows), "results": results,
              "telemetry": mon.summary}
    os.makedirs("results", exist_ok=True)
    with open("results/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # pretty table
    cols = ["system", "detection_recall", "asr_proxy", "fpr", "accuracy_3class",
            "flip_record_accuracy", "flip_pair_both_correct_rate"]
    print("\n" + " | ".join(f"{c:>14}" for c in cols))
    for r in results:
        print(" | ".join(f"{str(r.get(c, '-')):>14}" for c in cols))
    print(f"\nreport -> results/eval_report.json")


if __name__ == "__main__":
    main()
