"""
External-benchmark generalization check: PoisonedRAG (Zou et al., 2024) targeted
attacks against our (Q,C) detector. Recall-only (the benchmark has no benign
rows), reported honestly alongside the in-distribution numbers -- see paper
Limitations. Runs inside the vllm container (CUDA torch + peft):

    bash scripts/eval_poisonedrag.sh
"""
import os
import sys
import json
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.baselines import RegexBaseline
from src.poisonedrag_adapter import load_poisonedrag


def recall_only(rows, preds):
    n = len(rows)
    caught = sum(1 for p in preds if p != "safe")
    return {
        "n": n,
        "detection_recall": round(caught / n, 4) if n else None,
        "pred_dist": dict(Counter(preds)),
    }


def run_regex(rows):
    b = RegexBaseline()
    t0 = time.time()
    preds = [b.detect(r["query"], r["context"])["label"] for r in rows]
    latency_ms = (time.time() - t0) / len(rows) * 1000
    out = {"system": "B1-regex", "latency_ms_mean": round(latency_ms, 3)}
    out.update(recall_only(rows, preds))
    return out


def run_model(rows, adapter_path, base_model, tag, batch_size):
    from src.detector import ContextConsistencyDetector
    det = ContextConsistencyDetector(base_model=base_model, adapter_path=adapter_path)
    res = det.detect_batch(rows, batch_size=batch_size)
    preds = [r["label"] for r in res]
    lat = sorted(r["latency_ms"] for r in res)
    out = {"system": tag, "latency_ms_p50": round(lat[len(lat) // 2], 2)}
    out.update(recall_only(rows, preds))
    import torch, gc
    del det
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B"))
    ap.add_argument("--adapter", default="models/adapters")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    rows = load_poisonedrag()
    print(f"PoisonedRAG external eval: {len(rows)} (query, poisoned-context) pairs "
          f"(all ground-truth malicious; recall-only, no FPR)")

    results = []
    print("\n[B1] regex baseline..."); results.append(run_regex(rows))
    print("[B2] base 0.5B zero-shot...")
    results.append(run_model(rows, None, args.base_model, "B2-base-zeroshot", args.batch_size))
    print("[B3] tuned 0.5B (LoRA)...")
    results.append(run_model(rows, args.adapter, args.base_model, "B3-tuned-slm", args.batch_size))

    report = {"benchmark": "poisonedrag-adv_targeted", "n": len(rows), "results": results}
    os.makedirs("results", exist_ok=True)
    with open("results/eval_report_poisonedrag.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + " | ".join(f"{c:>18}" for c in ["system", "detection_recall", "n"]))
    for r in results:
        print(" | ".join(f"{str(r.get(c, '-')):>18}" for c in ["system", "detection_recall", "n"]))
    print("\nreport -> results/eval_report_poisonedrag.json")


if __name__ == "__main__":
    main()
