"""
Train/test lexical-diversity check (reviewer request): are train and test
semantically/lexically distinct, or does the generator produce near-duplicate
templates across the split? No GPU needed -- pure Python over
data/dataset.jsonl.

For the malicious-instruction contexts (the security-relevant class):
  1. Global 4-gram vocabulary overlap between train and test.
  2. Per-test-record nearest-neighbor Jaccard similarity against all train
     contexts of the same attack_type (max similarity = closest lexical match
     in train) -- flags near-duplicates if the max is close to 1.0.
  3. Two qualitative example pairs (same attack_type, train vs test).

Run: python3 -m eval.data_diversity_analysis
"""
import json
import re
from collections import defaultdict

WORD_RE = re.compile(r"[a-z0-9']+")


def ngrams(text: str, n: int = 4):
    toks = WORD_RE.findall(text.lower())
    return set(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load(path="data/dataset.jsonl"):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    rows = load()
    train = [r for r in rows if r.get("split") == "train" and r["label"] == "malicious-instruction"]
    test = [r for r in rows if r.get("split") == "test" and r["label"] == "malicious-instruction"]
    print(f"malicious-instruction: train={len(train)} test={len(test)}")

    # 1. Global 4-gram vocabulary overlap
    train_grams = set()
    for r in train:
        train_grams |= ngrams(r["context"])
    test_grams = set()
    for r in test:
        test_grams |= ngrams(r["context"])
    overlap = len(test_grams & train_grams)
    novel_rate = 1 - overlap / len(test_grams) if test_grams else None
    print(f"\nGlobal 4-gram vocab: train={len(train_grams)} test={len(test_grams)} "
          f"shared={overlap} novel_test_4gram_rate={novel_rate:.4f}")

    # 2. Per-test nearest-neighbor Jaccard within same attack_type
    by_type = defaultdict(list)
    for r in train:
        by_type[r["attack_type"]].append((r["id"], ngrams(r["context"])))

    max_sims = []
    for r in test:
        cand = by_type.get(r["attack_type"], [])
        best = 0.0
        best_id = None
        tg = ngrams(r["context"])
        for tid, gset in cand:
            s = jaccard(tg, gset)
            if s > best:
                best, best_id = s, tid
        max_sims.append((r["id"], r["attack_type"], best, best_id))

    sims = sorted((s for _, _, s, _ in max_sims), reverse=True)
    n = len(sims)
    mean_sim = sum(sims) / n if n else 0
    median_sim = sims[n // 2] if n else 0
    p90_sim = sims[int(n * 0.1)] if n else 0
    near_dup = sum(1 for s in sims if s >= 0.5)
    print(f"\nNearest-neighbor 4-gram Jaccard (test context vs closest train context, "
          f"same attack_type): mean={mean_sim:.4f} median={median_sim:.4f} "
          f"p90={p90_sim:.4f} max={sims[0]:.4f} min={sims[-1]:.4f}")
    print(f"near-duplicate rate (jaccard >= 0.5): {near_dup}/{n} = {near_dup/n:.4f}")

    # 3. Qualitative examples: most similar and a typical (median) pair
    max_sims.sort(key=lambda x: x[2], reverse=True)
    print("\n--- Closest train/test pair (highest lexical similarity) ---")
    tid, atype, sim, train_id = max_sims[0]
    test_row = next(r for r in test if r["id"] == tid)
    train_row = next(r for r in train if r["id"] == train_id)
    print(f"attack_type={atype} jaccard={sim:.3f}")
    print(f"TEST  [{tid}]: {test_row['context'][:300]}")
    print(f"TRAIN [{train_id}]: {train_row['context'][:300]}")

    median_idx = len(max_sims) // 2
    print("\n--- Median-similarity train/test pair (typical case) ---")
    tid, atype, sim, train_id = max_sims[median_idx]
    test_row = next(r for r in test if r["id"] == tid)
    train_row = next(r for r in train if r["id"] == train_id)
    print(f"attack_type={atype} jaccard={sim:.3f}")
    print(f"TEST  [{tid}]: {test_row['context'][:300]}")
    print(f"TRAIN [{train_id}]: {train_row['context'][:300]}")

    report = {
        "train_n": len(train), "test_n": len(test),
        "global_4gram_novel_test_rate": round(novel_rate, 4) if novel_rate is not None else None,
        "nn_jaccard_mean": round(mean_sim, 4), "nn_jaccard_median": round(median_sim, 4),
        "nn_jaccard_p90": round(p90_sim, 4), "nn_jaccard_max": round(sims[0], 4) if sims else None,
        "near_duplicate_rate_ge_0.5": round(near_dup / n, 4) if n else None,
    }
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/data_diversity_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nreport -> results/data_diversity_report.json")


if __name__ == "__main__":
    main()
