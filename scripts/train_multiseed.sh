#!/usr/bin/env bash
# 5-seed retraining + eval (reviewer request: training stability across seeds).
# Trains 5 LoRA adapters with different seeds, evaluates each on the held-out
# test set, and writes a combined report. Runs inside the vllm container.
#
# Usage: bash scripts/train_multiseed.sh [seed1 seed2 ...]
set -euo pipefail

IMAGE="vllm/vllm-openai:nightly-aarch64"
WORKDIR="$(pwd)"
HF_TOKEN="${HF_TOKEN:-}"
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(20260624 1 2 3 4)
fi

if docker ps --format '{{.Names}}' | grep -qE '^(teacher-gen|judge)$'; then
  echo "ERROR: a model server is holding the GPU; stop it first." >&2
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  echo "=== seed $seed: train ==="
  docker run --rm --gpus all \
    ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
    -e PYTHONPATH=/work \
    -v "$WORKDIR:/work" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -w /work \
    --entrypoint bash \
    "$IMAGE" -c "
      set -e
      python3 -m pip install --no-input -q --root-user-action=ignore -r requirements-train.txt
      python3 -u -m src.train --seed $seed --out models/adapters_seed${seed}
    "

  echo "=== seed $seed: eval ==="
  docker run --rm --gpus all \
    ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
    -e PYTHONPATH=/work \
    -v "$WORKDIR:/work" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -w /work \
    --entrypoint bash \
    "$IMAGE" -c "
      set -e
      python3 -m pip install --no-input -q --root-user-action=ignore -r requirements-train.txt
      python3 -u -m eval.run_eval --systems tuned --adapter models/adapters_seed${seed}
      mv results/eval_report.json results/eval_report_seed${seed}.json
    "
done

echo "=== aggregating ==="
python3 - "${SEEDS[@]}" <<'PYEOF'
import json, sys, statistics as st
seeds = sys.argv[1:]
metrics = {"detection_recall": [], "fpr": [], "accuracy_3class": [],
           "flip_record_accuracy": [], "flip_pair_both_correct_rate": []}
per_seed = {}
for s in seeds:
    with open(f"results/eval_report_seed{s}.json") as f:
        r = json.load(f)
    tuned = next(x for x in r["results"] if x["system"] == "B3-tuned-slm")
    per_seed[s] = tuned
    for k in metrics:
        metrics[k].append(tuned[k])

summary = {"seeds": seeds, "per_seed": per_seed, "aggregate": {}}
for k, vals in metrics.items():
    summary["aggregate"][k] = {
        "mean": round(st.mean(vals), 4), "stdev": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4), "max": round(max(vals), 4), "values": vals,
    }
with open("results/eval_report_multiseed.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary["aggregate"], indent=2))
print("\nreport -> results/eval_report_multiseed.json")
PYEOF
