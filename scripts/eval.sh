#!/usr/bin/env bash
# Run the evaluation harness inside the vllm container (CUDA torch + peft).
# GPU must be free (stop any model server first). Usage: bash scripts/eval.sh [args]
set -euo pipefail

IMAGE="vllm/vllm-openai:nightly-aarch64"
WORKDIR="$(pwd)"
HF_TOKEN="${HF_TOKEN:-}"

if docker ps --format '{{.Names}}' | grep -qE '^(teacher-gen|judge)$'; then
  echo "ERROR: a model server is holding the GPU; stop it first." >&2
  exit 1
fi

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
    python3 -u -m eval.run_eval $*
  "
