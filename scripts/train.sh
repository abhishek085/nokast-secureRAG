#!/usr/bin/env bash
# Fine-tune the student (Qwen2.5-0.5B) with LoRA inside the vllm container, which
# already has a CUDA-enabled torch for the GB10. The GPU must be FREE -- stop any
# vllm server first (docker rm -f teacher-gen judge).
#
# Usage:  bash scripts/train.sh [extra args passed to src.train]
set -euo pipefail

IMAGE="vllm/vllm-openai:nightly-aarch64"
WORKDIR="$(pwd)"
HF_TOKEN="${HF_TOKEN:-}"

# Refuse to run while a model server holds the GPU.
if docker ps --format '{{.Names}}' | grep -qE '^(teacher-gen|judge)$'; then
  echo "ERROR: a model server (teacher-gen/judge) is still running; stop it first." >&2
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
    python3 -u -m src.train $*
  "
