#!/usr/bin/env bash
# Merge the LoRA adapter into the base model -> standalone HF model in models/merged.
# Runs in the vllm container (CUDA torch + peft). GPU need not be free (CPU/merge is light).
set -euo pipefail

IMAGE="vllm/vllm-openai:nightly-aarch64"
WORKDIR="$(pwd)"
HF_TOKEN="${HF_TOKEN:-}"

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
    python3 -u -m src.export_model $*
  "
