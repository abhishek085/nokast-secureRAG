#!/usr/bin/env bash
# Launch the JUDGE (Nemotron-3-Super-120B-A12B-NVFP4) to cross-check teacher labels.
# MUST stop the teacher first — teacher (~66 GB) + judge (~65 GB) > 121 GB unified.
#   docker rm -f teacher-gen && bash scripts/serve_judge.sh
set -euo pipefail

NAME="${NAME:-judge}"
MODEL="${JUDGE_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
PORT="${PORT:-8000}"
IMAGE="vllm/vllm-openai:nightly-aarch64"

# Safety: refuse to start if the teacher is still up (would OOM).
if docker ps --format '{{.Names}}' | grep -q '^teacher-gen$'; then
  echo "ERROR: teacher-gen is still running. Stop it first: docker rm -f teacher-gen" >&2
  exit 1
fi

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" --gpus all -p "${PORT}:8000" \
  ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" "$MODEL" \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 16

echo "Launched $NAME ($MODEL). Poll: curl -s http://localhost:${PORT}/v1/models"
