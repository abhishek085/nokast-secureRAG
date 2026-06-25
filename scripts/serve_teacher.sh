#!/usr/bin/env bash
# Launch the TEACHER (Qwen3.6-35B-A3B-NVFP4) tuned for batched synthetic data
# generation on the DGX Spark (GB10). Stop any other model first — see the
# Phase 2 memory schedule (teacher and judge cannot co-reside on 121 GB).
set -euo pipefail

NAME="${NAME:-teacher-gen}"
MODEL="${TEACHER_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
PORT="${PORT:-8000}"
IMAGE="vllm/vllm-openai:nightly-aarch64"

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" --gpus all -p "${PORT}:8000" \
  ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" "$MODEL" \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 24 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --reasoning-parser qwen3

echo "Launched $NAME ($MODEL). Poll readiness with:"
echo "  curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool"
