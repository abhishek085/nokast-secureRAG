#!/usr/bin/env bash
# Supervisor for resumable data generation. Runs src.data_gen --resume; if the
# vLLM engine wedges (exit 2), restarts the teacher container and resumes, up to
# MAX_RESTARTS times. Any other non-zero exit aborts. Pass-through args go to
# data_gen, e.g.:  bash scripts/run_datagen.sh --n-benign 3000 --n-adversarial 2000 --n-flips 300
set -uo pipefail

MAX_RESTARTS="${MAX_RESTARTS:-30}"

wait_ready() {
  for _ in $(seq 1 120); do
    curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  echo "=== datagen attempt $attempt/$MAX_RESTARTS ==="
  python3 -u -m src.data_gen --resume "$@"
  code=$?
  case $code in
    0) echo "=== DATAGEN COMPLETE (attempt $attempt) ==="; exit 0 ;;
    2) echo "=== engine wedged; restarting teacher and resuming ===" ;;
    *) echo "=== datagen exited code=$code (not a wedge); aborting ==="; exit "$code" ;;
  esac
  docker rm -f teacher-gen >/dev/null 2>&1 || true
  bash scripts/serve_teacher.sh
  if ! wait_ready; then
    echo "ERROR: teacher did not become ready after restart"; exit 1
  fi
done

echo "ERROR: exhausted $MAX_RESTARTS restarts without completing"; exit 1
