#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-/localhome/local-yuhsu/vision-highres-runs/datasets/imagenet-1k-imagefolder}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_VAL="${MAX_VAL:-}"
STREAMING="${STREAMING:-0}"
HF_CACHE="${HF_CACHE:-/localhome/local-yuhsu/vision-highres-runs/hf-cache}"
LOG_DIR="${LOG_DIR:-/localhome/local-yuhsu/vision-highres-runs/prepare-imagenet}"
LOG_FILE="${LOG_DIR}/prepare.log"

mkdir -p "$OUTPUT_ROOT" "$HF_CACHE" "$LOG_DIR"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -z "${HF_TOKEN:-}" && -s "${HOME}/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
  export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN missing. ImageNet-1K on Hugging Face is gated." >&2
  echo "Run this in tmux first, then rerun:" >&2
  echo "  export HF_TOKEN='<your-huggingface-token-with-imagenet-access>'" >&2
  exit 2
fi

echo "=== prepare HF ImageNet $(date -Is) ==="
echo "output_root=${OUTPUT_ROOT}"
echo "hf_cache=${HF_CACHE}"
echo "log=${LOG_FILE}"
echo "max_train=${MAX_TRAIN:-full}"
echo "max_val=${MAX_VAL:-full}"
echo "streaming=${STREAMING}"

ARGS=(
  --dataset-id ILSVRC/imagenet-1k
  --output-root /data/imagenet-out
)
if [[ -n "$MAX_TRAIN" ]]; then ARGS+=(--max-train "$MAX_TRAIN"); fi
if [[ -n "$MAX_VAL" ]]; then ARGS+=(--max-val "$MAX_VAL"); fi
if [[ "$STREAMING" == "1" ]]; then ARGS+=(--streaming); fi

docker compose run --rm \
  -e HF_TOKEN \
  -e HF_HOME=/hf-cache \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -v "${HF_CACHE}:/hf-cache" \
  -v "${OUTPUT_ROOT}:/data/imagenet-out" \
  mamba2-vision bash -lc '
    set -euo pipefail
    python -m pip install -q "datasets>=3.0.0" >/dev/null
    python tools/prepare_hf_imagenet_imagefolder.py "$@"
    chown -R "${HOST_UID}:${HOST_GID}" /data/imagenet-out /hf-cache
  ' bash "${ARGS[@]}"

echo "=== prepared $(date -Is) ==="
find "$OUTPUT_ROOT" -maxdepth 2 -type d | sed -n "1,40p"
cat "$OUTPUT_ROOT/manifest.json"
