#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/imagenet-root [image_size] [compare_epochs]" >&2
  echo "expected layouts:" >&2
  echo "  root/train/<class>/*.JPEG and root/val/<class>/*.JPEG" >&2
  echo "  root/ILSVRC/Data/CLS-LOC/train and root/ILSVRC/Data/CLS-LOC/val" >&2
  exit 2
fi

DATA_ROOT="$(readlink -f "$1")"
IMAGE_SIZE="${2:-384}"
COMPARE_EPOCHS="${3:-2}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-imagenet-highres${IMAGE_SIZE}}"
RUN_ROOT="${RUN_ROOT:-/localhome/local-yuhsu/vision-highres-runs}"
MAMBA_D_MODEL="${MAMBA_D_MODEL:-192}"
MAMBA_DEPTH="${MAMBA_DEPTH:-4}"
MAMBA_D_STATE="${MAMBA_D_STATE:-64}"
MAMBA_HEADDIM="${MAMBA_HEADDIM:-64}"
MAMBA_EXPAND="${MAMBA_EXPAND:-2}"
LOG_DIR="${RUN_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/run.log"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -d "${DATA_ROOT}/train" && -d "${DATA_ROOT}/val" ]]; then
  HOST_TRAIN="${DATA_ROOT}/train"
  HOST_VAL="${DATA_ROOT}/val"
  CONTAINER_TRAIN="/data/imagenet/train"
  CONTAINER_VAL="/data/imagenet/val"
elif [[ -d "${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train" && -d "${DATA_ROOT}/ILSVRC/Data/CLS-LOC/val" ]]; then
  HOST_TRAIN="${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train"
  HOST_VAL="${DATA_ROOT}/ILSVRC/Data/CLS-LOC/val"
  CONTAINER_TRAIN="/data/imagenet/ILSVRC/Data/CLS-LOC/train"
  CONTAINER_VAL="/data/imagenet/ILSVRC/Data/CLS-LOC/val"
else
  echo "ImageNet train/val folders not found under: ${DATA_ROOT}" >&2
  exit 2
fi

TRAIN_CLASSES="$(find "$HOST_TRAIN" -mindepth 1 -maxdepth 1 -type d | wc -l)"
VAL_CLASSES="$(find "$HOST_VAL" -mindepth 1 -maxdepth 1 -type d | wc -l)"

echo "=== ImageNet high-res run $(date -Is) ==="
echo "data_root=${DATA_ROOT}"
echo "train=${HOST_TRAIN} classes=${TRAIN_CLASSES}"
echo "val=${HOST_VAL} classes=${VAL_CLASSES}"
echo "image_size=${IMAGE_SIZE}"
echo "batch_size=${BATCH_SIZE}"
echo "workers=${WORKERS}"
echo "smoke_epochs=${SMOKE_EPOCHS}"
echo "compare_epochs=${COMPARE_EPOCHS}"
echo "mamba_d_model=${MAMBA_D_MODEL}"
echo "mamba_depth=${MAMBA_DEPTH}"
echo "mamba_d_state=${MAMBA_D_STATE}"
echo "mamba_headdim=${MAMBA_HEADDIM}"
echo "mamba_expand=${MAMBA_EXPAND}"
echo "log=${LOG_FILE}"

if [[ "$TRAIN_CLASSES" -lt 900 || "$VAL_CLASSES" -lt 900 ]]; then
  echo "Class count too low for ImageNet-1K. Stop before wasting GPU." >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "=== Step 1: container dependency check ==="
docker compose run --rm \
  -e HOST_UID="$HOST_UID" \
  -e HOST_GID="$HOST_GID" \
  -v "${DATA_ROOT}:/data/imagenet:ro" \
  mamba2-vision bash -lc 'python -m pip install -q nvidia-ml-py >/dev/null 2>&1 || true
python - <<PY
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import mamba_ssm
import pynvml
print("mamba_ssm ok")
print("pynvml ok")
PY'

if [[ "$SMOKE_EPOCHS" -gt 0 ]]; then
  echo "=== Step 2: Vim high-res smoke ==="
  docker compose run --rm \
    -e HOST_UID="$HOST_UID" \
    -e HOST_GID="$HOST_GID" \
    -v "${DATA_ROOT}:/data/imagenet:ro" \
    mamba2-vision python train.py \
      --dataset imagefolder \
      --train-dir "$CONTAINER_TRAIN" \
      --val-dir "$CONTAINER_VAL" \
      --output-dir "/workspace/outputs/${RUN_NAME}-vim-smoke" \
      --model-arch vim \
      --image-size "$IMAGE_SIZE" \
      --patch-size 16 \
      --d-model "$MAMBA_D_MODEL" \
      --depth "$MAMBA_DEPTH" \
      --d-state "$MAMBA_D_STATE" \
      --headdim "$MAMBA_HEADDIM" \
      --epochs "$SMOKE_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --workers "$WORKERS" \
      --amp \
      --progress-mininterval 2
else
  echo "=== Step 2: Vim high-res smoke skipped ==="
fi

echo "=== Step 3: Vim vs ViT same-data compare ==="
docker compose run --rm \
  -e HOST_UID="$HOST_UID" \
  -e HOST_GID="$HOST_GID" \
  -v "${DATA_ROOT}:/data/imagenet:ro" \
  mamba2-vision bash -lc "python -m pip install -q nvidia-ml-py >/dev/null 2>&1 || true
PYTHONUNBUFFERED=1 python tools/fair_compare.py \
  --dataset imagefolder \
  --train-dir '$CONTAINER_TRAIN' \
  --val-dir '$CONTAINER_VAL' \
  --output-dir '/workspace/outputs/${RUN_NAME}-vim-vs-vit' \
  --reset-output-dir \
  --mamba-arch vim \
  --image-size '$IMAGE_SIZE' \
  --mamba-patch-size 16 \
  --mamba-d-model '$MAMBA_D_MODEL' \
  --mamba-depth '$MAMBA_DEPTH' \
  --mamba-d-state '$MAMBA_D_STATE' \
  --mamba-headdim '$MAMBA_HEADDIM' \
  --mamba-expand '$MAMBA_EXPAND' \
  --epochs '$COMPARE_EPOCHS' \
  --batch-size '$BATCH_SIZE' \
  --workers '$WORKERS' \
  --amp \
  --amp-dtype bfloat16 \
  --vit-model vit_b_16 \
  --progress-mininterval 2 \
  --csv-flush-every 20
chown -R '${HOST_UID}:${HOST_GID}' '/workspace/outputs/${RUN_NAME}-vim-smoke' '/workspace/outputs/${RUN_NAME}-vim-vs-vit' 2>/dev/null || true"

echo "=== Summary ==="
docker compose run --rm mamba2-vision \
  cat "/workspace/outputs/${RUN_NAME}-vim-vs-vit/summary.json"
echo "=== DONE $(date -Is) ==="
