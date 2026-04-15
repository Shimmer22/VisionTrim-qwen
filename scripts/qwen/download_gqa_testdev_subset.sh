#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-./playground/data/eval/gqa}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
SEED="${SEED:-42}"

BASE_URL="https://huggingface.co/datasets/AJN-AI/VoQA/resolve/0410274a0d5cf0e249dcc120f55a39f3cc570195/test/gqa"
Q_FILE="llava_gqa_testdev_balanced.jsonl"
IMG_ZIP="images.zip"

RAW_DIR="${ROOT_DIR}/raw"
DATA_DIR="${ROOT_DIR}/data"
SUBSET_DIR="${ROOT_DIR}/subset_${MAX_SAMPLES}"

mkdir -p "${RAW_DIR}" "${DATA_DIR}" "${SUBSET_DIR}"

if [[ ! -f "${RAW_DIR}/${Q_FILE}" ]]; then
  echo "[Download] ${Q_FILE}"
  curl -L "${BASE_URL}/${Q_FILE}" -o "${RAW_DIR}/${Q_FILE}"
fi

if [[ ! -f "${RAW_DIR}/${IMG_ZIP}" ]]; then
  echo "[Download] ${IMG_ZIP}"
  curl -L "${BASE_URL}/${IMG_ZIP}" -o "${RAW_DIR}/${IMG_ZIP}"
fi

if [[ ! -d "${DATA_DIR}/images" ]]; then
  echo "[Extract] images.zip"
  unzip -q "${RAW_DIR}/${IMG_ZIP}" -d "${DATA_DIR}"
fi

python scripts/qwen/make_gqa_subset.py \
  --input-jsonl "${RAW_DIR}/${Q_FILE}" \
  --input-images "${DATA_DIR}/images" \
  --output-jsonl "${SUBSET_DIR}/llava_gqa_testdev_balanced_${MAX_SAMPLES}.jsonl" \
  --output-images "${SUBSET_DIR}/images" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}"

echo "[Done] Root: ${ROOT_DIR}"
echo "[Done] Full questions: ${RAW_DIR}/${Q_FILE}"
echo "[Done] Full images: ${DATA_DIR}/images"
echo "[Done] Subset questions: ${SUBSET_DIR}/llava_gqa_testdev_balanced_${MAX_SAMPLES}.jsonl"
echo "[Done] Subset images: ${SUBSET_DIR}/images"
