#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-./playground/data/gqa}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
MAX_RETRIES="${MAX_RETRIES:-4}"

EVAL_PY="${ROOT_DIR}/eval/eval.py"
ZIP_PATH="${ROOT_DIR}/eval.zip"

mkdir -p "${ROOT_DIR}"

if [[ -f "${EVAL_PY}" ]]; then
  echo "[Skip] Found eval script: ${EVAL_PY}"
  exit 0
fi

download_with_retry() {
  local url="$1"
  echo "[Download] ${url}"
  curl -L \
    --fail \
    --retry "${MAX_RETRIES}" \
    --retry-delay 3 \
    --retry-connrefused \
    --connect-timeout 20 \
    --max-time "${TIMEOUT_SEC}" \
    -o "${ZIP_PATH}" \
    "${url}"
}

try_extract() {
  unzip -q -o "${ZIP_PATH}" -d "${ROOT_DIR}" || return 1
  [[ -f "${EVAL_PY}" ]]
}

URLS=(
  "https://nlp.stanford.edu/data/gqa/eval.zip"
  "http://nlp.stanford.edu/data/gqa/eval.zip"
  "https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip"
  "http://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip"
)

for url in "${URLS[@]}"; do
  if download_with_retry "${url}" && try_extract; then
    echo "[Done] GQA eval assets ready at ${ROOT_DIR}"
    exit 0
  fi
done

echo "[Error] Failed to download/prepare GQA eval assets."
echo "[Hint] Manually place eval.py under ${ROOT_DIR}/eval/ and re-run."
exit 1
