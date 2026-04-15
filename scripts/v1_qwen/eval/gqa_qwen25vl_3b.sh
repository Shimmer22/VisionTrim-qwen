#!/usr/bin/env bash
set -euo pipefail

# Example:
# conda activate visiontrim-qwen25vl
# bash scripts/v1_qwen/eval/gqa_qwen25vl_3b.sh

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
METHOD="${METHOD:-visiontrim}" # vanilla | visiontrim
RETAIN_RATIO="${RETAIN_RATIO:-0.33}"
DVTS_RATIO="${DVTS_RATIO:-0.75}"
TGVC_ITER="${TGVC_ITER:-1}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./playground/data/eval/gqa/data/images}"
OUT_ROOT="${OUT_ROOT:-./playground/data/eval/gqa/answers/qwen2_5_vl_3b}"
RAW_PRED="${OUT_ROOT}/llava_gqa_testdev_balanced_${METHOD}_raw.jsonl"
FINAL_PRED="${OUT_ROOT}/testdev_balanced_${METHOD}_predictions.json"

mkdir -p "${OUT_ROOT}"

# Useful on Apple Silicon to avoid hard failures for unsupported ops.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

python scripts/qwen/run_gqa_qwen.py \
  --model "${MODEL}" \
  --method "${METHOD}" \
  --retain-ratio "${RETAIN_RATIO}" \
  --dvts-ratio "${DVTS_RATIO}" \
  --tgvc-iter "${TGVC_ITER}" \
  --question-file "${QUESTION_FILE}" \
  --image-folder "${IMAGE_FOLDER}" \
  --answers-file "${RAW_PRED}" \
  --temperature 0 \
  --max-new-tokens 16

python scripts/convert_gqa_for_eval.py --src "${RAW_PRED}" --dst "${FINAL_PRED}"

echo "[Done] Converted predictions: ${FINAL_PRED}"
echo "[Hint] Use your local GQA eval command with: ${FINAL_PRED}"
