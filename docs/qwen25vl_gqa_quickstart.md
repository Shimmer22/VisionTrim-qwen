# Qwen2.5-VL-3B + GQA Quickstart

## 0) Download GQA testdev + make 10k subset

```bash
bash scripts/qwen/download_gqa_testdev_subset.sh
```

This creates:
- Full testdev questions: `./playground/data/eval/gqa/raw/llava_gqa_testdev_balanced.jsonl`
- Full testdev images: `./playground/data/eval/gqa/data/images`
- 10k subset questions: `./playground/data/eval/gqa/subset_10000/llava_gqa_testdev_balanced_10000.jsonl`
- 10k subset images: `./playground/data/eval/gqa/subset_10000/images`

Optional: download official GQA eval assets (`eval.py`) for scoring against ground truth:

```bash
bash scripts/qwen/download_gqa_eval_assets.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/qwen/download_gqa_eval_assets.ps1
```

## 1) Activate env

```bash
conda activate visiontrim-qwen25vl
```

## 2) Check Apple MPS

```bash
python -c "import torch; print(torch.__version__); print('mps_built=', torch.backends.mps.is_built()); print('mps_available=', torch.backends.mps.is_available())"
```

If `mps_available=False`, you can still run with fallback:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 3) Prepare GQA files

Expected defaults:
- Question file: `./playground/data/eval/gqa/raw/llava_gqa_testdev_balanced.jsonl`
- Image folder: `./playground/data/eval/gqa/data/images`

## 4) Run inference + convert format

```bash
# VisionTrim mode (default)
bash scripts/v1_qwen/eval/gqa_qwen25vl_3b.sh

# Vanilla baseline
METHOD=vanilla bash scripts/v1_qwen/eval/gqa_qwen25vl_3b.sh
```

Outputs:
- Raw jsonl: `./playground/data/eval/gqa/answers/qwen2_5_vl_3b/llava_gqa_testdev_balanced_<method>_raw.jsonl`
- Eval json: `./playground/data/eval/gqa/answers/qwen2_5_vl_3b/testdev_balanced_<method>_predictions.json`

## 5) Optional custom paths

```bash
MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
METHOD=visiontrim \
RETAIN_RATIO=0.33 \
DVTS_RATIO=0.75 \
TGVC_ITER=1 \
QUESTION_FILE=./playground/data/eval/gqa/subset_10000/llava_gqa_testdev_balanced_10000.jsonl \
IMAGE_FOLDER=./playground/data/eval/gqa/subset_10000/images \
OUT_ROOT=./playground/data/eval/gqa/answers/qwen2_5_vl_3b \
bash scripts/v1_qwen/eval/gqa_qwen25vl_3b.sh
```
