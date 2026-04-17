param(
    [string]$Model = "Qwen/Qwen2.5-VL-3B-Instruct",
    [ValidateSet("vanilla", "visiontrim")]
    [string]$Method = "visiontrim",
    [double]$RetainRatio = 0.33,
    [double]$DvtsRatio = 0.75,
    [int]$TgvcIter = 1,
    [string]$QuestionFile = "./playground/data/eval/gqa/subset_10000/llava_gqa_testdev_balanced_10000.jsonl",
    [string]$ImageFolder = "./playground/data/eval/gqa/subset_10000/images",
    [string]$OutRoot = "./playground/data/eval/gqa/answers/qwen2_5_vl_3b",
    [int]$NumSamples = 64,
    [int]$MaxPixels = 0,
    [int]$MinPixels = 0,
    [int]$ClearCudaCacheEvery = 0
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $QuestionFile)) { throw "Question file not found: $QuestionFile" }
if (!(Test-Path $ImageFolder)) { throw "Image folder not found: $ImageFolder" }

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$smokeQuestionFile = Join-Path (Split-Path $QuestionFile -Parent) ("llava_gqa_testdev_balanced_smoke_{0}.jsonl" -f $NumSamples)
python -c "from pathlib import Path; import itertools; src=Path(r'$QuestionFile'); dst=Path(r'$smokeQuestionFile'); dst.parent.mkdir(parents=True, exist_ok=True); lines=list(itertools.islice(src.open('r',encoding='utf-8'), $NumSamples)); dst.open('w',encoding='utf-8').writelines(lines)"

$rawPred = Join-Path $OutRoot ("llava_gqa_testdev_balanced_{0}_smoke_{1}_raw.jsonl" -f $Method, $NumSamples)
$finalPred = Join-Path $OutRoot ("testdev_balanced_{0}_smoke_{1}_predictions.json" -f $Method, $NumSamples)

python scripts/qwen/run_gqa_qwen.py `
    --model $Model `
    --method $Method `
    --retain-ratio $RetainRatio `
    --dvts-ratio $DvtsRatio `
    --tgvc-iter $TgvcIter `
    --question-file $smokeQuestionFile `
    --image-folder $ImageFolder `
    --answers-file $rawPred `
    --temperature 0 `
    --max-new-tokens 16 `
    --max-pixels $MaxPixels `
    --min-pixels $MinPixels `
    --clear-cuda-cache-every $ClearCudaCacheEvery

if ($LASTEXITCODE -ne 0) { throw "run_gqa_qwen.py failed with exit code $LASTEXITCODE" }

python scripts/convert_gqa_for_eval.py --src $rawPred --dst $finalPred

if ($LASTEXITCODE -ne 0) { throw "convert_gqa_for_eval.py failed with exit code $LASTEXITCODE" }

Write-Host "[Done] Smoke converted predictions: $finalPred"
