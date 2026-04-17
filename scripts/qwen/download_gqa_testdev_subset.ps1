param(
    [string]$RootDir = "./playground/data/eval/gqa",
    [int]$MaxSamples = 10000,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$baseUrl = "https://huggingface.co/datasets/AJN-AI/VoQA/resolve/0410274a0d5cf0e249dcc120f55a39f3cc570195/test/gqa"
$qFile = "llava_gqa_testdev_balanced.jsonl"
$imgZip = "images.zip"

$rawDir = Join-Path $RootDir "raw"
$dataDir = Join-Path $RootDir "data"
$subsetDir = Join-Path $RootDir ("subset_{0}" -f $MaxSamples)

New-Item -ItemType Directory -Force -Path $rawDir, $dataDir, $subsetDir | Out-Null

$qPath = Join-Path $rawDir $qFile
if (!(Test-Path $qPath)) {
    Write-Host "[Download] $qFile"
    curl.exe -L "$baseUrl/$qFile" -o $qPath
}

$zipPath = Join-Path $rawDir $imgZip
if (!(Test-Path $zipPath)) {
    Write-Host "[Download] $imgZip"
    curl.exe -L "$baseUrl/$imgZip" -o $zipPath
}

$imagesDir = Join-Path $dataDir "images"
if (!(Test-Path $imagesDir)) {
    Write-Host "[Extract] $imgZip"
    Expand-Archive -Path $zipPath -DestinationPath $dataDir -Force
}

$subsetJsonl = Join-Path $subsetDir ("llava_gqa_testdev_balanced_{0}.jsonl" -f $MaxSamples)
$subsetImages = Join-Path $subsetDir "images"

python scripts/qwen/make_gqa_subset.py `
    --input-jsonl $qPath `
    --input-images $imagesDir `
    --output-jsonl $subsetJsonl `
    --output-images $subsetImages `
    --max-samples $MaxSamples `
    --seed $Seed

Write-Host "[Done] Root: $RootDir"
Write-Host "[Done] Full questions: $qPath"
Write-Host "[Done] Full images: $imagesDir"
Write-Host "[Done] Subset questions: $subsetJsonl"
Write-Host "[Done] Subset images: $subsetImages"
