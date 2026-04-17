param(
    [string]$RootDir = "./playground/data/gqa",
    [int]$TimeoutSec = 180,
    [int]$MaxRetries = 4
)

$ErrorActionPreference = "Stop"

function Test-EvalReady {
    param([string]$Dir)
    return (Test-Path (Join-Path $Dir "eval/eval.py"))
}

function Invoke-WithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [int]$Retries = 4,
        [int]$TimeoutSeconds = 180
    )

    for ($i = 1; $i -le $Retries; $i++) {
        try {
            Write-Host "[Download] ($i/$Retries) $Url"
            Invoke-WebRequest -Uri $Url -OutFile $OutFile -TimeoutSec $TimeoutSeconds
            if ((Test-Path $OutFile) -and ((Get-Item $OutFile).Length -gt 0)) {
                return $true
            }
        }
        catch {
            Write-Warning "[Warn] Attempt $i failed: $($_.Exception.Message)"
            Start-Sleep -Seconds ([Math]::Min(20, 3 * $i))
        }
    }
    return $false
}

function Expand-ZipSafe {
    param(
        [Parameter(Mandatory = $true)][string]$ZipFile,
        [Parameter(Mandatory = $true)][string]$DestDir
    )
    try {
        Expand-Archive -Path $ZipFile -DestinationPath $DestDir -Force
        return $true
    }
    catch {
        Write-Warning "[Warn] Expand-Archive failed: $($_.Exception.Message)"
        return $false
    }
}

New-Item -ItemType Directory -Force -Path $RootDir | Out-Null

if (Test-EvalReady -Dir $RootDir) {
    Write-Host "[Skip] Found eval script: $(Join-Path $RootDir 'eval/eval.py')"
    exit 0
}

$zipPath = Join-Path $RootDir "eval.zip"
$urls = @(
    "https://nlp.stanford.edu/data/gqa/eval.zip",
    "http://nlp.stanford.edu/data/gqa/eval.zip",
    "https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip",
    "http://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip"
)

$ok = $false
foreach ($url in $urls) {
    if (Invoke-WithRetry -Url $url -OutFile $zipPath -Retries $MaxRetries -TimeoutSeconds $TimeoutSec) {
        if (Expand-ZipSafe -ZipFile $zipPath -DestDir $RootDir) {
            if (Test-EvalReady -Dir $RootDir) {
                Write-Host "[Done] GQA eval assets ready at $RootDir"
                $ok = $true
                break
            }
        }
    }
}

if (-not $ok) {
    throw "Failed to download/prepare GQA eval assets. You can manually place eval.py under $RootDir/eval/."
}
