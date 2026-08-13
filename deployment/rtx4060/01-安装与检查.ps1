param(
    [switch]$SkipInstall,
    [switch]$SkipTensorRTExport
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Config = Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml'
$FirstWeight = Join-Path $ProjectRoot 'models\first_person\best.pt'
$ThirdWeight = Join-Path $ProjectRoot 'models\third_person\best.pt'
$NasIndex = 'Y:\experiment_record_index.csv'
$NasArchive = 'Y:\VisionCortexExperimentArchive'
$ExpectedHashes = @{
    $FirstWeight = 'A541C59EF8B09158B9B22851DCADA6231DBAB0F2F1478AE824BCF609851C58EA'
    $ThirdWeight = 'EF5A867ABF21A8D790EABA054E92D114AE4567C1F867C041ED079CDE0A01A36B'
}

Set-Location -LiteralPath $ProjectRoot

foreach ($required in @($Config, $FirstWeight, $ThirdWeight, $NasIndex, $NasArchive)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}
foreach ($entry in $ExpectedHashes.GetEnumerator()) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Key).Hash
    if ($actual -ne $entry.Value) {
        throw "Model checksum verification failed: $($entry.Key)"
    }
}
foreach ($commandName in @('python', 'ffmpeg', 'ffprobe', 'nvidia-smi')) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "$commandName was not found in PATH."
    }
}

$versionText = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($versionText -notin @('3.11', '3.12')) {
    throw "Python 3.11 or 3.12 is required; detected $versionText."
}

foreach ($directory in @(
    'D:\VisionCortexLocal\Input',
    'D:\VisionCortexLocal\Runtime',
    'D:\VisionCortexLocal\Cache'
)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

if (-not (Test-Path -LiteralPath $Python)) {
    & python -m venv (Join-Path $ProjectRoot '.venv')
}
if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip setuptools wheel
    & $Python -m pip install torch torchvision --index-url 'https://download.pytorch.org/whl/cu128'
    & $Python -m pip install -e '.[dev,tensorrt]'
}

& $Python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.version.cuda)"
& $Python -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"

$UserKey = [Environment]::GetEnvironmentVariable('ARK_API_KEY', 'User')
if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY) -and -not [string]::IsNullOrWhiteSpace($UserKey)) {
    $env:ARK_API_KEY = $UserKey
}
if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY)) {
    $SecureKey = Read-Host 'Enter ARK_API_KEY (input is hidden)' -AsSecureString
    $Pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureKey)
    try {
        $PlainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Pointer)
        if ([string]::IsNullOrWhiteSpace($PlainKey)) { throw 'ARK_API_KEY cannot be empty.' }
        [Environment]::SetEnvironmentVariable('ARK_API_KEY', $PlainKey, 'User')
        $env:ARK_API_KEY = $PlainKey
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Pointer)
        $PlainKey = $null
    }
}

if (-not $SkipTensorRTExport) {
    & $Python -m labvision_evidence prepare-engine --config $Config
}
& $Python -m labvision_evidence validate-models --config $Config

Write-Host ''
Write-Host 'RTX4060 node validation completed.'
Write-Host "Fixed input index: $NasIndex"
Write-Host "Fixed output archive: $NasArchive\Six-View-Three-Hour-Experiment-2026-08-13"
Write-Host 'Local cache: D:\VisionCortexLocal\Cache'
Write-Host 'Next: run .\deployment\rtx4060\02-Start-Web.ps1 or the Chinese-named equivalent.'
