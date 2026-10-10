param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [switch]$PrepareEngines
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $ProjectRoot 'deployment\rtx4060\Runtime-Config.psm1') -Force
$Config = if ([IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $ProjectRoot $ConfigPath }
$PythonInstaller = Join-Path $ProjectRoot 'vendor\python\python-3.12.10-amd64.exe'
$PythonHome = Join-Path $ProjectRoot '.runtime\python312'
$BootstrapPython = Join-Path $PythonHome 'python.exe'
$Venv = Join-Path $ProjectRoot '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'
$Wheelhouse = Join-Path $ProjectRoot 'vendor\wheelhouse'
$FfmpegBin = Join-Path $ProjectRoot 'vendor\ffmpeg\bin'
$FirstWeight = Join-Path $ProjectRoot 'models\first_person\best.pt'
$ThirdWeight = Join-Path $ProjectRoot 'models\third_person\best.pt'
$ExpectedHashes = @{
    $FirstWeight = 'A541C59EF8B09158B9B22851DCADA6231DBAB0F2F1478AE824BCF609851C58EA'
    $ThirdWeight = 'EF5A867ABF21A8D790EABA054E92D114AE4567C1F867C041ED079CDE0A01A36B'
}

Set-Location -LiteralPath $ProjectRoot
& (Join-Path $PSScriptRoot 'Verify-Package.ps1') -PackageRoot $ProjectRoot

foreach ($Required in @(
    $Config,
    $PythonInstaller,
    $Wheelhouse,
    (Join-Path $FfmpegBin 'ffmpeg.exe'),
    (Join-Path $FfmpegBin 'ffprobe.exe'),
    $FirstWeight,
    $ThirdWeight
)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Required package item is missing: $Required" }
}
foreach ($Entry in $ExpectedHashes.GetEnumerator()) {
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Entry.Key).Hash
    if ($Actual -ne $Entry.Value) { throw "Model checksum verification failed: $($Entry.Key)" }
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'NVIDIA driver or nvidia-smi is unavailable. Install a current RTX 4090 driver first.'
}
$GpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1).Trim()
if ($GpuName -notmatch '4090') { throw "This package profile requires RTX 4090; detected: $GpuName" }

if (-not (Test-Path -LiteralPath $BootstrapPython -PathType Leaf)) {
    New-Item -ItemType Directory -Path $PythonHome -Force | Out-Null
    $Arguments = @(
        '/quiet',
        'InstallAllUsers=0',
        "TargetDir=$PythonHome",
        'Include_pip=1',
        'Include_test=0',
        'Include_launcher=0',
        'PrependPath=0',
        'Shortcuts=0'
    )
    $Process = Start-Process -FilePath $PythonInstaller -ArgumentList $Arguments -Wait -PassThru
    if ($Process.ExitCode -ne 0) { throw "Bundled Python installation failed: exit $($Process.ExitCode)" }
}
$Version = (& $BootstrapPython -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")').Trim()
if ($Version -ne '3.12') { throw "Bundled Python 3.12 is required; detected $Version" }

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    & $BootstrapPython -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $Python -m pip install --no-index --find-links $Wheelhouse --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Offline packaging-tool installation failed.' }
& $Python -m pip install --no-index --find-links $Wheelhouse 'visioncortex[tensorrt]==0.1.0'
if ($LASTEXITCODE -ne 0) { throw 'Offline VisionCortex dependency installation failed.' }

$env:PATH = "$FfmpegBin;$env:PATH"
$Site = Get-VisionCortexRuntimeConfiguration -ProjectRoot $ProjectRoot -ConfigPath $Config -PythonPath $Python
foreach ($Directory in @($Site.runtime_root, $Site.cache_root)) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

& $Python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'; name=torch.cuda.get_device_name(0); assert '4090' in name, name; print('GPU:', name); print('CUDA:', torch.version.cuda)"
if ($LASTEXITCODE -ne 0) { throw 'RTX 4090 CUDA validation failed.' }
& $Python -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'TensorRT validation failed.' }
& (Join-Path $FfmpegBin 'ffmpeg.exe') -hide_banner -version | Select-Object -First 1

if ($PrepareEngines) {
    & $Python -m visioncortex prepare-engine --config $Config
    if ($LASTEXITCODE -ne 0) { throw 'RTX 4090 TensorRT engine export failed.' }
}
& $Python -m visioncortex validate-models --config $Config
if ($LASTEXITCODE -ne 0) { throw 'The two 21-class YOLO models failed validation.' }

Write-Host ''
Write-Host 'VisionCortex RTX 4090 installation and validation completed.'
Write-Host 'Inputs are discovered from the NAS index; camera count is not hard-coded.'
Write-Host "Runtime: $($Site.runtime_root)"
Write-Host "Cache: $($Site.cache_root)"
Write-Host 'Next: deployment\rtx4090\02-Start-Web.ps1'
