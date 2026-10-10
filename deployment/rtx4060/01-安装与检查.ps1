param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$PythonPath = '',
    [switch]$InstallDependencies,
    [switch]$PrepareEngines
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $PSScriptRoot 'Runtime-Config.psm1') -Force
$Site = Get-VisionCortexRuntimeConfiguration -ProjectRoot $ProjectRoot -ConfigPath $ConfigPath -PythonPath $PythonPath
$Python = $Site.python
Set-Location -LiteralPath $ProjectRoot
foreach ($required in @($Site.index_csv, $Site.archive_root)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Configured site path is missing: $required" }
}
foreach ($role in @('first_person', 'third_person')) {
    $weight = $Site.models.$role
    $expected = [string]$Site.model_hashes.$role
    if ($expected -notmatch '^[0-9a-fA-F]{64}$') { throw "The private site must pin the $role model checksum." }
    if (-not (Test-Path -LiteralPath $weight -PathType Leaf)) { throw "Configured model is missing: $weight" }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $weight).Hash -ne $expected) {
        throw "Model checksum verification failed for $role."
    }
}
foreach ($command in @('ffmpeg', 'ffprobe', 'nvidia-smi')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "$command was not found in PATH." }
}
$PreviousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
    if ($InstallDependencies) {
        & $Python -m pip install torch torchvision --index-url 'https://download.pytorch.org/whl/cu128'
        if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime dependency installation failed.' }
        & $Python -m pip install -e '.[models,tensorrt]'
        if ($LASTEXITCODE -ne 0) { throw 'Model dependency installation failed.' }
    }
    if ($PrepareEngines) {
        & $Python -m visioncortex prepare-engine --config $Site.config_path
        if ($LASTEXITCODE -ne 0) { throw 'Explicit engine preparation failed.' }
    }
    & $Python -m visioncortex validate-models --config $Site.config_path
    if ($LASTEXITCODE -ne 0) { throw 'Model validation failed.' }
}
finally {
    if ($null -eq $PreviousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
    else { $env:PYTHONPATH = $PreviousPythonPath }
}
Write-Host 'Configured node validation completed. No credential was created or written.'
Write-Host "Input index: $($Site.index_csv)"
Write-Host "Archive root: $($Site.archive_root)"
Write-Host "Local runtime: $($Site.runtime_root)"
