param(
    [string]$Dev041Archive = 'Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-154357-29d1',
    [string]$Dev042Archive = 'Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-173813-fc76',
    [string]$ConfigPath = ''
)

$ErrorActionPreference = 'Stop'
$visionCortexProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$visionCortexPython = Join-Path $visionCortexProjectRoot '.venv\Scripts\python.exe'
$visionCortexSourceRoot = Join-Path $visionCortexProjectRoot 'src'
$visionCortexConfig = if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    Join-Path $visionCortexProjectRoot 'configs\rtx4060-laptop-production.yaml'
}
else {
    $ConfigPath
}

foreach ($requiredPath in @(
    $visionCortexPython,
    $visionCortexSourceRoot,
    $visionCortexConfig,
    $Dev041Archive,
    $Dev042Archive
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required read-only replay path is missing: $requiredPath"
    }
}

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $visionCortexSourceRoot

try {
    & $visionCortexPython -B -m labvision_evidence.runtime_preflight `
        --expected-source $visionCortexSourceRoot
    if ($LASTEXITCODE -ne 0) {
        throw "VisionCortex project runtime preflight failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m labvision_evidence.cli inspect-quality-ledger-inputs `
        --archive $Dev041Archive
    if ($LASTEXITCODE -ne 0) {
        throw "DEV-041 quality-ledger input preflight failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m labvision_evidence.cli inspect-quality-ledger-inputs `
        --archive $Dev042Archive
    if ($LASTEXITCODE -ne 0) {
        throw "DEV-042 quality-ledger input preflight failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m labvision_evidence.cli replay-quality-ledger `
        --archive $Dev041Archive `
        --config $visionCortexConfig
    if ($LASTEXITCODE -ne 0) {
        throw "DEV-041 quality-ledger replay failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m labvision_evidence.cli replay-quality-ledger `
        --archive $Dev042Archive `
        --config $visionCortexConfig
    if ($LASTEXITCODE -ne 0) {
        throw "DEV-042 quality-ledger replay failed: $LASTEXITCODE"
    }
}
finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $previousPythonPath
    }
}
