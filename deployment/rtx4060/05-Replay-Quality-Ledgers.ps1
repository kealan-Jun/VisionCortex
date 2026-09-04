param(
    [string]$CanonicalNasArchiveRoot = '\\192.168.66.149\video_database\VisionCortexExperimentArchive',
    [string]$MappedNasArchiveRoot = 'Y:\VisionCortexExperimentArchive',
    [string]$ConfigPath = ''
)

$ErrorActionPreference = 'Stop'
$visionCortexProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$visionCortexPython = Join-Path $visionCortexProjectRoot '.venv\Scripts\python.exe'
$visionCortexSourceRoot = Join-Path $visionCortexProjectRoot 'src'
$replayPathModule = Join-Path $PSScriptRoot 'Replay-Path-Resolution.psm1'
$visionCortexConfig = if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    Join-Path $visionCortexProjectRoot 'configs\rtx4060-laptop-production.yaml'
}
else {
    $ConfigPath
}

foreach ($requiredLocalPath in @(
    $visionCortexPython,
    $visionCortexSourceRoot,
    $visionCortexConfig,
    $replayPathModule
)) {
    if (-not (Test-Path -LiteralPath $requiredLocalPath)) {
        throw "Required local replay path is missing: $requiredLocalPath"
    }
}

$dev041RelativePath = '.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-154357-29d1'
$dev042RelativePath = '.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-173813-fc76'

Import-Module -Name $replayPathModule -Force

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $visionCortexSourceRoot

try {
    & $visionCortexPython -B -m visioncortex.runtime_preflight `
        --expected-source $visionCortexSourceRoot
    if ($LASTEXITCODE -ne 0) {
        throw "VisionCortex project runtime preflight failed: $LASTEXITCODE"
    }

    $archiveRoots = @($CanonicalNasArchiveRoot, $MappedNasArchiveRoot)
    $dev041Resolution = Resolve-ExactReplayArchive `
        -Label 'DEV-041' `
        -RelativePath $dev041RelativePath `
        -ArchiveRoots $archiveRoots
    $dev042Resolution = Resolve-ExactReplayArchive `
        -Label 'DEV-042' `
        -RelativePath $dev042RelativePath `
        -ArchiveRoots $archiveRoots
    Write-Host ($dev041Resolution | ConvertTo-Json -Depth 5)
    Write-Host ($dev042Resolution | ConvertTo-Json -Depth 5)
    $Dev041Archive = $dev041Resolution.resolved_archive
    $Dev042Archive = $dev042Resolution.resolved_archive

    & $visionCortexPython -B -m visioncortex.cli inspect-quality-ledger-inputs `
        --archive $Dev041Archive
    if ($LASTEXITCODE -ne 0) {
        throw "DEV-041 quality-ledger input preflight failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m visioncortex.cli inspect-quality-ledger-inputs `
        --archive $Dev042Archive
    if ($LASTEXITCODE -ne 0) {
        throw "DEV-042 quality-ledger input preflight failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m visioncortex.cli replay-quality-ledger `
        --archive $Dev041Archive `
        --config $visionCortexConfig
    if ($LASTEXITCODE -ne 0) {
        throw "DEV-041 quality-ledger replay failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m visioncortex.cli replay-quality-ledger `
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
