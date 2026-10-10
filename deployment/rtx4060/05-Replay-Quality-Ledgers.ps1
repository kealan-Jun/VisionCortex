param(
    [Parameter(Mandatory = $true)][string]$CanonicalNasArchiveRoot,
    [Parameter(Mandatory = $true)][string]$FirstRunRelativePath,
    [Parameter(Mandatory = $true)][string]$SecondRunRelativePath,
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$MappedNasArchiveRoot = ''
)

$ErrorActionPreference = 'Stop'
$visionCortexProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$visionCortexPython = Join-Path $visionCortexProjectRoot '.venv\Scripts\python.exe'
$visionCortexSourceRoot = Join-Path $visionCortexProjectRoot 'src'
$replayPathModule = Join-Path $PSScriptRoot 'Replay-Path-Resolution.psm1'
$visionCortexConfig = if ([IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $visionCortexProjectRoot $ConfigPath }
$runtimeConfigModule = Join-Path $PSScriptRoot 'Runtime-Config.psm1'

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


Import-Module -Name $replayPathModule -Force

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $visionCortexSourceRoot

try {
    & $visionCortexPython -B -m visioncortex.runtime_preflight `
        --expected-source $visionCortexSourceRoot
    if ($LASTEXITCODE -ne 0) {
        throw "VisionCortex project runtime preflight failed: $LASTEXITCODE"
    }

    Import-Module -Name $runtimeConfigModule -Force
    $null = Get-VisionCortexRuntimeConfiguration -ProjectRoot $visionCortexProjectRoot -ConfigPath $visionCortexConfig -PythonPath $visionCortexPython
    $archiveRoots = @($CanonicalNasArchiveRoot, $MappedNasArchiveRoot)
    $firstRunResolution = Resolve-ExactReplayArchive `
        -Label 'first-run' `
        -RelativePath $FirstRunRelativePath `
        -ArchiveRoots $archiveRoots
    $secondRunResolution = Resolve-ExactReplayArchive `
        -Label 'second-run' `
        -RelativePath $SecondRunRelativePath `
        -ArchiveRoots $archiveRoots
    Write-Host ($firstRunResolution | ConvertTo-Json -Depth 5)
    Write-Host ($secondRunResolution | ConvertTo-Json -Depth 5)
    $FirstRunArchive = $firstRunResolution.resolved_archive
    $SecondRunArchive = $secondRunResolution.resolved_archive

    & $visionCortexPython -B -m visioncortex.cli inspect-quality-ledger-inputs `
        --archive $FirstRunArchive
    if ($LASTEXITCODE -ne 0) {
        throw "first-run quality-ledger input preflight failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m visioncortex.cli inspect-quality-ledger-inputs `
        --archive $SecondRunArchive
    if ($LASTEXITCODE -ne 0) {
        throw "second-run quality-ledger input preflight failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m visioncortex.cli replay-quality-ledger `
        --archive $FirstRunArchive `
        --config $visionCortexConfig
    if ($LASTEXITCODE -ne 0) {
        throw "first-run quality-ledger replay failed: $LASTEXITCODE"
    }

    & $visionCortexPython -B -m visioncortex.cli replay-quality-ledger `
        --archive $SecondRunArchive `
        --config $visionCortexConfig
    if ($LASTEXITCODE -ne 0) {
        throw "second-run quality-ledger replay failed: $LASTEXITCODE"
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
