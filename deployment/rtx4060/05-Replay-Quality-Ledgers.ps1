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
$previousExpectedSource = $env:VISIONCORTEX_EXPECTED_SOURCE_ROOT
$env:PYTHONPATH = $visionCortexSourceRoot
$env:VISIONCORTEX_EXPECTED_SOURCE_ROOT = $visionCortexSourceRoot

try {
    $runtimeProbe = @'
import importlib
import json
import os
import pathlib
import sys

required = ["openpyxl", "cv2", "numpy", "pydantic", "yaml"]
versions = {}
for name in required:
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, "__version__", "available"))

import labvision_evidence
import labvision_evidence.cli
import labvision_evidence.replay_acceptance

expected = pathlib.Path(os.environ["VISIONCORTEX_EXPECTED_SOURCE_ROOT"]).resolve()
actual = pathlib.Path(labvision_evidence.__file__).resolve()
if expected not in actual.parents:
    raise RuntimeError(f"unexpected package source: {actual}")

print(json.dumps({
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "package_source": str(actual),
    "dependency_versions": versions,
}, ensure_ascii=True, sort_keys=True))
'@

    & $visionCortexPython -B -c $runtimeProbe
    if ($LASTEXITCODE -ne 0) {
        throw "VisionCortex project runtime preflight failed: $LASTEXITCODE"
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
    if ($null -eq $previousExpectedSource) {
        Remove-Item Env:VISIONCORTEX_EXPECTED_SOURCE_ROOT -ErrorAction SilentlyContinue
    }
    else {
        $env:VISIONCORTEX_EXPECTED_SOURCE_ROOT = $previousExpectedSource
    }
}
