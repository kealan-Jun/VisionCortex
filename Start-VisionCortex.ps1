param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [string]$Config,
    [string]$PythonExecutable,
    [switch]$NoBrowser,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $ProjectRoot 'configs\development-local.yaml'
}
elseif (-not [System.IO.Path]::IsPathRooted($Config)) {
    $Config = Join-Path $ProjectRoot $Config
}
$Config = (Resolve-Path -LiteralPath $Config).Path

$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not [string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $Python = (Resolve-Path -LiteralPath $PythonExecutable).Path
}
elseif (-not [string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV)) {
    $Python = (Resolve-Path -LiteralPath (Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe')).Path
}
elseif (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
    $Python = (Resolve-Path -LiteralPath (Join-Path $env:CONDA_PREFIX 'python.exe')).Path
}
elseif (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
    $Python = $VenvPython
}
else {
    $PythonCommand = @(
        Get-Command python3.12 -ErrorAction SilentlyContinue
        Get-Command python3.11 -ErrorAction SilentlyContinue
        Get-Command python -ErrorAction SilentlyContinue
    ) | Select-Object -First 1
    if (-not $PythonCommand) {
        throw 'Python was not found. Install Python 3.11 or 3.12 and try again.'
    }
    $Python = $PythonCommand.Source
}

& $Python (Join-Path $ProjectRoot 'tools\doctor.py') --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw 'The environment check failed. Follow the remediation shown above.'
}
$RootsScript = @'
import json
from pathlib import Path
import sys
from visioncortex.config import load_config
try:
    settings = load_config(Path(sys.argv[1]))
except Exception as exc:
    print('Configuration load failed: ' + type(exc).__name__, file=sys.stderr)
    raise SystemExit(1) from None
storage = settings['storage']
print(json.dumps({name: str(Path(storage[key]).expanduser().resolve()) for name, key in {
    'runtime': 'local_runtime_root', 'staging': 'local_staging_root',
    'cache': 'local_cache_root', 'input': 'local_input_root', 'archive': 'archive_root'
}.items()}))
'@
Push-Location -LiteralPath $ProjectRoot
try {
    $RootsJson = & $Python -c $RootsScript $Config
    if ($LASTEXITCODE -ne 0) { throw 'The selected configuration could not be loaded.' }
    $Roots = $RootsJson | ConvertFrom-Json
}
finally { Pop-Location }
$RuntimeRoot = $Roots.runtime
$RunRoot = $Roots.staging
$CacheRoot = $Roots.cache
$InputRoot = $Roots.input
$ArchiveRoot = $Roots.archive
if ($CheckOnly) {
    exit 0
}

$Url = "http://127.0.0.1:$Port"
try {
    $Health = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 3
}
catch {
    $Health = $null
}
if ($Health -and $Health.status -eq 'ok' -and $Health.product_name -eq 'VisionCortex' -and
    $Health.archive_root -and [System.IO.Path]::GetFullPath($Health.archive_root) -eq $ArchiveRoot) {
    Write-Host "VisionCortex is already running: $Url/#/home"
    if (-not $NoBrowser) { Start-Process "$Url/#/home" }
    exit 0
}

$Occupied = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($Occupied) {
    throw "Port $Port is occupied. Use -Port to select another port."
}

New-Item -ItemType Directory -Path $RuntimeRoot, $RunRoot, $CacheRoot, $InputRoot, $ArchiveRoot -Force | Out-Null
$StdoutLog = Join-Path $RuntimeRoot 'visioncortex-web.stdout.log'
$StderrLog = Join-Path $RuntimeRoot 'visioncortex-web.stderr.log'
$PidFile = Join-Path $RuntimeRoot 'visioncortex-web.pid'

$EnvironmentOverrides = [ordered]@{
    'VISIONCORTEX_CONFIG' = $Config
}
$SavedEnvironment = @{}
foreach ($Name in $EnvironmentOverrides.Keys) {
    $SavedEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
    [Environment]::SetEnvironmentVariable($Name, $EnvironmentOverrides[$Name], 'Process')
}
try {
    $Arguments = @(
        '-m', 'visioncortex', 'serve',
        '--host', '127.0.0.1',
        '--port', "$Port",
        '--config', ('"' + $Config + '"')
    )
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
}
finally {
    foreach ($Name in $SavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $SavedEnvironment[$Name], 'Process')
    }
}
Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii

$Ready = $false
for ($Attempt = 0; $Attempt -lt 120; $Attempt++) {
    if ($Process.HasExited) { break }
    try {
        $Health = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 2
        if ($Health.status -eq 'ok') {
            $Ready = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $Ready) {
    if (-not $Process.HasExited) { Stop-Process -Id $Process.Id -Force }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    $Details = if (Test-Path -LiteralPath $StderrLog) {
        (Get-Content -LiteralPath $StderrLog -Tail 30) -join [Environment]::NewLine
    }
    else {
        'No error log was created.'
    }
    throw "VisionCortex failed to start. Details: $Details"
}

Write-Host "VisionCortex started: $Url/#/home"
Write-Host "Configuration: $Config"
Write-Host "Logs: $RuntimeRoot"
if (-not $NoBrowser) { Start-Process "$Url/#/home" }
