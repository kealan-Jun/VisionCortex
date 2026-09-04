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
if ($Health -and $Health.status -eq 'ok') {
    Write-Host "VisionCortex is already running: $Url/#/home"
    if (-not $NoBrowser) { Start-Process "$Url/#/home" }
    exit 0
}

$Occupied = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($Occupied) {
    throw "Port $Port is occupied. Use -Port to select another port."
}

$RuntimeRoot = Join-Path $ProjectRoot 'outputs\development-runtime'
$RunRoot = Join-Path $ProjectRoot 'outputs\development-runs'
$CacheRoot = Join-Path $ProjectRoot 'outputs\development-cache'
$InputRoot = Join-Path $ProjectRoot 'outputs\development-input'
$ArchiveRoot = Join-Path $ProjectRoot 'outputs\clean-local-validation'
New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
$StdoutLog = Join-Path $RuntimeRoot 'visioncortex-web.stdout.log'
$StderrLog = Join-Path $RuntimeRoot 'visioncortex-web.stderr.log'
$PidFile = Join-Path $RuntimeRoot 'visioncortex-web.pid'

$EnvironmentOverrides = [ordered]@{
    'VISIONCORTEX_CONFIG' = $Config
    'VISIONCORTEX_NAS_INDEX_CSV' = (Join-Path $ProjectRoot 'examples\development-index.csv')
    'VISIONCORTEX_NAS_ARCHIVE_ROOT' = $ArchiveRoot
    'VISIONCORTEX_NAS_CACHE_ROOT' = $CacheRoot
    'VISIONCORTEX_LOCAL_INPUT_ROOT' = $InputRoot
    'VISIONCORTEX_LOCAL_RUNTIME_ROOT' = $RuntimeRoot
    'VISIONCORTEX_LOCAL_CACHE_ROOT' = $CacheRoot
    'VISIONCORTEX_LOCAL_STAGING_ROOT' = $RunRoot
    'VISIONCORTEX_OUTPUT_ROOT' = $RunRoot
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
Write-Host 'Mode: local development (no NAS access and no automatic model run)'
Write-Host "Logs: $RuntimeRoot"
if (-not $NoBrowser) { Start-Process "$Url/#/home" }
