param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Config = Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml'
$PidFile = 'D:\VisionCortexLocal\Runtime\visioncortex-web.pid'
$StdoutLog = 'D:\VisionCortexLocal\Runtime\visioncortex-web.stdout.log'
$StderrLog = 'D:\VisionCortexLocal\Runtime\visioncortex-web.stderr.log'

if (-not (Test-Path -LiteralPath $Python)) { throw 'Not installed. Run the setup script first.' }
if (-not (Test-Path -LiteralPath 'Y:\experiment_record_index.csv')) { throw 'Y: NAS or the index CSV is unavailable.' }

$UserKey = [Environment]::GetEnvironmentVariable('ARK_API_KEY', 'User')
if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY) -and -not [string]::IsNullOrWhiteSpace($UserKey)) {
    $env:ARK_API_KEY = $UserKey
}
if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY)) { throw 'ARK_API_KEY is not configured. Run setup again.' }

$env:VISIONCORTEX_CONFIG = $Config
$env:VISIONCORTEX_COARSE_DECODE_LANES = 'cuda,cuda,cuda,cuda,cpu,cpu'
$env:VISIONCORTEX_SOURCE_WORKERS = '6'
$env:VISIONCORTEX_DECODE_QUEUE_DEPTH = '4'
$env:VISIONCORTEX_CPU_DECODE_THREADS = '8'
$env:VISIONCORTEX_YOLO_INFERENCE_WORKERS = '2'
$env:VISIONCORTEX_MATERIALIZATION_WORKERS = '2'
$env:VISIONCORTEX_IO_WORKERS = '4'
$env:VISIONCORTEX_TENSORRT = 'required'

New-Item -ItemType Directory -Path (Split-Path -Parent $PidFile) -Force | Out-Null
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    try {
        $existingHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 3
    }
    catch { throw "Port $Port is occupied by a service that is not a healthy VisionCortex instance." }
    if ($existingHealth.status -ne 'ok') { throw "Port $Port does not expose a healthy VisionCortex instance." }
    if ($existingHealth.fixed_benchmark.submission_protocol_version -ne 1) {
        throw "Port $Port hosts an older VisionCortex Web process. Run 03-停止Web.ps1, then start Web again before submitting a benchmark."
    }
    Write-Host "Port $Port already hosts VisionCortex; opening the Web UI."
}
else {
    $quotedConfig = '"' + $Config + '"'
    $arguments = @('-m', 'visioncortex', 'serve', '--host', '127.0.0.1', '--port', "$Port", '--config', $quotedConfig)
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
    Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
            if ($health.status -eq 'ok' -and $health.fixed_benchmark.submission_protocol_version -eq 1) {
                $ready = $true
                break
            }
        }
        catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) {
        $details = if (Test-Path -LiteralPath $StderrLog) { (Get-Content -LiteralPath $StderrLog -Tail 20) -join [Environment]::NewLine } else { 'No stderr log.' }
        throw "The Web service did not become ready. PID=$($process.Id). Log: $details"
    }
}

$Url = "http://127.0.0.1:$Port/#/home"
Write-Host "VisionCortex Web: $Url"
Write-Host 'Use the home-page benchmark button to rerun the fixed six-view task.'
if (-not $NoBrowser) { Start-Process $Url }
