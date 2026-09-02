param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Config = Join-Path $ProjectRoot 'configs\rtx4090-production.yaml'
$FfmpegBin = Join-Path $ProjectRoot 'vendor\ffmpeg\bin'
$RuntimeRoot = 'D:\VisionCortex4090\Runtime'
$PidFile = Join-Path $RuntimeRoot 'visioncortex-web.pid'
$StdoutLog = Join-Path $RuntimeRoot 'visioncortex-web.stdout.log'
$StderrLog = Join-Path $RuntimeRoot 'visioncortex-web.stderr.log'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'VisionCortex is not installed. Run 01-Install-And-Validate.ps1 first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $FfmpegBin 'ffmpeg.exe') -PathType Leaf)) {
    throw 'Bundled FFmpeg is missing.'
}
$env:PATH = "$FfmpegBin;$env:PATH"
$env:VISIONCORTEX_CONFIG = $Config
$env:VISIONCORTEX_TENSORRT = 'required'
$env:VISIONCORTEX_LOCAL_RUNTIME_ROOT = $RuntimeRoot
$env:VISIONCORTEX_LOCAL_CACHE_ROOT = 'D:\VisionCortex4090\Cache'

$UserKey = [Environment]::GetEnvironmentVariable('ARK_API_KEY', 'User')
if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY) -and -not [string]::IsNullOrWhiteSpace($UserKey)) {
    $env:ARK_API_KEY = $UserKey
}
if ([string]::IsNullOrWhiteSpace($env:ARK_API_KEY)) {
    throw 'ARK_API_KEY is not configured. Run the installation script again.'
}

New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
$Existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($Existing) {
    try { $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 3 }
    catch { throw "Port $Port is occupied by another service." }
    if ($Health.status -ne 'ok') { throw "Port $Port is not a healthy VisionCortex service." }
    Write-Host "VisionCortex is already running on port $Port."
}
else {
    $QuotedConfig = '"' + $Config + '"'
    $Arguments = @('-m', 'visioncortex', 'serve', '--host', '127.0.0.1', '--port', "$Port", '--config', $QuotedConfig)
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
    Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 120; $Attempt++) {
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
            if ($Health.status -eq 'ok') { $Ready = $true; break }
        }
        catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $Ready) {
        $Details = if (Test-Path -LiteralPath $StderrLog) { (Get-Content -LiteralPath $StderrLog -Tail 30) -join [Environment]::NewLine } else { 'No stderr log.' }
        throw "VisionCortex did not become ready. PID=$($Process.Id). $Details"
    }
}

$Url = "http://127.0.0.1:$Port/#/home"
Write-Host "VisionCortex Web: $Url"
Write-Host 'The UI discovers actual camera/view count from the selected NAS collection.'
if (-not $NoBrowser) { Start-Process $Url }
