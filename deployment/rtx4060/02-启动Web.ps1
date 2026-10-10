param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$PythonPath = '',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $PSScriptRoot 'Runtime-Config.psm1') -Force
$Site = Get-VisionCortexRuntimeConfiguration -ProjectRoot $ProjectRoot -ConfigPath $ConfigPath -PythonPath $PythonPath
$OwnerFile = Join-Path $Site.runtime_root 'visioncortex-web.owner.json'
$StdoutLog = Join-Path $Site.runtime_root 'visioncortex-web.stdout.log'
$StderrLog = Join-Path $Site.runtime_root 'visioncortex-web.stderr.log'
$BaseUrl = "http://127.0.0.1:$Port"
$existing = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($existing.Count -gt 0) {
    if (-not (Test-Path -LiteralPath $OwnerFile -PathType Leaf)) {
        throw 'This port has no ownership receipt for the configured checkout; refusing to reuse it.'
    }
    $Owner = Get-Content -LiteralPath $OwnerFile -Raw | ConvertFrom-Json
    $null = Assert-VisionCortexProcessOwnership -Configuration $Site -Owner $Owner -Port $Port
    $Health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 3
    $Automation = Invoke-RestMethod -Uri "$BaseUrl/health/automation" -TimeoutSec 3
    Assert-VisionCortexHealthOwnership -Configuration $Site -Owner $Owner -Health $Health -Automation $Automation
    Write-Host 'The matching owned service is already running.'
}
else {
    if (Test-Path -LiteralPath $OwnerFile) {
        throw 'A prior ownership receipt exists; verify its process before replacing it.'
    }
    New-Item -ItemType Directory -Path $Site.runtime_root -Force | Out-Null
    $env:VISIONCORTEX_CONFIG = $Site.config_path
    $PreviousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
        $Arguments = @('-m', 'visioncortex', 'serve', '--host', '127.0.0.1', '--port', "$Port", '--config', ('"' + $Site.config_path + '"'))
        $Process = Start-Process -FilePath $Site.python -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
    }
    finally {
        if ($null -eq $PreviousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
        else { $env:PYTHONPATH = $PreviousPythonPath }
    }
    $Identity = Get-CimInstance Win32_Process -Filter "ProcessId = $($Process.Id)" -ErrorAction Stop
    if ($null -eq $Identity) { throw 'The spawned service exited before its process identity could be recorded.' }
    $Owner = [pscustomobject]@{
        schema_version = 'visioncortex-web-owner/1'; pid = $Process.Id; port = $Port
        created_utc = $Identity.CreationDate.ToUniversalTime().ToString('o')
        project_root = $Site.project_root; python = $Site.python; config_path = $Site.config_path
        settings_sha256 = $Site.settings_sha256; archive_root = $Site.archive_root
    }
    $Owner | ConvertTo-Json | Set-Content -LiteralPath $OwnerFile -Encoding utf8
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 120; $Attempt++) {
        try {
            $null = Assert-VisionCortexProcessOwnership -Configuration $Site -Owner $Owner -Port $Port
            $Health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 2
            $Automation = Invoke-RestMethod -Uri "$BaseUrl/health/automation" -TimeoutSec 2
            Assert-VisionCortexHealthOwnership -Configuration $Site -Owner $Owner -Health $Health -Automation $Automation
            $Ready = $true
            break
        }
        catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $Ready) {
        throw "The owned service did not become ready; inspect its preserved owner receipt and local log: $StderrLog"
    }
}
$Url = "$BaseUrl/#/home"
Write-Host "VisionCortex Web: $Url"
if (-not $NoBrowser) { Start-Process $Url }
