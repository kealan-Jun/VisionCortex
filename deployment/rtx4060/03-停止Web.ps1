param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$PythonPath = '',
    [ValidateRange(1, 65535)][int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $PSScriptRoot 'Runtime-Config.psm1') -Force
$Site = Get-VisionCortexRuntimeConfiguration -ProjectRoot $ProjectRoot -ConfigPath $ConfigPath -PythonPath $PythonPath
$OwnerFile = Join-Path $Site.runtime_root 'visioncortex-web.owner.json'
if (-not (Test-Path -LiteralPath $OwnerFile -PathType Leaf)) {
    throw 'No service ownership receipt exists for this checkout; refusing to stop any process.'
}
$Owner = Get-Content -LiteralPath $OwnerFile -Raw | ConvertFrom-Json
$null = Assert-VisionCortexProcessOwnership -Configuration $Site -Owner $Owner -Port $Port
$BaseUrl = "http://127.0.0.1:$Port"
$Health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 3
$Automation = Invoke-RestMethod -Uri "$BaseUrl/health/automation" -TimeoutSec 3
Assert-VisionCortexHealthOwnership -Configuration $Site -Owner $Owner -Health $Health -Automation $Automation
Assert-VisionCortexShutdownSafe -Health $Health
# Recheck identity after readiness inspection; never act on an unchecked PID.
$null = Assert-VisionCortexProcessOwnership -Configuration $Site -Owner $Owner -Port $Port
Stop-Process -Id $Owner.pid -ErrorAction Stop
Remove-Item -LiteralPath $OwnerFile -Force
Write-Host "Stopped the verified idle service, PID=$($Owner.pid)."
