$ErrorActionPreference = 'Stop'
$PidFile = 'D:\VisionCortex4090\Runtime\visioncortex-web.pid'
if (-not (Test-Path -LiteralPath $PidFile -PathType Leaf)) {
    Write-Host 'No recorded VisionCortex Web process.'
    exit 0
}
$ProcessId = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
$Process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
if ($null -eq $Process) {
    Remove-Item -LiteralPath $PidFile -Force
    Write-Host 'The process no longer exists; the PID file was removed.'
    exit 0
}
if ($Process.CommandLine -notmatch 'visioncortex' -or $Process.CommandLine -notmatch 'serve') {
    throw "PID $ProcessId is not a VisionCortex Web process; refusing to stop it."
}
Stop-Process -Id $ProcessId
Remove-Item -LiteralPath $PidFile -Force
Write-Host "Stopped VisionCortex Web, PID=$ProcessId."
