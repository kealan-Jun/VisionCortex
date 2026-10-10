@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-VisionCortex.ps1" %*
if errorlevel 1 (
  echo.
  echo VisionCortex did not start. Please read the message above.
  exit /b 1
)
