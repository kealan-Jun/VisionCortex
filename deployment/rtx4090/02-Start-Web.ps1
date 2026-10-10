param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$PythonPath = '',
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
# Both GPU profiles share the same checked instance ownership protocol.
$SharedEntry = Join-Path $PSScriptRoot '..\rtx4060\02-启动Web.ps1'
& $SharedEntry @PSBoundParameters
