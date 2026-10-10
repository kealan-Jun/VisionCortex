param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$PythonPath = '',
    [ValidateRange(1, 65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$SharedEntry = Join-Path $PSScriptRoot '..\rtx4060\03-停止Web.ps1'
& $SharedEntry @PSBoundParameters
