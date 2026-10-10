param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [string]$PythonPath = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Import-Module (Join-Path $PSScriptRoot 'Runtime-Config.psm1') -Force
$Site = Get-VisionCortexRuntimeConfiguration -ProjectRoot $ProjectRoot -ConfigPath $ConfigPath -PythonPath $PythonPath
$Registration = $Site.fixed_benchmark
if ($Registration.enabled -ne $true) { throw 'The private site has no enabled benchmark registration.' }
$Archive = Join-Path $Site.archive_root $Registration.archive_name
foreach ($Path in @($Site.index_csv, $Archive)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Configured path is missing: $Path" }
}
Import-Csv -LiteralPath $Site.index_csv |
    Where-Object { $_.experiment_id -eq $Registration.experiment_id } |
    ForEach-Object {
        $Segments = @($_.rgb_file -split ';' | Where-Object { $_ })
        [pscustomobject]@{ Camera = $_.camera_key; IndexView = $_.camera_view; SegmentCount = $Segments.Count }
    } | Format-Table -AutoSize
Write-Host "Runtime: $($Site.runtime_root)"
Write-Host "Cache: $($Site.cache_root)"
Write-Host "Configured archive: $Archive"
