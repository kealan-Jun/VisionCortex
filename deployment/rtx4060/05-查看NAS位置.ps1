$ErrorActionPreference = 'Stop'
$Index = 'Y:\experiment_record_index.csv'
$Archive = 'Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13'
$Continuous = Join-Path $Archive 'Original-Experiment-Videos\Normalized-Continuous'
$ExperimentId = 'exp_20260810_144014_e918b762'

foreach ($path in @($Index, $Archive, $Continuous)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "NAS path is missing: $path" }
}

Write-Host "Index CSV: $Index"
Write-Host "Fixed formal archive: $Archive"
Write-Host "Six continuous videos and clock CSV files: $Continuous"
Write-Host ''
Import-Csv -LiteralPath $Index |
    Where-Object { $_.experiment_id -eq $ExperimentId } |
    ForEach-Object {
        $segments = @($_.rgb_file -split ';' | Where-Object { $_ })
        [pscustomobject]@{
            Camera = $_.camera_key
            IndexView = $_.camera_view
            SegmentCount = $segments.Count
            FirstSegment = $segments[0]
            LastSegment = $segments[-1]
        }
    } |
    Format-Table -AutoSize -Wrap

Write-Host 'Local runtime directories:'
Write-Host '  Input retention: D:\VisionCortexLocal\Input'
Write-Host '  Runtime ledgers: D:\VisionCortexLocal\Runtime'
Write-Host '  YOLO/checkpoint cache: D:\VisionCortexLocal\Cache'
