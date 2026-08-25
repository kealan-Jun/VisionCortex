$ErrorActionPreference = 'Stop'
$Index = 'Y:\experiment_record_index.csv'
$Archive = 'Y:\VisionCortexExperimentArchive\CustomFlow_standard_correct_12_ABCFA_0001--exp_20260810_144014_e918b762'
$ExperimentId = 'exp_20260810_144014_e918b762'

foreach ($path in @($Index, $Archive)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "NAS path is missing: $path" }
}

Write-Host "Index CSV: $Index"
Write-Host "Fixed formal archive: $Archive"
Write-Host 'Original inputs: the rgb_file / frames_file 15-minute NAS segments below.'
Write-Host 'No 82-88 GB local continuous-video copy is required.'
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
Write-Host '  Runtime control/logs only: D:\VisionCortexLocal\Runtime'
Write-Host '  YOLO/checkpoint cache: D:\VisionCortexLocal\Cache'
Write-Host "  Formal output: $Archive"
