function Resolve-ExactReplayArchive {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string[]]$ArchiveRoots
    )

    $attempts = @()
    foreach ($archiveRoot in $ArchiveRoots) {
        if ([string]::IsNullOrWhiteSpace($archiveRoot)) {
            continue
        }
        $datasetRelativePath = Split-Path -Path $RelativePath -Parent
        $candidate = Join-Path -Path $archiveRoot -ChildPath $RelativePath
        $datasetRoot = Join-Path -Path $archiveRoot -ChildPath $datasetRelativePath
        $archiveRootVisible = Test-Path -LiteralPath $archiveRoot -PathType Container -ErrorAction SilentlyContinue
        $datasetRootVisible = Test-Path -LiteralPath $datasetRoot -PathType Container -ErrorAction SilentlyContinue
        $candidateVisible = Test-Path -LiteralPath $candidate -PathType Container -ErrorAction SilentlyContinue
        $attempts += [ordered]@{
            archive_root = $archiveRoot
            archive_root_visible = [bool]$archiveRootVisible
            dataset_staging_root = $datasetRoot
            dataset_staging_root_visible = [bool]$datasetRootVisible
            exact_run_root = $candidate
            exact_run_root_visible = [bool]$candidateVisible
        }
        if ($candidateVisible) {
            $resolved = (Resolve-Path -LiteralPath $candidate).Path
            $expectedRunId = Split-Path -Path $RelativePath -Leaf
            if ((Split-Path -Path $resolved -Leaf) -ne $expectedRunId) {
                throw "$Label replay root identity mismatch: expected $expectedRunId, resolved $resolved"
            }
            return [pscustomobject]@{
                label = $Label
                run_id = $expectedRunId
                resolved_archive = $resolved
                selected_archive_root = $archiveRoot
                path_resolution_attempts = $attempts
            }
        }
    }
    $diagnosis = $attempts | ConvertTo-Json -Depth 5 -Compress
    throw "$Label exact retained replay root is unavailable; bounded path diagnosis: $diagnosis"
}

Export-ModuleMember -Function Resolve-ExactReplayArchive
