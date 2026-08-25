param(
    [string]$NasHost = '192.168.66.149',
    [string]$NasShare = 'video_database',
    [string]$ArchiveDirectory = 'VisionCortexExperimentArchive',
    [string]$MappedArchiveRoot = 'Y:\VisionCortexExperimentArchive'
)

$ErrorActionPreference = 'Stop'
$classificationModule = Join-Path $PSScriptRoot 'Nas-Diagnostic-Classification.psm1'
if (-not (Test-Path -LiteralPath $classificationModule -PathType Leaf)) {
    throw "NAS diagnostic classification module is missing: $classificationModule"
}
Import-Module -Name $classificationModule -Force

$canonicalShareRoot = "\\$NasHost\$NasShare"
$canonicalArchiveRoot = Join-Path -Path $canonicalShareRoot -ChildPath $ArchiveDirectory
$dev041RelativePath = '.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-154357-29d1'
$dev042RelativePath = '.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-173813-fc76'

$workstation = Get-Service -Name 'LanmanWorkstation' -ErrorAction SilentlyContinue
$activeAdapters = @(
    Get-NetAdapter -ErrorAction SilentlyContinue |
        Where-Object { $_.Status -eq 'Up' } |
        Select-Object Name, InterfaceDescription, LinkSpeed, MediaConnectionState
)
$tcpProbe = Test-NetConnection `
    -ComputerName $NasHost `
    -Port 445 `
    -InformationLevel Detailed `
    -WarningAction SilentlyContinue
$tcp445Succeeded = [bool]$tcpProbe.TcpTestSucceeded

$mappedDrive = @(
    Get-SmbMapping -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPath -eq 'Y:' } |
        ForEach-Object {
            [pscustomobject]@{
                local_path = $_.LocalPath
                remote_path = $_.RemotePath
                status = [string]$_.Status
            }
        }
)
$nasConnections = @(
    Get-SmbConnection -ErrorAction SilentlyContinue |
        Where-Object { $_.ServerName -eq $NasHost } |
        ForEach-Object {
            [pscustomobject]@{
                server_name = $_.ServerName
                share_name = $_.ShareName
                dialect = [string]$_.Dialect
                num_opens = $_.NumOpens
                encrypted = [bool]$_.Encrypted
            }
        }
)

$canonicalShareVisible = $false
$canonicalArchiveVisible = $false
if ($tcp445Succeeded) {
    $canonicalShareVisible = [bool](
        Test-Path -LiteralPath $canonicalShareRoot -PathType Container -ErrorAction SilentlyContinue
    )
    if ($canonicalShareVisible) {
        $canonicalArchiveVisible = [bool](
            Test-Path -LiteralPath $canonicalArchiveRoot -PathType Container -ErrorAction SilentlyContinue
        )
    }
}
$mappedArchiveVisible = [bool](
    Test-Path -LiteralPath $MappedArchiveRoot -PathType Container -ErrorAction SilentlyContinue
)

$canonicalDev041 = Join-Path -Path $canonicalArchiveRoot -ChildPath $dev041RelativePath
$canonicalDev042 = Join-Path -Path $canonicalArchiveRoot -ChildPath $dev042RelativePath
$mappedDev041 = Join-Path -Path $MappedArchiveRoot -ChildPath $dev041RelativePath
$mappedDev042 = Join-Path -Path $MappedArchiveRoot -ChildPath $dev042RelativePath
$canonicalDev041Visible = $false
$canonicalDev042Visible = $false
$mappedDev041Visible = $false
$mappedDev042Visible = $false
if ($canonicalArchiveVisible) {
    $canonicalDev041Visible = [bool](
        Test-Path -LiteralPath $canonicalDev041 -PathType Container -ErrorAction SilentlyContinue
    )
    $canonicalDev042Visible = [bool](
        Test-Path -LiteralPath $canonicalDev042 -PathType Container -ErrorAction SilentlyContinue
    )
}
if ($mappedArchiveVisible) {
    $mappedDev041Visible = [bool](
        Test-Path -LiteralPath $mappedDev041 -PathType Container -ErrorAction SilentlyContinue
    )
    $mappedDev042Visible = [bool](
        Test-Path -LiteralPath $mappedDev042 -PathType Container -ErrorAction SilentlyContinue
    )
}
$dev041Visible = $canonicalDev041Visible -or $mappedDev041Visible
$dev042Visible = $canonicalDev042Visible -or $mappedDev042Visible
$diagnosis = Get-VisionCortexNasDiagnosis `
    -ActiveAdapterCount $activeAdapters.Count `
    -Tcp445Succeeded $tcp445Succeeded `
    -CanonicalShareVisible $canonicalShareVisible `
    -CanonicalArchiveVisible $canonicalArchiveVisible `
    -MappedArchiveVisible $mappedArchiveVisible `
    -Dev041Visible $dev041Visible `
    -Dev042Visible $dev042Visible

$result = [ordered]@{
    schema_version = 'visioncortex-nas-connectivity-diagnosis/1.0.0'
    status = 'completed'
    observed_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    target = [ordered]@{
        host = $NasHost
        tcp_port = 445
        canonical_share_root = $canonicalShareRoot
        canonical_archive_root = $canonicalArchiveRoot
        mapped_archive_root = $MappedArchiveRoot
    }
    local_client = [ordered]@{
        workstation_service_status = if ($null -eq $workstation) { 'not_found' } else { [string]$workstation.Status }
        active_adapter_count = $activeAdapters.Count
        active_adapters = $activeAdapters
    }
    network = [ordered]@{
        tcp_445_succeeded = $tcp445Succeeded
        remote_address = [string]$tcpProbe.RemoteAddress
        interface_alias = [string]$tcpProbe.InterfaceAlias
        source_address = [string]$tcpProbe.SourceAddress
        next_hop = [string]$tcpProbe.NetRoute.NextHop
    }
    smb = [ordered]@{
        mapped_drive_records = $mappedDrive
        existing_connection_records = $nasConnections
    }
    visibility = [ordered]@{
        canonical_share_root = [ordered]@{
            path = $canonicalShareRoot
            tested = $tcp445Succeeded
            visible = $canonicalShareVisible
        }
        canonical_archive_root = [ordered]@{
            path = $canonicalArchiveRoot
            tested = $canonicalShareVisible
            visible = $canonicalArchiveVisible
        }
        mapped_archive_root = [ordered]@{
            path = $MappedArchiveRoot
            tested = $true
            visible = $mappedArchiveVisible
        }
        dev041 = [ordered]@{
            run_id = 'collection-20260818-154357-29d1'
            canonical_path = $canonicalDev041
            mapped_path = $mappedDev041
            visible = $dev041Visible
        }
        dev042 = [ordered]@{
            run_id = 'collection-20260818-173813-fc76'
            canonical_path = $canonicalDev042
            mapped_path = $mappedDev042
            visible = $dev042Visible
        }
    }
    diagnosis = $diagnosis
    source_policy = [ordered]@{
        directory_enumerations = 0
        recursive_searches = 0
        nas_source_files_opened = 0
        video_operations = 0
        model_calls = 0
        token_usage = 0
        filesystem_writes = 0
        smb_mapping_changes = 0
        credential_reads_or_writes = 0
    }
}

$result | ConvertTo-Json -Depth 8
