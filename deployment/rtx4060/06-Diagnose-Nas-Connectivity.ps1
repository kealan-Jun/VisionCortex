param(
    [Parameter(Mandatory = $true)][string]$NasHost,
    [Parameter(Mandatory = $true)][string]$NasShare,
    [Parameter(Mandatory = $true)][string]$ArchiveDirectory,
    [Parameter(Mandatory = $true)][string]$FirstRunRelativePath,
    [Parameter(Mandatory = $true)][string]$SecondRunRelativePath,
    [string]$MappedArchiveRoot = ''
)

$ErrorActionPreference = 'Stop'
$classificationModule = Join-Path $PSScriptRoot 'Nas-Diagnostic-Classification.psm1'
if (-not (Test-Path -LiteralPath $classificationModule -PathType Leaf)) {
    throw "NAS diagnostic classification module is missing: $classificationModule"
}
Import-Module -Name $classificationModule -Force

if ($NasHost -match '[\\/]' -or $NasShare -match '[\\/]' -or
    [string]::IsNullOrWhiteSpace($NasHost) -or [string]::IsNullOrWhiteSpace($NasShare)) {
    throw 'Supply one NAS host and one share name explicitly.'
}
foreach ($relative in @($ArchiveDirectory, $FirstRunRelativePath, $SecondRunRelativePath)) {
    if ([string]::IsNullOrWhiteSpace($relative) -or [IO.Path]::IsPathRooted($relative) -or
        (($relative -split '[\\/]') -contains '..')) {
        throw 'Diagnostic run paths must stay relative to the explicit archive root.'
    }
}
$canonicalShareRoot = "\\$NasHost\$NasShare"
$canonicalArchiveRoot = Join-Path -Path $canonicalShareRoot -ChildPath $ArchiveDirectory

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
        Where-Object { -not [string]::IsNullOrWhiteSpace($MappedArchiveRoot) -and $_.LocalPath -eq [IO.Path]::GetPathRoot($MappedArchiveRoot).TrimEnd('\') } |
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
$mappedArchiveVisible = -not [string]::IsNullOrWhiteSpace($MappedArchiveRoot) -and [bool](Test-Path -LiteralPath $MappedArchiveRoot -PathType Container -ErrorAction SilentlyContinue)

$canonicalFirstRun = Join-Path -Path $canonicalArchiveRoot -ChildPath $FirstRunRelativePath
$canonicalSecondRun = Join-Path -Path $canonicalArchiveRoot -ChildPath $SecondRunRelativePath
$mappedFirstRun = if ($mappedArchiveVisible) { Join-Path -Path $MappedArchiveRoot -ChildPath $FirstRunRelativePath } else { $null }
$mappedSecondRun = if ($mappedArchiveVisible) { Join-Path -Path $MappedArchiveRoot -ChildPath $SecondRunRelativePath } else { $null }
$canonicalFirstRunVisible = $false
$canonicalSecondRunVisible = $false
$mappedFirstRunVisible = $false
$mappedSecondRunVisible = $false
if ($canonicalArchiveVisible) {
    $canonicalFirstRunVisible = [bool](
        Test-Path -LiteralPath $canonicalFirstRun -PathType Container -ErrorAction SilentlyContinue
    )
    $canonicalSecondRunVisible = [bool](
        Test-Path -LiteralPath $canonicalSecondRun -PathType Container -ErrorAction SilentlyContinue
    )
}
if ($mappedArchiveVisible) {
    $mappedFirstRunVisible = [bool](
        Test-Path -LiteralPath $mappedFirstRun -PathType Container -ErrorAction SilentlyContinue
    )
    $mappedSecondRunVisible = [bool](
        Test-Path -LiteralPath $mappedSecondRun -PathType Container -ErrorAction SilentlyContinue
    )
}
$firstRunVisible = $canonicalFirstRunVisible -or $mappedFirstRunVisible
$secondRunVisible = $canonicalSecondRunVisible -or $mappedSecondRunVisible
$diagnosis = Get-VisionCortexNasDiagnosis `
    -ActiveAdapterCount $activeAdapters.Count `
    -Tcp445Succeeded $tcp445Succeeded `
    -CanonicalShareVisible $canonicalShareVisible `
    -CanonicalArchiveVisible $canonicalArchiveVisible `
    -MappedArchiveVisible $mappedArchiveVisible `
    -FirstRunVisible $firstRunVisible `
    -SecondRunVisible $secondRunVisible

$result = [ordered]@{
    schema_version = 'visioncortex-nas-connectivity-diagnosis/2.0.0'
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
            tested = -not [string]::IsNullOrWhiteSpace($MappedArchiveRoot)
            visible = $mappedArchiveVisible
        }
        first_run = [ordered]@{
            run_id = Split-Path -Path $FirstRunRelativePath -Leaf
            canonical_path = $canonicalFirstRun
            mapped_path = $mappedFirstRun
            visible = $firstRunVisible
        }
        second_run = [ordered]@{
            run_id = Split-Path -Path $SecondRunRelativePath -Leaf
            canonical_path = $canonicalSecondRun
            mapped_path = $mappedSecondRun
            visible = $secondRunVisible
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
