function Get-VisionCortexNasDiagnosis {
    param(
        [Parameter(Mandatory = $true)][int]$ActiveAdapterCount,
        [Parameter(Mandatory = $true)][bool]$Tcp445Succeeded,
        [Parameter(Mandatory = $true)][bool]$CanonicalShareVisible,
        [Parameter(Mandatory = $true)][bool]$CanonicalArchiveVisible,
        [Parameter(Mandatory = $true)][bool]$MappedArchiveVisible,
        [Parameter(Mandatory = $true)][bool]$Dev041Visible,
        [Parameter(Mandatory = $true)][bool]$Dev042Visible
    )

    if ($Dev041Visible -and $Dev042Visible) {
        return [pscustomobject]@{
            code = 'retained_staging_reachable'
            replay_permitted_by_connectivity = $true
            explanation = 'Both exact retained run roots are visible through at least one authorized NAS path.'
        }
    }
    if ($CanonicalArchiveVisible -or $MappedArchiveVisible) {
        return [pscustomobject]@{
            code = 'exact_staging_unavailable'
            replay_permitted_by_connectivity = $false
            explanation = 'The archive root is visible, but one or both exact retained run roots are unavailable.'
        }
    }
    if ($ActiveAdapterCount -eq 0) {
        return [pscustomobject]@{
            code = 'local_network_adapter_unavailable'
            replay_permitted_by_connectivity = $false
            explanation = 'No active network adapter was observed.'
        }
    }
    if (-not $Tcp445Succeeded) {
        return [pscustomobject]@{
            code = 'nas_host_or_smb_port_unreachable'
            replay_permitted_by_connectivity = $false
            explanation = 'The NAS SMB endpoint did not accept a TCP 445 connection.'
        }
    }
    if (-not $CanonicalShareVisible) {
        return [pscustomobject]@{
            code = 'smb_share_session_or_authorization_unavailable'
            replay_permitted_by_connectivity = $false
            explanation = 'TCP 445 is reachable, but the canonical SMB share is not visible.'
        }
    }
    return [pscustomobject]@{
        code = 'archive_directory_unavailable'
        replay_permitted_by_connectivity = $false
        explanation = 'The canonical SMB share is visible, but the VisionCortex archive directory is not.'
    }
}

Export-ModuleMember -Function Get-VisionCortexNasDiagnosis
