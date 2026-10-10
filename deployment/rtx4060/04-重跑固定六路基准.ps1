param(
    [ValidateRange(1, 65535)][int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$BaseUrl = "http://127.0.0.1:$Port"
try { $Health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/health" -TimeoutSec 5 }
catch { throw 'The configured Web service is unavailable; no benchmark was started.' }
if ($Health.status -ne 'ok' -or $Health.analysis_ready -ne $true) { throw 'The service has not passed analysis preflight.' }
$Registration = $Health.fixed_benchmark
if ($Registration.enabled -ne $true -or $Registration.submission_protocol_version -ne 1 -or
    [string]::IsNullOrWhiteSpace([string]$Registration.experiment_id) -or
    [string]::IsNullOrWhiteSpace([string]$Registration.archive_name)) {
    throw 'The service has no enabled private benchmark registration; no benchmark was started.'
}
if (-not $Health.nas_available) { throw 'The configured NAS archive is unavailable.' }
if ($Health.mllm_enabled -and -not $Health.mllm_key_configured) { throw 'The selected model provider has no configured credential.' }
$Submission = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/benchmarks/six-view-three-hour/runs" -TimeoutSec 30
if ([string]::IsNullOrWhiteSpace([string]$Submission.run_id) -or
    $Submission.execution_owner -ne 'visioncortex_web_service' -or
    -not $Submission.client_process_independent -or $Submission.submission_protocol_version -ne 1) {
    throw 'The service did not confirm durable asynchronous submission ownership.'
}
Write-Host "SUBMITTED_RUN_ID=$($Submission.run_id)"
Write-Host "STATUS_URL=$BaseUrl$($Submission.status_url)"
Write-Host 'Submission completed. Monitor the returned status URL before starting another run.'
$Submission | ConvertTo-Json -Depth 6
