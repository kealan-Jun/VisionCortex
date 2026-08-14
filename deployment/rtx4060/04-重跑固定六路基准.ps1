param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$BaseUrl = "http://127.0.0.1:$Port"

try {
    $Health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/health" -TimeoutSec 5
}
catch {
    throw "VisionCortex Web is not available at $BaseUrl. Run deployment\rtx4060\02-启动Web.ps1 first. The benchmark is not started."
}

if ($Health.status -ne 'ok') { throw "VisionCortex Web health check failed at $BaseUrl." }
if ($Health.fixed_benchmark.submission_protocol_version -ne 1) {
    throw 'The running Web service uses an older benchmark submission protocol. Stop it with 03-停止Web.ps1 and start the updated Web service before submitting. The benchmark is not started.'
}
if (-not $Health.nas_available) { throw 'The configured NAS archive root is unavailable. The benchmark is not started.' }
if (-not $Health.ark_key_configured) { throw 'ARK_API_KEY is not available to the Web service. The benchmark is not started.' }
if ($Health.fixed_benchmark.archive_name -ne 'Six-View-Three-Hour-Experiment-2026-08-13') {
    throw "The Web service exposes an unexpected fixed benchmark: $($Health.fixed_benchmark.archive_name)"
}

$Submission = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/benchmarks/six-view-three-hour/runs" `
    -TimeoutSec 30

if ([string]::IsNullOrWhiteSpace([string]$Submission.run_id)) {
    throw 'The Web service accepted the request without returning a run_id.'
}
if ($Submission.execution_owner -ne 'visioncortex_web_service' -or -not $Submission.client_process_independent) {
    throw 'The Web service did not confirm durable asynchronous ownership.'
}
if ($Submission.submission_protocol_version -ne 1) {
    throw 'The Web service returned an unexpected submission protocol version.'
}

Write-Host "SUBMITTED_RUN_ID=$($Submission.run_id)"
Write-Host "STATUS_URL=$BaseUrl$($Submission.status_url)"
Write-Host "NAS_STAGING=$($Submission.nas_staging)"
Write-Host "SUBMISSION_RECEIPT=$($Submission.submission_receipt)"
Write-Host 'The submission command is complete. Do not start another run. Monitor with the status URL or NAS pipeline_status.json.'
$Submission | ConvertTo-Json -Depth 6
exit 0
