# DEV-20260817-035: RTX 4060 Quality Candidate Acceptance

## Mission

Run exactly one real, cold-start, six-view preprocessing acceptance on the RTX
4060 machine. This is an execution task, not a development task. It validates
that the DEV-032 quality fixes preserve the proven preprocessing speed while
restoring the reviewed five-experiment structure.

This run must stop after deterministic CV evidence selection. It must not call
the multimodal model, consume model tokens, materialize media, generate reports
or PDFs, or promote/replace the accepted formal NAS archive.

## Frozen source

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch for locating the candidate: `codex/dev032-quality-fixes`
- Immutable tag: `quality-candidate-dev035-20260817`
- Required full commit: `1b35f6106bc7a93b08be230ef395b6aa9270674f`
- GitHub coordination location: Issue `#3`

The operator must check out the immutable tag, not an advancing branch.

## Absolute prohibitions

- Do not modify source, configuration, tests, scripts, weights, dependencies,
  drivers, environment variables, thresholds, FPS, batch sizes or decoder
  lanes.
- Do not run pytest, compile checks, dry-runs, model validation, Web/API tests,
  FFmpeg probes outside the real run, or manually constructed samples.
- Do not retry, resume, tune or start a second run after success or failure.
- Do not reuse a prior CV cache identity or checkpoint as a cold-start result.
- Do not call the MLLM or print, inspect or change `ARK_API_KEY`.
- Do not export clips/key frames/key clips, create the daily report/PDF, or
  promote the fixed archive.
- Do not delete, clean, reset, stash, move or overwrite existing user data,
  staging results, cache directories or the accepted formal archive.
- Do not start or test the Web product. This task uses the preprocessing-only
  CLI directly.

## 1. Synchronize and verify the frozen candidate

Run these commands inside the already existing VisionCortex repository. Never
assume a drive letter and never clone a second copy.

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = (& git rev-parse --show-toplevel 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    throw 'Current directory is not inside the existing VisionCortex repository.'
}
Set-Location -LiteralPath $ProjectRoot
$OriginUrl = (& git remote get-url origin).Trim()
if ($OriginUrl -notmatch 'github\.com[/:]kealan-Jun/VisionCortex(?:\.git)?$') {
    throw "Unexpected origin: $OriginUrl"
}
$Dirty = @(git status --porcelain)
if ($Dirty.Count -gt 0) {
    $Dirty
    throw 'Working tree is dirty. Stop without changing user files.'
}
git fetch origin --tags
git switch --detach quality-candidate-dev035-20260817
$RunCommit = (& git rev-parse HEAD).Trim()
if ($RunCommit -ne '1b35f6106bc7a93b08be230ef395b6aa9270674f') {
    throw "Frozen SHA mismatch: $RunCommit"
}
$RemoteCandidate = (& git rev-parse origin/codex/dev032-quality-fixes).Trim()
Write-Host "PROJECT_ROOT=$ProjectRoot"
Write-Host "RUN_COMMIT=$RunCommit"
Write-Host "REMOTE_CANDIDATE=$RemoteCandidate"
git status --short --branch
```

If the working tree is dirty, the tag is missing, or the SHA differs, append
one `INCIDENT` to GitHub Issue #3 and stop. Do not repair the repository.

## 2. Fixed real input and output boundaries

- NAS index: `Y:\experiment_record_index.csv`
- Experiment record: `exp_20260810_144014_e918b762`
- Input: all six views and their fifteen 15-minute MP4/clock-CSV segments
  resolved from the index
- Input mode: `nas_segmented_virtual_timeline`
- Accepted formal archive (read-only for this run):
  `Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13`
- New isolated staging root:
  `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging`
- Configured local cache root: `D:\VisionCortexLocal\Cache`

The run must report six views, fifteen MP4 segments per view, and the associated
clock CSVs. `copied_source_bytes` and `continuous_source_copies_created` must
both be zero. Do not create six local three-hour concatenated videos.

The new code SHA must produce a new cache identity. Before starting, record the
resolved identity and prove that its detection work-unit count is zero. If the
identity already contains computed/reused detection ledgers, report an
`INCIDENT` and stop; do not delete or alter it.

## 3. Read-only machine and process evidence

Record, without changing settings:

```powershell
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber,CsTotalPhysicalMemory
nvidia-smi
Get-SmbConnection | Select-Object ServerName,ShareName,Dialect,NumOpens,Encrypted
Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object Name,InterfaceDescription,LinkSpeed
Get-PSDrive -Name @('Y') | Select-Object Name,Used,Free,Root
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match 'python|ffmpeg' -and $_.CommandLine -match 'labvision|run-fixed-benchmark' } |
  Select-Object ProcessId,Name,CommandLine
```

If another VisionCortex benchmark is active, report it and stop. Do not kill it.

Append one `ACK` to GitHub Issue #3 containing the task ID, frozen SHA, absolute
repository path, clean-tree proof, input inventory, cold cache proof and machine
evidence. Do not create another coordination document.

## 4. Execute exactly once

Use the repository's existing virtual environment. Do not install or update
packages. Do not wrap the command in a timeout and do not invoke the Web/API.

```powershell
Set-Location -LiteralPath $ProjectRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Existing project interpreter not found: $Python"
}
& $Python -m labvision_evidence.cli run-fixed-benchmark `
  --config (Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml') `
  --preprocessing-only
$PipelineExitCode = $LASTEXITCODE
Write-Host "PIPELINE_EXIT_CODE=$PipelineExitCode"
```

This is the sole authorized run. A nonzero exit code consumes the one-shot
opportunity. Preserve the resulting staging/cache exactly and do not retry.

During the run append a compact `HEARTBEAT` to GitHub Issue #3 on material stage
changes and approximately every five minutes. Each heartbeat must cite the
current staging JSON and include stage/work-unit progress, computed/reused,
GPU compute and memory, NVDEC/NVENC, temperature, power, CPU, RAM, host/process
network receive throughput, queue wait and actual batch fill. Use `unavailable`
when a metric cannot be proven; never estimate it.

## 5. Acceptance gates

The final result is `COMPLETED` only when every gate below is proven from the
new staging ledgers.

### Cold-start and performance

- preprocessing time is at most `1200.0s`;
- model/API calls and input/output/total Token are all zero;
- all motion/fine work required by the candidate is newly computed;
- copied source bytes are zero;
- no formal archive promotion or media/report/PDF materialization occurs;
- full phase timing and resource telemetry cover the whole preprocessing run.

### Reviewed experiment structure

- exactly five formal experiment groups;
- all reviewed start and end boundary errors are within `±8.0s`;
- G1: independent, one atomic experiment;
- G2: independent, one atomic experiment;
- G3: independent, one atomic experiment;
- G4: continuous, two atomic experiments;
- G5: independent, one atomic experiment;
- every formal group has valid first-person and third-person evidence;
- quarantined/single-view-only events have zero formal membership leakage;
- at least 80 selected key events.

### DEV-032 quality fixes

Prove the following rule receipts from `audit_layer.json`,
`boundary_precheck.json`, `progressive_fine_scan.json` and
`key_material_selection_preview.json`:

- `QF1-SINGLE-VIEW-BOUNDARY-CONTEXT`: valid same-action, shared non-hand-object
  leading context may extend G1's boundary but never becomes a formal event;
- `QF2-STABLE-OBJECT-IDENTITY`: G3 is not connected by class-name overlap alone;
  a continuity edge requires same-view stable track identity;
- `QF3-TEMPORAL-CLUSTER-COMPLETENESS`: every unresolved first-person temporal
  cluster is either closed by a third-person view or causes quality failure;
- `QF4-MOVEMENT-CORROBORATION`: the balance movement near G5 cannot open a
  formal experiment without a later dual-role, non-movement action on the same
  non-hand object;
- `QF5-NON-HAND-OBJECT-DEDUPLICATION`: generic `hand`/`gloved_hand` overlap alone
  cannot deduplicate distinct actions.

Every receipt must use schema `visioncortex-decision-receipt/1.0.0`, have a
deterministic `decision_id`, rule/version, verdict, reason codes, facts,
thresholds, subject IDs and evidence references.

### Timestamp-ledger hygiene

- adjacent physical-segment overlap timestamps are removed at write time;
- `duplicate_timestamp_frames_removed` and per-view counts are reported;
- final merged detection ledgers contain no duplicate local timestamp per view;
- timestamp cleanup must not be cited as a cause of G1/G3/G4/G5 repair.

## 6. Final GitHub result

Append exactly one final `COMPLETED` or `INCIDENT` to GitHub Issue #3. Include:

- task ID, run ID, exit code, frozen tag/SHA and final clean-tree proof;
- total and phase wall times;
- five-group/boundary/continuity/atomic-experiment table;
- selected-event count and per-action count;
- all five QF rule receipt counts and representative decision IDs;
- timestamp duplicate-removal and final uniqueness counts by view;
- motion/fine work units, frames, actual batch statistics and queue/decode wait;
- GPU/VRAM/NVDEC/NVENC/CPU/RAM/power/temperature/NAS throughput percentiles;
- input copy bytes, cache identity and computed/reused counts;
- model/API calls and input/output/total Token (all expected to be zero);
- exact NAS evidence paths;
- proof that the accepted formal archive was not changed;
- final task-process and FFmpeg-process count.

If any gate fails, use `INCIDENT`, preserve all evidence and stop. Do not retry,
change code, tune parameters, run tests, or launch the full pipeline.
