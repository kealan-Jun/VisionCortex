# DEV-20260819-050 — Read-Only NAS/SMB Diagnosis

## Objective

Classify why the RTX 4060 host cannot currently see either the canonical UNC
VisionCortex archive root or its mapped `Y:` compatibility root. If and only if
both exact retained DEV-041 and DEV-042 run roots are visible at diagnosis
time, execute the already bounded JSON/YAML inspections and deterministic
replays under this new authorization.

DEV-049 remains a closed incident and must not be retried. DEV-050 does not
authorize real video processing.

## Immutable gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch: `codex/post-dev043-deterministic-quality`
- Frozen SHA: the complete SHA in the unique `DEV-20260819-050 RELEASE`
- Coordination: GitHub Issue #3 only

Fetch and check out the exact RELEASE SHA in detached-HEAD mode. Prove local
HEAD, remote branch and RELEASE equality plus a clean worktree.

## Authorized diagnostic

Run once from the frozen repository root:

```powershell
$ErrorActionPreference = 'Stop'
$diagnosticScript = (Resolve-Path -LiteralPath '.\deployment\rtx4060\06-Diagnose-Nas-Connectivity.ps1').Path
$diagnosticJson = & powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $diagnosticScript
if ($LASTEXITCODE -ne 0) {
    throw "DEV-050 NAS diagnosis failed: $LASTEXITCODE"
}
$diagnosticText = $diagnosticJson -join [Environment]::NewLine
$diagnosticText
$diagnostic = $diagnosticText | ConvertFrom-Json
if ($diagnostic.diagnosis.replay_permitted_by_connectivity) {
    $replayScript = (Resolve-Path -LiteralPath '.\deployment\rtx4060\05-Replay-Quality-Ledgers.ps1').Path
    & powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $replayScript
    if ($LASTEXITCODE -ne 0) {
        throw "DEV-050 bounded ledger replay failed: $LASTEXITCODE"
    }
}
```

Do not alter parameters or run a second attempt. The diagnostic is allowed to
read only host network/SMB state and test the fixed NAS/share/archive/run paths.
It must not enumerate directories, mount/remount a share, read credentials or
repair the connection.

## Required evidence

Always report:

1. active adapter names/link speeds and selected TCP route fields;
2. `LanmanWorkstation` service status;
3. TCP 445 result for `192.168.66.149`;
4. sanitized `Y:` SMB mapping and existing NAS connection records;
5. canonical share/archive, mapped archive and both exact run visibility;
6. the emitted classification code and explanation;
7. proof of zero enumeration, zero writes and zero credential access.

Classification must be one of:

- `local_network_adapter_unavailable`;
- `nas_host_or_smb_port_unreachable`;
- `smb_share_session_or_authorization_unavailable`;
- `archive_directory_unavailable`;
- `exact_staging_unavailable`;
- `retained_staging_reachable`.

Only `retained_staging_reachable` permits the second read-only replay command.
If replay runs, include both ledger-input receipts, result hashes, groups,
QF1/QF2 decisions and the G3/G4 evidence grade. Otherwise, replay attempts must
remain zero.

## Forbidden actions

- no SMB mapping creation/removal, reconnect, credential prompt or credential
  read/write;
- no NAS enumeration, recursive search or broad file listing;
- no MP4, CSV, image, PDF, SQLite or semantic-cache read;
- no real pipeline, FFmpeg, OpenCV media operation, YOLO, TensorRT or MLLM;
- no Token use, dependency/config/model/cache/staging/archive modification;
- no Web/product API probe, development test, retry, commit, push, merge or tag.

The only external write is one final GitHub Issue #3 comment.

## Final response

Post exactly one `DEV-20260819-050 EVIDENCE RESULT` when the diagnostic
completed, even if the result proves external unreachability. Use INCIDENT only
when the diagnostic script itself fails. Include frozen identity, diagnostic
JSON, conditional replay status, zero-mutation proof, clean worktree and zero
relevant VisionCortex/FFmpeg processes.
