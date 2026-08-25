# DEV-20260819-047 — Module-Based Runtime Quality-Ledger Replay

## Objective

Replay the retained DEV-041 and DEV-042 JSON quality ledgers through the
repository virtual environment after replacing the failed inline Python
runtime probe with a versioned Python module. DEV-044, DEV-045 and DEV-046
remain closed incidents and must not be retried.

This is a new read-only diagnostic authorization. It does not consume a real
production one-shot.

## Immutable gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch: `codex/post-dev043-deterministic-quality`
- Frozen SHA: the complete SHA in the unique `DEV-20260819-047 RELEASE`
- Coordination: GitHub Issue #3 only

Fetch and check out the exact RELEASE SHA in detached-HEAD mode. Prove local
HEAD, remote branch and RELEASE equality plus a clean worktree before running
the script.

## Corrected execution design

Use only:

- `deployment\rtx4060\05-Replay-Quality-Ledgers.ps1`;
- `<repository>\.venv\Scripts\python.exe`;
- `python -B -m labvision_evidence.runtime_preflight`;
- `python -B -m labvision_evidence.cli replay-quality-ledger`.

There is no Python source string and no `python -c` invocation in this release.
PowerShell passes only ordinary module names and path arguments. The formal
runtime module imports all required dependencies, the full CLI and the replay
module, proves the package source belongs to the frozen checkout, and emits a
single JSON receipt before any ledger read.

## Exact command

Run once from the frozen repository root in a fresh PowerShell child process:

```powershell
$ErrorActionPreference = 'Stop'
$visionCortexReplayScript = (Resolve-Path -LiteralPath '.\deployment\rtx4060\05-Replay-Quality-Ledgers.ps1').Path
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $visionCortexReplayScript
if ($LASTEXITCODE -ne 0) {
    throw "DEV-047 quality-ledger replay failed: $LASTEXITCODE"
}
```

Do not change parameters or redirect output. On any nonzero exit, stop and post
one INCIDENT. Do not install packages, alter the environment, edit the script,
retry a command or launch a real collection.

## Required evidence

Report:

1. runtime preflight JSON: Python executable/version, package source and all
   dependency versions;
2. DEV-041 and DEV-042 `result_sha256`, evidence grade, counts and complete
   group structures;
3. every legacy QF1 boundary invariant requiring an explicit receipt;
4. independent `QF2-STABLE-OBJECT-IDENTITY` and
   `QF2-QUARANTINED-CONTEXT-CHAIN` verdicts and failed criteria;
5. whether DEV-042 G3 `+11.222s` and missing G4 are explained, each graded
   `PROVEN`, `PARTIAL_EVIDENCE` or `NOT_PROVEN`;
6. proof that quarantined context does not enter formal membership or selected
   key events;
7. MP4/CSV reads, video operations, inference/model calls and Token all zero.

Legacy runs lack exact persisted `boundary_candidates`; both replay results
must remain `degraded_legacy_ledger` and cannot be described as production
acceptance.

## Forbidden actions

- no MP4, timestamp CSV, image, PDF, SQLite or semantic-cache payload read;
- no pipeline launch, FFmpeg, OpenCV video operation, YOLO, TensorRT or MLLM;
- no Token use, retry, resume, tuning or real one-shot consumption;
- no dependency install/update, system Python, inline `python -c`, Web/API
  probe or development test;
- no code, config, model, cache, staging, registry or archive modification;
- no new NAS file and no commit, push, merge or tag.

The only external write is one final Issue #3 comment.

## Final response

Post exactly one `DEV-20260819-047 EVIDENCE RESULT` or one
`DEV-20260819-047 INCIDENT` containing frozen identity, runtime receipt, replay
comparison, evidence limitations, zero-mutation proof, clean worktree and zero
relevant VisionCortex/FFmpeg processes.

Do not start real video processing. A separate frozen task will authorize the
three real datasets after the replay conclusion is reviewed.
