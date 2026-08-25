# DEV-20260819-048 — Failed-Staging Ledger Input Replay

## Objective

Inspect both retained failed-staging ledger roots first, then replay DEV-041
and DEV-042 through deterministic quality rules. This closes the input-contract
defect exposed by DEV-047: a run stopped at `candidate_audit` has the canonical
early `Input-Manifests/manifest.yaml`, but has not yet reached the later package
stage that writes `run_manifest.json`.

DEV-044 through DEV-047 remain closed incidents and must not be retried. This
is a new read-only diagnostic authorization and does not consume a real video
one-shot.

## Immutable gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch: `codex/post-dev043-deterministic-quality`
- Frozen SHA: the complete SHA in the unique `DEV-20260819-048 RELEASE`
- Coordination: GitHub Issue #3 only

Fetch and check out the exact RELEASE SHA in detached-HEAD mode. Prove local
HEAD, remote branch and RELEASE equality plus a clean worktree before running
the script.

## Input-contract correction

The frozen code uses only two bounded manifest locations, in this order:

1. `JSON-Config-Files/run_manifest.json` for a completed package;
2. `JSON-Config-Files/Input-Manifests/manifest.yaml` for a failed pre-package
   staging run.

It never recursively searches the archive, follows a path from a receipt, or
opens any source path stored inside the manifest. Every selected ledger input
is reported with archive-relative path, format, byte size and SHA256.

The deployment script must complete, in order:

1. repository runtime preflight;
2. DEV-041 ledger-input inspection;
3. DEV-042 ledger-input inspection;
4. DEV-041 deterministic replay;
5. DEV-042 deterministic replay.

Both input inspections must pass before either replay starts.

## Exact command

Run once from the frozen repository root in a fresh PowerShell child process:

```powershell
$ErrorActionPreference = 'Stop'
$visionCortexReplayScript = (Resolve-Path -LiteralPath '.\deployment\rtx4060\05-Replay-Quality-Ledgers.ps1').Path
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $visionCortexReplayScript
if ($LASTEXITCODE -ne 0) {
    throw "DEV-048 quality-ledger replay failed: $LASTEXITCODE"
}
```

Do not alter parameters, redirect output, inspect paths manually or retry. On
any nonzero exit, stop and post one INCIDENT. Do not launch a real collection.

## Required evidence

Report:

1. runtime preflight JSON and frozen source binding;
2. both input inspection receipts, including selected relative paths, formats,
   byte sizes and SHA256 values;
3. DEV-041 and DEV-042 replay `result_sha256`, evidence grade, counts and full
   group structures;
4. every legacy QF1 boundary invariant requiring an explicit receipt;
5. independent `QF2-STABLE-OBJECT-IDENTITY` and
   `QF2-QUARANTINED-CONTEXT-CHAIN` verdicts and failed criteria;
6. whether DEV-042 G3 `+11.222s` and missing G4 are explained, each graded
   `PROVEN`, `PARTIAL_EVIDENCE` or `NOT_PROVEN`;
7. proof that quarantined context does not enter formal segment membership or
   selected key events;
8. MP4/CSV reads, video operations, inference/model calls and Token all zero.

The retained runs lack exact persisted `boundary_candidates`; both replay
results must remain `degraded_legacy_ledger`. A successful DEV-048 proves a
historical decision replay, not production video acceptance.

## Forbidden actions

- no MP4, timestamp CSV, image, PDF, SQLite or semantic-cache payload read;
- no pipeline launch, FFmpeg, OpenCV media operation, YOLO, TensorRT or MLLM;
- no Token use, retry, resume, tuning or real one-shot consumption;
- no dependency install/update, system Python, inline Python, Web/API probe or
  development test;
- no code, config, model, cache, staging, registry or archive mutation;
- no new NAS file and no commit, push, merge or tag.

The only external write is one final GitHub Issue #3 comment.

## Final response

Post exactly one `DEV-20260819-048 EVIDENCE RESULT` or one
`DEV-20260819-048 INCIDENT` with frozen identity, both ledger-input receipts,
replay comparison, evidence limitations, zero-mutation proof, clean worktree
and zero relevant VisionCortex/FFmpeg processes.

Do not start real video processing. The three real datasets require a separate
frozen task after this diagnostic result is reviewed.
