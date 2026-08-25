# DEV-20260819-049 — Canonical NAS Path Ledger Replay

## Objective

Resolve the exact retained DEV-041 and DEV-042 quality-ledger roots through the
canonical NAS UNC path, with mapped `Y:` only as a compatibility fallback,
then run the already frozen input inspections and deterministic replays.

DEV-048 remains a closed incident and must not be retried. DEV-049 is a new
read-only diagnostic authorization; it does not consume a real video one-shot.

## Immutable gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch: `codex/post-dev043-deterministic-quality`
- Frozen SHA: the complete SHA in the unique `DEV-20260819-049 RELEASE`
- Coordination: GitHub Issue #3 only

Fetch and check out the exact RELEASE SHA in detached-HEAD mode. Prove local
HEAD, remote branch and RELEASE equality plus a clean worktree before running
the script.

## Proven path correction

DEV-043 previously read the same retained roots and independently recorded the
normal NAS UNC root as:

`\\192.168.66.149\video_database\VisionCortexExperimentArchive`

DEV-048 failed before Python because its frozen deployment script required a
mapped `Y:` path. DEV-049 resolves only the two exact retained run IDs using:

1. canonical UNC archive root;
2. mapped `Y:\VisionCortexExperimentArchive` compatibility fallback.

There is no directory enumeration or recursive search. For each exact run, the
script emits archive-root, dataset-staging-root and exact-run-root visibility,
plus the selected root and resolved run ID. This distinguishes an unavailable
drive mapping from a missing retained directory without reading a ledger or
source file.

After both paths resolve, execution remains ordered as follows:

1. frozen repository runtime preflight;
2. exact DEV-041/DEV-042 path resolution;
3. both ledger-input inspections;
4. DEV-041 and DEV-042 deterministic replays.

## Exact command

Run once from the frozen repository root in a fresh PowerShell child process:

```powershell
$ErrorActionPreference = 'Stop'
$visionCortexReplayScript = (Resolve-Path -LiteralPath '.\deployment\rtx4060\05-Replay-Quality-Ledgers.ps1').Path
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $visionCortexReplayScript
if ($LASTEXITCODE -ne 0) {
    throw "DEV-049 canonical-path quality-ledger replay failed: $LASTEXITCODE"
}
```

Do not change parameters, inspect other NAS folders, redirect output or retry.
On any failure, stop and post one INCIDENT. Do not launch a real collection.

## Required evidence

Report:

1. runtime receipt and frozen source binding;
2. both exact path-resolution receipts and all bounded visibility fields;
3. both ledger-input receipts with relative path, format, bytes and SHA256;
4. both replay result hashes, evidence grades, counts and group structures;
5. all QF1 legacy invariants and independent QF2 stable-identity/context-chain
   verdicts;
6. whether DEV-042 G3 `+11.222s` and missing G4 are explained, graded
   `PROVEN`, `PARTIAL_EVIDENCE` or `NOT_PROVEN`;
7. quarantine non-leakage and MP4/CSV/model/Token zeros.

If neither root reaches an exact retained run, the path-resolution receipt must
be used to classify the failure. Do not manually search the NAS.

## Forbidden actions

- no directory enumeration or recursive search outside the two exact paths;
- no MP4, timestamp CSV, image, PDF, SQLite or semantic-cache read;
- no pipeline, FFmpeg, OpenCV media operation, YOLO, TensorRT or MLLM;
- no Token use, retry, resume, tuning or real one-shot consumption;
- no dependency, code, config, model, cache, staging, registry or archive
  mutation;
- no Web/API probe, development test, commit, push, merge or tag.

The only external write is one final GitHub Issue #3 comment.

## Final response

Post exactly one `DEV-20260819-049 EVIDENCE RESULT` or one
`DEV-20260819-049 INCIDENT` with frozen identity, runtime/path/input receipts,
replay comparison, evidence limitations, zero-mutation proof, clean worktree
and zero relevant VisionCortex/FFmpeg processes.

Do not start real video processing after this task.
