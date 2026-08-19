# DEV-20260819-046 — Project-Runtime Quality-Ledger Replay

## Objective

Replay the retained DEV-041 and DEV-042 JSON quality ledgers using the exact
repository virtual environment that previously executed the real RTX 4060
pipeline. DEV-044 and DEV-045 remain closed incidents and must not be retried.

This is a new read-only diagnostic authorization. It does not consume a
production one-shot.

## Immutable gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch: `codex/post-dev043-deterministic-quality`
- Frozen SHA: the complete SHA in the unique `DEV-20260819-046 RELEASE`
- Coordination: GitHub Issue #3 only

Fetch and check out the exact RELEASE SHA in detached-HEAD mode. Prove local
HEAD, remote branch and RELEASE equality plus a clean worktree before running
the script.

## Proven runtime correction

Earlier real three-dataset tasks used the repository-owned runtime:

`<repository>\.venv\Scripts\labvision.exe`

DEV-044/045 accidentally used the system interpreter
`D:\Miniconda3\python.exe`. The latter is not the production environment and
does not contain the complete project dependency set.

DEV-046 uses the versioned deployment script:

`deployment\rtx4060\05-Replay-Quality-Ledgers.ps1`

The script:

1. refuses to run when `.venv\Scripts\python.exe` is missing;
2. verifies both retained JSON run roots and the production config exist;
3. sets process-local `PYTHONPATH` to the frozen checkout's `src`;
4. imports `openpyxl`, `cv2`, `numpy`, `pydantic`, `yaml`, the full CLI and the
   replay module before any ledger read;
5. proves the imported package resolves beneath the frozen checkout;
6. uses `python -B`, writes no output file, and restores process environment;
7. runs DEV-041 first and DEV-042 only when DEV-041 succeeds.

## Exact command

Run once from the frozen repository root in a fresh PowerShell child process:

```powershell
$ErrorActionPreference = 'Stop'
$visionCortexReplayScript = (Resolve-Path -LiteralPath '.\deployment\rtx4060\05-Replay-Quality-Ledgers.ps1').Path
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $visionCortexReplayScript
if ($LASTEXITCODE -ne 0) {
    throw "DEV-046 quality-ledger replay failed: $LASTEXITCODE"
}
```

Do not alter parameters or redirect stdout to NAS. If runtime preflight or
either replay fails, stop and post one INCIDENT. Do not install dependencies,
change interpreters, modify the script or start a real collection.

## Required evidence

Report:

1. project interpreter path/version, dependency-presence/version matrix and
   imported package source path;
2. DEV-041 and DEV-042 replay `result_sha256`, evidence grade, counts and group
   structures;
3. every QF1 legacy boundary invariant requiring an explicit receipt;
4. independent `QF2-STABLE-OBJECT-IDENTITY` and
   `QF2-QUARANTINED-CONTEXT-CHAIN` verdicts with failed criteria;
5. whether DEV-042 G3 `+11.222s` and missing G4 are explained, graded as
   `PROVEN`, `PARTIAL_EVIDENCE` or `NOT_PROVEN`;
6. proof that quarantined context never enters formal membership or selected
   key events;
7. MP4/CSV reads, inference calls, model calls and Token all equal zero.

Legacy runs lack exact persisted `boundary_candidates`; both results must remain
`degraded_legacy_ledger` and cannot be described as production acceptance.

## Forbidden actions

- no MP4, timestamp CSV, image, PDF, SQLite or semantic-cache payload read;
- no pipeline launch, FFmpeg, OpenCV video operation, YOLO, TensorRT or MLLM;
- no Token use, retry, resume, tuning or real one-shot consumption;
- no dependency installation/update, system Python use, Web/API probe or test;
- no code, configuration, model, cache, staging, registry or archive mutation;
- no new NAS file and no commit, push, merge or tag.

The runtime preflight may import `cv2`; it must not call any image/video API.
The only permitted external write is one final Issue #3 comment.

## Final response

Post exactly one `DEV-20260819-046 EVIDENCE RESULT` or one
`DEV-20260819-046 INCIDENT` with frozen identity, runtime matrix, replay
results, evidence grades, zero-mutation proof, clean worktree and zero relevant
VisionCortex/FFmpeg process proof.

Do not start a real collection after this task. A separate frozen release will
authorize the three real datasets only after this replay is understood.
