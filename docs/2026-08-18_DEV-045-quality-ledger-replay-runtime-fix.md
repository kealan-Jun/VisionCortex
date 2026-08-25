# DEV-20260818-045 — Quality-Ledger Replay Runtime Fix

## Objective

Run the DEV-041 and DEV-042 read-only JSON quality-ledger replay after fixing
the DEV-044 task-book module-resolution omission. This is a new diagnostic
authorization. DEV-044 remains closed as an INCIDENT and must not be retried.

This task does not consume a production one-shot.

## Immutable gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch: `codex/post-dev043-deterministic-quality`
- Frozen SHA: the complete SHA in the unique `DEV-20260818-045 RELEASE`
- Coordination: GitHub Issue #3 only

Fetch the repository and check out the exact RELEASE SHA in detached-HEAD mode.
Before execution, prove local HEAD, remote branch and RELEASE SHA equality plus
a clean worktree.

## Root cause fixed by this task

The repository uses a `src/` package layout. DEV-044 invoked
`python -m labvision_evidence.cli` without installing the package or adding the
repository `src` directory to Python module search paths. The exact command
therefore failed before reading any ledger.

DEV-045 sets process-local `PYTHONPATH` to the frozen checkout's `src`
directory. It does not install or update any dependency. Every Python command
uses `-B` so the read-only task does not create bytecode files in the checkout.

## Allowed inputs

Read JSON files only beneath these retained run roots:

### DEV-041

`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-154357-29d1`

### DEV-042

`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-173813-fc76`

Configuration:

`configs/rtx4060-laptop-production.yaml`

The retained ledgers predate exact `boundary_candidates` persistence. Their
expected evidence grade is `degraded_legacy_ledger`; it is diagnostic evidence,
not exact raw-boundary production acceptance.

## Exact PowerShell procedure

Run this block once from the frozen repository root. Do not redirect stdout to
NAS and do not provide `--output`.

```powershell
$ErrorActionPreference = 'Stop'
$visionCortexRepoRoot = (Get-Location).Path
$visionCortexSourceRoot = Join-Path $visionCortexRepoRoot 'src'
$env:PYTHONPATH = $visionCortexSourceRoot

python -B -c "import pathlib, labvision_evidence; expected=pathlib.Path(r'$visionCortexSourceRoot').resolve(); actual=pathlib.Path(labvision_evidence.__file__).resolve(); assert expected in actual.parents, f'unexpected package path: {actual}'; print(actual)"
if ($LASTEXITCODE -ne 0) { throw "VisionCortex module preflight failed: $LASTEXITCODE" }

python -B -m labvision_evidence.cli replay-quality-ledger `
  --archive 'Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-154357-29d1' `
  --config 'configs/rtx4060-laptop-production.yaml'
if ($LASTEXITCODE -ne 0) { throw "DEV-041 ledger replay failed: $LASTEXITCODE" }

python -B -m labvision_evidence.cli replay-quality-ledger `
  --archive 'Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-173813-fc76' `
  --config 'configs/rtx4060-laptop-production.yaml'
if ($LASTEXITCODE -ne 0) { throw "DEV-042 ledger replay failed: $LASTEXITCODE" }
```

If the module preflight or either replay fails, stop immediately and post one
INCIDENT. Do not alter the environment, command, code or dependencies, and do
not run a real collection.

## Required evidence

For each ledger, report:

1. `result_sha256`, evidence grade, raw-segment source and boundary-candidate
   source.
2. Event, raw/normalized/formal segment, experiment-group and selected-event
   counts.
3. Every replayed group boundary, continuity type, atomic member IDs and view
   pair.
4. Every `QF2-STABLE-OBJECT-IDENTITY` receipt and its verdict, failed criteria,
   shared labels and stable identities.
5. Every independent `QF2-QUARANTINED-CONTEXT-CHAIN` receipt and its verdict,
   context IDs/fingerprints, object families, shared views/boundaries and
   per-criterion booleans.
6. Every legacy raw-segment invariant where
   `requires_explicit_qf1_receipt=true`, including direct accepted-event
   boundary deltas. State whether DEV-042 G3 `+11.222s` is represented.
7. Confirm quarantined context IDs appear only in decision receipts, never in
   formal membership or selected key events.
8. Compare the ledgers and grade each conclusion as `PROVEN`,
   `PARTIAL_EVIDENCE` or `NOT_PROVEN`.

## Forbidden actions

- no MP4, timestamp CSV, image, PDF, SQLite or semantic-cache payload read;
- no pipeline launch, FFmpeg, OpenCV, YOLO, TensorRT or MLLM call;
- no Token use, retry, resume, parameter tuning or one-shot consumption;
- no package installation, environment mutation beyond the process-local
  `PYTHONPATH` assignment above, development test, API/WebSocket probe or Web
  restart;
- no code, config, dependency, driver, model, cache, staging, registry or
  formal-archive modification;
- no new NAS file and no commit, push, merge or tag.

The only permitted external write is the final Issue #3 comment.

## Final response

Post exactly one `DEV-20260818-045 EVIDENCE RESULT` or one
`DEV-20260818-045 INCIDENT` to GitHub Issue #3 with:

- frozen identity, module preflight result and zero-mutation proof;
- both replay hashes and the required QF1/QF2 comparison;
- explicit legacy-ledger limitations;
- MP4/CSV reads `0`, inference/model calls `0`, Token `0`;
- final clean worktree and zero VisionCortex/FFmpeg process proof.

Do not start a real collection after this task.
