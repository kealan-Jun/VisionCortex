# DEV-20260818-044 — Deterministic Quality-Ledger Replay

## Objective

Use the frozen deterministic quality rules to replay the retained DEV-041 and
DEV-042 six-view JSON ledgers. Prove whether G3 boundary drift and the missing
G4 continuity decision can be explained from durable QF1/QF2 receipts before
another real-video one-shot is authorized.

This is a read-only ledger task. It does not consume a production one-shot.

## Immutable gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch: `codex/post-dev043-deterministic-quality`
- Frozen SHA: the complete SHA in the unique `DEV-20260818-044 RELEASE`
- Coordination: GitHub Issue #3 only

Fetch the repository, check out the exact RELEASE SHA in detached-HEAD mode,
and prove local HEAD, remote branch and RELEASE SHA equality plus a clean
worktree before reading any ledger.

## Allowed inputs

Read JSON files only beneath these retained run roots:

### DEV-041

`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-154357-29d1`

### DEV-042

`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-173813-fc76`

Use the frozen production configuration:

`configs/rtx4060-laptop-production.yaml`

The retained ledgers predate exact `boundary_candidates` persistence. Their
replay is therefore expected to report `degraded_legacy_ledger`; this is valid
diagnostic evidence for grouping and invariant auditing, but it must not be
reported as exact raw-boundary production acceptance.

## Exact commands

Run each command once. Do not redirect output to NAS or write an output file.

```powershell
python -m labvision_evidence.cli replay-quality-ledger `
  --archive 'Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-154357-29d1' `
  --config 'configs/rtx4060-laptop-production.yaml'

python -m labvision_evidence.cli replay-quality-ledger `
  --archive 'Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-173813-fc76' `
  --config 'configs/rtx4060-laptop-production.yaml'
```

If either command fails, stop and post one INCIDENT. Do not improvise a repair
or run the real pipeline.

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
5. Every independently emitted `QF2-QUARANTINED-CONTEXT-CHAIN` receipt and its
   verdict, context event IDs/fingerprints, object families, shared views,
   shared boundary and per-criterion booleans.
6. Every legacy raw-segment invariant where
   `requires_explicit_qf1_receipt=true`, including the direct accepted-event
   boundary delta. Explicitly identify whether the DEV-042 G3 `+11.222s`
   extension is visible in this list.
7. Confirm that quarantined context IDs appear only in receipts and never in
   formal group membership or selected key events.
8. Compare DEV-041 and DEV-042 and classify each conclusion as `PROVEN`,
   `PARTIAL_EVIDENCE` or `NOT_PROVEN`. Do not combine the two ledgers into one
   synthetic production result.

## Forbidden actions

- no MP4, timestamp CSV, image, PDF, SQLite or model-cache payload read;
- no pipeline launch, FFmpeg, OpenCV, YOLO, TensorRT or MLLM call;
- no Token use, retry, resume, parameter tuning or one-shot consumption;
- no development tests, product API/WebSocket probe or Web restart;
- no code, configuration, dependency, driver, model, cache, staging, registry
  or formal-archive modification;
- no new NAS file and no commit, push, merge or tag.

The only permitted external write is the final Issue #3 comment.

## Final response

Post exactly one `DEV-20260818-044 EVIDENCE RESULT` to GitHub Issue #3 with:

- frozen identity and zero-mutation proof;
- the two replay result hashes and comparison table;
- the QF1 invariant and independent QF2 receipt findings;
- explicit limitations caused by legacy missing boundary candidates;
- source MP4/CSV reads `0`, inference/model calls `0`, Token `0`;
- final clean worktree and zero VisionCortex/FFmpeg process proof.

Do not start a real collection after this task.
