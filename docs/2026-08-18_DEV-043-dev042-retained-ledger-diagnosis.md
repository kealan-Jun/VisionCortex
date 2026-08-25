# DEV-20260818-043 — DEV-041/042 Retained-Ledger Diagnosis

## Objective

Perform one read-only comparison of the retained DEV-041 and DEV-042 JSON
evidence ledgers.  Explain why the six-view cold run changed from the successful
DEV-041 ledger replay to six formal groups with a missed reviewed G4 in DEV-042,
and preserve the exact A path/cache evidence needed for a safe recovery run.

This is diagnosis only.  It does not consume a production one-shot.

## Immutable gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Branch: `codex/post-dev042-recovery`
- Frozen SHA: the complete SHA in the unique `DEV-20260818-043 RELEASE`
- Coordination: GitHub Issue #3 only

Check out the exact RELEASE SHA in detached-HEAD mode.  Prove local HEAD,
remote branch and RELEASE equality plus a clean worktree before reading ledgers.

## Allowed evidence

Read only the JSON/TXT ledgers below and Git metadata.  Do not open MP4 files,
clock CSV files, images, PDFs or SQLite databases.

### Six-view DEV-041

`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-154357-29d1`

### Six-view DEV-042

`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\collection-20260818-173813-fc76`

### A DEV-042

`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\CustomFlow-Standard-Correct-12-A-0004-2026-08-11\collection-20260818-181616-b350`

The local semantic cache may be inspected only by metadata and JSON content.
Never print, hash, enumerate or otherwise read credential values.

## Forbidden actions

- no pipeline launch, retry, resume or one-shot consumption;
- no FFmpeg, OpenCV, YOLO, TensorRT, model API or Token use;
- no source-video or timestamp-CSV access;
- no development test, product HTTP/WebSocket probe or Web restart;
- no code, config, model, dependency, driver, staging, cache, registry or formal
  archive modification;
- no commit, push, merge, tag or branch-tip substitution;
- no new NAS file.  The only allowed external write is the final Issue comment.

## Required six-view comparison

Read the existing `audit_layer.json`, `boundary_precheck.json`,
`progressive_fine_scan.json`, `key_material_selection_preview.json`,
`run_metrics.json`, `scan_runtime_motion_probe.json`, `scan_runtime_fine.json`
and all existing decision-receipt JSON files in both runs.

Report:

1. Exact raw, normalized and formal segment counts and boundaries in each run.
2. Every formal group's member segment IDs, event IDs, start/end, continuity
   type, atomic count, view pair and reviewed-baseline match state.
3. Event-level correspondence across runs using action type, nearest global
   timestamp, object families and supporting views.  Do not assume event IDs
   remain stable when the detected evidence differs.
4. The exact first decision where DEV-041 replay and DEV-042 diverge for:
   - reviewed G3 end (`+11.222s` in DEV-042);
   - false-positive `GROUP-0004` and `GROUP-0005`;
   - false-negative reviewed G4;
   - the additional Fine work (`44,098 -> 45,690` frames and `42 -> 48`
     persistent sessions).
5. Whether `QF1-SINGLE-VIEW-BOUNDARY-CONTEXT` and
   `QF2-QUARANTINED-CONTEXT-CHAIN` were evaluated, accepted or rejected, with
   their facts, thresholds and missing proof.
6. Progressive recall target/view/wave order, uncovered anchor clusters,
   accepted-versus-rejected state, stop reason and formal/quarantined outcome.
7. Whether the difference is proven to arise from scan coverage, event
   acceptance, segment normalization, continuity grouping, boundary extension
   or nondeterministic ordering.  Mark any unsupported claim `NOT_PROVEN`.
8. A minimal code-level fix recommendation that does not encode reviewed
   timestamps, group count or experiment-specific camera IDs as inference
   rules.  Identify the exact durable receipt required to make the next result
   auditable.

## Required A recovery evidence

Without reading media, report:

1. The exact normal UNC and extended UNC strings from the exception and their
   canonical relative artifact path.
2. All other retained A JSON fields containing either UNC representation, to
   determine whether the failure is isolated or systemic.
3. Experiment-understanding semantic-cache schema, fingerprint, status,
   model, input/output/total Token and reuse eligibility.  Do not print prompt
   bodies, image payloads or secrets.
4. Partial key-material counts by media/sidecar type and the exact first
   incomplete event.  Do not open the media.
5. Confirmation that Z still has no DEV-042 launch and its one-shot was not
   consumed.

## Final response

Post exactly one `DEV-20260818-043 EVIDENCE RESULT` to GitHub Issue #3 with:

- frozen identity and zero-mutation proof;
- the six-view divergence table and causal chain;
- the A path/cache recovery table;
- `PROVEN`, `PARTIAL_EVIDENCE` or `NOT_PROVEN` on every conclusion;
- exact retained JSON evidence paths;
- source MP4/CSV reads `0`, model calls `0`, Token `0`;
- final clean worktree and zero production/FFmpeg process proof.

Do not run a real collection after completing this task.
