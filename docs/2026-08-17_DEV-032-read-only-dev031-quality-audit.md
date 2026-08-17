# DEV-20260817-032: Read-Only DEV-031 Quality Audit

## Task type

This is an evidence-only audit on the RTX 4060 execution node. It is not a
development task, test task, benchmark run, or pipeline retry.

The only allowed outcome is one evidence-backed diagnosis of the preserved
DEV-031 staging and cache. Do not change code, configuration, models, weights,
cache, staging, accepted archives, Git branches, or runtime services.

## Source incident

- GitHub collaboration location: `kealan-Jun/VisionCortex` Issue `#3`
- DEV-031 incident comment:
  `https://github.com/kealan-Jun/VisionCortex/issues/3#issuecomment-5311054896`
- Run ID: `cli-20260817-095055-34d8`
- Runtime commit: `03f4ec0d4ac7432f55a3e5f030181ceea53b64ee`
- Cache identity: `9c3c60dcecccbe81fcd0`
- Staging:
  `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\cli-20260817-095055-34d8`
- Cache:
  `D:\VisionCortexLocal\Cache\Six-View-Three-Hour-Experiment-2026-08-13\9c3c60dcecccbe81fcd0`

DEV-031 passed the preprocessing performance SLA at `914.920986s`, completed
all 27 Fine sessions at `60,828 / 60,828` expected/actual frames, produced 103
accepted physical events and selected 87 key events, but failed the deterministic
quality precheck:

- G1 start error: `+8.55s`;
- G3: incorrectly grouped as `continuous / 2`;
- G4: expected `continuous / 2`, but no accepted match survived;
- G5 start error: `-14.65s`;
- merged detection ledgers contain 15 duplicate local-timestamp rows at physical
  segment boundaries: FP 10, AB 2, B439 3.

## Absolute prohibitions

Do not:

- run the pipeline, fixed benchmark, preprocessing, dry-run, Web submission, or
  any retry/resume command;
- read any MP4, clock CSV, RGB frame source, or original experiment file;
- invoke FFmpeg, FFprobe, YOLO, TensorRT, CUDA inference, model imports, MLLM,
  Ark/Doubao API, report generation, media materialization, or PDF generation;
- run pytest, compilation checks, installation, dependency updates, or any test;
- fetch, pull, switch, checkout, reset, stash, clean, commit, push, or otherwise
  change Git state;
- edit, create, delete, rename, copy, move, touch, or reserialize any file in the
  repository, staging, cache, formal archive, or NAS;
- create a new local or NAS handoff/report file;
- print or inspect the value of `ARK_API_KEY` or any credential;
- infer a cause that is not traceable to a preserved receipt or ledger field.

The audit may read only existing JSON/JSONL/Markdown/text ledgers under the
staging and cache paths above, plus read-only Git identity and process state.

## Start gate

Before inspecting evidence, append exactly one `ACK · DEV-20260817-032` comment
to GitHub Issue #3. The ACK must contain:

- actual repository path, branch, HEAD and upstream SHA;
- `git status --porcelain` result;
- confirmation that no benchmark or FFmpeg process is running;
- resolved staging and cache paths;
- file counts and byte counts for staging/cache before the audit;
- confirmation of every prohibition above.

If either source path is missing, Git worktree is dirty, a benchmark/FFmpeg
process is running, or the preserved hashes do not match, stop with one INCIDENT
comment. Do not repair anything.

Verify these staging SHA-256 values before analysis:

| Evidence | Expected SHA-256 |
|---|---|
| `run_status` / `pipeline_status` | `259d0557d7d19336a5bf120e61d78ff5c6780860b26eb7f1e497d6b03feb3102` |
| `run_metrics` | `97ce9652530631af35c072dadbca796b27e394c6c1b193118665e13297c6a273` |
| `scan_runtime_motion_probe` | `b6483ea6fd234909ce985a3b2ccc2e4479c4b9917325713f1c47157f736a425f` |
| `scan_runtime_fine` | `72ab7fbf4330b57d0526877b1f69c042c95432c679bbc0d6ab4d04e0414e3336` |
| `progressive_fine_scan` | `3e7eee37b6fd7624987508d1f55e844d774fc711de40ac110cf98e590dbcf76e` |
| `candidate_layer` | `d2ed0b4c1289f8b022d4797e8e3510eaf1a9fe63a22a0c2e02cc82bb8a0df908` |
| `audit_layer` | `88da73ca361fc6c71b3b15edb9f4496ccbc2776a34ed331785b8da2d27957d83` |
| `boundary_precheck` | `0b85111d176f64731a0a81e07738554455b0f201ec0a44ba548ee62129e82954` |
| `key_material_selection_preview` | `05e9a5dbf9fecffb6d68fda8ea0cbe7e9736f94319e57181e4b6e8abece3688d` |
| `resource_telemetry` | `0efc27627d2a9a072aabda0cf029f3efdf35763ead47548fc9a8c8c34700d94c` |
| `cache_identity` | `8b292838feaaf6f9684d5adb2549cfcfc04fd94b48d40b10bb75743ad40cba7d` |
| `nas_ingest` | `e9e348ddcb8b0daa2107c1a897bf3e554dff641f301fba1a0d8e5bd6cc0a2f3a` |

Resolve each logical name to its actual preserved file path and report the path.
Do not assume filenames when multiple candidates exist.

## Audit A: 15 boundary duplicate timestamps

Enumerate all 15 duplicate rows. For each duplicate, report every available
field without synthesizing missing data:

- role and view ID;
- global and local timestamp;
- physical MP4 segment path stored in the ledger;
- source session/work-unit ID;
- source frame index or expected frame index;
- the two or more conflicting ledger row IDs;
- whether their detection payloads are identical, different, or not provable;
- physical event IDs, atomic segment IDs, formal group IDs, boundary receipts,
  or continuity edges that reference or temporally contain the duplicate;
- whether removing only the duplicate row would alter a candidate, event,
  boundary, or continuity decision according to preserved receipts.

Produce a 15-row table and separate counts for FP, AB, and B439. Explicitly
classify the duplicate-to-quality relationship for G1, G3, G4, and G5 as one of:

- `PROVEN_CAUSAL`;
- `PROVEN_NOT_CAUSAL`;
- `CORRELATED_NOT_CAUSAL`;
- `NOT_PROVEN`.

Do not assume the 15 duplicates caused the grouping failures merely because they
exist.

## Audit B: G1 boundary chain

G1 is `independent / 1` but starts `+8.55s`, only `0.55s` outside tolerance.
Trace the full decision chain:

1. expected and predicted boundary;
2. earliest FP and TP candidates inside the reviewed pre-boundary interval;
3. earliest eligible dual-view physical event;
4. every earlier rejected event and its exact rejection reason;
5. the event/atomic segment that opened the final formal boundary;
6. all configured padding or clamping applied after that event;
7. whether the already-proven FP/RK paper evidence was unavailable, rejected,
   not converted to a physical event, or ignored by grouping.

Identify the smallest preserved decision where the missing `0.55s` tolerance is
introduced. If the ledgers cannot prove it, return `NOT_PROVEN` and list the
missing receipt fields.

## Audit C: G3 false continuity

Trace the incorrect G3 continuity receipt that cites `sample_bottle + tube +
tube_rack` across a `3.9s` gap:

- both atomic segment IDs and time intervals;
- every event and view supporting the edge;
- object identity source for each of the three shared objects;
- whether identity is a tracker ID, class-only match, spatial association, or
  another mechanism;
- whether any object is stationary workstation context rather than manipulated
  continuity evidence;
- the exact thresholds and acceptance receipt that created the edge;
- a counterfactual using only existing receipt fields: which minimal evidence
  requirement would reject this edge without making a claim about G4.

Do not reuse the old lone-tube diagnosis: DEV-031 already proves the lone-tube
guard held.

## Audit D: G4 missing continuity

Trace the expected G4 `continuous / 2` structure and the surviving
`7069.6s-7093.2s` independent atomic segment:

- identify the missing companion atomic segment;
- list both segments' formal events, views and object evidence;
- locate any candidate continuity edge and its rejection reason;
- if no edge was generated, identify the exact upstream absence;
- compare the G4 `tube + tube_rack` evidence with G3's accepted edge using the
  same stored fields and thresholds;
- determine why G3 was joined while G4 was separated.

The result must distinguish `edge_not_generated`, `edge_rejected`,
`segment_quarantined`, `segment_absent`, and `NOT_PROVEN`.

## Audit E: G5 early boundary

G5 is `independent / 1` but starts `-14.65s`. Trace:

1. expected and predicted boundary;
2. exact physical event and atomic segment that opened the boundary;
3. FP and TP observations supporting that event;
4. objects, action type, state change and confidence;
5. all padding, clamping and merge operations;
6. every alternative later boundary-opening event inside the reference window;
7. whether the early event is true experiment activity, preparation/context,
   single-action noise, or `NOT_PROVEN` from stored evidence.

No visual claim may be made because source video reads are forbidden.

## Audit F: event selection and isolation check

Confirm from preserved receipts only:

- 103 accepted physical events;
- 87 selected key events;
- 12 duplicate decisions and 4 per-action-cap decisions;
- no selected event originates from a quarantined/single-role-only segment;
- whether any of the 15 timestamp duplicates changed the 103 or 87 counts;
- per-group and per-action selected counts match the DEV-031 incident.

This audit must not rerun the selector.

## Required final GitHub result

Append exactly one final comment to Issue #3 titled:

`EVIDENCE RESULT · DEV-20260817-032`

The comment must contain:

1. audit identity, elapsed audit time, repository/HEAD/worktree state;
2. staging/cache before-and-after file count, byte count and core hashes;
3. the complete 15-row duplicate table;
4. G1/G3/G4/G5 decision-chain tables with event/segment/edge IDs;
5. separate causal verdicts for duplicates versus each quality failure;
6. a real issue list with symptom, evidence, proven cause, minimal developer
   change, and missing fields;
7. a minimal developer change set, limited to evidence proven by this audit;
8. explicit statements that MP4 reads, CSV reads, FFmpeg, inference, tests,
   model/API calls, Token use, source modifications, staging/cache changes and
   archive promotion were all zero;
9. final process count and clean Git state.

Allowed verdict vocabulary:

- `PROVEN`;
- `DISPROVEN`;
- `NOT_PROVEN`.

Do not report `COMPLETED` if any required table or decision chain is omitted.
Report `PARTIAL_EVIDENCE` with the exact missing receipt instead. This task never
authorizes a new real run.

## Acceptance gate

DEV-032 is accepted only when:

- all preserved hashes match and no evidence file changes;
- all 15 duplicate rows are enumerated;
- G1 and G5 each have a boundary-opening chain or an explicit missing-receipt
  proof;
- G3 and G4 each have a continuity-edge chain or an explicit missing-receipt
  proof;
- duplicates are not conflated with boundary/continuity causes;
- 103-to-87 selection integrity is checked without rerunning selection;
- no source video/CSV, model, API, test, or pipeline work occurs;
- there is one ACK and one final result comment only;
- no local/NAS report file is created and no Git state changes.

After the final comment, stop. Wait for a new developer-side frozen commit and a
new explicit one-shot authorization before any further RTX 4060 execution.
