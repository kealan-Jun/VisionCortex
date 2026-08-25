# DEV-20260817-038 — A/Z Natural-Dataset Full-Pipeline Acceptance

## Objective

Run one real cold-start full pipeline for each of these NAS collections on the RTX 4060 production runner:

1. `CustomFlow_standard_correct_12_A_0004`
2. `CustomFlow_standard_correct_12_z_0001`

This task validates dynamic view composition, naturally inferred experiment counts, object-aware key-material names, full model understanding, searchable evidence, the V2 daily report, the professional PDF, and atomic NAS promotion. Do not rerun the accepted six-view 3.3-hour benchmark.

This is a production-data execution task. The RTX 4060 executor must not modify code, configuration, models, thresholds, dependencies, drivers, reports, source data, staging, or caches; must not run tests or product API tests; and must not retry or continue a real run.

## Immutable release gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Frozen branch metadata: `codex/post-dev037-recall-quality`
- Taskbook: `docs/2026-08-17_DEV-038-AZ-full-pipeline-acceptance.md`
- Coordination: GitHub Issue #3 only
- Frozen SHA authority: the complete 40-character SHA in the unique `DEV-20260817-038 RELEASE` comment on Issue #3

The branch name is discovery metadata only. Never execute a moving branch tip. Fetch the branch, check out the RELEASE SHA in detached-HEAD mode, and prove all of the following before reading video payloads:

- `HEAD` equals the RELEASE SHA;
- `origin/codex/post-dev037-recall-quality` equals the RELEASE SHA at ACK time;
- this taskbook exists at `HEAD`;
- the worktree is clean;
- no VisionCortex/labvision/FFmpeg production process is active.

If any item differs, post one INCIDENT and do not consume either one-shot.

## Authoritative inputs

Resolve sources, frame-clock CSV files, collection identity, and view roles only through `Y:\experiment_record_index.csv`. Do not hard-code two, five, six, `1+4`, `2+4`, or any source-count limit. The current snapshot happens to contain one first-person and four third-person views for each collection; the live index and versioned role-resolution receipt remain authoritative.

### Dataset A

- User label: `CustomFlow_standard_correct_12_A_0004`
- `experiment_id`: `exp_20260811_161842_ca175e55`
- English archive: `CustomFlow-Standard-Correct-12-A-0004-2026-08-11`

### Dataset Z

- User label: `CustomFlow_standard_correct_12_z_0001`
- `experiment_id`: `exp_20260812_133147_53d6cbd4`
- English archive: `CustomFlow-Standard-Correct-12-Z-0001-2026-08-12`

The 15-minute physical MP4s must remain independent source segments behind a zero-copy virtual timeline. Do not concatenate, rename, relocate, or copy source payloads. Source-copy bytes must be zero.

## Exact execution order and commands

Run A and Z serially. Before each launch, resolve the repository-local CLI and config from the detached frozen checkout:

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Get-Location).Path
$Labvision = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot '.venv\Scripts\labvision.exe')).Path
$Config = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml')).Path
```

### 1. Dataset A — exactly one launch

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260811_161842_ca175e55' `
  --archive-name 'CustomFlow-Standard-Correct-12-A-0004-2026-08-11' `
  --config $Config
```

Record the process exit code, run ID, cold cache identity, staging/formal path, and terminal receipts before deciding whether Z may start.

### 2. Dataset Z — exactly one launch when permitted

After every A-owned Python/FFmpeg process has exited, run:

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260812_133147_53d6cbd4' `
  --archive-name 'CustomFlow-Standard-Correct-12-Z-0001-2026-08-12' `
  --config $Config
```

Each command is one independent authorization. The first process launch consumes that dataset's one-shot even if it exits immediately.

## Failure and continuation rules

- If A completes and promotes: proceed to Z after its processes exit and terminal receipts are durable.
- If A reaches a normal collection-specific content or quality gate and fails without corrupting shared state: retain A staging unchanged, classify the evidence, and proceed to Z.
- If A exposes a shared code/runtime, GPU/driver, TensorRT/model-loading, credential/API, NAS/storage, archive-promotion, repository-integrity, or process-ownership failure: stop before Z.
- If Z fails for any reason: retain its staging unchanged and stop.
- Never retry, resume, rerun, tune, switch cache identity, relax a quality gate, patch an artifact, or reuse an old CV/model ledger.
- Never delete failed staging, caches, history, or an existing formal archive.

A failure in report/PDF generation is a pipeline failure; do not manufacture or manually copy a report. Model service errors remain failures with their original call/retry receipt; do not issue an extra launch.

## Required release behavior

The run must prove the following without using reviewed labels as inference input:

1. The six-view reviewed baseline applies only to `exp_20260810_144014_e918b762`.
2. For A and Z, `baseline_selection.applied=false` with reason `experiment_id_not_applicable`.
3. A and Z must not be required to produce five experiments or at least 80 key events.
4. Experiment groups and independent/continuous structure arise from actual dual-view evidence. DEV-037 observed one A group, but that count is audit context, not a prior.
5. A naturally unobserved action category is not a failure. Its `Category.json` records `coverage_status=not_observed` and an evidence-based `absence_reason`.
6. Every formal experiment and key material has timestamp-aligned first-person, third-person, and side-by-side media. Missing-view evidence is quarantined and cannot silently open, extend, or bridge a formal group.
7. Key-material folders and files describe both action and object, for example `Contact-Hand-With-Weighing-Paper_EVT-...`, rather than only `Hand-Object-Contact`.
8. Key-material JSON keeps the unified top-level contract. Archive-classification fields live under `provenance.archive_classification` and in the category index.
9. Each archive contains `JSON-Config-Files/key_material_recall_eval.json`. Without applicable event-level human ground truth it says `not_evaluated`; it must not invent Precision/Recall and must not fail production.

## Full-chain output and per-stage NAS durability

Each successful collection must produce and atomically promote:

- zero-copy original-source references and input index;
- alignment transforms, nearest-neighbour matches, visual-anchor audit, and uncertainty;
- bounded experiments with first-person, third-person, side-by-side MP4s, and corresponding JSON;
- key frames, key clips, timestamps, hashes, cross-view associations, and unified event JSON, nested by experiment and the five physical-action categories;
- fine-grained experiment and material understanding with current step, next step, objects, observed facts, supported inference, uncertainty, and contradictions;
- searchable evidence database, stable IDs, artifact/evidence registries, integrity hashes, and physical-change records;
- branded `VC-LAB-DAILY-REPORT-V2` with representative evidence images;
- `VC-PROFESSIONAL-EVIDENCE-REPORT-V1`, distinct from the daily report and containing selected evidence images;
- total/stage timing and total/stage input, output, cached, and total Token receipts.

Every completed stage writes durable receipts/artifacts to the run staging area immediately. The formal archive remains all-or-nothing and may be replaced only after quality, evidence-package, report, PDF, index/integrity, and SHA256 promotion gates pass.

## Required performance and resource evidence

For A and Z separately record:

- preflight, alignment, Motion, coarse, Fine, audit, experiment understanding, experiment-media export, key-material export, material understanding, index/package, daily report, PDF, promotion, and total wall time;
- CPU, RAM, GPU compute, VRAM, power, temperature, clocks, NVDEC, and NVENC mean/P50/P95/max;
- wired NAS/host receive and process-tree read/write throughput, SMB errors/retries, decode/feed/queue/inference/materialization waits, queue-depth/full-time, and actual TensorRT batch distribution;
- exact indexed views, roles, segment counts, decode backend/lanes, sampled frames, work units, cache reuse, EOF reconciliation, and source-copy bytes;
- model request/success/failure/retry/reuse counts, latency min/median/P95/max, and input/output/cached/total Token by stage and run.

DEV-037's A preprocessing value of about 346.17 seconds is read-only context, not an acceptance prior and not a reason to tune in place. Z has no predeclared performance gate. Record observed results without weakening quality.

## NAS locations and state

- Index: `Y:\experiment_record_index.csv`
- Formal archive root: `Y:\VisionCortexExperimentArchive`
- Run staging: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging`
- System registry: `Y:\VisionCortexExperimentArchive\.VisionCortex-System\Collection-Processing-Registry.json`
- Runtime/cache: use and report the paths resolved by `configs\rtx4060-laptop-production.yaml`

The registry must distinguish `queued`, `processing`, `failed`, and `archived`. A collection is `archived` only after its full promotion receipt passes. Preserve all existing formal archives.

## GitHub reporting protocol

Use Issue #3 and task ID `DEV-20260817-038` as the sole coordination record:

1. one ACK with machine, branch metadata, complete RELEASE SHA, detached-HEAD proof, taskbook existence, clean-worktree proof, idle-process proof, and execution plan;
2. one PREFLIGHT with exact live index/role/segment/path counts, NAS/runtime/cache/free-space facts, model-engine hashes/status, credential-presence boolean, wired-link status, and confirmation that source payload bytes remain unread;
3. at most one HEARTBEAT every 5–10 minutes while a real process is active, covering dataset, stage, elapsed time, per-view progress, resource/NAS/batch/wait metrics, API/Token totals, and staging path;
4. one final COMPLETED only if both datasets complete their authorized terminal workflow and all required evidence is present; otherwise one final INCIDENT containing the full A/Z matrix and every retained evidence path.

The final response must also be written as one complete Markdown execution document, not only a chat summary. Never expose a key, secret, token value, or credential material in GitHub, logs, JSON, screenshots, terminal output, or reports.
