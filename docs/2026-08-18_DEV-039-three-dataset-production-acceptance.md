# DEV-20260818-039 — Three-Dataset Production Acceptance

## 1. Objective

On the RTX 4060 production runner, execute exactly one cold-start full pipeline for each of the three user-provided collections, in this order:

1. `Six-View-Three-Hour-Experiment-2026-08-13`;
2. `CustomFlow_standard_correct_12_A_0004`;
3. `CustomFlow_standard_correct_12_z_0001`.

The run validates the frozen dual-view CV baseline, natural collection discovery, bounded experiment grouping, at least the applicable key-event baseline, five-category/object-aware key materials, continuous atomic-action state receipts, fine-grained multimodal understanding, searchable JSON, branded daily report, professional PDF, per-stage timing/Token observability, and atomic NAS promotion.

This is a real production-data task. The RTX 4060 executor must not modify code, configuration, models, thresholds, dependencies, drivers, reports, source data, staging, caches, or Git history; must not run development tests or product HTTP/WebSocket probes; and must not retry, resume, or tune a real run.

## 2. Immutable release gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Frozen branch metadata: `codex/post-dev038-recall-planner`
- Taskbook: `docs/2026-08-18_DEV-039-three-dataset-production-acceptance.md`
- Coordination: GitHub Issue #3 only
- Frozen SHA authority: the complete 40-character SHA in the unique `DEV-20260818-039 RELEASE` comment on Issue #3

The branch name is discovery metadata only. Never execute a moving branch tip. Fetch the branch, check out the RELEASE SHA in detached-HEAD mode, and prove before reading video payloads:

- `HEAD` equals the RELEASE SHA;
- `origin/codex/post-dev038-recall-planner` equals the RELEASE SHA at ACK time;
- this taskbook exists at `HEAD`;
- the worktree is clean;
- no VisionCortex/labvision/FFmpeg production process is active;
- NAS index, formal archive, staging, runtime, and cache paths are reachable;
- model credentials are present as a boolean only, without printing their value.

If any item differs, post one INCIDENT and do not consume any one-shot.

## 3. Authoritative inputs and dynamic view discovery

Resolve experiment identity, physical MP4 segments, frame-clock CSV files, source count, and first-/third-person roles only through `Y:\experiment_record_index.csv` and the versioned role-resolution receipt.

Do not hard-code a source count, camera name, role count, or upper bound. The system must accept the actual indexed collection composition while enforcing the product contract that a formal experiment contains at least one valid first-person and one valid third-person view. Extra views are evidence candidates, not mandatory duplicates.

The 15-minute physical MP4 files remain independent source segments behind a zero-copy virtual timeline. Do not concatenate, rename, relocate, or copy source payloads. `source_copy_bytes` must remain zero.

| Order | Collection | experiment_id | English formal archive |
|---|---|---|---|
| 1 | Six-view fixed benchmark | `exp_20260810_144014_e918b762` | `Six-View-Three-Hour-Experiment-2026-08-13` |
| 2 | `CustomFlow_standard_correct_12_A_0004` | `exp_20260811_161842_ca175e55` | `CustomFlow-Standard-Correct-12-A-0004-2026-08-11` |
| 3 | `CustomFlow_standard_correct_12_z_0001` | `exp_20260812_133147_53d6cbd4` | `CustomFlow-Standard-Correct-12-Z-0001-2026-08-12` |

## 4. Exact launch commands

Resolve the frozen checkout's CLI and production configuration once:

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Get-Location).Path
$Labvision = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot '.venv\Scripts\labvision.exe')).Path
$Config = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml')).Path
```

Each command below is one independent authorization. The first process launch consumes that collection's one-shot even if it exits immediately.

### 4.1 Six-view fixed benchmark — exactly one launch

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260810_144014_e918b762' `
  --archive-name 'Six-View-Three-Hour-Experiment-2026-08-13' `
  --config $Config
```

### 4.2 Dataset A — exactly one launch when permitted

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260811_161842_ca175e55' `
  --archive-name 'CustomFlow-Standard-Correct-12-A-0004-2026-08-11' `
  --config $Config
```

### 4.3 Dataset Z — exactly one launch when permitted

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260812_133147_53d6cbd4' `
  --archive-name 'CustomFlow-Standard-Correct-12-Z-0001-2026-08-12' `
  --config $Config
```

After each command, wait for all collection-owned Python/FFmpeg processes to exit and persist terminal receipts before deciding whether the next collection may start.

## 5. Failure, continuation, and preservation rules

- If a collection completes and promotes, proceed to the next collection.
- If a collection reaches a normal collection-specific content or quality gate and fails without corrupting shared state, retain its staging/cache unchanged, record the failure, and proceed to the next collection.
- If any collection exposes a shared code/runtime, GPU/driver, TensorRT/model-loading, credential/API, NAS/storage, archive-promotion, repository-integrity, or process-ownership failure, stop before the next collection.
- Never retry, resume, rerun, tune, switch cache identity, relax a quality gate, patch an output, manually complete a report, or reuse an old CV/model ledger.
- Never delete failed staging, caches, historical receipts, or an existing formal archive.
- Existing formal archives are replaced only by a fully verified atomic promotion. A failed run must not alter them.

One final `COMPLETED` is allowed only when all three authorized workflows reach their required terminal state and the three-dataset matrix is complete. Otherwise post one final `INCIDENT` with the matrix and retained evidence paths.

## 6. Dataset-specific quality gates

### 6.1 Six-view fixed benchmark

The reviewed baseline applies only to `exp_20260810_144014_e918b762`:

- exactly five formal experiment groups;
- all reviewed start/end boundary errors within ±8 seconds;
- G2 is independent with one atomic experiment;
- G4 is continuous with two atomic experiments;
- at least 80 selected formal key events;
- every formal group has timestamp-aligned first-person and third-person support;
- quarantined single-view evidence never opens, extends, or bridges a formal boundary;
- no quarantined event leaks into selected key materials.

### 6.2 Natural datasets A and Z

For A and Z, `baseline_selection.applied=false` with reason `experiment_id_not_applicable`:

- no required experiment count or 80-event minimum may be borrowed from the six-view benchmark;
- groups and independent/continuous structure arise from actual dual-view evidence;
- naturally unobserved action categories are recorded as `coverage_status=not_observed`, not fabricated and not treated as failure;
- formal segments and materials still require aligned first-/third-person support;
- experiment and material quality must be evaluated against their own indexed duration and evidence receipts.

## 7. New frozen capability checks

The run must preserve the accepted CV boundary behavior while proving these additive capabilities:

1. `continuous_action_state_ledger.json` is written after the final CV audit.
2. The continuous-state layer is a shadow publication gate: `enforce_publish_gate=false`; it cannot mutate accepted CV groups or the six-view 80+ baseline.
3. Five action families expose timestamp-derived phases, object identity tokens, state before/after, phase completeness, incomplete reasons, and fragmentation receipts.
4. High-level liquid-transfer and panel-operation evidence may suppress duplicate publication of their lower-level contact/movement children, while those children remain indexed and traceable.
5. Key-material names state the action and concrete object, for example `Contact-Hand-With-Weighing-Paper_EVT-...`; generic `Hand-Object-Contact` names are insufficient when object evidence exists.
6. Every key-material JSON retains the unified event contract and references its key frames, key clips, timestamps, cross-view associations, hashes, evidence IDs, source segments, and provenance.
7. Model review receives only CV-compressed dual-view temporal evidence (before/peak/after), not full long videos, and cannot create a formal event unsupported by CV evidence.
8. Experiment understanding states current step, next step, objects, observed facts, supported inference, uncertainty, and contradictions at fine granularity.
9. When independent YOLO box ground truth is absent, per-class Precision/Recall/AP is explicitly `not_evaluated_no_ground_truth`; candidate cues must never be reported as formal accuracy.

## 8. Required archive structure

Every successful formal archive uses English file/folder names and contains at least:

- `Original-Experiment-Videos` — zero-copy source references/index, not duplicated payloads;
- `Experiment-Clips/<Experiment-Name>` — first-person, third-person, side-by-side MP4s plus their JSON;
- `Key-Materials/<Experiment-Name>/Key-Frames/<Five-Action-Category>`;
- `Key-Materials/<Experiment-Name>/Key-Clips/<Five-Action-Category>`;
- `JSON-Config-Files` — alignment, CV, boundary, state, model, timing, Token, index, integrity, and promotion receipts;
- `Lab-Daily-Reports` — branded concise operational daily report with representative images;
- `Professional-PDFs` — distinct evidence report with selected dual-view imagery and traceable references.

Each completed stage writes durable artifacts/receipts to staging immediately. The formal archive remains all-or-nothing until quality, evidence package, report, PDF, index/integrity, and SHA256 promotion gates pass.

## 9. Required performance and observability evidence

For each collection record:

- preflight, alignment, Motion, coarse, Fine, audit, experiment understanding, experiment media, key-material export, material understanding, searchable index/package, daily report, PDF, promotion, preprocessing, and total wall time;
- CPU, RAM, GPU compute, VRAM, power, temperature, clocks, NVDEC, and NVENC mean/P50/P95/max;
- wired NAS/host receive and process-tree read/write throughput, SMB errors/retries, decode/feed/queue/inference/materialization waits, queue-depth/full-time, and actual TensorRT batch distribution;
- indexed view/role/segment counts, decode backend/lanes, sampled frames, work units, cache reuse, EOF reconciliation, duplicate timestamps, and source-copy bytes;
- model request/success/failure/retry/reuse counts, min/median/P95/max latency, and input/output/cached/total Token by stage and total run.

The six-view preprocessing target remains ≤1,200 seconds. A and Z are lighter collections, so their wall times must be reported separately and must not be compared as if they had the six-view workload.

## 10. NAS paths and batch state

- Index: `Y:\experiment_record_index.csv`
- Formal archive: `Y:\VisionCortexExperimentArchive`
- Staging: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging`
- System registry: `Y:\VisionCortexExperimentArchive\.VisionCortex-System\Collection-Processing-Registry.json`
- Runtime/cache: use the paths resolved by `configs\rtx4060-laptop-production.yaml` and report them exactly

The registry distinguishes `queued`, `processing`, `failed`, and `archived`. A collection is `archived` only after its complete promotion receipt passes.

## 11. GitHub reporting protocol

Use Issue #3 and task ID `DEV-20260818-039` as the sole coordination record:

1. one `ACK` with machine identity, complete RELEASE SHA, detached-HEAD/remote equality, taskbook existence, clean worktree, idle process proof, and execution plan;
2. one `PREFLIGHT` with the three live index resolutions, dynamic roles/segments, exact paths/free space, engine hashes/status, credential-presence boolean, wired-link status, and confirmation that source payload bytes remain unread;
3. at most one `HEARTBEAT` every 5–10 minutes while a real process is active, with collection/stage/elapsed/per-view progress, CPU/GPU/NAS/batch/wait metrics, API/Token totals, and staging path;
4. one final `COMPLETED` or `INCIDENT` containing the full three-dataset quality, performance, resource, model/Token, output, promotion, and retained-evidence matrix.

The final Issue response must be a complete Markdown execution record, not only a short chat summary. Never expose API keys, tokens, secrets, credential material, or environment-variable values.
