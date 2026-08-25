# DEV-20260818-040 — Safe Archive Replacement and Three-Dataset Run

## Objective

Supersede the stopped DEV-039 execution with a new immutable release that reconciles an existing formal archive with verified atomic replacement.

Run exactly one cold-start full pipeline for each collection, in this order:

1. `exp_20260810_144014_e918b762` → `Six-View-Three-Hour-Experiment-2026-08-13`;
2. `exp_20260811_161842_ca175e55` → `CustomFlow-Standard-Correct-12-A-0004-2026-08-11`;
3. `exp_20260812_133147_53d6cbd4` → `CustomFlow-Standard-Correct-12-Z-0001-2026-08-12`.

DEV-039 consumed only the six-view launch at the CLI archive-name gate. It read zero source bytes and created no run ID. DEV-040 explicitly grants one new six-view launch. A and Z were never launched under DEV-039; DEV-040 grants their one-shot launches under this new frozen release.

## Immutable release gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Frozen branch metadata: `codex/post-dev039-archive-replacement`
- Taskbook: `docs/2026-08-18_DEV-040-safe-archive-replacement-and-three-dataset-run.md`
- Coordination: GitHub Issue #3 only
- Frozen SHA authority: the complete SHA in the unique `DEV-20260818-040 RELEASE` comment

Fetch the repository, check out the RELEASE SHA in detached-HEAD mode, and prove HEAD/remote equality, taskbook presence, clean worktree, and zero active VisionCortex/labvision/FFmpeg production processes before reading source payloads. Never execute a moving branch tip.

The RTX 4060 executor must not modify code, configuration, models, thresholds, dependencies, drivers, source data, staging, caches, formal archives, or Git history; must not run development tests or product HTTP/WebSocket probes; and must not retry, resume, tune, or manually repair a real run.

## Fixed archive replacement contract

The existing six-view formal archive is expected and is not a launch error.

Before the six-view launch, record without modifying it:

- formal path, directory count, file count, total bytes, and last-write time;
- SHA256 of `JSON-Config-Files/fixed_archive_promotion.json`, `quality_acceptance.json`, `evidence_package_eval.json`, timing/Token receipt, daily-report evaluation, and professional PDF when present;
- registry state and registration ID.

The new run writes only to its unique staging directory. The existing formal archive remains user-visible and unchanged until all pipeline, quality, evidence-package, report, PDF, and promotion checks pass.

On success, `promote_fixed_archive` must:

1. verify the complete staged package;
2. move the previous derived package to the unique run-history directory;
3. replace derived directories from staging;
4. verify SHA256 manifests;
5. preserve original media and publish zero-copy source references;
6. write `fixed_archive_promotion.json` with `previous_package_retained=true`.

On any failure before promotion, the formal archive counts, bytes, hashes, last-write time, and registry `archived` package must remain unchanged. Failed staging/cache remains intact. Never delete, rename, move, or pre-emptively back up the formal archive manually.

## Dynamic input contract

Resolve source count, roles, MP4 segments, CSV clocks, and paths only from `Y:\experiment_record_index.csv` and the versioned role-resolution receipt. Do not hard-code any view count or upper bound.

The indexed 15-minute MP4 files remain independent sources behind the zero-copy virtual timeline. Do not concatenate or copy them. `source_copy_bytes` must be zero.

## Exact commands

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Get-Location).Path
$Labvision = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot '.venv\Scripts\labvision.exe')).Path
$Config = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml')).Path
```

### Six-view — exactly one new DEV-040 launch

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260810_144014_e918b762' `
  --archive-name 'Six-View-Three-Hour-Experiment-2026-08-13' `
  --config $Config
```

### A — exactly one launch when permitted

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260811_161842_ca175e55' `
  --archive-name 'CustomFlow-Standard-Correct-12-A-0004-2026-08-11' `
  --config $Config
```

### Z — exactly one launch when permitted

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260812_133147_53d6cbd4' `
  --archive-name 'CustomFlow-Standard-Correct-12-Z-0001-2026-08-12' `
  --config $Config
```

The first process launch consumes that collection's one-shot even if it exits immediately. Wait for all collection-owned Python/FFmpeg processes and terminal receipts before the next launch.

## Continue/stop rules

- A normal collection-specific content or quality failure retains staging and may proceed to the next collection.
- A shared code/runtime, GPU/driver, TensorRT/model, credential/API, NAS/storage, archive-preservation/promotion, repository-integrity, or process-ownership failure stops all later launches.
- Any evidence that an existing formal archive changed before verified promotion is a shared failure; stop immediately.
- Never retry, resume, rerun, switch cache identity, weaken a gate, patch an output, or reuse an old CV/model ledger.

## Quality and output gates

For the six-view benchmark require five groups, reviewed boundaries within ±8 seconds, G2 independent/one atom, G4 continuous/two atoms, at least 80 selected formal key events, dual-view support, and zero quarantined-event leakage.

For A and Z, do not borrow the five-group or 80-event baseline. Their group count and continuity arise from actual dual-view evidence; naturally unobserved categories are recorded, not fabricated.

For every successful collection require:

- aligned first-person, third-person, and side-by-side experiment media plus JSON;
- object-aware key frames/clips/timestamps nested by experiment and five action categories;
- `continuous_action_state_ledger.json` in shadow mode without mutating CV acceptance;
- fine-grained current-step/next-step multimodal understanding;
- searchable evidence/index/integrity references;
- branded daily report and distinct professional PDF with evidence images;
- complete stage/total timing and input/output/cached/total Token receipts;
- English archive names and atomic formal promotion.

## Performance and observability

Record per collection all stage wall times, CPU/RAM, GPU compute/VRAM/power/temperature/clocks, NVDEC/NVENC, wired NAS throughput/SMB errors, decode/feed/queue/inference waits, batch distribution, indexed sources/roles/segments, source-copy bytes, model calls/retries/latencies, and stage/total Token.

Keep the six-view ≤1,200-second preprocessing target. Report A/Z separately because their workloads are lighter.

## NAS paths

- Index: `Y:\experiment_record_index.csv`
- Formal archive: `Y:\VisionCortexExperimentArchive`
- Staging: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging`
- History: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-History`
- Registry: `Y:\VisionCortexExperimentArchive\.VisionCortex-System\Collection-Processing-Registry.json`
- Runtime/cache: exact paths resolved by the frozen production config

## GitHub protocol

Use Issue #3 and task ID `DEV-20260818-040` only:

1. one ACK with frozen proofs and the prior DEV-039 preservation facts;
2. one PREFLIGHT with live dynamic index resolution, paths/capacity, engine hashes, credential-presence boolean, wired link, zero payload reads, and the existing six-view formal archive snapshot;
3. at most one HEARTBEAT every 5–10 minutes while a process is active;
4. one final COMPLETED or INCIDENT with the complete three-dataset quality/performance/resource/Token/output/promotion matrix and formal archive before/after proof.

Never expose API keys, secrets, credential values, or environment-variable contents.
