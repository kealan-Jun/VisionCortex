# DEV-20260818-041 — Three-Dataset Full-Pipeline Acceptance

## Objective

Execute exactly one new cold-start full pipeline for each indexed collection, in order:

1. `exp_20260810_144014_e918b762` → `Six-View-Three-Hour-Experiment-2026-08-13`;
2. `exp_20260811_161842_ca175e55` → `CustomFlow-Standard-Correct-12-A-0004-2026-08-11`;
3. `exp_20260812_133147_53d6cbd4` → `CustomFlow-Standard-Correct-12-Z-0001-2026-08-12`.

DEV-040 consumed only the six-view launch. A and Z were not started. DEV-041 grants one new launch for each collection under the new frozen release. A first process launch consumes that collection's one-shot even if it exits immediately. Never retry, resume, tune, change cache identity, or repair an output manually.

## Why this release exists

DEV-040 proved three shared issues that are corrected by this release:

- the source `experiment_id` was overwritten by the English archive name before baseline selection;
- one early cross-view event could close a long refined candidate while a later atomic action remained first-person-only;
- objectless motion intervals with no coarse YOLO support were still promoted to expensive 10 FPS Fine scanning.

The release keeps source identity and archive identity separate, audits temporal anchor clusters independently, retains the original motion envelope only for recall after coarse refinement, quarantines exhausted single-view context outside formal groups, and quarantines objectless motion with no coarse YOLO match before Fine.

DEV-040 retained-ledger replay is diagnostic evidence, not a new runtime result:

- G4 `REFINED-MOTION-FUSED-000009` changes from prematurely covered to `needs_third_person_supplement` and includes the late first-person transfer sequence;
- `MOTION-FUSED-000002` and `BURST-lubancat-e8cc0cb3_cam01-000007` become explicit non-formal single-view quarantine candidates;
- six objectless/no-coarse motion intervals represent approximately 1,730 selected video seconds and 17,301 10 FPS frames that no longer enter Fine;
- all five known six-view experiment windows retain object or coarse YOLO support.

Only the real one-shot run may prove final quality and wall-clock performance.

## Immutable release gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Frozen branch metadata: `codex/post-dev040-engineering-hardening`
- Taskbook: `docs/2026-08-18_DEV-041-three-dataset-full-pipeline.md`
- Coordination: GitHub Issue #3 only
- Frozen SHA authority: the complete SHA in the unique `DEV-20260818-041 RELEASE` comment

Fetch, check out the RELEASE SHA in detached-HEAD mode, and prove HEAD/remote equality, taskbook presence and SHA256, clean worktree, and zero active VisionCortex/labvision/FFmpeg production processes before source payload reads. Never execute a moving branch tip.

The RTX 4060 executor must not modify code, configuration, thresholds, models, dependencies, drivers, source data, caches, staging, formal archives, or Git history. It must not run development tests or product HTTP/WebSocket probes. It performs only the three authorized production commands and evidence collection.

## Dynamic input and storage contract

Resolve every source, role, MP4 segment, clock CSV, and path from `Y:\experiment_record_index.csv` plus the versioned role-resolution receipt. Do not hard-code source count, camera count, or an upper bound. A and Z currently resolve as two first-person views; that observation must not become a product-wide assumption.

Keep the original 15-minute MP4 and CSV segments immutable behind the virtual continuous timeline. Do not concatenate or copy source payloads. `source_copy_bytes` and continuous source-copy bytes must remain zero.

All runtime/cache work stays at the exact local paths resolved by the frozen production config. Every completed stage publishes its receipt/artifacts to the unique NAS staging package. Formal archives change only through verified atomic promotion after the complete package passes.

## Exact commands

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Get-Location).Path
$Labvision = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot '.venv\Scripts\labvision.exe')).Path
$Config = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml')).Path
```

### Six-view — exactly one DEV-041 launch

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260810_144014_e918b762' `
  --archive-name 'Six-View-Three-Hour-Experiment-2026-08-13' `
  --config $Config
```

### A — exactly one DEV-041 launch when permitted

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260811_161842_ca175e55' `
  --archive-name 'CustomFlow-Standard-Correct-12-A-0004-2026-08-11' `
  --config $Config
```

### Z — exactly one DEV-041 launch when permitted

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260812_133147_53d6cbd4' `
  --archive-name 'CustomFlow-Standard-Correct-12-Z-0001-2026-08-12' `
  --config $Config
```

Wait for the collection-owned Python/FFmpeg process tree and terminal receipts before launching the next collection.

## Continue and stop rules

- A normal collection-specific content or quality failure preserves staging and may proceed to the next collection.
- Stop all later launches for a shared code/runtime exception, source-identity/baseline-selection mismatch, GPU/driver/TensorRT/model failure, credential/API failure, NAS/storage failure, repository-integrity failure, archive-preservation failure, unsafe promotion, or orphaned process tree.
- For the six-view run, `baseline_selection.current_experiment_id` must equal `exp_20260810_144014_e918b762`, `applied` must be true, and the reason must be `experiment_id_matched`. Any other identity result is a shared failure and stops A/Z.
- For A and Z, the six-view baseline must be explicitly inapplicable; do not borrow its five-group, boundary, continuity, or 80-event expectations.
- Never weaken a gate or use a reviewed baseline to make an inference decision. The baseline remains evaluation-only.

## Quality and output gates

For the six-view benchmark require, in the same cold run:

- five formal experiment groups;
- reviewed starts and ends within ±8 seconds;
- G2 independent with one atom;
- G4 continuous with two atoms;
- at least 80 selected formal key events;
- formal experiment and key-material evidence aligned across first- and third-person views;
- no quarantined-event leakage into formal groups or key materials;
- no unresolved formal anchor cluster.

For A and Z, group count, experiment names, boundaries, continuity, action count, and available categories must arise from their own indexed video evidence. Naturally absent categories are recorded as absent and never fabricated.

For every successful collection require:

- aligned first-person, third-person, and side-by-side experiment MP4 plus matching JSON;
- object-aware key frames, clips, timestamps, and JSON nested by experiment and the five physical action families;
- continuous-action state receipts and searchable JSON/JSONL/SQLite evidence references;
- fine-grained current-step and evidence-supported next-step model understanding;
- branded daily report and a distinct professional PDF with evidence images;
- complete stage/total wall time and input/output/cached/total Token ledgers;
- English archive paths and verified atomic promotion.

## Performance and observability

The six-view preprocessing target remains at or below 1,200 seconds. A and Z are lighter workloads and must be reported separately; their absolute time must not be compared as if they were six-view/three-hour equivalents.

Record for each collection:

- all stage wall times and end-to-end time;
- source count, role count, segment count, source bytes, zero-copy bytes, and cache identity;
- Motion/coarse/Fine work units, windows, frames, calls, batch distribution, queue/decode/feed/inference waits, persistent sessions, and fallbacks;
- objectless motion quarantine count, temporal anchor-cluster closure, non-formal single-view quarantine, and unresolved formal clusters;
- CPU/RAM, GPU compute/VRAM/power/temperature/clocks, NVDEC/NVENC;
- wired Ethernet/SMB state, host receive throughput, process-tree reads/writes, retries, and errors;
- model call count, concurrency, latency distribution, retries/failures, and stage/total input/output/cached/total Token;
- every generated media/report/index file count, bytes, integrity receipt, and promotion result.

## NAS paths

- Index: `Y:\experiment_record_index.csv`
- Formal archive: `Y:\VisionCortexExperimentArchive`
- Staging: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging`
- History: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-History`
- Registry: `Y:\VisionCortexExperimentArchive\.VisionCortex-System\Collection-Processing-Registry.json`
- Runtime/cache: exact local paths resolved by the frozen production config

## GitHub protocol

Use Issue #3 and task ID `DEV-20260818-041` only:

1. one ACK with frozen proofs and explicit one-shot accounting;
2. one PREFLIGHT with dynamic index resolution, source/role matrix, path/capacity/engine hashes, credential-presence boolean, wired link, zero payload reads, and formal archive snapshots;
3. at most one HEARTBEAT every 5–10 minutes while a production process is active;
4. one final COMPLETED or INCIDENT containing the full three-dataset quality, performance, hardware, network, Token, output, quarantine, and promotion matrix with evidence paths.

Never expose API keys, credential values, secrets, or environment-variable contents.
