# DEV-20260817-037 — Three-Dataset Capability Acceptance

## Objective

Evaluate the current VisionCortex product capability on the three user-provided datasets without weakening the accepted CV, cross-view, boundary, or evidence rules:

1. accepted six-view benchmark archive;
2. `CustomFlow_standard_correct_12_A_0004`;
3. `CustomFlow_standard_correct_12_z_0001`.

This is a production-data acceptance task, not a development or tuning task. The RTX 4060 executor must not edit code, configuration, models, thresholds, dependencies, drivers, reports, or source data; must not run the test suite; and must not retry a real run.

## Release gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`.
- Branch: `codex/dev037-three-dataset-acceptance`.
- Frozen SHA: use the exact SHA posted by the developer together with this task. Do not infer or follow a moving branch tip.
- Coordination: GitHub Issue #3 only.
- Fetch first, check out the exact frozen SHA in detached-HEAD mode, and prove `HEAD == frozen SHA` with a clean worktree.
- Confirm that no VisionCortex Python or FFmpeg process is active before each real run.
- Do not start media access until the exact frozen SHA has been supplied.

## Why this release exists

It contains two input-compatibility fixes only:

- segmented views retain the true FPS of every physical MP4 instead of rejecting a legitimate per-segment nominal-FPS change;
- a clean FFmpeg session may reconcile exactly one absent terminal post-FPS sample at EOF, with an explicit receipt. Extra frames, interior-window shortfalls, and shortfalls greater than one remain fatal.

The release must not alter experiment grouping, CV thresholds, boundary rules, cross-view rules, or key-event selection.

## Authoritative source receipt

Resolve every source from `Y:\experiment_record_index.csv`. Never hard-code a source-count limit or a `2+4` layout.

### Dataset A

- User label: `CustomFlow_standard_correct_12_A_0004`
- `experiment_id`: `exp_20260811_161842_ca175e55`
- English archive: `CustomFlow-Standard-Correct-12-A-0004-2026-08-11`
- Current index snapshot: 5 logical views, 25 MP4 segments, 25 frame-clock CSV files.
- Current role receipt: 1 first-person and 4 third-person views.

### Dataset Z

- User label: `CustomFlow_standard_correct_12_z_0001`
- `experiment_id`: `exp_20260812_133147_53d6cbd4`
- English archive: `CustomFlow-Standard-Correct-12-Z-0001-2026-08-12`
- Current index snapshot: 5 logical views, 35 MP4 segments, 35 frame-clock CSV files.
- Current role receipt: 1 first-person and 4 third-person views.

The index and versioned role-resolution receipt are authoritative. If the live index differs, report the exact change. Stop before decode only if a collection no longer has at least one first-person and one third-person view. Never invent or manually relabel a view.

## Three-dataset execution plan

### Phase 1 — Existing six-view benchmark

Use the accepted archive at:

`Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13`

Perform a read-only release/reference verification:

- verify promotion, quality, evidence-package, index, report, PDF, media-reference, timing, and token receipts;
- verify the five accepted experiment groups, accepted boundary audit, dual-view clips, key materials, branded daily report, and professional PDF;
- report the existing accepted metrics from their authoritative receipts;
- do not read source MP4 payloads, rerun CV, regenerate model understanding, call the MLLM, or consume new Token;
- do not replace any benchmark archive file.

This benchmark is the regression reference. It is not a third expensive source rerun.

### Phase 2 — Dataset A

Run exactly once from a new cold cache identity:

```powershell
labvision run-index-collection `
  --experiment-id exp_20260811_161842_ca175e55 `
  --archive-name CustomFlow-Standard-Correct-12-A-0004-2026-08-11 `
  --config .\configs\rtx4060-laptop-production.yaml
```

### Phase 3 — Dataset Z

After every Dataset A process has exited and its terminal result is recorded, run exactly once from a different new cold cache identity:

```powershell
labvision run-index-collection `
  --experiment-id exp_20260812_133147_53d6cbd4 `
  --archive-name CustomFlow-Standard-Correct-12-Z-0001-2026-08-12 `
  --config .\configs\rtx4060-laptop-production.yaml
```

Do not run A and Z concurrently. If A has a content or quality failure after clean infrastructure execution, retain its staging evidence and continue to Z. If A exposes a shared infrastructure, code, model-loading, credential, storage, driver, or archive-promotion failure, stop before Z.

Each real-run authorization is consumed by its first launch. No retry, continuation, cache reuse, threshold adjustment, or second launch is authorized.

## Required full-chain capability for A and Z

Each new dataset must execute the product chain through the formal promotion gate:

1. index-driven, zero-copy segmented ingest;
2. global timestamp normalization, nearest-neighbour alignment, and visual-anchor audit;
3. motion probe, bounded YOLO candidate generation, progressive Fine scan, tracking, and candidate audit;
4. independent/continuous bounded experiment grouping;
5. aligned first-person, third-person, and side-by-side experiment MP4s plus their JSON;
6. five key-material categories: hand-object contact, object movement, liquid movement, container-state change, and device-panel operation;
7. per-experiment and per-category key frames, key clips, timestamps, cross-view links, source references, hashes, and normalized event JSON;
8. fine-grained experiment and material understanding, including current step, next step, objects, observed facts, supported inference, uncertainty, and contradictions;
9. searchable evidence index, `physical_change_log`, artifact registry, and durable media references;
10. branded `VC-LAB-DAILY-REPORT-V2` daily report and `VC-PROFESSIONAL-EVIDENCE-REPORT-V1` PDF with selected evidence images;
11. quality, evidence-package, retrieval/index, daily-report, and PDF evaluations;
12. SHA256-verified staging promotion to the exact English NAS archive only after every gate passes.

Do not impose the six-view benchmark's expected five groups or `>=80` key-event rule on A or Z. Report their naturally accepted groups and five-category distribution. Formal experiments and formal key materials must have valid first/third-person support. Single-view evidence may be quarantined but must not silently open, extend, or bridge a formal experiment. A zero category count is allowed only with an evidence-based explanation.

## NAS and archive rules

- Source index: `Y:\experiment_record_index.csv`
- Formal root: `Y:\VisionCortexExperimentArchive`
- Processing registry: `Y:\VisionCortexExperimentArchive\.VisionCortex-System\Collection-Processing-Registry.json`
- Failure staging: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging`
- Runtime cache: use the configured cache location and report its exact path and identity; do not move or concatenate source MP4s.
- Source-copy bytes must be zero.
- Keep the original hardware-generated 15-minute segment structure in `Original-Experiment-Videos` references; final accepted experiment clips must be continuous bounded MP4s.
- Archive each completed stage to staging immediately. Formal promotion remains all-or-nothing.
- All generated archive folder and file names must be English-safe.

## Performance, Token, and resource evidence

Capture telemetry from submission through final process exit:

- total wall time and every stage wall time;
- input, output, cached, and total Token per model stage and for the full run;
- model calls, retries, errors, and reuse count;
- CPU/process utilization and RAM peak;
- GPU compute, VRAM, power, temperature, clocks, NVDEC, and NVENC at p50/p95/max;
- TensorRT engines, contexts, actual batch distribution, OOM/contraction/fallback counts;
- decoded seconds/frames by source and stage, decoder lanes/backends, queue depth/full time, feed/decode/inference/materialization wait;
- wired NAS read/write throughput p50/p95/max, process bytes, SMB errors/retries, and path-stat time;
- source-copy bytes, cache identity, and `reused=0` for A and Z;
- terminal EOF reconciliation count and per-session receipts. It must be zero or an explicitly proven one-frame terminal reconciliation; no unreconciled mismatch is allowed.

Normalized preprocessing budget remains 363.64 seconds per collection-duration hour: approximately 380 seconds for A and 610 seconds for Z. Performance failure must not weaken quality.

## User-visible acceptance

After each terminal result, read-only Web verification is explicitly authorized. Use only GET/WebSocket reads and browser screenshots; no product POST/PUT/PATCH/DELETE request is allowed.

For every promoted dataset, verify that a user can navigate from batch state to:

- bounded experiments and aligned videos;
- detailed current/next-step understanding;
- five-category key materials;
- key frames, clips, timestamps, and evidence JSON;
- searchable physical changes and source references;
- timing/Token observability;
- branded daily report and professional PDF.

## GitHub reporting protocol

Use Issue #3 and task ID `DEV-20260817-037`.

1. One `ACK`: machine, exact frozen SHA, detached-HEAD proof, clean worktree, idle-process proof, and plan.
2. One `PREFLIGHT`: exact index rows/roles/segments, NAS paths/free space, model engines, credential-presence boolean, wired link, and benchmark archive verification plan.
3. Heartbeat every 5–10 minutes during each real run: dataset/stage, elapsed time, view/unit progress, CPU/GPU/NVDEC/NVENC/RAM/NAS, queue/wait/batch, API/Token totals, and staging path.
4. One final `COMPLETED` only if A and Z are promoted and the three-dataset matrix is complete. Otherwise one final `INCIDENT` with the complete matrix and all retained evidence paths.

The final matrix must contain, per dataset:

- input IDs, indexed roles and segment counts;
- run/cache ID or read-only benchmark reference;
- result and failure gate, if any;
- accepted/quarantined groups, continuity/atomic counts, boundaries, and supporting views;
- total and five-category key-material counts;
- experiment/material model-understanding results;
- searchable index, physical-change, and artifact-reference validation;
- daily/PDF template IDs, evidence-image counts, evaluation results, and hashes;
- total/stage timing and total/stage Token;
- resource/NAS distributions;
- formal archive or staging path, registry state, promotion hash, Web screenshots, and residual process count.

Never expose an API key, secret, or credential value in GitHub, logs, JSON, screenshots, terminal output, or reports.
