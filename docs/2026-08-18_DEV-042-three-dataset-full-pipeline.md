# DEV-20260818-042 — Three-Dataset Quality-Recovery Full Pipeline

## Objective

Execute exactly one new cold-start production pipeline for each indexed collection, in order:

1. `exp_20260810_144014_e918b762` → `Six-View-Three-Hour-Experiment-2026-08-13`;
2. `exp_20260811_161842_ca175e55` → `CustomFlow-Standard-Correct-12-A-0004-2026-08-11`;
3. `exp_20260812_133147_53d6cbd4` → `CustomFlow-Standard-Correct-12-Z-0001-2026-08-12`.

Each collection receives one launch only. The first process launch consumes that collection's one-shot even if it exits immediately. Never retry, resume, tune thresholds, switch cache identity, edit retained staging, or repair an output manually.

This is a real full-pipeline acceptance. It must continue through model understanding, media materialization, searchable evidence indexes, the laboratory daily report, the professional PDF, integrity checks, and verified formal-archive promotion whenever the quality gates pass.

## Why this release exists

DEV-041 proved the six-view cold preprocessing target at `1169.47s`, but all three collections stopped at `candidate_audit`. The retained ledgers identified one shared gate defect:

- rejected first-person candidates were correctly used to request supplemental third-person scanning;
- after every eligible third-person view was exhausted, clusters containing only `accepted=false` events were still treated as mandatory formal anchors;
- those weak audit candidates failed A, Z, and the six-view G4 even though accepted dual-view clusters already proved the formal experiments.

This release preserves the high-recall scan behavior but separates final evidence authority:

- accepted dual-view anchors remain mandatory formal evidence;
- an accepted single-view strong anchor remains blocking until corroborated;
- an exhausted cluster containing only rejected events remains searchable audit context, cannot fail an otherwise proven dual-view experiment, and cannot enter formal events or key materials;
- a rejected first-person context chain may connect two proven dual-view atomic fragments only when the fragments share the delivered FP/TP pair, the same recalled coarse boundary, a bounded context chain, and at least two normalized manipulated-object families; context events remain graph-only;
- a strong single-role trailing segment may extend clip time only inside the same recalled boundary with at least two shared non-hand objects; its events remain outside formal/key-material membership.

Read-only replay of the exact DEV-041 JSON ledgers, without MP4/CSV reads or model calls, produced:

- six-view: `5/5` groups, all reviewed boundaries within `±8s`, G4 `continuous / 2 atoms`, no unresolved formal anchor cluster;
- six-view G1 end error: `-29.775s → +0.525s`;
- A: one naturally derived dual-view experiment, no unresolved formal anchor cluster;
- Z: two naturally derived dual-view experiments, no unresolved formal anchor cluster;
- rejected context leakage into formal event membership: zero.

These are deterministic ledger-replay results, not a replacement for this real run.

## Immutable release gate

- Repository: `https://github.com/kealan-Jun/VisionCortex.git`
- Frozen branch: `codex/post-dev041-quality-gate`
- Taskbook: `docs/2026-08-18_DEV-042-three-dataset-full-pipeline.md`
- Coordination: GitHub Issue #3 only
- Frozen SHA authority: the complete SHA in the unique `DEV-20260818-042 RELEASE` comment

Fetch, check out the RELEASE SHA in detached-HEAD mode, then prove HEAD/remote/release equality, taskbook SHA256, clean worktree, and zero active VisionCortex/labvision/FFmpeg production processes before reading source payloads. Never execute a moving branch tip.

The RTX 4060 executor must not modify code, configuration, thresholds, models, dependencies, drivers, source data, caches, staging, formal archives, or Git history. It must not run development tests or product HTTP/WebSocket probes. It performs only the three authorized production commands and evidence collection.

## Dynamic input and role contract

Resolve every source, camera role, MP4 segment, clock CSV, and path from `Y:\experiment_record_index.csv` and the versioned role-resolution receipt. Do not hard-code source count, role count, camera IDs, or product-wide upper bounds.

DEV-041 resolved the actual runtime matrix as:

- six-view: `1 first-person + 5 third-person`;
- A: `1 first-person + 4 third-person`;
- Z: `1 first-person + 4 third-person`.

This is evidence for these three indexed collections only. If DEV-042 resolves a different matrix, record it and stop that collection as an identity/role-resolution incident; do not override the index manually.

Keep original 15-minute MP4 and CSV segments immutable behind the virtual continuous timeline. Do not concatenate or copy source payloads. Source-copy and continuous-copy bytes must remain zero.

## Exact commands

```powershell
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Get-Location).Path
$Labvision = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot '.venv\Scripts\labvision.exe')).Path
$Config = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot 'configs\rtx4060-laptop-production.yaml')).Path
```

### Six-view — one launch

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260810_144014_e918b762' `
  --archive-name 'Six-View-Three-Hour-Experiment-2026-08-13' `
  --config $Config
```

### A — one launch when permitted

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260811_161842_ca175e55' `
  --archive-name 'CustomFlow-Standard-Correct-12-A-0004-2026-08-11' `
  --config $Config
```

### Z — one launch when permitted

```powershell
& $Labvision run-index-collection `
  --experiment-id 'exp_20260812_133147_53d6cbd4' `
  --archive-name 'CustomFlow-Standard-Correct-12-Z-0001-2026-08-12' `
  --config $Config
```

Wait for the collection-owned Python/FFmpeg process tree and terminal receipts before launching the next collection.

## Continue and stop rules

- A normal collection-specific content or quality failure preserves staging and may proceed to the next collection.
- Stop all later launches for a shared code/runtime exception, source-identity or role-resolution mismatch, baseline-selection mismatch, GPU/driver/TensorRT/model failure, credential/API failure, NAS/storage failure, repository-integrity failure, archive-preservation failure, unsafe promotion, or orphaned process tree.
- Six-view baseline selection must use `exp_20260810_144014_e918b762`, with `applied=true` and reason `experiment_id_matched`.
- A and Z must explicitly reject the six-view reviewed baseline as inapplicable. Never borrow its five-group, boundary, continuity, or 80-event expectations.
- Never weaken a gate or use the reviewed baseline as an inference rule.

## Quality gates

For the six-view benchmark require in the same cold run:

- exactly five formal experiment groups;
- every reviewed start and end within `±8s`;
- G2 independent with one atomic experiment;
- G4 continuous with two atomic experiments;
- at least 80 selected formal key events;
- no unresolved formal anchor cluster;
- no rejected/quarantined context event in formal groups or key materials;
- every formal experiment and key material has aligned first-person and third-person evidence or an explicit aligned counterpart clip at the same global timestamp.

For A and Z, names, group count, boundaries, continuity, action count, and available categories must arise from their own indexed evidence. Naturally absent categories are recorded as absent and never fabricated. Report the observed counts, but do not turn the DEV-041 replay counts into hard-coded inference constraints.

For every successful collection require:

- aligned first-person, third-person, and side-by-side experiment MP4 plus matching JSON;
- object-specific English key-material names nested by experiment and the five action families;
- key frames, key clips, global timestamps, counterpart references, and schema-compliant event JSON;
- searchable JSON/JSONL/SQLite indexes with stable evidence IDs and relative archive references;
- fine-grained current-step and evidence-supported next-step model understanding;
- a branded laboratory daily report and a distinct professional PDF with representative evidence images;
- complete stage/total wall time and stage/total input/output/cached/total Token ledgers;
- verified atomic promotion. Existing six-view files may change only through the safe replacement transaction; failed staging must never replace the formal archive.

## Performance and observability

The six-view preprocessing gate is `≤1200s`. A and Z are lighter collections; report their times separately and never compare them as if they were six-view/three-hour equivalents.

Record per collection:

- all stage and end-to-end wall times;
- source/role/segment counts, source bytes, zero-copy bytes, and cache identity;
- Motion/coarse/Fine windows, frames, calls, batch distribution, queue/decode/feed/inference wait, persistent sessions, and fallbacks;
- weak-anchor recall attempts, exhausted weak-cluster quarantine, strong blocking clusters, context-chain bridges, tail-context boundary extensions, and leakage checks;
- CPU/RAM, GPU compute/VRAM/power/temperature/clocks, NVDEC/NVENC;
- wired Ethernet/SMB state, NAS throughput, process-tree reads/writes, retries, and errors;
- model call count, concurrency, latency distribution, retries/failures, and stage/total input/output/cached/total Token;
- generated media/report/index counts, bytes, integrity receipts, archive backup/replacement receipt, and promotion result.

## NAS paths

- Index: `Y:\experiment_record_index.csv`
- Formal archive: `Y:\VisionCortexExperimentArchive`
- Staging: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging`
- History: `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-History`
- Registry: `Y:\VisionCortexExperimentArchive\.VisionCortex-System\Collection-Processing-Registry.json`
- Runtime/cache: exact local paths resolved by the frozen production config

## GitHub protocol

Use Issue #3 and task ID `DEV-20260818-042` only:

1. one ACK with frozen proofs and one-shot accounting;
2. one PREFLIGHT with dynamic index/role resolution, capacity/engine hashes, credential-presence boolean, wired link, zero payload reads, and formal archive snapshots;
3. at most one HEARTBEAT every 5–10 minutes while a production process is active;
4. one final COMPLETED or INCIDENT with the complete three-dataset quality, performance, hardware, network, Token, output, quarantine, and promotion matrix plus evidence paths.

Never expose API keys, credential values, secrets, or environment-variable contents.
