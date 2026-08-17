# DEV-20260817-036 — Two New NAS Collections + Report Refresh

## Authorization and launch gate

- Executor: RTX 4060 Laptop production machine.
- Repository: `https://github.com/kealan-Jun/VisionCortex.git`.
- Release branch: `codex/index-driven-ingest`.
- Frozen SHA: **PENDING DEVELOPER RELEASE COMMENT**.
- GitHub coordination: repository Issue #3 only.
- ACK this task immediately, sync the repository, and complete metadata-only preflight.
- Do **not** start video decode or a product run until the developer posts the exact frozen SHA in Issue #3 as `DEV-036 RELEASE`.
- After release, verify `HEAD == upstream == origin/<release-branch> == frozen SHA` and a clean worktree.
- Never modify source, config, models, dependencies, drivers, reports, or thresholds. Do not run tests. Do not tune after seeing results.

## Product rule

Source count and view composition are dynamic. Do not impose a six-source, `2+4`, or any other fixed limit. Use exactly the rows resolved from `Y:\experiment_record_index.csv` for each `experiment_id`. Decoder lanes are capacity slots, not a user-facing source limit; excess sources advance in bounded waves without being dropped.

## Authorized collections

### Collection A

- User label: `CustomFlow_standard_correct_12_A_0004`
- Exact source `experiment_id`: `exp_20260811_161842_ca175e55`
- Indexed interval: `2026-08-11T16:18:42.738206+08:00` to `2026-08-11T17:21:17.100807+08:00`
- Expected from current index snapshot: 5 sources, 25 MP4 segments, 25 frame-clock CSV files.
- Current indexed roles: 1 first-person + 4 third-person.
- Formal English archive name: `CustomFlow-Standard-Correct-12-A-0004-2026-08-11`

### Collection Z

- User label: `CustomFlow_standard_correct_12_z_0001`
- Exact source `experiment_id`: `exp_20260812_133147_53d6cbd4`
- Indexed interval: `2026-08-12T13:31:47.787418+08:00` to `2026-08-12T15:12:21.924402+08:00`
- Expected from current index snapshot: 5 sources, 35 MP4 segments, 35 frame-clock CSV files.
- Current indexed roles: 1 first-person + 4 third-person.
- Formal English archive name: `CustomFlow-Standard-Correct-12-Z-0001-2026-08-12`

The index and the versioned device-role receipt are authoritative. If the release resolves a different actual composition, report the exact receipt and stop before decode if the first/third-person quality gate is incomplete. Never invent a view or manually remap it without an approved receipt.

## Existing benchmark report refresh

Refresh the accepted archive at:

`Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13`

Requirements:

- Reuse its existing accepted evidence package, key materials, experiment clips, model understanding, and token ledger.
- Do not decode source video, run YOLO/TensorRT, call the MLLM, or consume new model tokens.
- Preserve hashes and a recoverable copy of the previous daily/PDF presentation files under the run-history area before replacement.
- Regenerate with `VC-LAB-DAILY-REPORT-V2` and `VC-PROFESSIONAL-EVIDENCE-REPORT-V1`.
- Daily report: concise operating summary, experiment timeline, five action-category counts, exceptions, next actions, selected evidence images, Logo and theme color.
- Professional PDF: evidence-oriented narrative, methods and provenance, cross-view support, key images, step understanding, uncertainty/contradictions, performance and token appendix, Logo and theme color.
- Validate both report evaluations before retaining the refreshed presentation.
- Register the benchmark source collection as `archived` in the central collection-processing registry without pretending that this report-only refresh is a new full pipeline run.

## Execution order and one-shot policy

1. Metadata-only preflight for both collections and the existing benchmark archive.
2. Refresh and validate the existing benchmark daily report/PDF without model calls.
3. Run Collection A exactly once with a new cold cache identity.
4. After every A process has exited and the final result is recorded, run Collection Z exactly once with a different new cold cache identity.

Each collection has one authorized real run. No retry, continuation, parameter adjustment, cache reuse, or second launch. Run A and Z sequentially; do not make them compete for GPU, NAS bandwidth, model rate limits, or telemetry ownership.

If A fails a content/quality gate after clean execution, retain its staging evidence and continue to Z so that the second dataset still contributes diversity evidence. If A fails because of shared infrastructure, code, model loading, driver, storage, archive promotion, credential, or API availability, stop before Z and report the shared blocker.

## Metadata-only preflight

Before the frozen release is posted, you may only:

- fetch and inspect Git metadata;
- read the two exact index row groups;
- verify the expected 5 rows and 1+4 role receipt per collection;
- stat the 60 indexed MP4 paths and 60 CSV paths concurrently without opening media payloads;
- verify NAS archive/cache/runtime paths and free space;
- verify the two TensorRT engines can be located, without inference;
- verify the model credential environment variable is present, without printing its value;
- verify no VisionCortex Python/FFmpeg process is running;
- record wired NAS interface/link state and an idle bandwidth baseline.

Do not concatenate or copy 15-minute segments. Original media remains in its indexed NAS locations and is represented as one virtual global timeline. Source-copy bytes must be zero.

## Full-pipeline acceptance for each new collection

The released product entry must run the full chain:

1. index-driven zero-copy ingest and source receipt;
2. timestamp normalization, nearest-neighbor matching, and visual-anchor correction;
3. motion probe, coarse candidate generation, bounded progressive fine scan, tracking, and candidate audit;
4. bounded real experiment grouping as independent or continuous experiments;
5. accepted multiview experiment clips and per-view/side-by-side MP4 + JSON;
6. five-category key material extraction: hand-object contact, object movement, liquid movement, container-state change, device-panel operation;
7. key frames, key clips, aligned timestamps, cross-view associations, provenance, and normalized event JSON;
8. fine-grained experiment and material understanding: current step, next step, objects, observed facts, supported inference, uncertainty, contradictions;
9. searchable evidence package and material/artifact references;
10. latest branded lab daily report and professional PDF with selected evidence images;
11. quality/evidence/report evaluations;
12. SHA256-verified staging promotion to the exact formal NAS archive only after every promotion gate passes.

Do not impose a benchmark-specific expected experiment count or an `>=80` material count on these new collections. Report the naturally accepted groups and category distribution. Every formal group and key material must have valid supporting multiview evidence; single-view content may be quarantined and must not silently open, extend, or bridge a formal cross-view experiment. A category count of zero is allowed only with an explicit evidence-based explanation.

All archive folders and files must use English-safe names. Key materials must be nested first by experiment and then by the five action categories while preserving stable JSON references.

## Performance and resource evidence

Collect telemetry from before preflight through final process exit, not merely the first minutes of Fine:

- wall time from submission to terminal state;
- every stage wall time, including ingest, source validation, alignment, motion, coarse, Fine waves, candidate audit, MLLM, media export, indexing, daily report, PDF, validation, and promotion;
- per-stage input/output/cached model tokens and API call/retry/error counts;
- CPU total and process utilization; RAM used/peak;
- GPU compute, VRAM, power, temperature, clocks, NVDEC and NVENC utilization at p50/p95/max;
- actual TensorRT engine, execution contexts, batch distribution, OOM/contraction/fallback count;
- decoder backends and lanes actually used; decoded frames/video seconds per stage and source;
- queue depth/full time, feed wait, decode wait, inference wall time, and materialization wait;
- wired NAS throughput p50/p95/max, process read bytes, path-stat time, SMB errors/retries, and total bytes read/written;
- zero-copy source bytes and local/NAS cache identity, with `reused=0` for A and Z.

The normalized preprocessing budget remains at most 363.64 seconds per input-video hour, equivalent to 20 minutes for 3.3 hours. This implies evidence targets of approximately 380 seconds for A and 610 seconds for Z. These are performance gates, not reasons to weaken quality or retry.

## Archive and batch-state rules

- Central state ledger: `Y:\VisionCortexExperimentArchive\.VisionCortex-System\Collection-Processing-Registry.json`.
- State vocabulary: `queued`, `processing`, `failed`, `archived`.
- A batch is `archived` only after quality, evidence-package, daily-report, professional-PDF, and promotion verification all pass.
- Cache or staging presence is never equivalent to processed/retained.
- On failure, record `failed`, staging path, failure stage, and error; do not create or replace a formal archive.
- Formal outputs must be visible in the Web experiment pages and referenced by the central ledger.

## GitHub reporting protocol

Use Issue #3 and task ID `DEV-20260817-036`.

1. One `ACK` comment immediately: machine, current branch/SHA, worktree, release gate waiting state, preflight plan.
2. One `PREFLIGHT` comment after metadata-only checks: exact rows, roles, segment counts, stat results, NAS/runtime/cache/free space, engines, credential-presence boolean, idle processes and wired-link baseline.
3. Heartbeat every 5–10 minutes during each real run: current collection/stage, elapsed time, source progress, GPU/NVDEC/NVENC/CPU/RAM/NAS snapshots, queue/wait/batch metrics, current API/token totals, staging path.
4. One final `COMPLETED` only if both authorized runs reach their terminal results and all requested receipts are present; otherwise `INCIDENT` with explicit per-collection state.

The final result must include:

- frozen SHA and clean-worktree proof;
- exact run IDs/cache identities;
- collection names, source IDs, view/segment counts and source-copy bytes;
- formal archive or staging path per collection;
- accepted/quarantined groups, independent/continuous structure, atomic count, boundaries and views;
- total and five-category material counts;
- full stage timing and total timing tables;
- total and stage token tables split into input/output/cache;
- all resource/NAS/performance distributions;
- report/PDF template IDs, hashes, visual counts and evaluation results;
- batch-state ledger entries;
- residual process count and formal archive hash proof;
- a concise list of real issues with evidence, likely cause, status, and one next action.

Never expose any API key or secret in GitHub, logs, JSON, screenshots, terminal output, or reports.
