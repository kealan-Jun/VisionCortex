# DEV-20260817-033: RTX 4060 Capability Freeze Evidence Capture

## Task type

This is a read-only capability-freeze inventory on the RTX 4060 execution node.
It is not a development task, quality-fix task, test task, benchmark run, or
pipeline retry.

The purpose is to preserve a trustworthy description of what is already proven
before the developer machine changes quality rules or begins modularization.
Do not modify, tag, repackage, or promote anything on the RTX 4060 node.

## Authoritative context

- Repository: `kealan-Jun/VisionCortex`
- Collaboration location: GitHub Issue `#3`
- DEV-031 run ID: `cli-20260817-095055-34d8`
- DEV-031 runtime commit:
  `03f4ec0d4ac7432f55a3e5f030181ceea53b64ee`
- DEV-031 staging:
  `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\cli-20260817-095055-34d8`
- DEV-031 cache:
  `D:\VisionCortexLocal\Cache\Six-View-Three-Hour-Experiment-2026-08-13\9c3c60dcecccbe81fcd0`
- Existing formal archive:
  `Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13`
- DEV-032 final evidence:
  `https://github.com/kealan-Jun/VisionCortex/issues/3#issuecomment-5311341855`

DEV-031 proved the preprocessing performance candidate can finish in
`914.920986s`, but it failed the deterministic quality gate. DEV-032 proved the
15 adjacent-segment duplicate grid timestamps were not causal for G1/G3/G4/G5.
The current performance commit must therefore be described as a performance
candidate, not as a production release.

## Absolute prohibitions

Do not:

- run or resume the pipeline, preprocessing, fixed benchmark, dry-run, Web
  submission, report generation, or any validation run;
- read any MP4 frame, original RGB payload, clock CSV content, or source image;
- run FFmpeg/FFprobe against media, YOLO, TensorRT inference, CUDA inference,
  model imports that allocate GPU memory, MLLM, Ark/Doubao API, or any API test;
- run pytest, compilation, frontend build, installation, dependency update,
  model conversion, TensorRT engine build, or performance benchmark;
- fetch, pull, switch, checkout, reset, stash, clean, merge, tag, commit, push,
  or otherwise change Git state;
- edit, create, delete, rename, copy, move, touch, or reserialize repository,
  configuration, model, staging, cache, formal archive, or NAS files;
- create a local or NAS freeze report; the only output is GitHub Issue #3;
- compute SHA256 by reading original videos, experiment MP4 outputs, key clips,
  or key-frame image payloads;
- print, inspect, export, or record any credential or the value of
  `ARK_API_KEY`.

Allowed operations are metadata-only Git/process/system queries, SHA256 of
code/config/model/ledger/report files explicitly allowed below, NAS file
existence/size enumeration, and reading existing JSON/JSONL/Markdown/TXT
receipts. `ffmpeg -version` and `ffmpeg -encoders` are allowed because they do
not open media. `nvidia-smi` metadata queries are allowed; GPU workloads are not.

## Start gate

Append exactly one comment titled:

`ACK · DEV-20260817-033`

The ACK must include:

- actual repository path, branch, HEAD, upstream and remote branch SHA;
- exact `git status --porcelain` output;
- confirmation that benchmark, pipeline, Python worker and FFmpeg processes are
  not running, except any pre-existing Web service which must remain untouched;
- resolved existence of the formal archive, DEV-031 staging and DEV-031 cache;
- before-task file count and byte count for staging/cache/formal archive,
  enumerated from metadata only;
- acknowledgement that remote may be ahead of the DEV-031 runtime SHA and that
  no synchronization is authorized;
- acknowledgement of every prohibition above.

If Git is dirty, a pipeline/benchmark/FFmpeg process is running, or any of the
three evidence roots is missing, stop with one `INCIDENT · DEV-20260817-033`
comment. Do not repair the state.

## Inventory A: immutable code and configuration identity

Record path, byte size and SHA256 for the files that actually exist. Missing
files must be reported as `NOT_PRESENT`; do not create substitutes.

1. Git identities:
   - local HEAD and commit date;
   - upstream SHA;
   - remote tracking SHA already present locally;
   - nearest tag, if any;
   - dirty/untracked file count.
2. Configuration and package identity:
   - `configs/default.yaml`;
   - `configs/rtx4060-laptop-production.yaml`;
   - the manifest/config used by DEV-031 when preserved in staging;
   - `pyproject.toml`;
   - dependency lock or requirements files, if present;
   - Web static files as one Git tree identity, not individual ad-hoc copies.
3. Model identity, using local files only:
   - first-person `.pt` and `.engine`;
   - third-person `.pt` and `.engine`;
   - path, size, modification time and SHA256;
   - do not deserialize or import the model.

Do not read any secret. Record only `ARK_API_KEY configured=true/false`, based on
presence, never its value, length, prefix, suffix, or hash.

## Inventory B: execution environment identity

Record metadata only:

- Windows edition/build;
- CPU model, physical/logical core count;
- total RAM;
- GPU model, driver version, total memory, power limit and current temperature;
- CUDA version reported by `nvidia-smi`;
- Python executable and version used by the repository;
- installed versions of PyTorch, Ultralytics, OpenCV, Pydantic and TensorRT,
  using package metadata only; do not import or initialize CUDA/model modules;
- FFmpeg and FFprobe version/build text without opening media;
- presence of `h264_nvenc` and `libx264` in the encoder list;
- local input/cache/runtime disk total/free space;
- active Ethernet adapter name, status and negotiated link speed;
- SMB mapping/connection metadata for `Y:` and its UNC target;
- no network speed test and no synthetic NAS transfer.

## Inventory C: formal end-to-end archive capability

Inspect the existing formal archive with metadata and existing receipts only.
Do not run generation or repair.

Report file count and byte count for these English-named areas when present:

- `Original-Experiment-Videos`;
- `Experiment-Clips`;
- `JSON-Config-Files`;
- `Key-Materials`;
- `Lab-Daily-Reports`;
- `Professional-PDFs`.

For each area, report required capability evidence:

- original-video references are retained without hashing video payloads;
- experiment group count, atomic experiment count and dual-view video count;
- key-event count, key-frame count and key-clip count;
- normalized event JSON and evidence-package presence;
- timing and Token ledger presence and totals;
- daily report JSON/Markdown/HTML presence;
- professional PDF presence;
- quality/eval receipt and formal promotion receipt presence;
- evidence index manifest, SQLite database, artifact registry and evidence
  registry presence when available.

SHA256 is allowed for JSON, JSONL, SQLite, Markdown, TXT, CSV, XLSX and PDF
receipts. Do not hash MP4/JPG/PNG media. If an existing artifact registry already
contains media SHA256 values, report its completeness and stored verification
status without recomputing media hashes. If no registry exists, report
`NOT_PRESENT`; do not backfill it.

Do not claim the formal archive was produced by the current performance SHA
unless an existing receipt proves that exact relationship.

## Inventory D: DEV-031 performance candidate

Re-verify the 13 small DEV-031 ledger hashes already proven by DEV-032 and cite
their resolved paths. Do not read media or recompute detection.

Extract and report:

- total/preprocessing/stage durations;
- 27 Fine sessions and `60,828 / 60,828` frame accounting;
- candidate, accepted-event and selected-event counts;
- API call and Token totals;
- quality-gate outcome and G1/G3/G4/G5 failures;
- DEV-032 conclusion for each proven cause;
- the fact that the 15 duplicate grid timestamps are ledger hygiene only and
  not a quality fix;
- staging/cache file counts, bytes and hashes before and after this task.

The capability classification must be exactly:

- performance SLA: `PROVEN`;
- end-to-end production quality for DEV-031: `NOT_PROVEN`;
- production release readiness for DEV-031: `NOT_PROVEN`.

## Inventory E: freeze matrix and gaps

Produce one matrix with these rows:

1. source ingest and zero-copy segmented timeline;
2. alignment;
3. motion/coarse/Fine preprocessing;
4. dual-view experiment grouping and boundaries;
5. key-event selection;
6. experiment-level MLLM understanding;
7. key-material MLLM understanding;
8. media materialization;
9. JSON/index lineage and integrity;
10. daily report;
11. professional PDF;
12. NAS formal promotion;
13. Web archive visibility.

Columns:

- capability;
- best preserved evidence path/run;
- code SHA when proven;
- status: `PROVEN`, `PARTIAL_EVIDENCE`, `FAILED`, or `NOT_PRESENT`;
- quality result;
- performance result;
- Token result;
- missing receipt or release blocker.

Keep the successful formal end-to-end archive and the fast-but-quality-failing
DEV-031 preprocessing candidate as separate evidence lines. Do not combine their
claims into one imaginary release.

## Required final GitHub result

Append exactly one final comment titled:

`FREEZE EVIDENCE · DEV-20260817-033`

It must contain:

1. elapsed audit time and final repository/process state;
2. code/config/model SHA256 table;
3. environment and hardware table;
4. formal archive inventory and receipt/integrity table;
5. DEV-031 performance and quality table;
6. the 13-row capability freeze matrix;
7. a concise real issue list with evidence and release blocker;
8. explicit separation of `proven full end-to-end archive` from
   `fast preprocessing candidate`;
9. explicit statement that the task performed zero pipeline runs, media reads,
   inference, model/API calls, Tokens, tests, source modifications, artifact
   changes and archive promotions;
10. after-task counts/bytes/hashes proving staging/cache/formal archive remained
    unchanged;
11. recommended freeze inputs for the developer machine, but no Git tag or
    release action executed on RTX 4060.

Allowed conclusion vocabulary:

- `PROVEN`;
- `PARTIAL_EVIDENCE`;
- `FAILED`;
- `NOT_PRESENT`;
- `NOT_PROVEN`.

Do not call the current fast preprocessing commit production-ready. Do not use
`COMPLETED` as a synonym for release readiness.

## Acceptance gate

DEV-033 is accepted only when:

- Git, repository, models, configurations and environment are identified;
- formal archive and DEV-031 candidate are inventoried separately;
- no media payload is opened or rehashed;
- every claim cites a preserved path/receipt or is marked missing;
- code/config/model hashes and final evidence-root immutability are reported;
- the capability matrix contains all 13 rows;
- API calls and Token use are zero;
- there is exactly one ACK and one final result, or one ACK and one INCIDENT;
- no local/NAS file or Git state changes occur.

After the final comment, stop. Wait for a new developer-side task. Do not start a
real six-view run until a new frozen quality-fix SHA and explicit one-shot
authorization are provided.
