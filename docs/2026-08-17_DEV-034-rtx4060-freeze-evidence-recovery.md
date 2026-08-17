# DEV-20260817-034: RTX 4060 Freeze Evidence Recovery

## Task type and purpose

This is a fresh, explicitly authorized read-only recovery of the capability
freeze inventory that DEV-033 correctly terminated after three prohibited
loopback product-API GET requests.

This task does not authorize a video run, retry, test, product API request,
model call, report generation, code change, Git synchronization, tag, package,
or archive promotion. It exists only to finish the static freeze inventory from
preserved files and system metadata.

## Source evidence

- Repository: `kealan-Jun/VisionCortex`
- Collaboration location: GitHub Issue `#3`
- DEV-031 run ID: `cli-20260817-095055-34d8`
- DEV-031 runtime commit:
  `03f4ec0d4ac7432f55a3e5f030181ceea53b64ee`
- DEV-032 audit:
  `https://github.com/kealan-Jun/VisionCortex/issues/3#issuecomment-5311341855`
- DEV-033 ACK:
  `https://github.com/kealan-Jun/VisionCortex/issues/3#issuecomment-5311536939`
- DEV-033 INCIDENT:
  `https://github.com/kealan-Jun/VisionCortex/issues/3#issuecomment-5311688034`
- Formal archive:
  `Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13`
- DEV-031 staging:
  `Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging\Six-View-Three-Hour-Experiment-2026-08-13\cli-20260817-095055-34d8`
- DEV-031 cache:
  `D:\VisionCortexLocal\Cache\Six-View-Three-Hour-Experiment-2026-08-13\9c3c60dcecccbe81fcd0`

## Corrected protocol

The conflict in DEV-033 is removed:

- Do not call any VisionCortex product endpoint, including loopback GET, health,
  archives, runs, files, metrics, or WebSocket endpoints.
- GitHub Issue API/CLI access is allowed only to post the required ACK and final
  result. It is not counted as a VisionCortex product API call.
- Web archive visibility must be derived only from already-preserved static
  source/routes, archived receipts and Web assets. Live runtime visibility must
  be reported as `NOT_PROVEN` because product API access is forbidden.
- The three DEV-033 loopback responses are tainted for freeze acceptance and
  must not be reused as evidence.

## Absolute prohibitions

Do not:

- run or resume the pipeline, preprocessing, fixed benchmark, dry-run, Web
  submission, report generation, media materialization, or validation;
- issue any VisionCortex HTTP/WebSocket request, including read-only GET;
- read any MP4 frame, source RGB payload, clock CSV content, key-frame image,
  key-clip payload, or original experiment media payload;
- run FFmpeg/FFprobe against media, YOLO, TensorRT/CUDA inference, model imports
  that allocate GPU memory, MLLM, Ark/Doubao, or any model/product API;
- run pytest, compilation, frontend build, installation, dependency update,
  model conversion, engine build, benchmark, or network speed test;
- fetch, pull, switch, checkout, reset, stash, clean, merge, rebase, tag, commit,
  push, or otherwise change Git state;
- edit, create, delete, rename, copy, move, touch, or reserialize repository,
  config, model, staging, cache, formal archive, report, or NAS files;
- create a local/NAS recovery or freeze report;
- hash MP4, JPG, PNG or original-video payloads;
- inspect, print, export, record or hash any credential value.

Allowed operations are read-only Git/process/system metadata, package metadata
without importing GPU/model modules, `nvidia-smi` metadata, `ffmpeg -version`,
`ffprobe -version`, `ffmpeg -encoders`, file existence/size enumeration, hashes
of code/config/model/JSON/JSONL/SQLite/Markdown/TXT/CSV/XLSX/PDF files, and reads
of existing non-media ledgers. Model files may be hashed but not deserialized.

## Start gate

Append exactly one comment titled:

`ACK · DEV-20260817-034`

Before the ACK, verify:

- repository path, branch, local HEAD, upstream and live remote SHA;
- `git status --porcelain=v1 --untracked-files=all` is empty;
- benchmark, pipeline, FFmpeg/FFprobe and non-Web worker process counts are zero;
- the two pre-existing Web-service processes remain untouched;
- formal archive, staging and cache exist;
- metadata-only file/byte counts equal:
  - staging: `45 / 12,353,628`;
  - cache: `29 / 55,311,020`;
  - formal archive: `1,081 / 45,314,893,035`;
- preserved manifest hashes equal:
  - DEV-031 13-ledger manifest:
    `3a618d476065a6d95ffb395a428730f447506ee6d6401bdfc9239f43da89da22`;
  - cache 29-file JSON/JSONL manifest:
    `e7c8f71b9e32d0599ff1bed9931d5d81a0eb7230f789c13878c0163d33f69858`;
  - formal 10-receipt manifest:
    `1be2af35e0884f3ba6c7f44b94c0f0f56129c5715b4f546d00b94bf4a39b063a`.

Use the exact manifest-hash definition from DEV-033: sorted relative path,
byte-size and per-file SHA256 rows joined with UTF-8 LF. Do not include or hash
media payloads.

The ACK must state that the remote is ahead but synchronization is forbidden,
and acknowledge every prohibition above. If any start gate differs, post one
`INCIDENT · DEV-20260817-034` and stop without repair.

## Inventory A: code, configuration and model identity

For every existing item, report resolved path, size, modification time and
SHA256. Missing items are `NOT_PRESENT`; do not create substitutes.

1. Git identity:
   - local HEAD/commit date, upstream SHA, live remote SHA, nearest tag;
   - dirty and untracked counts.
2. Configuration/package identity:
   - `configs/default.yaml`;
   - `configs/rtx4060-laptop-production.yaml`;
   - preserved DEV-031 manifest/effective config when present;
   - `pyproject.toml` and lock/requirements files when present;
   - Git tree identity for `src/labvision_evidence/web`.
3. Local model identity:
   - first-person `.pt` and `.engine`;
   - third-person `.pt` and `.engine`;
   - do not deserialize, inspect tensors, import the model, or allocate GPU.

Record only `ARK_API_KEY configured=true/false`; never inspect its value,
length, prefix, suffix or hash.

## Inventory B: execution environment identity

Record metadata only:

- Windows edition/build;
- CPU model and physical/logical cores;
- total RAM;
- GPU model, driver, reported CUDA version, total memory, power limit and
  temperature;
- Python executable/version actually associated with the repository;
- package metadata versions for PyTorch, Ultralytics, OpenCV, Pydantic and
  TensorRT without importing GPU/model modules;
- FFmpeg/FFprobe version/build and presence of `h264_nvenc`/`libx264`, without
  opening media;
- local runtime/cache disk total/free space;
- active Ethernet adapter/status/negotiated link speed;
- `Y:` SMB mapping and UNC connection metadata;
- no bandwidth test and no synthetic transfer.

## Inventory C: formal end-to-end archive

Use metadata and existing non-media receipts only. Report count and bytes for:

- `Original-Experiment-Videos`;
- `Experiment-Clips`;
- `JSON-Config-Files`;
- `Key-Materials`;
- `Lab-Daily-Reports`;
- `Professional-PDFs`.

Report, without generation or repair:

- retained original-video references, without media hashing;
- experiment groups, atomic experiments and dual-view output counts;
- key events, key frames and key clips;
- normalized event JSON and evidence-package presence;
- timing and Token ledgers and totals;
- daily report JSON/Markdown/HTML presence;
- professional PDF presence;
- quality/eval and formal-promotion receipt presence;
- evidence index manifest, SQLite, artifact registry and evidence registry
  presence;
- completeness of stored artifact SHA256/integrity receipts, without media
  rehashing.

SHA256 is allowed only for non-media receipt files listed by the protocol. Do
not claim the formal archive was generated by DEV-031 or its SHA unless an
existing receipt proves that relationship.

## Inventory D: DEV-031 fast preprocessing candidate

Re-verify the 13 preserved small-ledger hashes and report their paths. Extract:

- total/preprocessing and stage durations;
- 27 Fine sessions and `60,828 / 60,828` accounting;
- candidate, accepted-event and selected-event counts;
- VisionCortex/model API and Token totals;
- G1/G3/G4/G5 quality failures;
- the five DEV-032 proven quality causes;
- the fact that 15 duplicate grid timestamps are non-causal ledger hygiene;
- before/after staging/cache identities.

Required classification:

- preprocessing performance SLA: `PROVEN`;
- DEV-031 end-to-end production quality: `NOT_PROVEN`;
- DEV-031 production release readiness: `NOT_PROVEN`.

## Inventory E: capability freeze matrix

Produce exactly 13 rows:

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
13. Web archive capability.

Columns:

- capability;
- best preserved evidence path/run;
- code SHA when proven;
- status: `PROVEN`, `PARTIAL_EVIDENCE`, `FAILED`, `NOT_PRESENT` or
  `NOT_PROVEN`;
- quality result;
- performance result;
- Token result;
- missing receipt/release blocker.

For Web archive capability, report static route/UI availability from preserved
source only. Live Web archive visibility must be `NOT_PROVEN`; do not call any
endpoint. Keep the successful formal end-to-end archive and fast-but-failing
DEV-031 preprocessing candidate as separate evidence lines.

## Required final GitHub result

Append exactly one final comment titled:

`FREEZE EVIDENCE · DEV-20260817-034`

It must contain:

1. elapsed inventory time and final repository/process state;
2. code/config/model SHA256 table;
3. environment/hardware table;
4. formal archive inventory and integrity/receipt table;
5. DEV-031 performance/quality table;
6. the complete 13-row freeze matrix;
7. real issue list with evidence and release blocker;
8. separation of full formal archive evidence from the fast candidate;
9. statement that Web static capability is evidence-only and live visibility is
   `NOT_PROVEN`;
10. explicit zero counts for VisionCortex product API calls, pipeline runs,
    media reads, inference, model APIs, Tokens, tests, mutations and promotions;
11. final counts/bytes/hashes proving all evidence roots remained unchanged;
12. recommended inputs for developer-side release freezing, without performing
    a tag/package/release action on RTX 4060.

GitHub Issue ACK/result posting is authorized and must not be counted as a
VisionCortex product API call.

## Acceptance gate

DEV-034 is accepted only when:

- all start-gate identities match;
- code/config/model/environment identity is complete or explicitly missing;
- formal archive and DEV-031 candidate remain separate;
- the 13-row matrix is complete;
- Web live visibility is `NOT_PROVEN` without endpoint access;
- no media payload is opened or rehashed;
- every claim cites preserved evidence or is marked not proven;
- VisionCortex product/model API calls and Tokens are zero;
- there is one ACK and one final result, or one ACK and one INCIDENT;
- no local/NAS file or Git state changes occur.

After the final comment, stop. Do not start a real run. Wait for a new frozen
developer-side quality-fix commit and explicit one-shot authorization.
