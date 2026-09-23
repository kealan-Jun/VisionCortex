# VisionCortex Ubuntu RTX 3090 Ti deployment

This is an independent deployment profile for the actual workstation. It does
not modify or claim compatibility with the frozen Windows RTX 4090 package.

## Validated host identity

- Ubuntu 22.04 x86_64
- Intel Core i7-13700KF (16 cores / 24 threads)
- NVIDIA GeForce RTX 3090 Ti, 24 GiB, compute capability 8.6
- NVIDIA driver 580.173.02
- 32 GiB system RAM
- Local environment/engine/runtime volume: `/srv/sentinel-data`
- NAS mount: `/home/x1/桌面/nas`

Runtime phases may request batch 16, but the TensorRT FP16 dynamic engine is
built with a maximum batch of 4. Ultralytics expands a dynamic 640 profile as
far as 1920x1920 during engine construction; the former build batch of 16 ran
out of builder memory in the YOLO attention block on this actual RTX 3090 Ti.
The scanner reads the engine metadata and splits larger runtime requests into
safe batches of at most 4. Batch or worker increases require a bounded,
telemetry-backed acceptance run; they must not be changed during a production
collection.

The fine-scan profile sets six decode slots per scanner's role group, with
per-source limits of four for first-person and six for third-person video. Setting
`fine_active_decode_slots` to one prevents those per-source limits from being
used. A same-role supplemental wave with three scanner contexts shares the
six slots (two per context). Mixed-role concurrent scanners receive separate
budgets; six is not a service-wide limit. Actual decoder concurrency is also
bounded by the available physical decode sessions. A single session can
still use only one decoder.

New Web submissions read the current profile. Persisted offline jobs retain
their submitted performance settings, while the device/day service adopts
changed settings after its active work drains. Editing the YAML alone is
not evidence that a running task uses it. Verify the task's
`runtime_fine_*.json` receipts (`active_decode_slot_budget`,
`ordered_source_decode_workers`, and `total_ordered_source_decode_workers`)
after the changed settings are adopted. These fields prove allocation, not a
speedup. Throughput remains `NOT_PROVEN` without comparable real-video runs
at budgets one and six, using the same inputs, models and cache conditions,
with stage wall time, resource telemetry and output-quality checks.

## Python environment

The normal PATH exposes Python 3.13, while the project requires Python 3.11 or
3.12. The installer reuses the existing
`/home/x1/anaconda3/envs/gaoqing/bin/python` 3.12 interpreter only as a seed,
then creates the isolated deployment venv on the new volume. It does not create
another Conda environment.

```bash
export VISIONCORTEX_PYTHON='/home/x1/anaconda3/envs/gaoqing/bin/python'
./deployment/rtx3090ti-ubuntu/01-Install-And-Validate.sh --skip-api-key-check
```

After installing the pinned environment and retaining the verified offline
wheelhouse, the latest preflight measured 74 GiB free locally. Both installation
and production use a 30 GiB local safety gate. The old 100 GiB production gate
was deliberately removed: original videos, formal output, NAS run staging, and
the persistent YOLO/checkpoint cache are on the NAS. Local storage contains the
Python environment, target-built TensorRT engines, small control logs, and
temporary decode/export scratch only.

## Preflight

```bash
export VISIONCORTEX_PYTHON='/srv/sentinel-data/VisionCortex3090Ti/.venv/bin/python'
./deployment/rtx3090ti-ubuntu/00-Preflight.sh --install
./deployment/rtx3090ti-ubuntu/00-Preflight.sh --production
```

The production preflight requires the live writable NAS index/archive and at
least 30 GiB free on the local runtime volume. It reports NAS free space but
does not impose one fixed NAS threshold: before each collection the pipeline
estimates that collection's delivery/candidate requirement from its real source
sizes and duration, then checks the run-specific NAS staging directory. The
preflight reads metadata only; it does not open source videos.

Storage placement for this profile:

- NAS source video: paths resolved from `experiment_record_index.csv`;
- NAS formal output/staging: `.VisionCortex-Run-Staging` followed by promotion;
- NAS persistent cache: existing `/home/x1/桌面/nas/VisionCortexExperimentCache`;
- local: `.venv`, `Models`, `Engines`, PID/log files, model-quality receipts,
  isolated third-party settings, and auto-cleaned `Runtime/tmp`.

The local temporary directory is intentional. FFmpeg and TensorRT need a small
fast scratch area while a clip or engine is actively being built; treating that
ephemeral scratch as NAS cache adds network I/O without making data safer.

The deployment must reuse the existing `VisionCortexExperimentArchive`,
`.VisionCortex-Run-Staging`, and `VisionCortexExperimentCache` roots. Preflight
fails if any required NAS root is absent; deployment scripts must not create a
replacement archive, output, or cache root.

## Install and build engines

Installation downloads pinned Linux wheels. The nearby PyPI mirror serves the
same upstream wheel paths and SHA-256 link metadata to avoid the extremely slow
direct route observed on this host; TensorRT remains on NVIDIA's official
package index. TensorRT engines are always built on this RTX 3090 Ti and are
stored outside the source tree.

The deployment also pins ONNX, ONNX Runtime GPU, and ONNX Slim because
Ultralytics needs them before TensorRT export. Child processes are forced onto
the deployment `.venv/bin` PATH so an export cannot auto-install into the
user's base Conda environment.

The source repository intentionally excludes every model binary. Before the
installer starts, provision the two project-trained 21-class weights at the
paths frozen by `configs/models/closed-set-yolo.json`:

- `/srv/sentinel-data/VisionCortex3090Ti/Models/ClosedSetYOLO/first_person/best.pt`
- `/srv/sentinel-data/VisionCortex3090Ti/Models/ClosedSetYOLO/third_person/best.pt`

`visioncortex install-closed-set-models` installs explicit source files only after
hash verification and refuses to overwrite a non-matching destination;
`visioncortex validate-closed-set-models` revalidates both the hash and exact
21-class ontology. The installer refuses a checksum mismatch. It then downloads and verifies the pinned public
YOLO-World, CLIP, Grounding DINO and SAM2.1 assets, builds both TensorRT engines,
and runs one fail-closed validation covering all six local model artifacts.
Subsequent installs reuse assets only after rechecking SHA-256.

```bash
export VISIONCORTEX_PYTHON='/srv/sentinel-data/VisionCortex3090Ti/.venv/bin/python'
visioncortex validate-closed-set-models \
  --registry configs/models/closed-set-yolo.json
./deployment/rtx3090ti-ubuntu/01-Install-And-Validate.sh --skip-api-key-check
```

Before Web or a real run, provide `ARK_API_KEY` through a hidden/secure
user-controlled input. On this workstation the startup script may read the raw
key from `/home/x1/.config/VisionCortex/ark_api_key`; the file must be owned by
the current user, must not be a symbolic link, and must have mode `600`. The new
volume does not enforce Unix file modes and therefore must not hold this secret.
Never add the key to YAML, shell history, receipts, screenshots or Git.

## Web lifecycle

```bash
./deployment/rtx3090ti-ubuntu/02-Start-Web.sh --open-browser
./deployment/rtx3090ti-ubuntu/03-Stop-Web.sh
```

This manual lifecycle binds only to `127.0.0.1:8000`. PID ownership is checked
before a process is reused or stopped. It is intended for administrator-side
validation, not for other computers on the LAN.

For the local no-NAS appliance experience, install the persistent user service
and desktop launcher once:

```bash
./deployment/rtx3090ti-ubuntu/05-Install-Local-Service.sh
```

The installer enables the Web service at boot, installs a **VisionCortex**
application entry, and opens the product in a dedicated browser window after
the desktop user logs in. The launcher uses the production LAN service when it
is enabled; otherwise it starts the local no-NAS service. Both modes use
`127.0.0.1:8000` from this workstation. Browser auto-open happens at graphical
login, while user-service lingering keeps the enabled service available before
login. This convenience does not weaken production preflight, authentication,
NAS, model, or quality gates.

The launcher detects a primary display width of 3200 pixels or greater and
starts the dedicated Chromium profile at a 1.5 device scale. This keeps both
the browser chrome and the product readable on the workstation's 3840x2160
display while the Web UI independently expands its dashboard and video layouts
at wide CSS viewports. Set `VISIONCORTEX_BROWSER_SCALE=1.0` (or another value
from 1.0 through 2.0) before launching to override the detected value; ordinary
1080p displays are not forced to scale.

For the local NAS-backed appliance flow used on the RTX 3050, install the
separate capture-monitor/analysis service once:

```bash
./deployment/rtx3090ti-ubuntu/09-Install-Analysis-Service.sh
```

This enables `visioncortex-analysis.service` at boot on
`127.0.0.1:8001`. It automatically discovers every top-level `*_cam*`
recorder directory, scans the newest 32 recordings per camera every 30
seconds, and groups files only when their recorder-issued
`recording_session_id` matches and their cross-camera time windows overlap.
Unregistered camera roles and incomplete recorder sidecars stay visible but
blocked. The existing `visioncortex-local.service` remains an independent
no-NAS fallback on port 8000. When the analysis service is installed, the
desktop launcher opens the NAS-backed page on port 8001.

## Authenticated LAN server

The normal team entry point is the production LAN service. Team members do not
clone the repository or install Python, CUDA, TensorRT, or model assets. They
open the URL printed by the installer and sign in through the browser.

Before the first installation, keep the Ark key in the existing owner-only
credential file. The commands below do not place the value in shell history:

```bash
install -d -m 700 /home/x1/.config/VisionCortex
IFS= read -r -s -p 'Ark API key: ' VISIONCORTEX_ARK_INPUT; printf '\n'
umask 077
printf '%s\n' "$VISIONCORTEX_ARK_INPUT" > /home/x1/.config/VisionCortex/ark_api_key
chmod 600 /home/x1/.config/VisionCortex/ark_api_key
unset VISIONCORTEX_ARK_INPUT
```

Install and inspect the server:

```bash
./deployment/rtx3090ti-ubuntu/07-Install-LAN-Server.sh
./deployment/rtx3090ti-ubuntu/08-Server-Status.sh
```

The installer securely prompts twice for the browser password for user
`visioncortex`, disables the alternative no-NAS service if it would conflict on
port 8000, runs the production preflight, and starts
`visioncortex-lan.service`. The service loads the production profile, Ark
credential, TensorRT engine paths, NAS index/archive/cache roots, and listens on
`0.0.0.0:8000`. Application middleware then rejects clients outside loopback,
RFC 1918 IPv4, and local IPv6 ranges and requires HTTP Basic authentication for
every route, including files and task submission.

All Web-submitted GPU workflows first enter a SQLite queue under the configured
local runtime root. A cross-process lease allows only one upload, fixed
benchmark, or indexed collection to use the 3090 Ti at a time. Waiting jobs and
their Web-visible state survive service or host restarts; a claimed job whose
lease expires is reclaimed with the same run identity so durable pipeline
checkpoints can be reused.

Large browser uploads use durable resumable sessions in the same local SQLite
state database. The browser declares each real file size before sending data;
the server subtracts outstanding reservations from the live NAS free space and
reserves source bytes plus configurable processing and safety headroom. There
is no fixed total-upload-size, duration, or view-count ceiling. Accepted files
arrive in 16 MiB chunks directly into `Original-Experiment-Videos` on the NAS,
resume from the server-confirmed byte offset after a disconnect, receive a full
SHA-256 before the GPU job is queued, and are not duplicated on the local
runtime volume. An incomplete session expires only after seven days without
progress; finalized/queued jobs keep their reservation until execution ends.
The indexed NAS collection path remains the preferred zero-copy path when the
source files already exist in the recorder collection.

These contracts and deterministic tests do not prove a real seven-view,
eight-hour transfer or full analysis. Record that scenario as `NOT_PROVEN`
until it is exercised against the deployed 3090 Ti host and NAS.

For startup before the desktop user logs in, enable user-service lingering once:

```bash
sudo loginctl enable-linger x1
```

This is a LAN-only contract, not an Internet deployment. Do not add router port
forwarding or place a local reverse proxy in front of it without a separate
public TLS, identity, rate-limit, and audit design. The service fails closed if
the NAS, GPU, production preflight, Ark key, or Web password is unavailable.
HTTP Basic protects access but does not encrypt LAN traffic; use only a trusted
wired/VLAN network until a separately reviewed HTTPS endpoint is deployed.

## Production boundary

Do not run the three real datasets until all of the following pass:

1. production preflight;
2. pinned dependency installation;
3. RTX 3090 Ti CUDA and TensorRT import checks;
4. two target-built TensorRT engines and 21-class model validation;
5. local Web health;
6. a bounded non-production media acceptance run with telemetry;
7. an explicit review of NAS staging/promotion paths on Linux.

`visioncortex accept-local-models` is the bounded real-GPU model-execution gate;
it executes both TensorRT role engines, YOLO-World, Grounding DINO, LabPics and
SAM2 using a local public annotated image and derived nine-frame clip. Passing
proves runtime wiring only. `visioncortex model-certification-readiness` reports
the exact held-out event and participant-box truth deficits without opening a
production archive. Neither command can certify production-domain quality or
replace the cold Doubao call required by a formal end-to-end run.

Final participant grounding is selective and bounded in the 3090 Ti profile.
Liquid, container-state and pipette events are always verified; other events
escalate only when closed-set participant evidence is ambiguous. The decision,
budget and measured local-model time are persisted in
`JSON-Config-Files/final_key_material_annotation.json`. Set
`VISIONCORTEX_SELECTIVE_KEY_MATERIAL_VERIFICATION=false` to preserve the prior
supplement path without changing models or archive schemas.
