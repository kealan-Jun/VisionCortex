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

`labvision install-closed-set-models` installs explicit source files only after
hash verification and refuses to overwrite a non-matching destination;
`labvision validate-closed-set-models` revalidates both the hash and exact
21-class ontology. The installer refuses a checksum mismatch. It then downloads and verifies the pinned public
YOLO-World, CLIP, Grounding DINO and SAM2.1 assets, builds both TensorRT engines,
and runs one fail-closed validation covering all six local model artifacts.
Subsequent installs reuse assets only after rechecking SHA-256.

```bash
export VISIONCORTEX_PYTHON='/srv/sentinel-data/VisionCortex3090Ti/.venv/bin/python'
labvision validate-closed-set-models \
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

The service binds only to `127.0.0.1:8000`. PID ownership is checked before a
process is reused or stopped.

## Production boundary

Do not run the three real datasets until all of the following pass:

1. production preflight;
2. pinned dependency installation;
3. RTX 3090 Ti CUDA and TensorRT import checks;
4. two target-built TensorRT engines and 21-class model validation;
5. local Web health;
6. a bounded non-production media acceptance run with telemetry;
7. an explicit review of NAS staging/promotion paths on Linux.

`labvision accept-local-models` is the bounded real-GPU model-execution gate;
it executes both TensorRT role engines, YOLO-World, Grounding DINO, LabPics and
SAM2 using a local public annotated image and derived nine-frame clip. Passing
proves runtime wiring only. `labvision model-certification-readiness` reports
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
