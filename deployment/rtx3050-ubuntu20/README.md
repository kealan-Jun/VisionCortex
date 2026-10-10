# RTX 3050 6GB / Ubuntu 20.04 offline deployment

Chinese operator and end-user instructions are consolidated in
[`docs/VisionCortex-RTX3050-离线部署与使用交付手册.md`](../../docs/VisionCortex-RTX3050-离线部署与使用交付手册.md).

This profile targets the provided machine contract only: Ubuntu 20.04, kernel
5.15 or newer, RTX 3050 with 6 GiB VRAM, NVIDIA driver 570 or newer, 15 GiB
system RAM, and at least 2 GiB swap.

The verified USB payload includes a portable CPython 3.12 runtime, an offline
wheelhouse, registered source weights and public sidecar assets. Model identities
and offline hashes remain mandatory. Engines from another GPU are excluded.
Format the drive as **exFAT or ext4**: the pinned TensorRT wheel exceeds FAT32's
single-file limit. Copy and verify the complete package directory.

Install dependencies and assets from the USB drive:

```bash
bash ./Verify-Package.sh
bash ./Install-VisionCortex.sh --install-root /opt/visioncortex-rtx3050
/opt/visioncortex-rtx3050/Start-VisionCortex.sh --local
```

The installation prefix is configurable through `--install-root` or
`VISIONCORTEX_INSTALL_ROOT`. An existing target is never overwritten. The
installer requires 18 GiB free before installation and keeps wheel downloads on
the USB. Default installation does not prepare or invoke models, access NAS, or
request a provider key. Local startup uses the actual checkout's
`configs/development-local.yaml`, its configured local storage roots and port
8003. It needs only the core environment for page inspection. Direct source
clones can use `bash deployment/rtx3050-ubuntu20/Start-VisionCortex.sh --local`
after the [development installation](../../docs/guides/local-development.zh-CN.md).

For model execution, prepare a private site configuration with actual local
weights, engine destinations, input/index and storage roots. Set
`project.site_configuration_required: false`, declare `project.run_purpose`
(`analysis` or `production`) and set `runtime.local_only: false`. The committed
hardware profile is an unprepared template. Set the selected provider and its
credential environment name only when that provider is enabled; supply a key
through that environment variable or an owned mode-600 file selected by
`VISIONCORTEX_MODEL_API_KEY_FILE`. Keys are never embedded in the package or
service unit. Read [configuration guidance](../../configs/README.md) before
using a site overlay.

Explicit model preparation during a new installation:

```bash
bash ./Install-VisionCortex.sh --install-root /opt/visioncortex-rtx3050 \
  --prepare-models --config /absolute/path/to/private-site.yaml
```

This opt-in path checks the configured site before GPU access, builds engines
on the target RTX 3050, retains bounded static-batch autotuning and writes a
synthetic engine smoke receipt to the configured local runtime root. Existing
installations require a separately planned model preparation operation;
rerunning the installer will refuse to replace them.

Start a prepared production instance:

```bash
/opt/visioncortex-rtx3050/Start-VisionCortex.sh --production \
  --config /absolute/path/to/private-site.yaml --port 8003
```

The startup reads storage and models from the effective configuration. It does
not guess a NAS mount or copy source media. Missing configuration, unavailable
configured storage, hardware/decode requirements or provider credentials fail
closed. The Web instance listens on loopback; LAN access requires the
[dedicated server deployment](../rtx3090ti-ubuntu/README.md).

Install an independent user service and desktop entry:

```bash
bash /opt/visioncortex-rtx3050/app/deployment/rtx3050-ubuntu20/Install-Autostart.sh --local
```

The default unit is `visioncortex-rtx3050.service`, using port 8003. Customize
`VISIONCORTEX_SERVICE_NAME` and `VISIONCORTEX_WEB_PORT` for another instance. The
installer renders and verifies a unit for the actual checkout; it rejects an
existing unit owned by another checkout or an active unit with changed
configuration. It does not stop other instances. Production service installation
uses `--production --config /absolute/path/to/private-site.yaml`. Desktop
login autostart is opt-in with `VISIONCORTEX_DESKTOP_AUTOSTART=1`; enabling user
lingering remains an explicit administrator action.

An install receipt proves only the operations actually completed. Synthetic
engine execution does not prove real-video accuracy or throughput. A cold run
with representative video, provider invocation receipts, resource telemetry and
archive evidence is still required. Formal production also requires the
applicable independently reviewed model certification configured for that site.

## Source and asset selection

Build from a reviewed immutable Git commit. Supply `--output`, `--wheelhouse`,
`--python-runtime`, `--uv`, `--runtime` and `--ffmpeg-archive` explicitly.
The runtime asset root contains `Models/ClosedSetYOLO/...` and `Engines/...`;
CLIP is read from `Engines/ViT-B-32.pt`, without using a packager's home cache.
Pinned model and FFmpeg hashes remain mandatory. The customer source export
includes its local documentation dependencies and deployment helpers, while
private agent guidance, site evidence and history navigation are excluded.
