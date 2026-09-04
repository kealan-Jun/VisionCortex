# RTX 3050 6GB / Ubuntu 20.04 offline deployment

Chinese operator and end-user instructions are consolidated in
[`docs/VisionCortex-RTX3050-离线部署与使用交付手册.md`](../../docs/VisionCortex-RTX3050-离线部署与使用交付手册.md).

This profile targets the provided machine contract only: Ubuntu 20.04, kernel
5.15 or newer, RTX 3050 with 6 GiB VRAM, NVIDIA driver 570 or newer, 15 GiB
system RAM, and at least 2 GiB swap.

The USB directory is self-contained. It includes a glibc 2.17-compatible
CPython 3.12 runtime, an offline wheelhouse, both registered 21-class source
weights, every production public sidecar, and the evaluated research
candidates with receipts. A checksum-pinned portable FFmpeg build supplies
CUDA decode and NVENC without changing the target's system packages. The
package deliberately excludes TensorRT engines built on
another GPU. The installer builds new engines on the RTX 3050 and performs a
bounded static-batch autotune over 16, 8, 4, 2, then 1. It accepts the first
descending candidate that passes repeated real TensorRT execution while
preserving the configured VRAM reserve for hardware decode. A second execution
smoke test writes `Runtime/rtx3050-engine-smoke.json`.

Format the USB drive as **exFAT or ext4, not FAT32**. The pinned TensorRT
runtime wheel is 4,304,296,018 bytes, which is slightly larger than FAT32's
single-file limit. Copy the complete package directory; do not split or omit
the wheelhouse. The package builder removes the unused private Python terminfo
database and rejects any remaining case-insensitive path collision, so an
exFAT copy cannot silently merge differently-cased payload names.

Run from the USB drive:

```bash
bash ./Verify-Package.sh
bash ./Install-VisionCortex.sh
```

The fixed install root is `/opt/visioncortex-rtx3050`. Wheels remain on the USB,
so they do not consume the target's constrained system disk. The installer
requires 18 GiB free before installation and 8 GiB afterward. It prunes only
TensorRT builder resources for GPU architectures other than SM86 after both
engines have been built.

Start without NAS access for local inspection:

```bash
/opt/visioncortex-rtx3050/Start-VisionCortex.sh --local
```

Start the production service after the existing NAS paths are mounted:

```bash
/opt/visioncortex-rtx3050/Start-VisionCortex.sh --production
```

Production startup detects either `$HOME/桌面/nas` or
`/mnt/visioncortex-nas`. It fails closed if the existing index, archive, cache,
secure Ark credential, CUDA decode, NVENC, model hashes, or free-space contract
is unavailable. The API key is never included in the package; the installer
can accept it once through hidden terminal input and stores it as a mode-600
user file.

The package and synthetic engine smoke prove reproducible installation and
actual model execution, not real-video quality or end-to-end throughput. Those
claims require a cold target-machine run over representative six-view media,
with the normal run metrics, Token usage, GPU/RAM telemetry, and archive
receipts retained.

Formal production additionally requires a current independently reviewed model
certification receipt at
`/opt/visioncortex-rtx3050/Runtime/Model-Quality/production_model_certification.json`.
It is intentionally not fabricated or inferred from model confidence and is
not part of the generic USB payload.
