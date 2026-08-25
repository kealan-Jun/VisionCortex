from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LABPICS_CLASSES = (
    "Vessel",
    "V Label",
    "V Cork",
    "V Parts GENERAL",
    "Ignore",
    "Liquid GENERAL",
    "Liquid Suspension",
    "Foam",
    "Gel",
    "Solid GENERAL",
    "Granular",
    "Powder",
    "Solid Bulk",
    "Vapor",
    "Other Material",
    "Filled",
)
REPORT_CLASSES = (
    "Vessel",
    "Filled",
    "Liquid GENERAL",
    "Liquid Suspension",
    "Foam",
    "Gel",
    "Solid GENERAL",
    "Granular",
    "Powder",
    "Solid Bulk",
    "Vapor",
)

_MODEL_CACHE: dict[tuple[str, str, str, str], Any] = {}
_VALIDATED: dict[tuple[str, str], tuple[int, int]] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(
        ((config.get("models") or {}).get("liquid_semantic_sidecar") or {})
    )


def validate_liquid_semantic_runtime(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    if not settings.get("enabled"):
        return {"enabled": False, "status": "disabled"}
    checkpoint = Path(str(settings.get("checkpoint_path") or "")).resolve()
    expected = str(settings.get("checkpoint_sha256") or "").strip().lower()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"LabPics liquid semantic checkpoint is missing: {checkpoint}"
        )
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise RuntimeError("LabPics liquid semantic checkpoint_sha256 is invalid")
    key = (str(checkpoint), expected)
    file_stat = checkpoint.stat()
    fingerprint = (file_stat.st_size, file_stat.st_mtime_ns)
    if _VALIDATED.get(key) != fingerprint:
        actual = _sha256(checkpoint)
        if actual != expected:
            raise RuntimeError(
                "LabPics liquid semantic checkpoint hash mismatch: "
                f"expected={expected} actual={actual}"
            )
        # Re-read the stat after hashing so a concurrent replacement cannot
        # populate the validation cache with the pre-hash fingerprint.
        verified_stat = checkpoint.stat()
        verified_fingerprint = (verified_stat.st_size, verified_stat.st_mtime_ns)
        if verified_fingerprint != fingerprint:
            raise RuntimeError(
                "LabPics liquid semantic checkpoint changed during validation"
            )
        _VALIDATED[key] = verified_fingerprint
    device = str(settings.get("device") or "cuda")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LabPics liquid semantic CUDA runtime is unavailable")
    return {
        "enabled": True,
        "status": "validated",
        "backend": "LabPics-PSPNet-ResNet101",
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": expected,
        "device": device,
        "scope": "bounded_final_key_frames_only",
        "classes": list(REPORT_CLASSES),
        "policy": "observation_only_never_action_confirmation",
        "source_record": str(settings.get("source_record") or ""),
        "license": str(settings.get("license") or ""),
    }


def _build_model(checkpoint: Path, device: str, use_half: bool) -> Any:
    """Build a modern runtime-compatible form of the published LabPics net.

    Architecture and class order follow the CC-BY-4.0 artifact published in
    Zenodo record 3697767. The compatibility implementation avoids executing
    downloaded Python and never downloads an encoder during construction.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torchvision.models import resnet101

    class LabPicsPSPNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.Encoder = resnet101(weights=None)
            self.PSPScales = (1.0, 0.5, 0.25, 0.125)
            self.PSPLayers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(2048, 1024, 3, padding=1, bias=True)
                    )
                    for _ in self.PSPScales
                ]
            )
            self.PSPSqueeze = nn.Sequential(
                nn.Conv2d(4096, 512, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            self.SkipConnections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(1024, 512, 1, bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(512, 256, 1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(256, 256, 1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                ]
            )
            self.SqueezeUpsample = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(1024, 512, 1, bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(768, 256, 1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(512, 256, 1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                ]
            )
            self.OutLayersList = nn.ModuleList(
                [nn.Conv2d(256, 2, 3, padding=1, bias=False) for _ in LABPICS_CLASSES]
            )

        def forward(self, images: Any) -> list[Any]:
            x = self.Encoder.conv1(images)
            x = self.Encoder.bn1(x)
            x = self.Encoder.relu(x)
            x = self.Encoder.maxpool(x)
            skips = []
            x = self.Encoder.layer1(x)
            skips.append(x)
            x = self.Encoder.layer2(x)
            skips.append(x)
            x = self.Encoder.layer3(x)
            skips.append(x)
            x = self.Encoder.layer4(x)
            psp_size = x.shape[-2:]
            psp = []
            for scale, layer in zip(self.PSPScales, self.PSPLayers, strict=True):
                size = tuple(max(1, int(value * scale)) for value in psp_size)
                branch = functional.interpolate(x, size=size, mode="bilinear")
                branch = layer(branch)
                psp.append(
                    functional.interpolate(branch, size=psp_size, mode="bilinear")
                )
            x = self.PSPSqueeze(torch.cat(psp, dim=1))
            for index, skip_layer in enumerate(self.SkipConnections):
                size = skips[-1 - index].shape[-2:]
                x = functional.interpolate(x, size=size, mode="bilinear")
                x = torch.cat((skip_layer(skips[-1 - index]), x), dim=1)
                x = self.SqueezeUpsample[index](x)
            return [
                functional.interpolate(layer(x), size=images.shape[-2:], mode="bilinear")
                for layer in self.OutLayersList
            ]

    model = LabPicsPSPNet()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if use_half and device.startswith("cuda"):
        model.half()
    return model


def _load_model(config: dict[str, Any]) -> tuple[Any, dict[str, Any], float]:
    runtime = validate_liquid_semantic_runtime(config)
    settings = _settings(config)
    checkpoint = Path(runtime["checkpoint"])
    device = str(runtime["device"])
    use_half = bool(settings.get("half", True))
    key = (str(checkpoint), str(runtime["checkpoint_sha256"]), device, str(use_half))
    started = time.perf_counter()
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _build_model(checkpoint, device, use_half)
    return _MODEL_CACHE[key], runtime, time.perf_counter() - started


def predict_liquid_masks(
    frame: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("LabPics liquid semantic input must be a BGR image")
    import torch

    settings = _settings(config)
    model, runtime, load_seconds = _load_model(config)
    maximum = max(64, int(settings.get("maximum_image_size", 640)))
    scale = min(1.0, maximum / max(frame.shape[:2]))
    resized = (
        cv2.resize(
            frame,
            (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else frame
    )
    device = str(runtime["device"])
    use_half = bool(settings.get("half", True)) and device.startswith("cuda")
    tensor = torch.from_numpy(resized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    means = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
    tensor = (tensor - means) / 65.0
    tensor = tensor.to(device=device, dtype=torch.float16 if use_half else torch.float32)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = [torch.softmax(item, dim=1)[0, 1] for item in logits]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    threshold = float(settings.get("mask_threshold", 0.5))
    masks = {
        name: (probability.detach().float().cpu().numpy() >= threshold)
        for name, probability in zip(LABPICS_CLASSES, probabilities, strict=True)
        if name in REPORT_CLASSES
    }
    receipt = {
        **runtime,
        "status": "completed",
        "input_shape": list(frame.shape),
        "inference_shape": list(resized.shape),
        "threshold": threshold,
        "model_load_seconds": round(load_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "gpu_peak_allocated_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 3)
            if device.startswith("cuda")
            else 0.0
        ),
        "token_usage": 0,
        "ark_calls": 0,
        "source_copy_bytes": 0,
        "full_timeline_inference": False,
    }
    return masks, receipt


def analyze_liquid_semantics(
    frame: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
    *,
    artifact_stem: str,
) -> dict[str, Any]:
    masks, receipt = predict_liquid_masks(frame, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    union_liquid = np.zeros(next(iter(masks.values())).shape, dtype=bool)
    for name in ("Liquid GENERAL", "Liquid Suspension", "Foam", "Gel"):
        union_liquid |= masks.get(name, False)
    selected = {
        "Vessel": masks["Vessel"],
        "Filled": masks["Filled"],
        "Liquid": union_liquid,
    }
    palette = {
        "Vessel": (255, 200, 0),
        "Filled": (0, 220, 255),
        "Liquid": (255, 80, 20),
    }
    overlay = cv2.resize(
        frame,
        (next(iter(selected.values())).shape[1], next(iter(selected.values())).shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    mask_index = np.zeros(overlay.shape[:2], dtype=np.uint8)
    class_records = []
    for index, (name, mask) in enumerate(selected.items(), 1):
        pixels = int(mask.sum())
        bbox = None
        if pixels:
            ys, xs = np.where(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            color = np.asarray(palette[name], dtype=np.uint8)
            overlay[mask] = ((overlay[mask].astype(np.uint16) + color) // 2).astype(np.uint8)
            mask_index[mask] = index
        class_records.append(
            {
                "class_name": name,
                "positive_pixels": pixels,
                "coverage": round(pixels / max(1, mask.size), 8),
                "bbox_xyxy": bbox,
            }
        )
    overlay_path = output_dir / f"{artifact_stem}-Liquid-State-Overlay.jpg"
    mask_path = output_dir / f"{artifact_stem}-Liquid-State-Mask.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to write liquid semantic overlay: {overlay_path}")
    if not cv2.imwrite(str(mask_path), mask_index):
        raise RuntimeError(f"Failed to write liquid semantic mask: {mask_path}")
    return {
        **receipt,
        "policy": "observation_only_never_action_confirmation",
        "classes": class_records,
        "overlay_path": str(overlay_path),
        "mask_path": str(mask_path),
    }
