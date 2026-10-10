"""Receipted, causal SAM2 masks for explicitly pinned sampled-frame windows.

This reuses the production predictor; it neither discovers recordings nor changes
production configuration. JPEG input is a declared derivative, not source pixels.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image

from . import temporal_segmentation as temporal

REQUEST_SCHEMA = "visioncortex-temporal-mask-request/1"
OBSERVATION_REQUEST_SCHEMA = "visioncortex-temporal-mask-request/2"
RESULT_SCHEMA = "visioncortex-temporal-mask-result/1"


def _hash(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_request(request: dict, root: Path) -> None:
    version = request.get("schema_version")
    if version not in {REQUEST_SCHEMA, OBSERVATION_REQUEST_SCHEMA}:
        raise ValueError("Unsupported temporal request")
    source = request["source"]
    development = source["split"] in {"train", "val"}
    if version == OBSERVATION_REQUEST_SCHEMA:
        purpose = request.get("data_use", {}).get("purpose")
        if purpose == "production_observation":
            allowed = (source["split"] is None
                       and source.get("schema_version") == "labprism-local-observation/1"
                       and source.get("training_use_authorized") is False
                       and source.get("independent_ground_truth") is False)
        else:
            allowed = purpose == "development" and development
        if not allowed:
            raise ValueError("Invalid observation purpose or training exposure")
    elif not development:
        raise ValueError("Version 1 requires development exposure identity")
    if (source["camera_role"] not in {"first_person", "third_person"}
            or not source["camera_id"] or not _hash(source["source_sha256"])
            or not _hash(request["parent_result_sha256"])):
        raise ValueError("Missing source, role or exposure identity")
    width, height = request["dimensions"]
    if any(type(v) is not int or not 1 <= v <= 16384 for v in (width, height)):
        raise ValueError("Invalid dimensions")
    if request["input_transform"] != {"encoding": "JPEG", "quality": 95, "resize": False}:
        raise ValueError("Undeclared input transform")
    previous_index, previous_time = -1, -float("inf")
    for window_index, window in enumerate(request["windows"]):
        if window["id"] != window_index or not 1 <= len(window["frames"]) <= 50:
            raise ValueError("Invalid bounded window")
        prompts = window["prompts"]
        if not 1 <= len(prompts) <= 8 or len({p["id"] for p in prompts}) != len(prompts):
            raise ValueError("Invalid prompt identities")
        for prompt in prompts:
            x1, y1, x2, y2 = prompt["box"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and prompt["label"]):
                raise ValueError("Invalid prompt box")
        for position, frame in enumerate(window["frames"]):
            if (frame["frame_index"] <= previous_index or frame["timestamp_ms"] <= previous_time
                    or not math.isfinite(frame["timestamp_ms"])
                    or not _hash(frame["rgb_sha256"]) or not _hash(frame["input_rgb_sha256"])
                    or not isinstance(frame["clip_pts"], int)
                    or Fraction(frame["time_base"]) <= 0):
                raise ValueError("Invalid sampled frame identity/order")
            name = f"frames/{window_index:04d}/{position:05d}.jpg"
            path = root / name
            if (frame["input_file"] != name or path.is_symlink()
                    or not path.resolve().is_relative_to(root.resolve())
                    or temporal._sha256(path) != frame["input_sha256"]):
                raise ValueError("Temporal input file identity mismatch")
            with Image.open(path) as image:
                if image.format != "JPEG" or image.size != (width, height):
                    raise ValueError("Temporal image encoding/dimensions mismatch")
                rgb = np.asarray(image.convert("RGB"))
            if hashlib.sha256(rgb.tobytes()).hexdigest() != frame["input_rgb_sha256"]:
                raise ValueError("Decoded temporal input identity mismatch")
            previous_index, previous_time = frame["frame_index"], frame["timestamp_ms"]
        directory = root / "frames" / f"{window_index:04d}"
        if {p.name for p in directory.iterdir()} != {f"{i:05d}.jpg" for i in range(len(window["frames"]))}:
            raise ValueError("Unreceipted predictor input files")
    if not request["windows"]:
        raise ValueError("Empty temporal request")


def mask_record(mask: np.ndarray, output: Path, name: str) -> dict:
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected native binary mask")
    if not cv2.imwrite(str(output / name), mask):
        raise OSError("Temporal mask write failed")
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        if len(polygon) >= 3:
            polygons.append(polygon.tolist())
    return {"visible_pixels": int(mask.sum()), "mask_contours": polygons,
            "mask": {"file": name, "sha256": temporal._sha256(output / name)}}


def run(request_path: Path, config: dict, output: Path) -> dict:
    import torch

    request = json.loads(request_path.read_text())
    validate_request(request, request_path.parent)
    settings = temporal._settings(config)
    if not settings.get("enabled"):
        raise ValueError("Temporal predictor disabled")
    device = str(settings.get("device") or "cuda")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frames = []
    window_times = []
    with temporal._MODEL_LOCK, torch.inference_mode():
        predictor, runtime = temporal._load_predictor(config)
        runtime["scope"] = "explicit_bounded_sample_windows"
        cast = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
        with cast:
            for window in request["windows"]:
                state = None
                window_started = time.perf_counter()
                try:
                    state = predictor.init_state(
                        str(request_path.parent / "frames" / f'{window["id"]:04d}'),
                        offload_video_to_cpu=True, offload_state_to_cpu=False)
                    for object_id, prompt in enumerate(window["prompts"], 1):
                        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=object_id,
                                                        box=np.asarray(prompt["box"], np.float32))
                    seen = set()
                    for index, ids, logits in predictor.propagate_in_video(state, start_frame_idx=0):
                        if index in seen or not 0 <= index < len(window["frames"]):
                            raise ValueError("Unexpected predictor frame identity")
                        seen.add(index)
                        expected_ids = set(range(1, len(window["prompts"]) + 1))
                        if len(ids) != len(expected_ids) or set(ids) != expected_ids:
                            raise ValueError("Temporal object identity changed")
                        if tuple(logits.shape) != (len(ids), 1, request["dimensions"][1], request["dimensions"][0]):
                            raise ValueError("Temporal mask coordinates changed")
                        if not torch.isfinite(logits).all():
                            raise ValueError("Nonfinite temporal logits")
                        masks = (logits[:, 0] > 0).to("cpu").numpy().astype(np.uint8)
                        frame = window["frames"][index]
                        instances = []
                        for object_id, mask in zip(ids, masks, strict=True):
                            prompt = window["prompts"][object_id - 1]
                            name = f'temporal-{frame["frame_index"]:06d}-{object_id - 1:02d}.png'
                            instances.append({"id": f'w{window["id"]}-object{object_id - 1}',
                                "label": prompt["label"], "display_name": prompt.get("display_name", prompt["label"]),
                                "seed_object_id": prompt["id"],
                                "seed_frame_index": window["frames"][0]["frame_index"],
                                "state": "prompted" if index == 0 else "memory_propagated",
                                "confidence": None, "status": "unreviewed_model_proposal",
                                **mask_record(mask, output, name),
                                **({"proposal_group": prompt["proposal_group"]} if "proposal_group" in prompt else {})})
                        frames.append({**frame, "window_id": window["id"], "temporal_instances": instances})
                    if seen != set(range(len(window["frames"]))):
                        raise ValueError("Temporal predictor skipped sampled frames")
                finally:
                    if state is not None:
                        predictor.reset_state(state)
                        del state
                window_times.append(time.perf_counter() - window_started)
                print(json.dumps({"window": window["id"], "frames": len(seen),
                                  "seconds": window_times[-1]}), flush=True)
    result = {"schema_version": RESULT_SCHEMA, "source": request["source"],
              "request_sha256": temporal._sha256(request_path),
              "parent_result_sha256": request["parent_result_sha256"],
              "dimensions": request["dimensions"], "input_transform": request["input_transform"],
              "model": runtime, "frames": sorted(frames, key=lambda f: f["frame_index"]),
              "identity_across_windows": False, "quality_status": "unreviewed_model_proposals",
              "metrics": {"stage_seconds": time.perf_counter() - started,
                          "window_seconds": window_times, "accuracy": None, "temporal_quality": None}}
    if request["schema_version"] == OBSERVATION_REQUEST_SCHEMA:
        result["data_use"] = request["data_use"]
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    receipt = {"schema_version": "visioncortex-temporal-mask-receipt/1",
               "request_sha256": result["request_sha256"],
               "files": {p.name: temporal._sha256(p) for p in output.iterdir() if p.is_file()},
               "implementation": {p.name: temporal._sha256(p) for p in [Path(__file__), Path(temporal.__file__)]},
               "config": settings, "model": runtime, "production_service_changed": False}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.request, json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
