"""Resident detector client; PyTorch lives in its own project environment/process."""

import base64
import hashlib
import json
import math
import os
import re
from pathlib import Path
from field_recognition.runtime.paths import PROJECT_ROOT
import selectors
import subprocess
import threading
import time

import cv2


class PanelDetector:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.config = None
        self.state = {
            "status": "not_loaded",
            "resident": False,
            "model": "YOLO11n",
            "load_count": 0,
        }

    def enabled(self):
        return os.environ.get("FIELD_PANEL_DETECTOR_ENABLED", "0").lower() in {"1", "true", "yes"}

    def snapshot(self):
        return dict(self.state, enabled=self.enabled())

    def close(self):
        process, self.process = self.process, None
        if process:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    pipe.close()
        self.state.update(resident=False)

    def _receive(self, timeout):
        with selectors.DefaultSelector() as poll:
            poll.register(self.process.stdout, selectors.EVENT_READ)
            if not poll.select(timeout):
                raise TimeoutError("panel detector timeout")
            raw = self.process.stdout.readline(1024 * 1024)
        if not raw or not raw.endswith(b"\n"):
            raise RuntimeError("panel detector protocol interrupted")
        result = json.loads(raw)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def _start(self):
        if self.process and self.process.poll() is None:
            return
        self.close()
        self.state.update(status="loading", error=None)
        path = Path(os.environ["FIELD_PANEL_MODEL_MANIFEST"])
        config = json.loads(path.read_text())
        weights = Path(config["weights"])
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        from field_recognition.recognition.panel_layout import normalize_classes

        config["classes"] = normalize_classes(config["classes"])
        if digest != config["weights_sha256"]:
            raise ValueError("panel model manifest mismatch")
        self.config = config
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
        env.update(
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            YOLO_AUTOINSTALL="false",
            YOLO_OFFLINE="true",
            YOLO_CONFIG_DIR=str(path.parent / "runtime"),
        )
        device = os.environ.get("FIELD_PANEL_DEVICE", "cpu").strip().lower()
        if not re.fullmatch(r"cpu|gpu:\d+", device):
            raise ValueError("FIELD_PANEL_DEVICE must be cpu or gpu:<index>")
        if os.environ.get("FIELD_OCR_SERVICE_URL") and device != "cpu":
            raise ValueError(
                "The standalone GPU service owns CUDA; local panel detection requires cpu"
            )
        # The child sees only the explicitly selected host GPU, at its local index 0.
        env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else device.split(":")[1]
        self.process = subprocess.Popen(
            [
                config["python"],
                "-u",
                str(PROJECT_ROOT / "panel_detector_worker.py"),
                str(weights),
                str(config.get("imgsz", 960)),
                str(config.get("confidence", 0.25)),
                "cpu" if device == "cpu" else "gpu:0",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
        )
        ready = self._receive(60)
        if ready.get("status") != "ready" or ready.get("names") != {
            key: value["name"] for key, value in config["classes"].items()
        }:
            raise ValueError("unexpected detector classes")
        self.state.update(
            status="ready",
            resident=True,
            error=None,
            device=device,
            weights_sha256=digest,
            model_version=config["version"],
            load_count=self.state["load_count"] + 1,
            confidence_threshold=config.get("confidence", 0.25),
        )

    def warmup(self):
        if not self.enabled():
            return False
        with self.lock:
            try:
                self._start()
                return True
            except Exception as exc:
                self.close()
                self.state.update(status="error", error=type(exc).__name__)
                return False

    def predict(self, image, *, photo=False):
        with self.lock:
            started = time.monotonic()
            try:
                self._start()
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError("detector input encoding failed")
                self.process.stdin.write(
                    json.dumps(
                        {"image": base64.b64encode(encoded).decode(), "photo": photo}
                    ).encode()
                    + b"\n"
                )
                self.process.stdin.flush()
                result = self._receive(10)
                boxes = []
                diagnostics = dict(result.get("diagnostics") or {})
                diagnostics.setdefault("raw_candidate_count", len(result["boxes"]))
                diagnostics.setdefault("raw_count_basis", "worker_returned_boxes")
                excluded = list(diagnostics.get("excluded_candidates", []))
                excluded.extend(dict(b, reason="candidate_limit") for b in result["boxes"][6:])
                h, w = image.shape[:2]
                for item in result["boxes"][:6]:
                    key = str(item["class_id"])
                    values = item["xyxy"] + [item["confidence"]]
                    if key not in self.config["classes"] or not all(
                        math.isfinite(v) for v in values
                    ):
                        raise ValueError("invalid detector output")
                    x1, y1, x2, y2 = item["xyxy"]
                    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
                    x2, y2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        excluded.append(dict(item, reason="region_too_small"))
                        continue
                    boxes.append(
                        item
                        | {
                            "xyxy": [x1, y1, x2, y2],
                            "identity_evidence": {
                                "basis": "detector_manifest_asset_mapping"
                                if self.config["classes"][key].get("instrument_id")
                                else "detector_manifest_type_mapping",
                                "class_id": item["class_id"],
                                "model_version": self.config["version"],
                                "weights_sha256": self.state["weights_sha256"],
                            },
                            **{
                                field: self.config["classes"][key].get(field)
                                for field in ("instrument_id", "type_id", "layout", "measurements")
                            },
                        }
                    )
                diagnostics.update(quality_filtered_count=len(boxes), excluded_candidates=excluded)
                boxes.sort(key=lambda b: (b["class_id"], b["xyxy"][0], b["xyxy"][1]))
                self.state.update(
                    status="ready",
                    error=None,
                    last_seconds=round(time.monotonic() - started, 3),
                    recovery=result.get("recovery"),
                )
                return {
                    "status": "completed",
                    "boxes": boxes,
                    "model": "YOLO11n",
                    "weights_sha256": self.state["weights_sha256"],
                    "model_version": self.config["version"],
                    "device": self.state["device"],
                    "wall_seconds": self.state["last_seconds"],
                    "speed_ms": result.get("speed_ms"),
                    "recovery": result.get("recovery"),
                    "diagnostics": diagnostics,
                }
            except Exception as exc:
                self.close()
                self.state.update(status="error", error=type(exc).__name__)
                return {
                    "status": "failed",
                    "boxes": [],
                    "error": type(exc).__name__,
                    "model": "YOLO11n",
                }
