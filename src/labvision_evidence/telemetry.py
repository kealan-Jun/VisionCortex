from __future__ import annotations

import json
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


class ResourceMonitor:
    """Low-overhead CPU/RAM/GPU/NVDEC/NVENC telemetry sampled by stage."""

    def __init__(self, destination: Path, interval_seconds: float = 1.0):
        self.destination = destination
        self.interval = max(0.25, interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stage = "not_started"
        self._samples: list[dict[str, Any]] = []

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval * 2))
        report = self.report()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _gpu() -> dict[str, Any]:
        if not shutil.which("nvidia-smi"):
            return {}
        fields = ["utilization.gpu", "utilization.decoder", "utilization.encoder", "memory.used",
                  "memory.total", "temperature.gpu", "power.draw", "clocks.sm"]
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, check=False, text=True, timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        parsed: dict[str, Any] = {}
        for key, value in zip(fields, values, strict=False):
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = None
        return parsed

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.wait(self.interval):
            memory = psutil.virtual_memory()
            self._samples.append({
                "timestamp": datetime.now(timezone.utc).isoformat(), "stage": self._stage,
                "cpu_percent": psutil.cpu_percent(interval=None), "memory_percent": memory.percent,
                "memory_used_bytes": memory.used, "gpu": self._gpu(),
            })

    def report(self) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            ordered = sorted(values)
            return {"mean": round(sum(values) / len(values), 3), "max": round(max(values), 3),
                    "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3)}

        grouped: dict[str, list[dict[str, Any]]] = {}
        for sample in self._samples:
            grouped.setdefault(sample["stage"], []).append(sample)
        summaries: dict[str, Any] = {}
        mappings = {
            "gpu_compute_percent": "utilization.gpu", "nvdec_percent": "utilization.decoder",
            "nvenc_percent": "utilization.encoder", "gpu_memory_used_mib": "memory.used",
            "gpu_temperature_c": "temperature.gpu", "gpu_power_w": "power.draw",
        }
        for stage, samples in grouped.items():
            summary: dict[str, Any] = {
                "sample_count": len(samples),
                "cpu_percent": stats([sample["cpu_percent"] for sample in samples]),
                "memory_percent": stats([sample["memory_percent"] for sample in samples]),
            }
            for output_name, gpu_name in mappings.items():
                summary[output_name] = stats([
                    sample["gpu"][gpu_name] for sample in samples
                    if sample["gpu"].get(gpu_name) is not None
                ])
            summaries[stage] = summary
        return {"sample_count": len(self._samples), "stage_summaries": summaries, "samples": self._samples}
