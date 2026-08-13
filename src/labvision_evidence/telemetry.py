from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
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
        self._previous_network = psutil.net_io_counters()
        self._previous_process_io = self._process_tree_io()
        self._previous_perf = time.perf_counter()

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

    @staticmethod
    def _process_tree_io() -> dict[str, int]:
        read_bytes = 0
        write_bytes = 0
        process_count = 0
        try:
            root = psutil.Process()
            processes = [root, *root.children(recursive=True)]
        except (psutil.Error, OSError):
            processes = []
        for process in processes:
            try:
                counters = process.io_counters()
                read_bytes += int(counters.read_bytes)
                write_bytes += int(counters.write_bytes)
                process_count += 1
            except (psutil.Error, OSError, AttributeError):
                continue
        return {
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "process_count": process_count,
        }

    @staticmethod
    def _smb_connections() -> list[dict[str, Any]]:
        if os.name != "nt" or not shutil.which("powershell"):
            return []
        command = (
            "Get-SmbConnection -ErrorAction SilentlyContinue | "
            "Select-Object ServerName,ShareName,Dialect,NumOpens,Encrypted | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True, check=False, text=True, timeout=8,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []
            payload = json.loads(result.stdout)
            return payload if isinstance(payload, list) else [payload]
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return []

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.wait(self.interval):
            now_perf = time.perf_counter()
            elapsed = max(1e-6, now_perf - self._previous_perf)
            memory = psutil.virtual_memory()
            network = psutil.net_io_counters()
            process_io = self._process_tree_io()
            network_received = max(0, int(network.bytes_recv - self._previous_network.bytes_recv))
            network_sent = max(0, int(network.bytes_sent - self._previous_network.bytes_sent))
            process_read = max(
                0, int(process_io["read_bytes"] - self._previous_process_io["read_bytes"])
            )
            process_write = max(
                0, int(process_io["write_bytes"] - self._previous_process_io["write_bytes"])
            )
            self._samples.append({
                "timestamp": datetime.now(timezone.utc).isoformat(), "stage": self._stage,
                "cpu_percent": psutil.cpu_percent(interval=None), "memory_percent": memory.percent,
                "memory_used_bytes": memory.used,
                "gpu": self._gpu(),
                "host_network": {
                    "received_bytes_delta": network_received,
                    "sent_bytes_delta": network_sent,
                    "received_mib_per_second": round(network_received / elapsed / 1024**2, 4),
                    "sent_mib_per_second": round(network_sent / elapsed / 1024**2, 4),
                    "scope": "host_all_network_interfaces",
                },
                "pipeline_process_tree_io": {
                    "read_bytes_delta": process_read,
                    "write_bytes_delta": process_write,
                    "read_mib_per_second": round(process_read / elapsed / 1024**2, 4),
                    "write_mib_per_second": round(process_write / elapsed / 1024**2, 4),
                    "sampled_process_count": process_io["process_count"],
                },
            })
            self._previous_network = network
            self._previous_process_io = process_io
            self._previous_perf = now_perf

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
            for output_name, section, field in (
                ("host_network_receive_mib_s", "host_network", "received_mib_per_second"),
                ("host_network_send_mib_s", "host_network", "sent_mib_per_second"),
                ("pipeline_read_mib_s", "pipeline_process_tree_io", "read_mib_per_second"),
                ("pipeline_write_mib_s", "pipeline_process_tree_io", "write_mib_per_second"),
            ):
                summary[output_name] = stats([sample[section][field] for sample in samples])
            summary["host_network_received_bytes"] = sum(
                sample["host_network"]["received_bytes_delta"] for sample in samples
            )
            summary["host_network_sent_bytes"] = sum(
                sample["host_network"]["sent_bytes_delta"] for sample in samples
            )
            summary["pipeline_read_bytes"] = sum(
                sample["pipeline_process_tree_io"]["read_bytes_delta"] for sample in samples
            )
            summary["pipeline_write_bytes"] = sum(
                sample["pipeline_process_tree_io"]["write_bytes_delta"] for sample in samples
            )
            summaries[stage] = summary
        return {
            "schema_version": "visioncortex-resource-telemetry/2",
            "sample_count": len(self._samples),
            "sampling_interval_seconds": self.interval,
            "network_scope_note": (
                "Host NIC counters include unrelated host traffic; pipeline process-tree I/O is "
                "reported separately and includes local and NAS file I/O."
            ),
            "smb_connections": self._smb_connections(),
            "stage_summaries": summaries,
            "samples": self._samples,
        }
