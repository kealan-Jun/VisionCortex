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

from .run_insights import hardware_summary


class _NvmlSampler:
    def __init__(self) -> None:
        import pynvml

        self.module = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._errors: list[tuple[str, Exception]] = []

    @classmethod
    def create(cls) -> "_NvmlSampler | None":
        try:
            return cls()
        except Exception:
            # GPU telemetry is optional.  pynvml raises its own exception
            # hierarchy when the package exists but the driver is unavailable.
            return None

    def sample(self) -> dict[str, Any]:
        nvml = self.module

        def value(name: str, call, scale: float = 1.0) -> float | None:
            try:
                return float(call()) / scale
            except Exception as exc:  # NVML raises driver-specific subclasses.
                self._errors.append((name, exc))
                return None

        utilization = None
        try:
            utilization = nvml.nvmlDeviceGetUtilizationRates(self.handle)
        except Exception as exc:
            self._errors.append(("utilization", exc))
        memory = None
        try:
            memory = nvml.nvmlDeviceGetMemoryInfo(self.handle)
        except Exception as exc:
            self._errors.append(("memory", exc))
        decoder = value(
            "decoder", lambda: nvml.nvmlDeviceGetDecoderUtilization(self.handle)[0]
        )
        encoder = value(
            "encoder", lambda: nvml.nvmlDeviceGetEncoderUtilization(self.handle)[0]
        )
        return {
            "utilization.gpu": float(utilization.gpu) if utilization is not None else None,
            "utilization.decoder": decoder,
            "utilization.encoder": encoder,
            "memory.used": float(memory.used) / 1024**2 if memory is not None else None,
            "memory.total": float(memory.total) / 1024**2 if memory is not None else None,
            "temperature.gpu": value(
                "temperature",
                lambda: nvml.nvmlDeviceGetTemperature(
                    self.handle, nvml.NVML_TEMPERATURE_GPU
                )
            ),
            "power.draw": value(
                "power", lambda: nvml.nvmlDeviceGetPowerUsage(self.handle), 1000.0
            ),
            "clocks.sm": value(
                "clock",
                lambda: nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM)
            ),
        }

    def drain_errors(self) -> list[tuple[str, Exception]]:
        errors = list(self._errors)
        self._errors.clear()
        return errors

    def close(self) -> None:
        try:
            self.module.nvmlShutdown()
        except Exception:
            pass


class ResourceMonitor:
    """Low-overhead CPU/RAM/GPU/NVDEC/NVENC telemetry sampled by stage."""

    def __init__(
        self,
        destination: Path,
        interval_seconds: float = 1.0,
        live_mirror_destination: Path | None = None,
    ):
        self.destination = destination
        self.live_destination = destination.with_name(f"{destination.stem}_live.json")
        self.live_mirror_destination = live_mirror_destination
        self.journal_destination = destination.with_suffix(".jsonl")
        self.interval = max(0.25, interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stage = "not_started"
        self._samples: list[dict[str, Any]] = []
        self._previous_network = psutil.net_io_counters()
        self._previous_process_io = self._process_tree_io()
        self._previous_perf = time.perf_counter()
        self._sampling_errors: list[dict[str, Any]] = []
        try:
            self._nvml = _NvmlSampler()
        except Exception as exc:
            self._nvml = None
            self._record_sampling_error("nvml_initialization", exc)
        self._thread_ended_unexpectedly = False

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(10.0, self.interval * 5))
        report = self.report()
        self._write_atomic(self.destination, report)
        latest = self._samples[-1] if self._samples else None
        self._write_atomic(
            self.live_destination,
            {
                "schema_version": "visioncortex-resource-telemetry-live/1",
                "status": "completed",
                "sample_count": len(self._samples),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "latest": latest,
            },
        )
        if self.live_mirror_destination is not None:
            self._write_atomic(
                self.live_mirror_destination,
                {
                    "schema_version": "visioncortex-resource-telemetry-live/1",
                    "status": "completed",
                    "sample_count": len(self._samples),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "latest": latest,
                },
            )
        if self._nvml is not None:
            self._nvml.close()
            self._nvml = None
        return report

    def _record_sampling_error(self, component: str, exc: Exception) -> None:
        self._sampling_errors.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": self._stage,
                "component": component,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        if len(self._sampling_errors) > 100:
            del self._sampling_errors[:-100]

    def _write_live_payload(self, payload: dict[str, Any]) -> None:
        for component, destination in (
            ("live_destination", self.live_destination),
            ("live_mirror_destination", self.live_mirror_destination),
        ):
            if destination is None:
                continue
            try:
                self._write_atomic(destination, payload)
            except Exception as exc:
                # Telemetry is evidence, but a transient SMB/live-file failure
                # must not silently kill all later sampling or the pipeline.
                self._record_sampling_error(component, exc)

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{threading.get_ident()}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _gpu_nvidia_smi() -> dict[str, Any]:
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

    def _gpu(self) -> dict[str, Any]:
        if self._nvml is not None:
            values = self._nvml.sample()
            for component, exc in self._nvml.drain_errors():
                self._record_sampling_error(f"nvml_{component}", exc)
            if any(value is not None for value in values.values()):
                return values
        return self._gpu_nvidia_smi()

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
        try:
            psutil.cpu_percent(interval=None)
            while not self._stop.wait(self.interval):
                try:
                    now_perf = time.perf_counter()
                    elapsed = max(1e-6, now_perf - self._previous_perf)
                    memory = psutil.virtual_memory()
                    network = psutil.net_io_counters()
                    process_io = self._process_tree_io()
                    network_received = max(
                        0, int(network.bytes_recv - self._previous_network.bytes_recv)
                    )
                    network_sent = max(
                        0, int(network.bytes_sent - self._previous_network.bytes_sent)
                    )
                    process_read = max(
                        0,
                        int(
                            process_io["read_bytes"]
                            - self._previous_process_io["read_bytes"]
                        ),
                    )
                    process_write = max(
                        0,
                        int(
                            process_io["write_bytes"]
                            - self._previous_process_io["write_bytes"]
                        ),
                    )
                    sample = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "stage": self._stage,
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "memory_percent": memory.percent,
                        "memory_used_bytes": memory.used,
                        "memory_total_bytes": memory.total,
                        "sample_interval_seconds": elapsed,
                        "gpu": self._gpu(),
                        "host_network": {
                            "received_bytes_delta": network_received,
                            "sent_bytes_delta": network_sent,
                            "received_mib_per_second": round(
                                network_received / elapsed / 1024**2, 4
                            ),
                            "sent_mib_per_second": round(
                                network_sent / elapsed / 1024**2, 4
                            ),
                            "scope": "host_all_network_interfaces",
                        },
                        "pipeline_process_tree_io": {
                            "read_bytes_delta": process_read,
                            "write_bytes_delta": process_write,
                            "read_mib_per_second": round(
                                process_read / elapsed / 1024**2, 4
                            ),
                            "write_mib_per_second": round(
                                process_write / elapsed / 1024**2, 4
                            ),
                            "sampled_process_count": process_io["process_count"],
                        },
                    }
                    self._samples.append(sample)
                    # Advance counters before publishing the live payload. A
                    # failed NAS write must not inflate the next sample.
                    self._previous_network = network
                    self._previous_process_io = process_io
                    self._previous_perf = now_perf
                    try:
                        self.journal_destination.parent.mkdir(parents=True, exist_ok=True)
                        with self.journal_destination.open("a", encoding="utf-8") as journal:
                            journal.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    except OSError as exc:
                        self._record_sampling_error("sample_journal", exc)
                    self._write_live_payload(
                        {
                            "schema_version": "visioncortex-resource-telemetry-live/1",
                            "status": "running",
                            "sample_count": len(self._samples),
                            "sampling_error_count": len(self._sampling_errors),
                            "updated_at": sample["timestamp"],
                            "latest": sample,
                        }
                    )
                except Exception as exc:
                    self._record_sampling_error("sample_loop", exc)
        except BaseException as exc:
            self._thread_ended_unexpectedly = True
            if isinstance(exc, Exception):
                self._record_sampling_error("monitor_thread", exc)

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
            "gpu_sm_clock_mhz": "clocks.sm",
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
        report = {
            "schema_version": "visioncortex-resource-telemetry/2",
            "sample_count": len(self._samples),
            "sampling_interval_seconds": self.interval,
            "gpu_telemetry_backend": "nvml" if self._nvml is not None else "nvidia-smi",
            "network_scope_note": (
                "Host NIC counters include unrelated host traffic; pipeline process-tree I/O is "
                "reported separately and includes local and NAS file I/O."
            ),
            "smb_connections": self._smb_connections(),
            "monitor_health": {
                "sampling_error_count": len(self._sampling_errors),
                "sampling_errors": list(self._sampling_errors),
                "thread_ended_unexpectedly": self._thread_ended_unexpectedly,
                "thread_alive_after_stop": bool(
                    self._thread is not None and self._thread.is_alive()
                ),
            },
            "stage_summaries": summaries,
            "samples": self._samples,
        }
        report["hardware_summary"] = hardware_summary(report)
        return report
