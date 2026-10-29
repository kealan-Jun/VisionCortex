"""Read-only, source-backed hardware and token summaries for a saved run."""
from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Any


def with_operation_refreshes(root, metrics: dict) -> dict:
    """Use the same durable follow-up ledger in the UI and exported reports."""
    metrics = copy.deepcopy(metrics)
    calls = metrics.setdefault("mllm_calls", [])
    seen = {call.get("refresh_call_id") for call in calls}
    changed = False
    for path in sorted((root / "JSON-Config-Files/Stage-Refreshes").glob("*.json")):
        try:
            if path.stat().st_size > 32 * 1024 * 1024:
                continue
            receipt = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(receipt, dict) or receipt.get("scope") not in {"operations", "gap_review"}:
            continue
        for index, call in enumerate(receipt.get("model_calls") or []):
            identity = f"{path.name}:{index}"
            if isinstance(call, dict) and identity not in seen:
                calls.append({**call, "stage": "gap_review" if receipt["scope"] == "gap_review" else "operation_review_refresh", "refresh_call_id": identity})
                seen.add(identity)
                changed = True
    if changed:
        metrics.setdefault("tokens", {})["run_total"] = token_summary(metrics)["totals"]
    return metrics


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _epoch(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def _stats(values) -> dict[str, Any]:
    values = sorted(v for value in values if (v := _number(value)) is not None)
    if not values:
        return {"count": 0, "mean": None, "max": None, "p95": None}
    return {"count": len(values), "mean": round(sum(values) / len(values), 3),
            "max": values[-1], "p95": values[max(0, math.ceil(len(values) * .95) - 1)]}


def hardware_summary(telemetry: dict, live: dict | None = None,
                     run_started_at: str | None = None, max_points: int = 180) -> dict:
    """Do not turn missing samples into zero or mix a previous attempt into this one."""
    start = _epoch(run_started_at)
    rows = []
    excluded = 0
    for sample in telemetry.get("samples") or []:
        timestamp = _epoch(sample.get("timestamp"))
        if timestamp is None or (start is not None and timestamp < start):
            excluded += 1
            continue
        gpu = sample.get("gpu") or {}
        used, total = _number(gpu.get("memory.used")), _number(gpu.get("memory.total"))
        rows.append({"timestamp": sample["timestamp"], "epoch": timestamp,
                     "stage": sample.get("stage") or "unknown",
                     "gpu_percent": gpu.get("utilization.gpu"),
                     "vram_percent": 100 * used / total if used is not None and total else None,
                     "vram_used_mib": used, "vram_total_mib": total,
                     "cpu_percent": sample.get("cpu_percent"),
                     "ram_percent": sample.get("memory_percent"),
                     "nvdec_percent": gpu.get("utilization.decoder"),
                     "nvenc_percent": gpu.get("utilization.encoder"),
                     "power_w": gpu.get("power.draw"),
                     "temperature_c": gpu.get("temperature.gpu"),
                     "network_receive_mib_s": (sample.get("host_network") or {}).get("received_mib_per_second"),
                     "process_read_mib_s": (sample.get("pipeline_process_tree_io") or {}).get("read_mib_per_second")})
    rows.sort(key=lambda row: row["epoch"])
    fields = ("gpu_percent", "vram_percent", "vram_used_mib", "vram_total_mib", "cpu_percent",
              "ram_percent", "nvdec_percent", "nvenc_percent", "power_w", "temperature_c",
              "network_receive_mib_s", "process_read_mib_s")

    def summarize(items):
        return {"sample_count": len(items), **{key: _stats(row[key] for row in items) for key in fields}}

    stages = defaultdict(list)
    for row in rows:
        stages[row["stage"]].append(row)
    # At most max_points + stage_count buckets. Stage boundaries never blend.
    width = max(1, math.ceil(len(rows) / max(1, max_points)))
    timeline = []
    for stage, items in stages.items():
        for i in range(0, len(items), width):
            bucket = items[i:i + width]
            timeline.append({"stage": stage, "timestamp": bucket[0]["timestamp"],
                             "end_timestamp": bucket[-1]["timestamp"],
                             "offset_seconds": round(bucket[0]["epoch"] - rows[0]["epoch"], 3),
                             "sample_count": len(bucket),
                             **{key: _stats(row[key] for row in bucket)
                                for key in ("gpu_percent", "vram_percent", "cpu_percent")}})
    timeline.sort(key=lambda row: row["offset_seconds"])
    latest = (live or {}).get("latest") or {}
    latest_time = _epoch(latest.get("timestamp"))
    if latest_time is None or (start is not None and latest_time < start):
        latest = {}
    return {"available": bool(rows), "sample_count": len(rows),
            "excluded_sample_count": excluded,
            "sampling_interval_seconds": telemetry.get("sampling_interval_seconds"),
            "started_at": rows[0]["timestamp"] if rows else None,
            "ended_at": rows[-1]["timestamp"] if rows else None,
            "scope": "host CPU/RAM and device 0 GPU; includes other processes",
            "aggregation": "sample arithmetic mean; peaks use all valid raw samples",
            "overall": summarize(rows), "stages": {key: summarize(items) for key, items in stages.items()},
            "timeline": timeline, "latest": latest,
            "live_status": (live or {}).get("status"),
            "monitor_health": telemetry.get("monitor_health") or {}}


def token_summary(metrics: dict) -> dict:
    """Count all stages and retries once; local reuse is not a new paid request."""
    fields = ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")
    calls = metrics.get("mllm_calls")
    ledger_total = (metrics.get("tokens") or {}).get("run_total") or {}
    if not isinstance(calls, list) or not calls:
        return {"source": "merged_run_metrics.tokens.run_total", "available": bool(ledger_total),
                "totals": {key: ledger_total.get(key) for key in fields},
                "executed_calls": None, "reused_calls": None, "attempts": None,
                "unknown_attempts": None, "stages": [], "models": [],
                "reconciled": None, "coverage": "aggregate_only"}

    def summarize(items):
        executed = [item for item in items if not item.get("cache_reused")]
        unknown = 0
        attempts = 0
        missing_fields = {key: 0 for key in fields}
        for item in executed:
            usage = item.get("usage") or {}
            receipts = item.get("attempt_receipts") or []
            count = _number(item.get("attempts"))
            count = int(count) if count is not None else max(1, len(receipts))
            attempts += count
            reported_unknown = _number(usage.get("unknown_attempt_count"))
            if reported_unknown is not None:
                unknown += int(reported_unknown)
            elif receipts:
                unknown += sum(_number((r.get("usage") or {}).get("total_tokens")) is None for r in receipts)
                unknown += max(0, count - len(receipts))
            elif count:
                # Legacy successful receipt with multiple attempts does not prove
                # the earlier attempts were free. Keep their usage unresolved.
                unknown += count if _number(usage.get("total_tokens")) is None else max(0, count - 1)
            for key in fields:
                missing_fields[key] += count > 0 and _number(usage.get(key)) is None
        totals = {}
        for key in fields:
            values = [_number((item.get("usage") or {}).get(key)) for item in executed]
            known = [value for value in values if value is not None]
            totals[key] = int(sum(known)) if known else (0 if not executed or attempts == 0 else None)
        return {"totals": totals, "executed_calls": len(executed),
                "reused_calls": len(items) - len(executed), "attempts": attempts,
                "unknown_attempts": unknown, "missing_usage_calls": missing_fields}

    stages, models = defaultdict(list), defaultdict(list)
    for call in calls:
        stages[call.get("stage") or "unknown"].append(call)
        models[(call.get("provider") or "未记录厂商", call.get("response_model") or call.get("model") or "未记录模型")].append(call)
    summary = summarize(calls)
    comparisons = [summary["totals"][key] == ledger_total[key] for key in fields[:3]
                   if _number(ledger_total.get(key)) is not None and summary["totals"][key] is not None]
    return {"available": True, "source": "merged_run_metrics.mllm_calls", **summary,
            "coverage": "partial" if summary["unknown_attempts"] or any(summary["missing_usage_calls"][f] for f in fields[:3]) else "reported",
            "reconciled": all(comparisons) if comparisons else None,
            "stages": [{"stage": key, **summarize(items)} for key, items in stages.items()],
            "models": [{"provider": key[0], "model": key[1], **summarize(items)} for key, items in models.items()]}


def timing_summary(metrics: dict, scans: dict | None = None) -> dict:
    """Wall stages and overlapping worker/request time are separate measures."""
    stages = []
    total = _number(metrics.get("total_duration_seconds"))
    for row in metrics.get("stage_durations") or []:
        seconds = _number(row.get("duration_seconds"))
        if seconds is not None:
            stages.append({"stage": row.get("stage", "unknown"), "seconds": seconds,
                           "status": row.get("status"),
                           "share_percent": round(100*seconds/total, 2) if total else None})
    calls = [c for c in metrics.get("mllm_calls") or [] if not c.get("cache_reused")]
    attempts = [a for c in calls for a in c.get("attempt_receipts") or []]
    intervals = []
    for a in attempts:
        lo, hi = _epoch(a.get("started_at")), _epoch(a.get("ended_at"))
        if lo is not None and hi is not None and hi > lo:
            intervals.extend([(lo, 1), (hi, -1)])
    active = peak = 0
    busy = weighted = 0.0
    previous = None
    for stamp, delta in sorted(intervals):
        if previous is not None and active:
            busy += stamp-previous
            weighted += (stamp-previous)*active
        active += delta
        peak = max(peak, active)
        previous = stamp

    def known_sum(rows, field):
        values = [_number(row.get(field)) for row in rows]
        known = [v for v in values if v is not None]
        return round(sum(known), 3) if known else None

    expected = sum(int(_number(c.get("attempts")) or len(c.get("attempt_receipts") or []) or 1) for c in calls)
    scan_rows = []
    for phase, scan in (scans or {}).items():
        diagnosis = scan.get("bottleneck_diagnosis") or {}
        scan_rows.append({"phase": phase, "available": bool(diagnosis),
                          **{k: diagnosis.get(k) for k in (
                              "queue_wait_seconds", "inference_seconds", "tracking_and_ledger_seconds",
                              "actual_batch_size_mean", "batch_fill_ratio", "inference_frame_count",
                              "classification", "component_timings", "component_profiled_workers", "component_expected_workers")}})
    return {"available": bool(stages), "wall_seconds": total, "stages": stages,
            "largest_stage": max(stages, key=lambda s: s["seconds"], default=None),
            "stage_sum_seconds": round(sum(s["seconds"] for s in stages), 3),
            "model": {"scope": "all ledger calls, including later reviews; overlaps are not wall-time additions",
                      "call_seconds_sum": known_sum(calls, "latency_seconds"),
                      "attempt_seconds_sum": known_sum(attempts, "latency_seconds"),
                      "failed_attempt_seconds_sum": known_sum([a for a in attempts if a.get("status") == "failed"], "latency_seconds"),
                      "backoff_seconds_sum": known_sum(attempts, "retry_wait_seconds"),
                      "rate_limit_attempts": sum(a.get("http_status") == 429 or a.get("status_code") == 429 for a in attempts),
                      "backoff_recorded_attempts": sum(_number(a.get("retry_wait_seconds")) is not None for a in attempts),
                      "timed_attempts": len(intervals)//2, "expected_attempts": expected,
                      "concurrency_coverage": "complete" if intervals and len(intervals)//2 == expected else "partial" if intervals else "unavailable",
                      "peak_concurrency": peak if intervals else None,
                      "mean_active_concurrency": round(weighted/busy, 3) if busy else None,
                      "request_busy_wall_seconds": round(busy, 3) if intervals else None},
            "scans": scan_rows,
            "read_decode_separately_measured": False,
            "measurement_note": "Frame supply wait includes storage, decode and scheduling; it is not isolated decode time."}
