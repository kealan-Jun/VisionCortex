"""Offline, fail-closed comparison of bounded fine-scan benchmark receipts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import subprocess

ROLES = {"first_person", "third_person"}


def _checkout_identity() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    return sha, bool(dirty)


def _decode(text: str):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(text, parse_constant=invalid)


def _json(path: Path):
    return _decode(path.read_text(encoding="utf-8"))


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Receipt path escapes the benchmark output directory")
    return resolved


def _load(root: Path) -> tuple[dict, dict]:
    result = _json(_inside(root, root / "result.json"))
    if result["phase"] != "fine" or not result["rows"]:
        raise ValueError("A completed nonempty fine scan is required")
    rows = {}
    for row in result["rows"]:
        view = row["view_id"]
        if not isinstance(view, str) or Path(view).name != view or view in rows:
            raise ValueError("Invalid or duplicated view identity")
        directory = _inside(root, root / view)
        ledger_path = Path(row["ledger"])
        ledger = _inside(root, ledger_path if ledger_path.is_absolute() else root / ledger_path)
        frames = [_decode(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        audit = _json(_inside(root, directory / "ActivityAudit.json"))
        candidates = _json(_inside(root, directory / "Candidates.json"))
        runtimes = [_json(_inside(root, p)) for p in sorted(directory.rglob("runtime_fine*.json"))]
        if not frames or not runtimes or not isinstance(candidates, list):
            raise ValueError(f"Incomplete frame, candidate or runtime evidence for {view}")
        for key in ("selected_key_events", "intervals", "events"):
            if not isinstance(audit[key], list):
                raise ValueError(f"Missing action audit list: {key}")
        rows[view] = {"input": {k: row[k] for k in ("source", "source_stat", "windows", "role")},
                      "frames": frames, "audit": audit, "candidates": candidates, "runtime": runtimes}
    return result, rows


def _sampling(frames: list[dict], view: str, source: dict) -> list[dict]:
    identities = []
    for frame in frames:
        physical = frame["source_frame"]
        if (frame["view_id"] != view or frame["role"] != source["role"]
                or physical["status"] != "resolved"
                or any(physical[k] is None for k in ("packet_position", "source_pts", "time_base"))
                or not re.fullmatch(r"[0-9a-f]{64}", physical["decoded_pixels_sha256"])
                or physical["source_path"] != source["source"]
                or [physical["source_size_bytes"], physical["source_mtime_ns"]] != source["source_stat"]):
            raise ValueError(f"Frame provenance does not match input: {view}")
        identities.append({k: frame[k] for k in ("view_id", "role", "frame_index", "local_ms", "global_ms", "source_frame")})
    return identities


def _changes(before: list, after: list) -> dict:
    def counts(items):
        return Counter(json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False) for item in items)
    left, right = counts(before), counts(after)
    return {"baseline_count": len(before), "candidate_count": len(after),
            "removed": [json.loads(item) for item in (left - right).elements()],
            "added": [json.loads(item) for item in (right - left).elements()]}


def compare_fine_scans(baseline: Path, candidate: Path, target_batch: int = 16) -> dict:
    """Read only owned receipt files; never inspect media, engines or models."""
    if isinstance(target_batch, bool) or not isinstance(target_batch, int) or target_batch < 1:
        raise ValueError("target_batch must be a positive integer")
    missing, differences, views, role_batches = [], [], {}, {}
    report = {"scope": "bounded_same_source_fine_scan_consistency_not_human_accuracy",
              "target_batch": target_batch, "promotion_ready": False,
              "missing_evidence": missing, "differences": differences, "views": views,
              "role_batches": role_batches,
              "baseline_basis": "reviewed_existing_engine_with_aligned_source_weights",
              "remaining_gates": ["production_concurrency_acceptance", "broader_real_video_action_quality"]}
    runs = []
    for label, path in (("baseline", baseline), ("candidate", candidate)):
        try:
            runs.append(_load(Path(path).resolve(strict=True)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            missing.append({"run": label, "gate": "complete_receipts", "reason": str(exc)})
    if len(runs) == 2:
        (before, left), (after, right) = runs
        try:
            current_sha, current_dirty = _checkout_identity()
            if current_dirty:
                missing.append({"gate": "clean_current_checkout"})
            for label, result in (("baseline", before), ("candidate", after)):
                identity = result["comparison_identity"]
                for key, size in (("code_sha", 40), ("config_sha256", 64), ("manifest_sha256", 64)):
                    if not isinstance(identity[key], str) or not re.fullmatch(f"[0-9a-f]{{{size}}}", identity[key]):
                        raise ValueError(f"Invalid comparison identity: {label}.{key}")
                if identity.get("git_dirty") is not False or identity["code_sha"] != current_sha:
                    missing.append({"run": label, "gate": "current_clean_source_identity"})
            if before["comparison_identity"] != after["comparison_identity"]:
                differences.append({"gate": "comparison_identity", "baseline": before["comparison_identity"],
                                    "candidate": after["comparison_identity"]})
            for role in sorted(ROLES):
                for result in (before, after):
                    if not re.fullmatch(r"[0-9a-f]{64}", result["models"][role]["sha256"]):
                        raise ValueError(f"Missing source weights identity: {role}")
                if before["models"][role]["sha256"] != after["models"][role]["sha256"]:
                    differences.append({"gate": "source_weights", "role": role})
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            missing.append({"gate": "comparison_identity", "reason": str(exc)})
        for role in sorted(ROLES):
            try:
                engine = after["models"][f"{role}_engine"]
                build = engine["build_receipt"]
                for field in ("engine_sha256", "weights_sha256"):
                    if not re.fullmatch(r"[0-9a-f]{64}", build[field]):
                        raise ValueError(f"Invalid candidate build identity: {field}")
                if (build["engine_sha256"] != engine["sha256"]
                        or build["weights_sha256"] != after["models"][role]["sha256"]):
                    differences.append({"gate": "candidate_build_identity", "role": role})
            except (ValueError, KeyError, TypeError) as exc:
                missing.append({"gate": "candidate_build_identity", "role": role, "reason": str(exc)})
        if set(left) != set(right):
            differences.append({"gate": "view_set", "baseline": sorted(left), "candidate": sorted(right)})
        for label, rows in (("baseline", left), ("candidate", right)):
            if {r["input"]["role"] for r in rows.values()} != ROLES:
                missing.append({"run": label, "gate": "both_view_roles"})
        for view in sorted(set(left) & set(right)):
            a, b = left[view], right[view]
            if a["input"] != b["input"]:
                differences.append({"gate": "same_input", "view_id": view, "baseline": a["input"], "candidate": b["input"]})
            try:
                if _sampling(a["frames"], view, a["input"]) != _sampling(b["frames"], view, b["input"]):
                    differences.append({"gate": "same_sampled_frames", "view_id": view})
                detail = views[view] = {"frames": len(b["frames"]), "action_changes": {}}
                for key in ("intervals", "selected_key_events", "events", "candidates"):
                    change = _changes(a[key] if key == "candidates" else a["audit"][key],
                                      b[key] if key == "candidates" else b["audit"][key])
                    detail["action_changes"][key] = change
                    if change["removed"] or change["added"]:
                        differences.append({"gate": "same_actions", "view_id": view, "field": key, **change})
                if not b["audit"]["selected_key_events"]:
                    detail["quality_limit"] = "No selected action in this window; action recall is NOT_PROVEN"
                detail["detection_frames_changed"] = sum(x.get("detections") != y.get("detections")
                                                          for x, y in zip(a["frames"], b["frames"], strict=False))
            except (ValueError, KeyError, TypeError) as exc:
                missing.append({"gate": "frame_or_action_evidence", "view_id": view, "reason": str(exc)})
            for label, row, result in (("baseline", a, before), ("candidate", b, after)):
                for runtime in row["runtime"]:
                    try:
                        role = row["input"]["role"]
                        engine = result["models"][f"{role}_engine"]
                        if (runtime["role"] != role or runtime["phase"] != "fine"
                                or runtime["backend"] != "TensorRT" or runtime["model_path"] != engine["path"]
                                or not re.fullmatch(r"[0-9a-f]{64}", engine["sha256"])):
                            raise ValueError("Runtime engine identity differs from the fine-scan model receipt")
                    except (ValueError, KeyError, TypeError) as exc:
                        missing.append({"gate": "runtime_engine_identity", "run": label, "view_id": view, "reason": str(exc)})
            for runtime in b["runtime"]:
                try:
                    role = b["input"]["role"]
                    shared = runtime.get("shared_inference")
                    if runtime.get("inference_statistics_scope") == "caller_submissions_not_physical_gpu_batches" and not shared:
                        raise ValueError("Shared inference requires physical backend batch statistics")
                    maximum = shared["engine_batch_size_max"] if shared else runtime["actual_batch_size_max"]
                    contractions = shared["oom_batch_contractions"] if shared else runtime["oom_batch_contractions"]
                    if type(maximum) is not int or not isinstance(contractions, list):
                        raise ValueError("Missing physical batch or contraction evidence")
                    if shared:
                        counts = shared["engine_batch_size_counts"]
                        if (shared["scope"] != "process_role_pool_cumulative_not_per_video" or not isinstance(counts, dict) or not counts
                                or any(not k.isdigit() or int(k) < 1 or type(v) is not int or v < 1 for k, v in counts.items())
                                or max(map(int, counts)) != maximum):
                            raise ValueError("Invalid cumulative physical batch histogram")
                    role_batches[role] = max(role_batches.get(role, 0), maximum)
                    if contractions:
                        differences.append({"gate": "no_batch_contractions", "view_id": view, "contractions": contractions})
                    effective = shared["effective_batch_size"] if shared else runtime["final_effective_batch_size"]
                    if runtime["engine_build_batch"] < target_batch or effective < target_batch:
                        differences.append({"gate": "batch_capacity", "view_id": view, "target": target_batch})
                except (ValueError, KeyError, TypeError) as exc:
                    missing.append({"gate": "physical_batch_evidence", "view_id": view, "reason": str(exc)})
        for role in sorted(ROLES):
            if role not in role_batches:
                missing.append({"gate": "physical_batch_evidence", "role": role})
            elif role_batches[role] < target_batch:
                differences.append({"gate": "target_batch_not_observed", "role": role, "observed": role_batches[role]})
        report["scan_wall_seconds"] = {"baseline": before.get("wall_seconds"), "candidate": after.get("wall_seconds")}
    report["comparison_status"] = "rejected" if differences else "insufficient_evidence" if missing else "passed"
    report["evidence_status"] = "PARTIAL_EVIDENCE" if report["comparison_status"] == "passed" else "NOT_PROVEN"
    timings = report.get("scan_wall_seconds", {})
    if report["comparison_status"] == "passed" and timings and all(type(v) in (int, float) and v > 0 for v in timings.values()):
        report["bounded_scan_speedup"] = timings["baseline"] / timings["candidate"]
    else:
        report["speed_evidence"] = "NOT_PROVEN; raw wall times do not establish an attributable speedup"
    return report
