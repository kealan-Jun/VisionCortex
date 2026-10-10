"""Portable reference graph projected from the existing disposable evidence index."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def export_speech_references(root: Path, groups: list[dict]) -> dict:
    from . import speech_search, speech_worker
    control = root / "JSON-Config-Files/speech.json"
    result = {"status": "not_available", "sources": [], "utterances": [], "step_links": [],
              "physical_action_confirmation": False}
    if not control.is_file():
        return result
    try:
        identity = speech_worker.sha256(control)
        payload = speech_worker.read_json(control, 16 * 1024 * 1024)
        rows = speech_search._build(root, payload, identity)["rows"]
        if speech_worker.sha256(control) != identity:
            raise ValueError("转写版本在导出期间变化")
        keep = ("reference_id", "source_id", "chunk_id", "view_id", "text", "speaker_identity",
                "start_seconds", "end_seconds", "playback_start_seconds", "playback_end_seconds",
                "aligned_start_ms", "aligned_end_ms", "recorded_start_global_us", "recorded_end_global_us",
                "alignment", "alignment_basis", "transcript_path", "transcript_sha256", "audio")
        utterances = [{key: row.get(key) for key in keep} for row in rows]
        known = {row["reference_id"] for row in utterances}
        links, unresolved = [], []
        for group in groups:
            for ordinal, step in enumerate((group.get("model_understanding") or {}).get("steps") or [], 1):
                ids = list(dict.fromkeys(step.get("speech_segment_ids") or []))
                if not ids:
                    continue
                missing = [ref for ref in ids if ref not in known]
                unresolved.extend(missing)
                links.append({"group_id": group.get("group_id"), "step_number": ordinal,
                              "step_id": step.get("step_id"), "speech_reference_ids": ids,
                              "unresolved_reference_ids": missing, "relation": "model_referenced_speech",
                              "physical_action_confirmation": False})
        return {**result, "status": "references_incomplete" if unresolved else "available",
                "transcription_status": payload.get("status"), "speech_sha256": identity,
                "sources": [{key: source.get(key) for key in ("id", "view_id", "segment_ordinal", "original", "alignment", "alignment_basis")}
                            for source in payload.get("sources", [])],
                "utterances": utterances, "step_links": links,
                "unresolved_reference_ids": sorted(set(unresolved))}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**result, "status": "unavailable", "reason": type(exc).__name__}


def export_reference_index(root: Path, groups: list[dict]) -> dict:
    database = root / "JSON-Config-Files/evidence_index.sqlite"
    result = {"schema_version": "visioncortex-report-references/1",
              "source": "JSON-Config-Files/evidence_index.sqlite",
              "status": "not_available", "formal_acceptance_implied": False,
              "speech": export_speech_references(root, groups)}
    if not database.is_file():
        return result
    if not database.resolve().is_relative_to(root.resolve()):
        return {**result, "status": "unavailable", "reason": "index_outside_archive"}
    try:
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            events = [{**json.loads(row[1]), "event_uid": row[0]} for row in connection.execute("SELECT event_uid, event_json FROM key_events ORDER BY event_uid")]
            artifacts = [json.loads(row[0]) for row in connection.execute("SELECT artifact_json FROM artifacts ORDER BY artifact_uid")]
            evidence = [json.loads(row[0]) for row in connection.execute("SELECT evidence_json FROM evidence ORDER BY evidence_uid")]
    except (OSError, sqlite3.Error, ValueError) as exc:
        return {**result, "status": "unavailable", "reason": type(exc).__name__}
    by_id = {event["event_id"]: event for event in events}
    steps, unresolved = [], []
    for group in groups:
        for ordinal, step in enumerate((group.get("model_understanding") or {}).get("steps") or [], 1):
            refs = list(dict.fromkeys(step.get("supporting_event_ids") or []))
            missing = [ref for ref in refs if ref not in by_id]
            unresolved.extend(missing)
            steps.append({"group_id": group.get("group_id"), "step_number": ordinal,
                          "step_id": step.get("step_id"), "event_ids": refs,
                          "event_uids": [by_id[ref]["event_uid"] for ref in refs if ref in by_id],
                          "unresolved_event_ids": missing})
    return {**result, "status": "references_incomplete" if unresolved else "available",
            "steps": steps, "events": events, "artifacts": artifacts,
            "evidence": evidence, "unresolved_event_ids": sorted(set(unresolved))}
