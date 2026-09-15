"""Background capture processing and read-only artifact browsing for v1 archives."""
from __future__ import annotations

import html
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from .device_day import DeviceDayRunner, append_comment
from .device_day_contract import (
    DIRECTORIES, VERSION, archive_name, atomic_json, read_json, safe_child, validate_archive_name,
)


def published_recording_status(data):
    """Expose readiness from current published clips and understanding, not old receipts."""
    understood = {item["segment_id"] for item in data.get("understandings", [])}
    by_recording = {}
    for segment in data.get("segments", []):
        by_recording.setdefault(segment["recording_id"], set()).add(segment["segment_id"])
    return data | {"recordings": [row | {
        "preprocessing_ready": bool(by_recording.get(row["recording_id"])),
        "current_results_ready": bool(by_recording.get(row["recording_id"]))
        and by_recording[row["recording_id"]].issubset(understood)
    } for row in data.get("recordings", [])]}


class DeviceDayService:
    def __init__(self, settings_factory, gpu_lock):
        self.settings_factory = settings_factory
        self.gpu_lock = gpu_lock
        self.stop_event = threading.Event()
        self.wakeup = threading.Event()
        self.thread = None
        self.last_result = {"schema_version": VERSION, "status": "not_started"}
        self._runner = None
        self._settings_key = None
        self._vision_lock = threading.Lock()
        self._vision_users = 0
        self._inventory_lock = threading.Lock()

    def start(self):
        from .runtime_process import role
        if role(self.settings_factory()) == 'web':
            return
        if not (self.settings_factory().get("device_day") or {}).get("enabled"):
            return
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="device-day-pipeline", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.wakeup.set()
        if self.thread is not None:
            # The service owns daemon discovery/dispatch threads. Returning
            # after five seconds let ASGI exit while workers still exported
            # evidence, leaving leases and ffmpeg children unfinished. Stop
            # admitting work, then wait for the existing executor drain.
            self.thread.join()

    def submit(self, settings, recordings):
        identifier = "device-day-" + uuid.uuid4().hex
        root = Path(settings["storage"]["local_runtime_root"]) / "device-day" / "requests"
        atomic_json(root / f"{identifier}.json", {"schema_version": VERSION, "status": "queued",
                    "recordings": recordings, "request_id": identifier})
        self.start()
        self.wakeup.set()
        return {"request_id": identifier, "status": "queued", "workflow": "device_day",
                "archive_url": "/device-days", "capture_deletion": "disabled_by_user"}

    def observe(self, settings, inventory):
        """Consume the existing NAS monitor snapshot, without a second scanner."""
        if not (settings.get("device_day") or {}).get("enabled"):
            return
        # Metrics must not enter recording payloads or invalidate media receipts.
        from .device_day_latency import observe
        import logging
        import sqlite3
        try:
            observe(settings, inventory.get('recordings', []), inventory.get('discovery'))
        except (OSError, sqlite3.Error, ValueError):
            logging.getLogger(__name__).warning('Discovery latency persistence unavailable')
        from .observed_inventory import observe as upsert_observations
        root = Path(settings["storage"]["local_runtime_root"]) / "device-day"
        if not upsert_observations(root, inventory, start_date=settings["device_day"].get("start_date")):
            return
        self.wakeup.set()

    def monitor_status(self, inventory):
        """Attach durable per-slice states to the existing batch monitor UI."""
        if self._runner is None:
            return inventory | {"automatic_processing": {"enabled": True, "status": "starting", "queue": {}}}
        states = {}
        for stage, queue in self._runner.queues.items():
            with queue.connect() as db:
                for row in db.execute("SELECT recording_id,payload,status,wall_seconds,result,queued_at FROM recordings"):
                    payload = json.loads(row["payload"])
                    states.setdefault(row["recording_id"], {})[stage] = {
                        "status": row["status"], "wall_seconds": row["wall_seconds"],
                        "queued_at": row["queued_at"], "source_signature": payload.get("source_signature"),
                        "audio_signature": payload.get("audio", {}).get("source_signature"),
                        "result": json.loads(row["result"] or "{}")}
        recordings = []
        for record in inventory.get("recordings", []):
            stages = {s: value for s, value in states.get(record["recording_id"], {}).items()
                      if value["source_signature"] == record["source_signature"] and
                      (s == "vision" or value["audio_signature"] == record.get("audio", {}).get("source_signature"))}
            target = archive_name(record["camera_key"], record["recording_start_us"]) if record.get("recording_start_us", 0) > 0 else None
            recordings.append(record | {"processing": stages, "archive": target,
                                       "archive_url": f"/api/device-days/{quote(target)}/files/{quote(DIRECTORIES[3])}/LaboratoryDailyReport.html" if target else None})
        by_id = {r["recording_id"]: r for r in recordings}
        batches = []
        for batch in inventory.get("batches", []):
            rows = [by_id[r["recording_id"]] for r in batch["recordings"] if r["recording_id"] in by_id]
            complete = sum(r["processing"].get("report", {}).get("status") == "completed" for r in rows)
            running = any(s["status"] == "running" for r in rows for s in r["processing"].values())
            failed = any(s["status"] == "failed" for r in rows for s in r["processing"].values())
            status = "completed" if rows and complete == len(rows) else "failed" if failed else "running" if running else "queued" if any(r["processing"] for r in rows) else "waiting"
            batches.append(batch | {"processing": {"status": status, "completed_slices": complete,
                            "total_slices": len(rows), "recordings": rows}})
        return inventory | {"recordings": recordings, "batches": batches, "automatic_processing": {
            "enabled": True, "status": self.last_result.get("status", "watching"), "queue": self._runner.queue_snapshot(),
            "storage": self.last_result.get("stages", {}).get("retention", {}).get("storage"), "archive_url": "/device-days"}}

    def _request_complete(self, request):
        from .device_day import load_context, visual_input
        from .device_day_contract import STAGES
        # Keep the full request resumable when only preprocessing is authorized.
        # Do not rescan every downstream receipt on every scheduler tick.
        from .device_day_provider_gate import ProviderGate
        if self._runner.settings.get("preprocessing_only") or ProviderGate(self._runner.config).blocks("understanding"):
            return False
        for original in request["recordings"]:
            layout = self._runner.layout(original)
            record = original | {"archive_date": layout.name[:10],
                                 "processing_priority": 1 if request.get("kind") == "historical_backfill" else 0}
            receipts = {}
            for stage in STAGES:
                path = self._runner._receipt(layout, record, stage)
                if not path.is_file():
                    return False
                receipt = read_json(path)
                inputs = (record if stage == "retention" else visual_input(receipts["retention"]) if stage == "vision"
                          else receipts["retention"] if stage == "stt" else
                          {"vision": receipts["vision"], "stt": receipts["stt"], "context": load_context(layout, record)}
                          if stage == "understanding" else receipts["understanding"])
                if receipt.get("status") != "completed" or not self._runner._accepts_receipt(receipt, self._runner._key(stage, record, inputs)):
                    return False
                receipts[stage] = receipt
        return True

    def _storage_available(self, runner, inventory):
        # Reserve space for a bounded retention batch and its derived material.
        # Existing retained media consume no additional original-copy allowance.
        pending = {r["recording_id"]: r for r in runner.queues["retention"].pending()}
        pending.update({r["recording_id"]: r for r in inventory.get("recordings", [])})
        sizes = []
        for record in sorted(pending.values(), key=lambda r: (-r.get("size_bytes", 0), r["recording_start_us"])):
            if not record.get("processable", record.get("available")) or record.get("configured_role") not in {"first_person", "third_person"}:
                continue
            target = runner.layout(record).source_path(record, "video", Path(record["video_path"]))
            if not target.is_file():
                sizes.append(record["size_bytes"])
                if len(sizes) == 8:
                    break
        available = shutil.disk_usage(runner.archive_root).free
        required = sum(sizes) * 3 + 20 * 1024**3
        return {"available_bytes": available, "required_bytes": required, "ready": available >= required,
                "capture_deletion": "disabled_by_user", "scope": "largest_eight_new_sources_and_derived_material_reserve"}

    def _acquire_vision(self):
        # The service owns one shared GPU reservation against the legacy pipeline.
        # Each slice inside that reservation uses a separate bounded worker.
        while not self.stop_event.is_set():
            with self._vision_lock:
                if self._vision_users or self.gpu_lock.acquire(blocking=False):
                    self._vision_users += 1
                    return True
            self.stop_event.wait(.1)
        return False

    def _release_vision(self):
        with self._vision_lock:
            self._vision_users -= 1
            if not self._vision_users:
                self.gpu_lock.release()

    def _loop(self):
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import ExitStack
        from .device_day_contract import STAGES
        jobs, results, next_poll = {}, {}, {}
        publication_job = None
        recovery_job, overview_job = None, None
        last_recovery, last_overview = 0, 0
        from .device_day_recovery import RetentionRecovery
        recovery = RetentionRecovery()
        timeline_job, last_timeline = None, 0
        timeline_generation = None
        publication_dirty = True
        timeline_pending = set()
        observed_mtime, observed_inventory = None, {"recordings": []}
        last_publication = time.monotonic()
        def publish(runner):
            from .device_day_capture_files import reconcile_capture_files
            from .device_day_publication import reconcile_outputs
            from .capture_link_cleanup import resume_pending
            resume_pending(runner)
            from .publication_journal import PublicationJournal, reconcile
            reconcile(runner)
            capture = reconcile_capture_files(runner)
            reconcile_outputs(runner)
            return {"capture_files": capture,
                    "pending_publications": bool(PublicationJournal(runner.runtime_root).pending(1))}
        def run_stage(runner, inventory, stage):
            from .device_day_night_schedule import stage_admitted, night_schedule
            if not stage_admitted(runner.config, stage):
                return {"status": "waiting_for_night_window", "schedule": night_schedule(runner.config)}
            if runner.settings.get("preprocessing_only") and stage not in {"retention", "vision"}:
                return {"status": "paused_by_user", "message": "当前仅执行预处理，云端理解与STT暂停"}
            from .device_day_provider_gate import ProviderGate
            gate = ProviderGate(runner.config)
            if gate.blocks(stage):
                block = gate.state(stage)
                return {"status": "waiting_for_provider", "provider": runner.config.get("mllm", {}).get("provider"),
                        "reason": block.get("reason"), "message": block.get("message")}
            retry = bool(runner.settings.get("failure_retry_limit"))
            if stage == "retention" and runner.archive_root.is_dir():
                storage = self._storage_available(runner, inventory)
                if not storage["ready"]:
                    return {"status": "waiting_for_storage", "storage": storage}
            if stage != "vision" or (runner.config.get("runtime", {}).get("resource_limits", {}).get("vision")):
                # Auxiliary files are reconciled by the independent publication
                # worker. A busy audio sidecar must not delay the RGB receipt
                # notification or occupy a retention slot after RGB is sealed.
                return runner.run_once(inventory, stage=stage, retry=retry, max_jobs=1, stop_event=self.stop_event)
            if self._acquire_vision():
                try:
                    return runner.run_once(inventory, stage=stage, retry=retry, max_jobs=1, stop_event=self.stop_event)
                finally:
                    self._release_vision()
            return {"status": "stopped"}
        settings = self.settings_factory().get("device_day") or {}
        capacities = {"retention": settings.get("retention_workers", 2), "vision": settings.get("vision_workers", 2),
                      "stt": settings.get("stt_workers", 1),
                      "understanding": settings.get("understanding_workers", 2), "report": 1}
        with ExitStack() as executor_stack:
            workers = executor_stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="device-publication"))
            slot_workers = {}
            probe_worker = executor_stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="provider-health"))
            timeline_worker = executor_stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="day-timeline"))
            recovery_worker = executor_stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="retention-recovery"))
            overview_worker = executor_stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="archive-overview"))
            from .device_day_overview import ArchiveOverview
            overview = ArchiveOverview()
            probe_job = None
            while not self.stop_event.is_set():
                try:
                    settings = self.settings_factory()
                    if not (settings.get("device_day") or {}).get("enabled"):
                        break
                    changed_stages = set()
                    if overview_job is not None and overview_job.done():
                        try:
                            results['overview'] = overview_job.result()
                        except Exception as exc:
                            results['overview'] = {'status': 'failed', 'error_type': type(exc).__name__}
                        overview_job = None
                    if recovery_job is not None and recovery_job.done():
                        try:
                            results['retention_recovery'] = recovery_job.result()
                            if results['retention_recovery'].get('status') == 'restored_input_queued':
                                next_poll.clear()
                                self._runner._prepared.pop('retention', None)
                            if results['retention_recovery'].get('status') == 'completed':
                                changed_stages.add('retention')
                                for child in ('vision', 'stt', 'understanding', 'report'):
                                    self._runner._prerequisite_generation[child] += 1
                        except Exception as exc:
                            results['retention_recovery'] = {'status': 'failed', 'error_type': type(exc).__name__}
                        recovery_job = None
                    for slot, job in list(jobs.items()):
                        stage = slot[0]
                        if job.done():
                            del jobs[slot]
                            try:
                                results[stage] = job.result()
                            except Exception as exc:
                                results[stage] = {"status": "failed", "error_type": type(exc).__name__}
                            changed = bool(results[stage].get("results"))
                            next_poll[slot] = time.monotonic() + (0 if changed else 30)
                            if changed:
                                changed_stages.add(stage)
                                if stage in {'vision', 'understanding', 'stt'}:
                                    timeline_pending.update(item['archive'][:10] for item in results[stage].get('results', [])
                                                            if item.get('archive'))
                    from .device_day_contract import DEPENDENCIES
                    for child, parents in DEPENDENCIES.items():
                        if changed_stages.intersection(parents):
                            for ordinal in range(capacities[child]):
                                next_poll[(child, ordinal)] = 0
                    settings_key = json.dumps(settings, sort_keys=True, default=str)
                    if self._runner is None or self._settings_key != settings_key:
                        if jobs or recovery_job is not None or overview_job is not None:
                            self.wakeup.wait(1)
                            self.wakeup.clear()
                            continue
                        self._runner = DeviceDayRunner(settings)
                        self._settings_key = settings_key
                        overview = ArchiveOverview()
                    from .observed_inventory import read_inventory, revision
                    modified = revision(self._runner.runtime_root)
                    if modified != observed_mtime:
                        publication_dirty = True
                        # A new closed slice must wake idle slots immediately,
                        # not inherit the no-work backoff from a prior scan.
                        next_poll.clear()
                        observed_inventory = read_inventory(self._runner.runtime_root)
                        observed_mtime = modified
                        next_poll.clear()
                    from .device_day_provider_gate import ProviderGate
                    gate = ProviderGate(self._runner.config)
                    from .device_day_night_schedule import stage_admitted
                    if stage_admitted(self._runner.config, "understanding") and (probe_job is None or probe_job.done()):
                        if gate.blocks("understanding") and time.time() >= gate.state().get("next_probe_at", 0):
                            probe_job = probe_worker.submit(gate.probe_if_due)
                    inventory = observed_inventory
                    records = {r["recording_id"]: r for r in inventory["recordings"]}
                    for request_path in sorted((self._runner.runtime_root / "requests").glob("*.json")):
                        request = read_json(request_path)
                        if request.get("status") not in {"queued", "running"}:
                            continue
                        if request["status"] == "running" and self._request_complete(request):
                            atomic_json(request_path, request | {"status": "completed"})
                            continue
                        for record in request["recordings"]:
                            records.setdefault(record["recording_id"], record | {
                                "processing_priority": 1 if request.get("kind") == "historical_backfill" else 0})
                        if request["status"] == "queued":
                            atomic_json(request_path, request | {"status": "running"})
                            next_poll.clear()
                    from .device_day_schedule import priority_date, scheduling_record
                    focus_date = priority_date(self._runner.runtime_root)
                    records = {key: scheduling_record(row, focus_date=focus_date) for key, row in records.items()}
                    inventory = inventory | {"recordings": list(records.values())}
                    if recovery_job is None and time.monotonic() - last_recovery >= 1:
                        recovery_job = recovery_worker.submit(recovery.tick, self._runner)
                        last_recovery = time.monotonic()
                    if overview_job is None and time.monotonic() - last_overview >= 30:
                        overview_job = overview_worker.submit(overview.publish, self._runner)
                        last_overview = time.monotonic()
                    if self._runner.settings.get("camera_lanes"):
                        camera_count = max(1, len({r.get("camera_key") for r in records.values()
                                                  if r.get("configured_role") in {"first_person", "third_person"}}))
                        from .device_day import stage_worker_capacity
                        capacities.update({stage: stage_worker_capacity(self._runner.settings, stage, camera_count)
                                           for stage in STAGES})
                    if records or any(q.pending() for q in self._runner.queues.values()):
                        for stage in STAGES:
                            if not stage_admitted(self._runner.config, stage):
                                continue
                            for ordinal in range(capacities[stage]):
                                slot = (stage, ordinal)
                                if slot not in jobs and time.monotonic() >= next_poll.get(slot, 0):
                                    if slot not in slot_workers:
                                        slot_workers[slot] = executor_stack.enter_context(ThreadPoolExecutor(
                                            max_workers=1, thread_name_prefix=f"device-{stage}-{ordinal}"))
                                    jobs[slot] = slot_workers[slot].submit(run_stage, self._runner, inventory, stage)
                    from .timeline_invalidation import TimelineInvalidations
                    invalidations = TimelineInvalidations(self._runner.runtime_root)
                    if timeline_job is not None and timeline_job.done():
                        try:
                            results['timeline'] = timeline_job.result()
                            if timeline_generation and not results['timeline'].get('publication_pending'):
                                invalidations.complete(timeline_generation['day'], timeline_generation['token'])
                        except Exception as exc:
                            results['timeline'] = {'status': 'failed', 'error_type': type(exc).__name__}
                        timeline_job = None
                    if timeline_job is None and time.monotonic() - last_timeline >= 30 and records:
                        from .device_day_timeline import refresh_timeline
                        pending_days = invalidations.pending()
                        # Historical local caches get a one-time bootstrap. No
                        # new capture means no recurring all-date recomputation.
                        days = sorted({archive_name(r['camera_key'], r['recording_start_us'])[:10]
                                       for r in records.values() if r.get('recording_start_us')})
                        missing = [d for d in days if not (self._runner.runtime_root / 'DayTimeline' / f'{d}.json').is_file()]
                        day = pending_days[0]['day'] if pending_days else (sorted(timeline_pending)[0] if timeline_pending else (missing[0] if missing else None))
                        if day:
                            timeline_pending.discard(day)
                            timeline_generation = next((item for item in pending_days if item['day'] == day), None)
                            timeline_job = timeline_worker.submit(refresh_timeline, self._runner.config, day)
                            last_timeline = time.monotonic()
                    publication_dirty = publication_dirty or bool(changed_stages)
                    storage_wait = results.get("retention", {}).get("status") == "waiting_for_storage"
                    self.last_result = {"schema_version": VERSION, "status": "waiting_for_storage" if storage_wait else "running" if jobs else "waiting_for_nas_monitor",
                                        "stages": results, "parallel_capacity": dict(capacities),
                                        "queue": self._runner.queue_snapshot(),
                                        "capture_deletion": "disabled_by_user"}
                    atomic_json(self._runner.runtime_root / "service.json", self.last_result)
                    if publication_job is not None and publication_job.done():
                        try:
                            results["publication"] = publication_job.result()
                            publication_dirty = publication_dirty or results["publication"].get('pending_publications', False)
                        except Exception as exc:
                            results["publication"] = {"status": "failed", "error_type": type(exc).__name__}
                            publication_dirty = True
                        publication_job = None
                    if publication_job is None and publication_dirty and time.monotonic() - last_publication >= 30:
                        publication_job = workers.submit(publish, self._runner)
                        last_publication = time.monotonic()
                        publication_dirty = False
                except Exception as exc:
                    self.last_result = {"schema_version": VERSION, "status": "failed", "error_type": type(exc).__name__}
                self.wakeup.wait(1)
                self.wakeup.clear()


def install_routes(app, settings_factory, service):
    from .device_day_timeline import install_routes as install_timeline
    install_timeline(app, settings_factory)
    @app.get('/archive-overview', response_class=HTMLResponse)
    def overview_document():
        path = Path(settings_factory()['storage']['local_runtime_root']) / 'device-day' / 'ArchiveOverview.html'
        if not path.is_file():
            return HTMLResponse('<h1>实验数据总览正在生成</h1><p>请稍后刷新，预处理继续运行。</p>', status_code=503,
                                headers={'Retry-After': '30'})
        return FileResponse(path, media_type='text/html', headers={'Cache-Control': 'no-store'})

    def archive(name):
        try:
            root = safe_child(Path(settings_factory()["storage"]["archive_root"]), validate_archive_name(name))
            if not (root / DIRECTORIES[1] / "Index.json").is_file():
                raise ValueError("Unknown device/day archive")
            return root
        except (ValueError, OSError) as exc:
            from .artifact_reader import storage_status
            code, message = storage_status(exc)
            raise HTTPException(code, message) from exc

    @app.get("/api/device-days")
    def listing():
        root = Path(settings_factory()["storage"]["archive_root"])
        items = []
        if root.is_dir():
            for child in sorted(root.iterdir(), reverse=True):
                try:
                    validate_archive_name(child.name)
                    if child.is_symlink():
                        continue
                    index = child / DIRECTORIES[1] / "Index.json"
                    if index.is_file():
                        data = published_recording_status(read_json(index))
                        items.append({"archive": child.name, "segment_count": len(data.get("segments", [])),
                                      "transcript_count": sum(len((r.get("transcription") or {}).get("comments") or []) for r in data.get("recordings", [])),
                                      "updated_at": data.get("updated_at"), "recordings": [
                                          {k: r.get(k) for k in ("recording_id", "start_us", "end_us", "capture_complete", "stages", "current_results_ready", "preprocessing_ready")}
                                          for r in data.get("recordings", [])],
                                      "report_url": f"/api/device-days/{quote(child.name)}/files/{quote(DIRECTORIES[3])}/LaboratoryDailyReport.html"})
                except (ValueError, OSError):
                    continue
        queue = service._runner.queue_snapshot() if service._runner else {}
        observed = Path(settings_factory()["storage"]["local_runtime_root"]) / "device-day" / "observed-inventory.json"
        from .observed_inventory import read_inventory
        errors = read_inventory(observed.parent).get("errors", [])
        return {"schema_version": VERSION, "archives": items, "service": service.last_result, "queue": queue,
                "discovery_errors": errors,
                "capture_deletion": "disabled_by_user"}

    @app.get("/api/device-days/{name}/index")
    def index(name: str):
        return published_recording_status(read_json(archive(name) / DIRECTORIES[1] / "Index.json"))

    @app.get("/api/device-days/{name}/files/{relative:path}")
    def file(name: str, relative: str, request: Request):
        root = archive(name)
        try:
            from .artifact_reader import resolve
            path = resolve(root, relative, directories=DIRECTORIES,
                           suffixes={".json", ".jsonl", ".html", ".jpg", ".png", ".mp4", ".opus", ".wav", ".m4a", ".mp3", ".flac", ".ogg", ".csv", ".txt", ".srt", ".vtt", ".log"})
        except (ValueError, OSError) as exc:
            from .artifact_reader import storage_status
            code, message = storage_status(exc)
            raise HTTPException(code, message) from exc
        # Top-level MP4 documents can stay black in embedded Chromium even
        # while decoding. Media/range/API requests must still get exact bytes.
        if (path.suffix.lower() == '.mp4' and request.headers.get('sec-fetch-dest') == 'document'
                and request.headers.get('sec-fetch-mode') == 'navigate'):
            return RedirectResponse(f'/device-days/{quote(name, safe="")}/watch/{quote(relative, safe="/")}',
                                    headers={'Cache-Control': 'no-store'})
        from .web_player import MediaFileResponse
        response = MediaFileResponse if path.suffix.lower() == '.mp4' else FileResponse
        return response(path, headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})

    @app.get('/device-days/{name}/watch/{relative:path}', response_class=HTMLResponse)
    def watch(name: str, relative: str):
        from .web_player import render_player
        root = archive(name)
        try:
            from .artifact_reader import resolve
            resolve(root, relative, directories=DIRECTORIES, suffixes={'.mp4'})
        except (ValueError, OSError) as exc:
            from .artifact_reader import storage_status
            code, message = storage_status(exc)
            raise HTTPException(code, message) from exc
        return HTMLResponse(render_player(name, relative), headers={'Cache-Control': 'no-store'})

    @app.post("/api/device-days/{name}/comments", status_code=201)
    def comment(name: str, payload: dict):
        archive(name)
        settings = settings_factory()
        try:
            result = append_comment(Path(settings["storage"]["archive_root"]), name, payload,
                                    Path(settings["storage"]["local_runtime_root"]))
        except (ValueError, OSError) as exc:
            raise HTTPException(400, "comment内容或时间范围无效") from exc
        service.wakeup.set()
        return result

    @app.put("/api/device-days/{name}/protocol")
    def protocol(name: str, payload: dict):
        root = archive(name)
        if not isinstance(payload.get("text"), str) or not payload["text"].strip() or len(payload["text"]) > 100000:
            raise HTTPException(400, "protocol需要文本内容，最多100000字符")
        from .device_day import exclusive
        from .device_day_contract import digest
        target = root / DIRECTORIES[4] / "Protocol.json"
        settings = settings_factory()
        with exclusive(Path(settings["storage"]["local_runtime_root"]) / "device-day" / "locks" / f"{name}.comments.lock"):
            if target.is_file():
                old = read_json(target)
                atomic_json(root / DIRECTORIES[4] / "History" / f"Protocol-{digest(old)}.json", old)
            atomic_json(target, {"schema_version": VERSION, "text": payload["text"],
                                 "version": str(payload.get("version") or "未标明"), "source": "user_supplied_protocol"})
        service.wakeup.set()
        return {"status": "saved"}

    @app.get("/device-days", response_class=HTMLResponse)
    def browser():
        # Preserve bookmarks while bringing archive browsing into the app shell.
        return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta http-equiv="refresh" content="0;url=/#/device-days">'
                '<title>VisionCortex 设备日归档</title>'
                '<a href="/#/device-days">进入应用内设备日归档</a></html>')

    @app.get("/device-days/{name}", response_class=HTMLResponse)
    def archive_browser(name: str):
        from .device_day_browser import render_archive
        root = archive(name)
        return render_archive(root, read_json(root / DIRECTORIES[1] / "Index.json"))

    @app.get("/device-days/{name}/stt", response_class=HTMLResponse)
    def stt_browser(name: str):
        data = read_json(archive(name) / DIRECTORIES[1] / "Index.json")
        from .device_day_reports import clock
        sections = []
        def href(path):
            return f'/api/device-days/{quote(name)}/files/{quote(path, safe="/")}'
        for record in data.get("recordings", []):
            transcript = record.get("transcription") or {}
            comments = transcript.get("comments") or []
            if not comments:
                continue
            source = comments[0].get("audio_ref")
            player = f'<audio controls preload="none" src="{href(source["path"])}"></audio>' if source else ''
            paragraphs = ''.join(f'<p><strong>{clock(c["start_us"])}</strong> {html.escape(c["text"])} '
                                 f'<a href="{href(c["transcript_path"])}">识别来源</a></p>' for c in comments)
            sections.append(f'<section>{player}{paragraphs}</section>')
        return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
               '<title>录音识别</title><style>body{font:16px/1.8 system-ui;max-width:1000px;margin:40px auto;padding:20px}section{border-top:1px solid #ccc;padding-top:20px}audio{width:100%}</style>' \
               f'<a href="/device-days">返回设备日归档</a><h1>{html.escape(name)} 录音识别</h1><p>STT自动识别，未经人工复核；不能仅据录音确认实验动作。时间对齐依据保留在识别来源中。</p>' \
               + (''.join(sections) or '<p>尚无识别文本；可在总索引中查看无录音、排队或失败状态。</p>') + '</html>'
