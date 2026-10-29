"""Generated pixels through the real LiveScanner hook; no field acceptance."""

from dataclasses import replace
from datetime import datetime
import hashlib
import json
import subprocess
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest
import qrcode

from field_recognition.capture.capture_actions import qr_matches
from field_recognition.capture.live_scan import LiveScanner
from field_recognition.capture.qr_decode import decode_qr
from visioncortex.lab.domain import Conflict
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.processing import Worker
from visioncortex.lab.scan_source import (
    ENV,
    SCHEMA,
    ScanSourcePublisher,
    load_publisher_from_environment,
)
from visioncortex.lab.scan_source_scanner import attach_to_runtime, attach_to_scanner
from visioncortex.lab.timeline_runtime import TimelineRuntime

from .test_operator_scan import scene, rows

__all__ = ["scene"]


def configuration(root, cameras=("fixture-camera-0",)):
    incoming = root / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    return dict(
        schema_version=SCHEMA,
        enabled=True,
        runtime_root=str(root),
        incoming_root=str(incoming),
        max_bytes=32 * 1024**2,
        max_frame_bytes=4 * 1024**2,
        max_events=20,
        retention_seconds=86400,
        source_kind="fixture",
        cameras=[
            dict(camera_id=c, source_instance="manual:" + c, scan_kind="equipment")
            for c in cameras
        ],
    )


def pixels(values):
    parts = [np.asarray(qrcode.make(value).convert("RGB")) for value in values]
    rgb = np.full(
        (480, max(640, sum(p.shape[1] for p in parts) + 40), 3), 255, dtype=np.uint8
    )
    x = 20
    for part in parts:
        rgb[20 : 20 + part.shape[0], x : x + part.shape[1]] = part
        x += part.shape[1]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def metadata(index=1, *, camera="fixture-camera-0", clock=True):
    epoch = int(
        datetime.fromisoformat("2026-10-29T09:00:00+08:00").timestamp() * 1000000
    )
    return dict(
        source_camera_key=camera,
        camera_id="cam02",
        sender_id="fixture-sender",
        decoded_frame_id=index,
        sequence=index,
        timestamp_us=index * 1000000,
        stream="main",
        global_timestamp_us=epoch + index * 1000000,
        clock_sync_valid=clock,
        clock_uncertainty_us=0,
        frame_age_ms=0,
    )


def session(camera="fixture-camera-0", *, owner="browser"):
    return dict(
        session_id=str(uuid4()),
        owner=owner,
        camera_id=camera,
        operator="fixture operator",
        status="scanning",
        frames_scanned=0,
        consecutive_decode_errors=0,
        message="scanning",
        after_frame_id=0,
    )


def receipts(root):
    return [
        json.loads(p.read_text()) for p in sorted(root.rglob("CaptureReceipt.json"))
    ]


def scanner(publisher, frames, *, camera="fixture-camera-0", owner="browser"):
    # A deterministic finite camera port runs the actual native decoder and
    # LiveScanner._run loop. Only the camera transport is a fixture.
    iterator = iter(frames)
    core = dict(
        now=lambda: "2026-10-29T09:30:00+08:00",
        qr_matches=lambda values, points: qr_matches({}, values, points),
    )
    live = LiveScanner(core)

    def frame():
        try:
            return next(iterator)
        except StopIteration:
            live.shutdown.set()
            raise ValueError("finite fixture camera ended") from None

    core["receiver_camera"] = SimpleNamespace(
        target=camera,
        frame=frame,
        snapshot=lambda: dict(
            service_status_available=True, service_status=dict(online=True)
        ),
    )
    if publisher is not None:
        attach_to_scanner(live, publisher)
    live.session = session(camera, owner=owner)
    live.last_seen = time.monotonic()
    live._run(live.session["session_id"])
    assert live.session["status"] != "failed", live.last_error
    return live


def test_no_configuration_or_automation_has_no_drop_or_callback(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert load_publisher_from_environment() is None
    untouched = tmp_path / "untouched"
    scanner(None, [])
    publisher = ScanSourcePublisher(configuration(tmp_path / "runtime"))
    scanner(publisher, [(pixels([str(uuid4())]), metadata())], owner="automation")
    assert not receipts(publisher.root)
    assert not (publisher.runtime / "scan-source-state").exists()
    assert not untouched.exists()


def test_factory_installs_optional_callback_without_starting_it(tmp_path, monkeypatch):
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(configuration(tmp_path / "runtime")))
    monkeypatch.setenv(ENV, str(manifest))
    core = dict(
        receiver_camera=SimpleNamespace(target="fixture-camera-0"),
        qr_matches=lambda *a: {},
    )
    live = LiveScanner(core)
    runtime = dict(live_scanner=live)
    assert attach_to_runtime(runtime) is runtime
    assert live._scan_source_attached
    assert not (tmp_path / "runtime/scan-source-state").exists()
    monkeypatch.delenv(ENV)
    other = LiveScanner(core)
    attach_to_runtime(dict(live_scanner=other))
    assert other.core is core and not getattr(other, "_scan_source_attached", False)


def test_preview_thread_cannot_replace_scanned_frame_or_publish_after_stop(tmp_path):
    publisher = ScanSourcePublisher(configuration(tmp_path / "runtime"))
    first, later = pixels([str(uuid4())]), pixels([str(uuid4())])
    local = threading.local()

    def frame():
        return (
            (later, metadata(2))
            if getattr(local, "preview", False)
            else (first, metadata(1))
        )

    live = LiveScanner(
        dict(
            receiver_camera=SimpleNamespace(target="fixture-camera-0", frame=frame),
            qr_matches=lambda values, points: qr_matches({}, values, points),
        )
    )
    attach_to_scanner(live, publisher)
    live.session, live.last_seen = session(), time.monotonic()
    image, _ = live.camera().frame()
    decoded = decode_qr(image)

    def preview():
        local.preview = True
        live.camera().frame()

    thread = threading.Thread(target=preview)
    thread.start()
    thread.join()
    live.core["qr_matches"](*decoded[:2])
    saved = receipts(publisher.root)
    assert (
        len(saved) == 1
        and saved[0]["scan_source"]["frame_metadata"]["decoded_frame_id"] == 1
    )
    assert saved[0]["scan_source"]["decoded_values"] == decoded[0]
    live.camera().frame()
    live.session["status"] = "stopped"
    live.core["qr_matches"](*decoded[:2])
    assert len(receipts(publisher.root)) == 1


def test_real_live_loop_keeps_bare_uuid_before_old_filter_and_A_B_A(tmp_path):
    publisher = ScanSourcePublisher(configuration(tmp_path / "runtime"))
    a, b = str(uuid4()), str(uuid4())
    live = scanner(
        publisher, [(pixels([v]), metadata(i)) for i, v in enumerate([a, a, b, a], 1)]
    )
    saved = receipts(publisher.root)
    assert len(saved) == 3
    assert sorted(r["scan_source"]["decoded_values"][0] for r in saved) == sorted(
        [a, b, a]
    )
    assert live.session["frames_scanned"] == 4
    assert "scan" not in live.session  # Legacy JSON matcher rejected every bare UUID.
    for r in saved:
        assert r["scan_intent"]["intent"] == "operator_scan"
        assert r["scan_source"]["source_epoch_status"] == "unknown"
        assert (
            r["scan_source"]["person_id"] is None
            and "person_id" not in r["scan_intent"]
        )


def test_empty_and_multiple_raw_qr_kept_once_without_fabricated_identity(tmp_path):
    publisher = ScanSourcePublisher(configuration(tmp_path / "runtime"))
    values = [str(uuid4()), str(uuid4())]
    scanner(
        publisher,
        [
            (pixels([]), metadata(1)),
            (pixels([]), metadata(2)),
            (pixels(values), metadata(3)),
        ],
    )
    saved = receipts(publisher.root)
    assert len(saved) == 2
    assert sorted(len(r["scan_source"]["decoded_values"]) for r in saved) == [0, 2]
    assert all(
        r["scan_source"]["decoder_result_is_authoritative"] is False for r in saved
    )
    assert all("equipment_id" not in r for r in saved)


def test_same_frame_retry_restarts_idempotently_and_mutation_fails(tmp_path):
    document = configuration(tmp_path / "runtime")
    publisher = ScanSourcePublisher(document)
    image = pixels([str(uuid4())])
    meta, active = metadata(), session()
    decoded = decode_qr(image)
    first = publisher.publish(image, meta, active, decoded)
    restarted = ScanSourcePublisher(document)
    assert restarted.publish(image, meta, active, decoded)["status"] == "duplicate"
    changed = image.copy()
    changed[-2, -2] = [0, 50, 200]
    with pytest.raises(Conflict, match="physical scan frame"):
        restarted.publish(changed, meta, active, decoded)
    assert len(receipts(publisher.root)) == 1
    assert first["capture_id"] == receipts(publisher.root)[0]["capture_id"]


def test_two_cameras_same_session_and_frame_ids_are_isolated(tmp_path):
    publisher = ScanSourcePublisher(
        configuration(tmp_path / "runtime", ("fixture-camera-0", "fixture-camera-1"))
    )
    image = pixels([str(uuid4())])
    active = session()
    results = []
    for camera in ("fixture-camera-0", "fixture-camera-1"):
        results.append(
            publisher.publish(
                image,
                metadata(camera=camera),
                dict(active, camera_id=camera),
                decode_qr(image),
            )
        )
    assert results[0]["capture_id"] != results[1]["capture_id"]
    assert len(receipts(publisher.root)) == 2
    assert (
        publisher.publish(
            image,
            metadata(camera="other"),
            dict(active, camera_id="other"),
            decode_qr(image),
        )["status"]
        == "camera_not_allowed"
    )


def test_unknown_clock_preserves_original_fields_and_never_uses_receive_time(tmp_path):
    publisher = ScanSourcePublisher(configuration(tmp_path / "runtime"))
    image = pixels([str(uuid4())])
    meta = metadata()
    del meta["clock_uncertainty_us"]
    publisher.publish(image, meta, session(), decode_qr(image))
    receipt = receipts(publisher.root)[0]
    assert receipt["captured_at"] is None
    assert receipt["time_basis"]["clock_sync_valid"] is False
    assert (
        receipt["time_basis"]["original_frame_clock"]["global_timestamp_us"]
        == meta["global_timestamp_us"]
    )


def test_unknown_clock_storage_partition_is_pinned_across_midnight_retry(
    tmp_path, monkeypatch
):
    from visioncortex.lab import scan_source

    class Clock(datetime):
        value = "2026-10-29T15:59:59+00:00"

        @classmethod
        def now(cls, zone=None):
            return datetime.fromisoformat(cls.value)

    monkeypatch.setattr(scan_source, "datetime", Clock)
    publisher = ScanSourcePublisher(configuration(tmp_path / "runtime"))
    image, active, meta = pixels([str(uuid4())]), session(), metadata(clock=False)
    first = publisher.publish(image, meta, active, decode_qr(image))
    before = receipts(publisher.root)[0]
    assert first["relative_path"].startswith(
        "2026-10-29_fixture-camera-0/MetaData/VideoEvidence/"
    )
    assert before["scan_source"]["archive_day_basis"] == "receipt_day_not_event_time"
    assert before["captured_at"] is None
    Clock.value = "2026-10-29T16:00:01+00:00"
    second = ScanSourcePublisher(publisher.config).publish(
        image, meta, active, decode_qr(image)
    )
    assert (
        second["status"] == "duplicate"
        and second["relative_path"] == first["relative_path"]
    )
    assert receipts(publisher.root) == [before]


def test_source_failure_leaves_no_committed_receipt_and_identical_retry_resumes(
    tmp_path, monkeypatch
):
    publisher = ScanSourcePublisher(configuration(tmp_path / "runtime"))
    image, meta, active = pixels([str(uuid4())]), metadata(), session()
    original = publisher._immutable

    def interrupted(relative, raw, guard):
        if relative.endswith("CaptureReceipt.json"):
            raise OSError("fixture interrupted before commit marker")
        return original(relative, raw, guard)

    monkeypatch.setattr(publisher, "_immutable", interrupted)
    with pytest.raises(OSError):
        publisher.publish(image, meta, active, decode_qr(image))
    assert not receipts(publisher.root)
    monkeypatch.setattr(publisher, "_immutable", original)
    assert (
        publisher.publish(image, meta, active, decode_qr(image))["status"] == "resumed"
    )
    assert len(receipts(publisher.root)) == 1


def test_capacity_missing_frame_and_storage_failure_fail_closed(tmp_path, monkeypatch):
    config = configuration(tmp_path / "runtime")
    config["max_events"] = 1
    publisher = ScanSourcePublisher(config)
    image, active = pixels([str(uuid4())]), session()
    decoded = decode_qr(image)
    bad = metadata()
    del bad["decoded_frame_id"]
    with pytest.raises(ValueError, match="stable native frame"):
        publisher.publish(image, bad, active, decoded)
    from visioncortex.lab import scan_source

    original = scan_source.verify_root
    monkeypatch.setattr(
        scan_source,
        "verify_root",
        lambda root: (_ for _ in ()).throw(OSError("fixture changed mount")),
    )
    with pytest.raises(OSError, match="changed mount"):
        publisher.publish(image, metadata(), active, decoded)
    assert not receipts(publisher.root)
    monkeypatch.setattr(scan_source, "verify_root", original)
    publisher.publish(image, metadata(), active, decoded)
    other = pixels([str(uuid4())])
    with pytest.raises(Conflict, match="capacity"):
        publisher.publish(other, metadata(2), active, decode_qr(other))


def lossless_codec(path, originals):
    height, width = originals[0].shape[:2]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            "1",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            "-threads",
            "1",
            str(path),
        ],
        input=b"".join(p.tobytes() for p in originals),
        check=True,
        timeout=10,
    )
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        timeout=10,
    )
    decoded = list(np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3))
    assert len(decoded) == len(originals)
    assert all(np.array_equal(a, b) for a, b in zip(originals, decoded, strict=True))
    return decoded


def test_native_codec_live_hook_discovery_qr_worker_scan_context_report(
    scene, tmp_path
):
    s = scene
    settings = replace(s["runtime"].settings, processing_profile="timeline-v1")
    publisher = ScanSourcePublisher(configuration(settings.runtime_root))
    originals = [pixels([s["devices"][i]]) for i in (0, 1, 0)]
    (tmp_path / "GeneratedInput.json").write_text(
        json.dumps(
            dict(
                fixture_only=True,
                expected_source_uuid_sequence=[s["devices"][i] for i in (0, 1, 0)],
                original_rgb_sha256=[
                    hashlib.sha256(
                        cv2.cvtColor(p, cv2.COLOR_BGR2RGB).tobytes()
                    ).hexdigest()
                    for p in originals
                ],
                codec_pts_seconds=[0, 1, 2],
                synthetic_receiver_timestamp_offset_seconds=1,
                expected_values_written_before_encoding_or_decoding=True,
            ),
            indent=2,
        )
        + "\n"
    )
    decoded = lossless_codec(tmp_path / "generated-native-qr.mkv", originals)
    scanner(publisher, [(bgr, metadata(i)) for i, bgr in enumerate(decoded, 1)])
    ingest = Ingest(s["core"], settings)
    assert ingest.discover_once()["processed"] == 3
    assert ingest.discover_once()["skipped"] == 3
    jobs = [Worker(s["core"], settings).run_once() for _ in range(3)]
    assert all(j["status"] == "completed" for j in jobs)
    runtime = TimelineRuntime(s["core"], settings)
    outcomes = runtime.scan_intake.tick()["outcomes"]
    assert [o["status"] for o in outcomes] == ["declared_equipment"] * 3
    assert runtime.scan_intake.tick()["consumed"] == 0
    current = runtime.operator_scans.context(
        as_of="2026-10-29T09:30:00+08:00", camera_id=s["cameras"][0]
    )
    assert current["equipment_id"] == s["devices"][0]
    assert len(current["switches"]) == 3 and not current["experiment_end_implied"]
    assert current["person_id"] is None
    snapshot = runtime.views.snapshot("2026-10-29")
    report = snapshot["document"]
    activities = report["unassigned_activities"]
    assert len([a for a in activities if "operator_scan_event" in a["payload"]]) == 3
    assert all(a["evidence_ids"] for a in activities)
    assert rows(s, "quantity_ledger_entry") == [] and rows(s, "fact_revision") == []
    with s["core"].db.transaction(readonly=True) as conn:
        inbox = conn.execute(
            "SELECT payload FROM inbox WHERE customer_id=%s", (s["core"].customer,)
        ).fetchall()
    assert len(inbox) == 3
    assert all(
        r["payload"]["scan_source"]["original_rgb_sha256"]
        in {
            hashlib.sha256(cv2.cvtColor(p, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
            for p in originals
        }
        for r in inbox
    )
    (tmp_path / "ChainResult.json").write_text(
        json.dumps(
            dict(
                fixture_only=True,
                customer_id=str(s["core"].customer),
                outcomes=outcomes,
                operator_context=current,
                snapshot=snapshot,
                inbox=[r["payload"] for r in inbox],
                fact_revision_count=0,
                ledger_entry_count=0,
                decoder_patched=False,
            ),
            indent=2,
        )
        + "\n"
    )


def test_live_hook_empty_multiple_and_unknown_clock_reach_pending_not_context(scene):
    s = scene
    settings = replace(s["runtime"].settings, processing_profile="timeline-v1")
    publisher = ScanSourcePublisher(configuration(settings.runtime_root))
    frames = [
        (pixels([]), metadata(1)),
        (pixels(s["devices"]), metadata(2)),
        (pixels([s["devices"][0]]), metadata(3, clock=False)),
    ]
    scanner(publisher, frames)
    assert Ingest(s["core"], settings).discover_once()["processed"] == 3
    assert all(
        Worker(s["core"], settings).run_once()["status"] == "completed" for _ in frames
    )
    runtime = TimelineRuntime(s["core"], settings)
    outcomes = runtime.scan_intake.tick()["outcomes"]
    assert {o["status"] for o in outcomes} == {
        "no_qr_decoded",
        "ambiguous_multiple_qr",
        "unknown_time",
    }
    current = runtime.operator_scans.context(
        as_of="2026-10-29T09:30:00+08:00", camera_id=s["cameras"][0]
    )
    assert current["equipment_id"] is None
    assert rows(s, "quantity_ledger_entry") == [] and rows(s, "fact_revision") == []
