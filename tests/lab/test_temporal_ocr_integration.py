"""Real codecs/PTS, localhost OCR and isolated Unix-socket PostgreSQL; no real models."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import subprocess

import cv2
import numpy as np
import pytest

from visioncortex.lab.domain import Conflict, fingerprint
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.temporal_video import (
    TemporalVideo,
    TemporalVideoRejected,
    decode_frames,
    validate_registration,
    resolve_dynamic_context,
    jsonb_stable,
)
from visioncortex.lab.timeline_runtime import TimelineRuntime
from psycopg.types.json import Jsonb
from visioncortex.lab.timeline_report import render_report
from ._fixtures import (
    db as evidence_db,
    settings as evidence_settings,
)
from .test_timeline import at, hj as timeline_fixture, DAY
from .test_timeline_runner import fake_service, configured


@pytest.fixture(scope="session", name="db")
def isolated_db():
    return evidence_db.__wrapped__()


@pytest.fixture
def settings(core, tmp_path):
    return evidence_settings.__wrapped__(core, tmp_path)


@pytest.fixture(name="hj")
def context(core, settings):
    return timeline_fixture.__wrapped__(core, settings)


from ._support import decoder_python

DECODER = decoder_python()


def provider(**extra):
    return dict(
        schema_version="vc-lab-temporal-video-provider/1",
        enabled=True,
        revision="fixture-decoder/1",
        decoder_python=DECODER,
        max_input_frames=600,
        decoder_wait_seconds=30,
        **extra,
    )


def original(hj, name="native", inverted=True):
    # Actual VFR codec with nonzero container PTS. Incoming is the original; no
    # large-source media cache is used by this explicit native-original fixture.
    path = hj["settings"].incoming / "MetaData" / name / "rgb.mp4"
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(38)
    image = rng.integers(80, 240, (120, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (38, 25), (202, 100), (15, 15, 15), -1)
    cv2.putText(image, "26.2", (48, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 2)
    image[:12, :12] = [17, 39, 231]
    raw = np.ascontiguousarray(image[::-1, ::-1] if inverted else image).tobytes() * 20
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "240x120",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-vf",
            r"settb=1/1000,setpts=200+N*40+mod(N\,2)*10",
            "-vsync",
            "0",
            "-enc_time_base",
            "1/1000",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            "-video_track_timescale",
            "1000",
            str(path),
        ],
        input=raw,
        check=True,
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    result = Ingest(hj["core"], hj["settings"]).accept(
        dict(
            capture_id=name,
            source_instance="fixture-camera",
            source_revision="fixture/1",
            kind="video",
            session_id="fixture-session",
            relative_path=str(path.relative_to(hj["settings"].incoming)),
            captured_at=at("09:00:00"),
            time_basis={"basis": "fixture", "uncertainty_us": 0},
            start_seconds="0",
            end_seconds="0.8",
            source_kind="fixture",
            sha256=sha,
            native_record={"fixture_only": True},
        ),
        native_original=True,
    )
    return result, path, image, sha


def registration(hj, sha, intervals=None, **extra):
    return dict(
        schema_version="vc-lab-temporal-video-ocr/1",
        enabled=True,
        revision="fixture-fields/1",
        source_sha256=sha,
        media_version=1,
        source_stream_key="fixture-camera",
        source_session_key="fixture-session",
        rotation_degrees_clockwise=180,
        orientation_basis="synthetic inverted source fixture",
        time_basis="original_media_pts",
        world_clock_status="verified",
        source_intervals=intervals or [dict(start_seconds="0", end_seconds="0.8")],
        fields=[
            dict(
                screen_id="fixture-screen",
                field_key="temperature",
                profile_id="fixture-lcd/1",
                role="process_value",
                channel="PV",
                roi_box=["0", "0", "1", "1"],
                layout=dict(
                    revision="fixture-layout/1",
                    display_type="lcd",
                    dynamic_boxes=[[0.15, 0.2, 0.86, 0.85]],
                    registration_mode="rigid",
                ),
            )
        ],
        **extra,
    )


def test_real_vfr_decoder_preserves_nonzero_pts_and_variable_intervals(hj):
    _, path, upright, _ = original(hj)
    frames = list(decode_frames(path, "0", "0.8", provider()))
    assert len(frames) == 20
    stamps = [Fraction(meta["pts"]) * Fraction(meta["time_base"]) for _, meta in frames]
    assert stamps[0] > 0 and len(set(b - a for a, b in zip(stamps, stamps[1:]))) > 1
    assert np.array_equal(frames[0][0], upright[::-1, ::-1])
    assert frames[0][1]["stream_origin"] != "0"
    with pytest.raises(TemporalVideoRejected):
        list(decode_frames(path, "0", "0.8", dict(provider(), max_input_frames=1)))


def test_video_ocr_is_only_created_by_explicit_source_registration(hj):
    m, _, _, sha = original(hj)
    runtime = TimelineRuntime(hj["core"], hj["settings"])
    with hj["core"].db.transaction(readonly=True) as conn:
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM hj_timeline_task WHERE customer_id=%s AND kind='ocr'",
                (hj["core"].customer,),
            ).fetchone()["n"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT storage FROM media_location WHERE customer_id=%s AND media_id=%s",
                (hj["core"].customer, m["media_id"]),
            ).fetchone()["storage"]
            == "incoming"
        )
    doc = registration(hj, sha)
    saved = runtime.tasks.register_temporal_ocr(
        "explicit",
        m["media_id"],
        doc,
        actor="fixture",
        reason="synthetic temporal fixture",
    )
    assert saved == runtime.tasks.register_temporal_ocr(
        "explicit",
        m["media_id"],
        doc,
        actor="fixture",
        reason="synthetic temporal fixture",
    )
    claim = runtime.tasks.claim("claim", "ocr", "fixture/1")
    assert claim["request"]["input_mode"] == "temporal_display"
    assert claim["request"]["temporal_display"] == doc
    assert (
        len(claim["request"]["windows"]) == 1
        and len(claim["request"]["windows"][0]["input_spans"]) == 1
    )
    assert not list(hj["settings"].cache.rglob("*.mp4"))


@pytest.mark.parametrize(
    "change",
    [
        {"enabled": False},
        {"source_sha256": "0" * 64},
        {"source_stream_key": "another-camera"},
        {"source_session_key": "another-session"},
        {"rotation_degrees_clockwise": 90},
        {"time_basis": "wallclock_assumed"},
    ],
)
def test_registration_rejects_unpinned_source_or_implicit_input_policy(hj, change):
    m, _, _, sha = original(hj)
    doc = registration(hj, sha)
    doc.update(change)
    with pytest.raises((TemporalVideoRejected, Conflict)):
        TimelineRuntime(hj["core"], hj["settings"]).tasks.register_temporal_ocr(
            "bad", m["media_id"], doc, actor="fixture", reason="must reject"
        )


def test_actual_claim_decode_ocr_checkpoint_submit_trace_report(hj, monkeypatch):
    m, path, upright, sha = original(hj)
    with fake_service("ocr") as (state, model):
        model["temporal_display"] = provider()
        runtime = TimelineRuntime(hj["core"], configured(hj, model, "ocr", monkeypatch))
        runtime.tasks.register_temporal_ocr(
            "real-chain",
            m["media_id"],
            registration(hj, sha),
            actor="fixture",
            reason="exercise actual chain",
        )
        result = runtime.runner.run_once()
        assert result["status"] == "completed", result
        assert state["posts"] > 0
        import base64

        first = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(state["bodies"][0]["media_base64"]), dtype=np.uint8
            ),
            cv2.IMREAD_COLOR,
        )
        assert np.array_equal(first[:12, :12], upright[:12, :12])
    trace = runtime.views.trace(at("09:00:00"), at("09:00:02"))
    assert trace["activities"]
    item = trace["activities"][-1]
    field = item["payload"]["fields"][0]
    assert (
        item["source_type"] == "ocr"
        and item["evidence"][0]["media_id"] == m["media_id"]
    )
    assert (
        field["status"] in {"uncertain", "unreadable"}
        and field["value"] is None
        and field["value_text"] is None
    )
    assert field["auto_confirmed"] is False
    assert item["payload"]["temporal_display"]["source_refs"][0]["pts"] > 0
    assert item["payload"]["external_confirmation"] == "not_provided"
    report = runtime.views.snapshot(DAY)
    text = render_report(report["document"])
    assert "temperature" in text and ("uncertain" in text or "unreadable" in text)
    with hj["core"].db.transaction(readonly=True) as conn:
        task = conn.execute(
            "SELECT result FROM hj_timeline_task WHERE customer_id=%s AND id=%s",
            (hj["core"].customer, result["task_id"]),
        ).fetchone()["result"]
        assert task["execution_verification"] == "local_transport_and_payload_verified"
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM hj_timeline_window_result WHERE customer_id=%s",
                (hj["core"].customer,),
            ).fetchone()["n"]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM fact_revision WHERE customer_id=%s",
                (hj["core"].customer,),
            ).fetchone()["n"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM field_candidate WHERE customer_id=%s",
                (hj["core"].customer,),
            ).fetchone()["n"]
            == 0
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_checkpoint_retry_does_not_repeat_completed_video_window(hj, monkeypatch):
    m, _, _, sha = original(hj)
    intervals = [
        dict(start_seconds="0", end_seconds="0.4"),
        dict(start_seconds="0.4", end_seconds="0.8"),
    ]
    calls = []
    run = TemporalVideo.run

    def interrupted(self, *args, window_index=0, **kwargs):
        calls.append(window_index)
        if calls == [0, 1]:
            raise OSError("synthetic interruption before second decoder")
        return run(self, *args, window_index=window_index, **kwargs)

    monkeypatch.setattr(TemporalVideo, "run", interrupted)
    with fake_service("ocr") as (state, model):
        model["temporal_display"] = provider()
        runtime = TimelineRuntime(hj["core"], configured(hj, model, "ocr", monkeypatch))
        runtime.tasks.register_temporal_ocr(
            "resume",
            m["media_id"],
            registration(hj, sha, intervals),
            actor="fixture",
            reason="synthetic resume",
        )
        first = runtime.runner.run_once()
        assert first["status"] == "awaiting_provider", first
        with hj["core"].db.transaction() as conn:
            conn.execute(
                "UPDATE hj_timeline_task SET next_attempt_at='1970-01-01Z' WHERE customer_id=%s AND id=%s",
                (hj["core"].customer, first["task_id"]),
            )
        second = runtime.runner.run_once()
        assert second["status"] == "completed", second
        assert calls == [0, 1, 1]
        with hj["core"].db.transaction(readonly=True) as conn:
            assert (
                conn.execute(
                    "SELECT count(*) AS n FROM hj_timeline_window_result WHERE customer_id=%s",
                    (hj["core"].customer,),
                ).fetchone()["n"]
                == 2
            )


def test_submission_rechecks_source_and_rejects_uncertain_nonnull_values(hj):
    m, _, _, sha = original(hj)
    tasks = TimelineRuntime(hj["core"], hj["settings"]).tasks
    tasks.register_temporal_ocr(
        "submit-check",
        m["media_id"],
        registration(hj, sha),
        actor="fixture",
        reason="source pinning",
    )
    claim = tasks.claim("lease", "ocr", "fixture/1")
    result = dict(
        schema_version="vc-lab-timeline-model-result/1",
        input_fingerprint=claim["request"]["input_fingerprint"],
        covered_window_indices=[0],
        events=[
            dict(
                kind="reading",
                text="fixture",
                spans=[
                    dict(media_id=m["media_id"], start_seconds="0", end_seconds="0.1")
                ],
                fields=[
                    dict(
                        field_key="temperature",
                        screen_id="fixture-screen",
                        profile_id="fixture-lcd/1",
                        role="process_value",
                        channel="PV",
                        unit=None,
                        status="uncertain",
                        value="26.2",
                        auto_confirmed=False,
                    )
                ],
            )
        ],
    )
    with pytest.raises(ValueError, match="uncertain"):
        tasks.submit(
            "bad-value", claim["task_id"], claim["lease_token"], "fixture/1", result
        )
    result["events"][0]["fields"][0]["value"] = None
    with hj["core"].db.transaction() as conn:
        conn.execute(
            "UPDATE media_asset SET version=version+1 WHERE customer_id=%s AND id=%s",
            (hj["core"].customer, m["media_id"]),
        )
    with pytest.raises(Conflict, match="changed"):
        tasks.submit(
            "bad-source", claim["task_id"], claim["lease_token"], "fixture/1", result
        )


def test_bad_profile_and_oversized_roi_are_rejected_before_inference(hj):
    _, _, _, sha = original(hj)
    doc = registration(hj, sha)
    invalid = deepcopy(doc)
    invalid["fields"][0]["role"] = "guessed"
    with pytest.raises(TemporalVideoRejected):
        validate_registration(invalid)
    invalid = deepcopy(doc)
    invalid["fields"][0]["roi_box"] = ["0", "0", "2", "1"]
    with pytest.raises(TemporalVideoRejected):
        validate_registration(invalid)


def supplied_uncertain(claim, media_id, *, clock="verified"):
    return dict(
        schema_version="vc-lab-timeline-model-result/1",
        input_fingerprint=claim["request"]["input_fingerprint"],
        covered_window_indices=[0],
        events=[
            dict(
                kind="reading",
                text="synthetic uncertain readout",
                spans=[dict(media_id=media_id, start_seconds="0", end_seconds="0.1")],
                fields=[
                    dict(
                        field_key="temperature",
                        screen_id="fixture-screen",
                        profile_id="fixture-lcd/1",
                        role="process_value",
                        channel="PV",
                        unit=None,
                        status="uncertain",
                        value=None,
                        value_text=None,
                        auto_confirmed=False,
                    )
                ],
                payload=dict(
                    field_mapping="temporal_registered_video",
                    temporal_world_clock_status=clock,
                ),
            )
        ],
    )


def test_temporal_evidence_pins_actual_media_revision(hj):
    m, _, _, sha = original(hj)
    with hj["core"].db.transaction() as conn:
        conn.execute(
            "UPDATE media_asset SET version=2 WHERE customer_id=%s AND id=%s",
            (hj["core"].customer, m["media_id"]),
        )
    tasks = TimelineRuntime(hj["core"], hj["settings"]).tasks
    doc = registration(hj, sha)
    doc["media_version"] = 2
    tasks.register_temporal_ocr(
        "v2-register",
        m["media_id"],
        doc,
        actor="fixture",
        reason="same bytes explicitly new evidence revision",
    )
    claim = tasks.claim("v2-claim", "ocr", "fixture/1")
    tasks.submit(
        "v2-submit",
        claim["task_id"],
        claim["lease_token"],
        "fixture/1",
        supplied_uncertain(claim, m["media_id"]),
    )
    with hj["core"].db.transaction(readonly=True) as conn:
        rows = conn.execute(
            "SELECT media_version FROM evidence_ref WHERE customer_id=%s AND media_id=%s ORDER BY media_version",
            (hj["core"].customer, m["media_id"]),
        ).fetchall()
    assert rows[-1]["media_version"] == 2


def test_uncalibrated_temporal_clock_keeps_world_time_and_attribution_unknown(hj):
    m, _, _, sha = original(hj)
    runtime = TimelineRuntime(hj["core"], hj["settings"])
    doc = registration(hj, sha)
    doc["world_clock_status"] = "uncalibrated"
    runtime.tasks.register_temporal_ocr(
        "unknown-clock",
        m["media_id"],
        doc,
        actor="fixture",
        reason="source clock not independently calibrated",
    )
    claim = runtime.tasks.claim("unknown-clock-claim", "ocr", "fixture/1")
    result = runtime.tasks.submit(
        "unknown-clock-submit",
        claim["task_id"],
        claim["lease_token"],
        "fixture/1",
        supplied_uncertain(claim, m["media_id"], clock="uncalibrated"),
    )
    with hj["core"].db.transaction(readonly=True) as conn:
        row = conn.execute(
            "SELECT * FROM hj_activity WHERE customer_id=%s AND id=%s",
            (hj["core"].customer, result["activity_ids"][0]),
        ).fetchone()
    assert row["start_at"] is None and row["end_at"] is None
    assert (
        row["association_status"] == "unassigned"
        and row["payload"]["temporal_world_clock_status"] == "uncalibrated"
    )
    pending = runtime.views.trace(at("09:00:00"), at("09:00:02"))[
        "unknown_time_temporal_activities"
    ]
    assert pending["activities"][0]["id"] == result["activity_ids"][0]
    assert pending["activities"][0]["start_at"] is None


def dynamic_registration(hj, sha):
    doc = registration(hj, sha)
    doc["dynamic_context"] = {"enabled": True}
    doc["fields"][0].update(
        dynamic_panel_id="fixture-panel", dynamic_track_id="fixture-track"
    )
    return doc


def test_actual_dynamic_query_unknown_does_not_invent_instrument(hj, monkeypatch):
    m, _, _, sha = original(hj)
    with fake_service("ocr") as (_, model):
        model["temporal_display"] = provider()
        runtime = TimelineRuntime(hj["core"], configured(hj, model, "ocr", monkeypatch))
        runtime.tasks.register_temporal_ocr(
            "dynamic-source",
            m["media_id"],
            dynamic_registration(hj, sha),
            actor="fixture",
            reason="actual unknown dynamic query",
        )
        completed = runtime.runner.run_once()
    assert completed["status"] == "completed", completed
    items = runtime.views.trace(at("09:00:00"), at("09:00:02"))["activities"]
    assert items and all(item["instrument_id"] is None for item in items)
    for item in items:
        field = item["payload"]["fields"][0]
        assert field["reading_status"] == field["status"]
        assert field["association_status"] == "uncertain"
        assert field["dynamic_context"]["current_focus_used_for_identity"] is False


def dynamic_response(ref, doc, start, end, *, revision=None, epoch=1):
    supported = Fraction(start) == Fraction(end)
    return dict(
        schema_version="vc-lab-dynamic-context/1",
        status="candidate",
        revision=revision or 1,
        revision_id="fixture-revision",
        media_id=ref["media_id"],
        source_session_key=doc["source_session_key"],
        source_stream_key=doc["source_stream_key"],
        per_panel=[
            dict(
                panel_id="fixture-panel",
                track_id="fixture-track",
                epoch=epoch,
                instrument_id="fixture-instrument" if supported else None,
                physical_association_status="spatial_identity_supported"
                if supported
                else "uncertain",
                ocr_identity_key=["fixture-instrument", epoch] if supported else None,
                business_context_status="uncertain_crosses_context_boundary",
                segments=[],
            )
        ],
        current_focus={"instrument_id": "must-never-be-used"},
        ocr_notifications=[],
    )


def test_final_association_uses_actual_support_range_not_last_frame_focus():
    doc = dict(
        source_stream_key="camera",
        source_session_key="session",
        revision="fixture/1",
        world_clock_status="uncalibrated",
        dynamic_context={"enabled": True},
    )
    field = dict(
        screen_id="screen",
        field_key="temperature",
        profile_id="layout/1",
        role="process_value",
        dynamic_panel_id="fixture-panel",
        dynamic_track_id="fixture-track",
    )
    ref = dict(
        media_id="fixture-media", sha256="a" * 64, start_seconds="0", end_seconds="1"
    )
    refs = [
        dict(
            source_sha256=ref["sha256"],
            media_id=ref["media_id"],
            media_offset_seconds=str(v),
        )
        for v in [".1", ".4"]
    ]
    queries = []

    def query(media, session, start, end, **kwargs):
        queries.append((start, end))
        return dynamic_response(ref, doc, start, end, **kwargs)

    result = dict(
        status="uncertain", raw_text="26.2", candidates=["26.2"], source_refs=refs
    )
    event = TemporalVideo._event(
        dict(
            result=result,
            field=field,
            source_ref=refs[-1],
            dynamic_context={"revision": 1},
        ),
        ref,
        doc,
        query,
    )
    assert queries == [("1/10", "2/5")]
    assert event["instrument_id"] is None
    assert event["fields"][0]["association_status"] == "uncertain"
    assert (
        event["fields"][0]["dynamic_context"]["current_focus_used_for_identity"]
        is False
    )


def test_native_roi_epoch_reset_ignores_revision_only_changes(hj, monkeypatch):
    from visioncortex.lab.temporal_display import TemporalDisplay

    m, path, _, sha = original(hj)
    doc = dynamic_registration(hj, sha)
    ref = dict(
        media_id=m["media_id"],
        sha256=sha,
        media_version=1,
        start_seconds="0",
        end_seconds=".8",
    )
    resets, calls = [], []
    reset = TemporalDisplay.reset

    def record(self, identity=None, *, reason="explicit_reset"):
        resets.append(reason)
        return reset(self, identity, reason=reason)

    monkeypatch.setattr(TemporalDisplay, "reset", record)

    def query(media, session, start, end, **kwargs):
        calls.append((start, end))
        return dynamic_response(
            ref,
            doc,
            start,
            end,
            revision=kwargs.get("revision") or len(calls),
            epoch=1 if Fraction(start) < Fraction(".4") else 2,
        )

    events, receipts = TemporalVideo(hj["settings"]).run(
        path, ref, doc, None, provider(), query_context=query
    )
    assert resets == ["dynamic_panel_identity_or_epoch_changed"]
    assert events and receipts[-1]["temporal_execution"]["input_frame_count"] == 20
    assert receipts[-1]["temporal_execution"]["ocr_provider_invocations"] == 0
    assert all(e["fields"][0]["value_text"] is None for e in events)


def test_dynamic_context_rejects_another_source_before_using_identity():
    doc = dict(
        source_stream_key="camera",
        source_session_key="session",
        dynamic_context={"enabled": True},
    )
    field = dict(dynamic_panel_id="fixture-panel", dynamic_track_id="fixture-track")
    ref = dict(media_id="fixture-media")

    def query(*args, **kwargs):
        response = dynamic_response(ref, doc, "0", "0")
        response["media_id"] = "other-media"
        return response

    with pytest.raises(TemporalVideoRejected, match="source/session"):
        resolve_dynamic_context(query, field, ref, doc, 0, 0)


def test_temporal_fingerprint_survives_actual_jsonb_roundtrip(db):
    result = jsonb_stable(
        dict(
            matrix=[[-0.0, 1.0, 1e17], [1e-17, 0.5, 0.0]],
            source_refs=[
                dict(
                    pts=840024907,
                    time_base="1/1000000",
                    media_offset_seconds="840024907/1000000",
                )
            ],
        )
    )
    with db.transaction(readonly=True) as conn:
        stored = conn.execute("SELECT %s::jsonb AS value", (Jsonb(result),)).fetchone()[
            "value"
        ]
    assert fingerprint(stored) == fingerprint(result)
    assert type(stored["source_refs"][0]["pts"]) is int
    assert stored["source_refs"][0]["media_offset_seconds"] == "840024907/1000000"
    for nonfinite in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(TemporalVideoRejected, match="Nonfinite"):
            jsonb_stable(dict(value=nonfinite))


def pixel_producer_config(**extra):
    return dict(
        schema_version="vc-lab-dynamic-video/1",
        enabled=True,
        revision="fixture-original-pixels/1",
        max_frames=20,
        batch_frames=1,
        max_buffer_bytes=32 * 1024**2,
        max_seconds=300,
        panels=[],
        **extra,
    )


def test_real_pixel_producer_commits_each_native_pts_before_actual_ocr_task(
    hj, monkeypatch
):
    m, _, _, sha = original(hj)
    doc = dynamic_registration(hj, sha)
    doc["dynamic_video"] = pixel_producer_config()
    with fake_service("ocr") as (state, model):
        model["temporal_display"] = provider()
        runtime = TimelineRuntime(hj["core"], configured(hj, model, "ocr", monkeypatch))
        runtime.tasks.register_temporal_ocr(
            "pixel-chain",
            m["media_id"],
            doc,
            actor="fixture",
            reason="actual QR decoder and original-pixel append route",
        )
        fail = runtime.tasks.fail

        def detailed(*args, **kwargs):
            import sys

            exc = sys.exc_info()[1]
            value = fail(*args, **kwargs)
            return dict(value, fixture_error=str(exc), fixture_cause=str(exc.__cause__))

        monkeypatch.setattr(runtime.tasks, "fail", detailed)
        result = runtime.runner.run_once()
    assert result["status"] == "completed", result
    assert state["posts"] > 0
    with hj["core"].db.transaction(readonly=True) as conn:
        frames = conn.execute(
            "SELECT * FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s ORDER BY source_seconds",
            (hj["core"].customer, m["media_id"]),
        ).fetchall()
        checkpoint = conn.execute(
            "SELECT * FROM hj_timeline_window_result WHERE customer_id=%s",
            (hj["core"].customer,),
        ).fetchone()
    assert len(frames) == 20
    assert frames[0]["payload"]["source_pts"] == 200
    assert frames[0]["payload"]["stream_origin"] == "1/5"
    assert frames[-1]["payload"]["source_pts"] == 970
    assert all(
        f["payload"]["tracks"] == [] and f["payload"]["qr_occurrences"] == []
        for f in frames
    )
    receipt = checkpoint["execution_receipt"]["invocations"][-1]["temporal_execution"]
    assert (
        receipt["dynamic_video"]["qr_calls"]
        == receipt["dynamic_video"]["db_commits"]
        == 20
    )
    assert (
        receipt["dynamic_video"]["input_frame_count"]
        == receipt["input_frame_count"]
        == 20
    )
    assert receipt["combined_producer_roi_peak_bytes"] <= 64 * 1024**2
    assert (
        fingerprint(checkpoint["normalized_result"]) == checkpoint["result_fingerprint"]
    )
    items = runtime.views.trace(at("09:00:00"), at("09:00:02"))["activities"]
    assert items and all(a["instrument_id"] is None for a in items)
    assert items[-1]["payload"]["fields"][0]["dynamic_context"]["revision"] is not None


def test_uncommitted_pixel_batch_is_rejected_before_using_old_identity(hj):
    m, path, _, sha = original(hj)
    doc = dynamic_registration(hj, sha)
    doc["dynamic_video"] = pixel_producer_config()
    ref = dict(
        media_id=m["media_id"],
        sha256=sha,
        media_version=1,
        start_seconds="0",
        end_seconds=".8",
    )

    class Pending:
        def observe(self, *args):
            return dict(status="buffered", pending_frames=1)

        def flush(self):
            return dict(status="committed", pending_frames=0)

        def receipt(self):
            return dict(buffer_bytes=0)

    def no_query(*args, **kwargs):
        raise AssertionError(
            "Old dynamic identity must not be queried before producer commit"
        )

    with pytest.raises(TemporalVideoRejected, match="commit before"):
        TemporalVideo(hj["settings"]).run(
            path,
            ref,
            doc,
            None,
            provider(),
            frame_producer=Pending(),
            query_context=no_query,
        )
    invalid = deepcopy(doc)
    invalid["dynamic_video"]["batch_frames"] = 5
    with pytest.raises(TemporalVideoRejected, match="single-frame"):
        validate_registration(invalid)


def test_native_codec_pixels_switch_candidate_focus_and_keep_each_panel_identity(
    hj, monkeypatch
):
    from visioncortex.lab.dynamic_context import DynamicContext
    from .test_dynamic_context import GEOMETRY
    from .test_dynamic_video import native_frame, configuration

    core = hj["core"]
    devices = []
    for n in range(2):
        issued = core.issue_equipment(
            "pixel-codec-reserve-" + str(n), actor="generated fixture"
        )
        device = core.activate_equipment(
            "pixel-codec-activate-" + str(n),
            issued["device_id"],
            verified_at=at("08:00:00"),
            actor="generated fixture",
            evidence={"fixture_only": True},
        )
        devices.append(device["device_id"])
    images = [
        native_frame(devices, n * 40 if n <= 3 else 120, (n - 3) * 40 if n > 3 else 0)
        for n in range(7)
    ]
    path = hj["settings"].incoming / "MetaData/pixel-focus/rgb.mp4"
    path.parent.mkdir(parents=True)
    raw = b"".join(np.ascontiguousarray(im[::-1, ::-1]).tobytes() for im in images)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "640x320",
            "-r",
            "10",
            "-i",
            "pipe:0",
            "-vf",
            "settb=1/1000000,setpts=200000+N*100000",
            "-vsync",
            "0",
            "-enc_time_base",
            "1/1000000",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            "-video_track_timescale",
            "1000000",
            str(path),
        ],
        input=raw,
        check=True,
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    m = Ingest(core, hj["settings"]).accept(
        dict(
            capture_id="pixel-focus-codec",
            source_instance="fixture-camera",
            source_revision="fixture/1",
            kind="video",
            session_id="fixture-session",
            relative_path=str(path.relative_to(hj["settings"].incoming)),
            captured_at=at("09:00:00"),
            time_basis={"basis": "generated_fixture", "uncertainty_us": 0},
            start_seconds="0",
            end_seconds=".7",
            source_kind="fixture",
            sha256=sha,
            native_record={"fixture_only": True},
        ),
        native_original=True,
    )
    dynamic = DynamicContext(core)
    links = []
    for n, device in enumerate(devices):
        links.append(
            dynamic.register_panel_link(
                "pixel-codec-link-" + str(n),
                device,
                "panel-" + str(n),
                "fixture-layout/1",
                GEOMETRY,
                actor="generated fixture geometry",
                reason="lossless generated QR/control fixture",
                evidence_ids=[m["evidence_id"]],
            )
        )
    doc = registration(hj, sha, intervals=[dict(start_seconds="0", end_seconds=".7")])
    doc["dynamic_context"] = {"enabled": True}
    doc["dynamic_video"] = configuration(links, images[0])
    doc["dynamic_video"]["max_seconds"] = 60
    doc["fields"] = []
    for n in range(2):
        field = deepcopy(registration(hj, sha)["fields"][0])
        field.update(
            screen_id="screen-" + str(n),
            field_key="value-" + str(n),
            dynamic_panel_id="panel-" + str(n),
            dynamic_track_id="pixel-track-" + str(n),
            roi_box=[
                str((15 + 320 * n) / 640),
                str(15 / 320),
                str((220 + 320 * n) / 640),
                str(120 / 320),
            ],
        )
        field["layout"]["dynamic_boxes"] = [[0.1, 0.1, 0.6, 0.6]]
        doc["fields"].append(field)
    with fake_service("ocr") as (state, model):
        model["temporal_display"] = provider()
        runtime = TimelineRuntime(core, configured(hj, model, "ocr", monkeypatch))
        runtime.tasks.register_temporal_ocr(
            "pixel-codec-task",
            m["media_id"],
            doc,
            actor="generated fixture",
            reason="decode to actual QR/fit/control to context/reading",
        )
        result = runtime.runner.run_once()
    assert result["status"] == "completed", result
    assert state["posts"] > 0
    a = dynamic.query_window(m["media_id"], "fixture-session", ".3", ".3")
    b = dynamic.query_window(m["media_id"], "fixture-session", ".6", ".6")
    assert a["current_focus"]["instrument_id"] == devices[0]
    assert b["current_focus"]["instrument_id"] == devices[1]
    assert all(
        p["physical_association_status"] == "spatial_identity_supported"
        for p in b["per_panel"]
    )
    values = runtime.views.trace(at("09:00:00"), at("09:00:01"))["activities"]
    identities = {
        a["payload"]["fields"][0]["screen_id"]: a["instrument_id"] for a in values
    }
    assert identities == {"screen-0": devices[0], "screen-1": devices[1]}
    assert all(a["payload"]["business_confirmation"] is False for a in values)
    with core.db.transaction(readonly=True) as conn:
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM hj_dynamic_frame_event WHERE customer_id=%s",
                (core.customer,),
            ).fetchone()["n"]
            == 7
        )
        assert (
            conn.execute(
                "SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s",
                (core.customer,),
            ).fetchone()["n"]
            == 0
        )
