"""Bounded local native refresh; no models, codec, business DB or real NAS."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from visioncortex.lab.components import ROOT
from visioncortex.lab.delivery import NativeDelivery
from visioncortex.lab.native_linked_layout import module
from visioncortex.lab.native_refresh import NativeRefresh, _read
from visioncortex.lab.native_timeline_links import build_sidecar, association_path
import visioncortex.lab.native_refresh as refresh_module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def prepare(root, customer, camera="fixture-camera", *, index=True):
    archive = "2026-10-29_" + camera
    day = root / archive
    day.mkdir(parents=True)
    start = int(datetime.fromisoformat("2026-10-29T09:00:00+08:00").timestamp() * 1e6)
    seg = dict(
        segment_id="fixture-segment",
        recording_id="fixture-recording",
        activity="inactive",
        activity_label="unverified fixture",
        start_ms=0,
        end_ms=4000,
        start_us=start,
        end_us=start + 4000000,
        source_ref=dict(
            path="MetaData/fixture/rgb.mp4", sha256="a" * 64, size_bytes=123
        ),
        key_frames=[],
        scene_frames=[],
        json_path="ProcessedClips/Clips/09-00-00_09-00-04_NoExperimentActivity_fixture/NoExperimentActivity.json",
    )
    record = dict(
        recording_id=seg["recording_id"],
        start_us=start,
        end_us=start + 4000000,
        camera_key=camera,
        stages={"understanding": {"status": "pending"}},
        sources=[],
    )
    idx = dict(
        archive=archive,
        recordings=[record],
        segments=[seg],
        understandings=[],
        context={},
    )
    if index:
        write(day / "ProcessedClips/Index.json", idx)
    document, _ = module("device_day_content").segment_documents(
        idx, day, seg, None, None, record, []
    )
    main = (
        day
        / "MultimodalUnderstanding/09-00-00_09-00-04_NoExperimentActivity_fixture/Understanding.json"
    )
    module("device_day_content")._write(main, document)
    historical = (
        day / "MultimodalUnderstanding/.Evidence/Versions/fixture/Understanding.json"
    )
    write(historical, dict(document, immutable=True))
    return dict(
        root=root,
        day=day,
        archive=archive,
        index=idx,
        main=main,
        historical=historical,
        customer=customer,
    )


def sidecar(f, revision=1):
    # This is only a fixture rules identity for the pure builder test. Public
    # scheduling verifies the final actual Rules lock in a separate PG test.
    rules = dict(
        revision="fixture-candidate",
        sha256=hashlib.sha256(
            (ROOT / "contracts/lab/rules.lock.json").read_bytes()
        ).hexdigest(),
    )
    value = build_sidecar(
        archive=f["archive"],
        customer_id=f["customer"],
        revision=revision,
        snapshot_fingerprint=str(revision) * 64,
        rules=rules,
        canonical_report=f["archive"] + "/LaboratoryDailyReport/Timeline/fixture.json",
        activities=[],
        person_projection={},
    )
    relative = association_path(
        f["archive"], revision, value["snapshot_fingerprint"], rules["sha256"]
    )
    path = f["root"] / relative
    write(path, value)
    raw = path.read_bytes()
    return dict(
        relative_path=relative,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
    )


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    customer = uuid4()
    core = SimpleNamespace(customer=customer)
    settings = SimpleNamespace(runtime_root=tmp_path / "runtime")
    f = prepare(root, customer)
    f["engine"] = NativeRefresh(core, settings)
    f["config"] = dict(
        runtime={"local_only": True},
        storage={
            "archive_root": str(root),
            "local_runtime_root": str(settings.runtime_root),
        },
    )
    return f


def invoke(f, committed):
    with module("device_day_write_guard").storage_guard(f["config"]) as guard:
        return f["engine"]._refresh(f["root"], f["archive"], committed, guard)


def test_existing_native_index_refresh_uses_original_writers_without_models_or_clip_export(
    fixture, monkeypatch
):
    f = fixture
    old = json.loads(f["main"].read_text())
    history = f["historical"].read_bytes()
    source = (f["day"] / "ProcessedClips/Index.json").read_bytes()

    def forbidden(*a, **k):
        raise AssertionError("No model or video materializer is allowed")

    monkeypatch.setattr(module("device_day_content"), "publish_content", forbidden)
    monkeypatch.setattr(module("application.clip_delivery"), "publish", forbidden)
    monkeypatch.setattr(
        module("application.retained_video"), "publish_selected", forbidden
    )
    monkeypatch.setattr(module("device_day_synthesis"), "synthesize", forbidden)
    original = module("device_day_content")._write
    calls = []

    def writer(path, value):
        calls.append(str(path))
        return original(path, value)

    monkeypatch.setattr(module("device_day_content"), "_write", writer)
    result = invoke(f, sidecar(f))
    new = json.loads(f["main"].read_text())
    assert (
        result["status"] == "refreshed"
        and result["model_calls"] == result["clip_exports"] == 0
    )
    assert {k: v for k, v in new.items() if k != "关联时间线"} == old and new[
        "status"
    ] == "pending"
    assert calls == [str(f["main"])] and f["historical"].read_bytes() == history
    assert (f["day"] / "ProcessedClips/Index.json").read_bytes() == source
    report = json.loads(
        (f["day"] / "LaboratoryDailyReport/LaboratoryDailyReport.json").read_text()
    )
    assert (
        report["全天综合理解"]["status"] == "pending"
        and report["关联时间线"]["native_entry_refresh_status"]
        == "included_by_native_builder"
    )
    assert (
        new["关联时间线"]["segment_association_status"]
        == "pending_no_exact_source_evidence"
    )


def test_late_revision_replays_original_writer_and_duplicate_is_byte_idempotent(
    fixture,
):
    f = fixture
    r1 = sidecar(f)
    invoke(f, r1)
    day = f["day"] / "LaboratoryDailyReport/LaboratoryDailyReport.json"
    old = day.read_bytes()
    side1 = (f["root"] / r1["relative_path"]).read_bytes()
    assert invoke(f, r1)["status"] == "unchanged" and day.read_bytes() == old
    r2 = sidecar(f, 2)
    assert invoke(f, r2)["status"] == "refreshed"
    assert json.loads(day.read_text())["关联时间线"]["snapshot_revision"] == 2
    latest = day.read_bytes()
    assert invoke(f, r2)["status"] == "unchanged" and day.read_bytes() == latest
    assert (f["root"] / r1["relative_path"]).read_bytes() == side1


def test_camera_refresh_never_overwrites_other_camera(fixture):
    f = fixture
    other = prepare(f["root"], f["customer"], "other-camera")
    other["engine"] = f["engine"]
    other["config"] = f["config"]
    invoke(f, sidecar(f))
    invoke(other, sidecar(other))
    before = other["main"].read_bytes()
    invoke(f, sidecar(f, 2))
    assert other["main"].read_bytes() == before
    report = json.loads(
        (other["day"] / "LaboratoryDailyReport/LaboratoryDailyReport.json").read_text()
    )
    assert (
        report["archive"] == other["archive"]
        and report["关联时间线"]["snapshot_revision"] == 1
    )


def test_missing_index_remains_pending_without_creating_native_main(fixture):
    f = fixture
    (f["day"] / "ProcessedClips/Index.json").unlink()
    old = f["main"].read_bytes()
    assert invoke(f, sidecar(f))["status"] == "pending_native_index"
    assert (
        f["main"].read_bytes() == old
        and not (f["day"] / "LaboratoryDailyReport/LaboratoryDailyReport.json").exists()
    )


@pytest.mark.parametrize("fault", ["customer", "receipt_hash", "camera_index"])
def test_uncommitted_cross_customer_or_cross_camera_inputs_fail_closed(fixture, fault):
    f = fixture
    committed = sidecar(f)
    before = f["main"].read_bytes()
    if fault == "customer":
        f["engine"].core.customer = uuid4()
    elif fault == "receipt_hash":
        committed["sha256"] = "0" * 64
    else:
        idx = deepcopy(f["index"])
        idx["archive"] = "2026-10-29_other-camera"
        write(f["day"] / "ProcessedClips/Index.json", idx)
    if fault == "camera_index":
        with pytest.raises(ValueError, match="another camera"):
            invoke(f, committed)
    else:
        assert invoke(f, committed)["status"] == "pending_committed_association"
    assert (
        f["main"].read_bytes() == before
        and not (f["day"] / "LaboratoryDailyReport/LaboratoryDailyReport.json").exists()
    )


def test_failed_day_writer_recovers_without_touching_historical_evidence(
    fixture, monkeypatch
):
    f = fixture
    committed = sidecar(f)
    reports = module("device_day_reports")
    original = reports.render_day
    history = f["historical"].read_bytes()

    def fail(*a, **k):
        raise OSError("synthetic original writer failure")

    monkeypatch.setattr(reports, "render_day", fail)
    with pytest.raises(OSError):
        invoke(f, committed)
    monkeypatch.setattr(reports, "render_day", original)
    assert invoke(f, committed)["status"] == "refreshed"
    assert f["historical"].read_bytes() == history


def test_unknown_schema_and_output_bound_preserve_existing_main(fixture):
    f = fixture
    committed = sidecar(f)
    old = f["main"].read_bytes()
    with module("device_day_write_guard").storage_guard(f["config"]) as guard:
        with pytest.raises(ValueError, match="budget"):
            f["engine"]._refresh(f["root"], f["archive"], committed, guard, 1)
    assert f["main"].read_bytes() == old
    value = json.loads(old)
    value["schema_version"] = "fixture-immutable-execution/1"
    write(f["main"], value)
    with pytest.raises(ValueError, match="not refreshable"):
        invoke(f, committed)


def test_nas_reader_rejects_changed_mount_before_read_or_write(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(refresh_module, "is_nas", lambda _: True)

    class Changed:
        def validate(self, **kwargs):
            raise OSError("synthetic mount changed")

        def mount_for(self, *a):
            raise AssertionError("must not touch missing mount")

    with pytest.raises(OSError, match="mount changed"):
        _read(f["root"], f["archive"] + "/ProcessedClips/Index.json", Changed())


def test_nas_reader_uses_pinned_descriptor_and_does_not_create_absent_directory(
    fixture, monkeypatch
):
    f = fixture
    monkeypatch.setattr(refresh_module, "is_nas", lambda _: True)
    fd = os.open(f["root"], os.O_RDONLY | os.O_DIRECTORY)
    calls = []

    class Pinned:
        def validate(self, **kwargs):
            calls.append(kwargs)

        def mount_for(self, path):
            return SimpleNamespace(
                path=f["root"], descriptor=fd, snapshot={"device": os.fstat(fd).st_dev}
            ), path

    try:
        result = _read(f["root"], f["archive"] + "/ProcessedClips/Index.json", Pinned())
        assert result[0]["archive"] == f["archive"] and calls[0] == calls[-1] == {}
        assert (
            _read(
                f["root"],
                "2026-10-29_absent-camera/ProcessedClips/Index.json",
                Pinned(),
            )
            is None
        )
        assert not (f["root"] / "2026-10-29_absent-camera").exists()
    finally:
        os.close(fd)


def test_timeline_delivery_disabled_publication_keeps_previous_default_without_db_or_models():
    core = SimpleNamespace()
    settings = SimpleNamespace(processing_profile="timeline-v1", timeline_publish=False)
    assert NativeDelivery(core, settings).tick() == {
        "created": 0,
        "dependencies_ready": 0,
        "profile": "timeline-v1",
    }


def test_main_source_range_mismatch_is_not_rebound_to_new_index(fixture):
    f = fixture
    committed = sidecar(f)
    document = json.loads(f["main"].read_text())
    document["source_video"]["sha256"] = "b" * 64
    write(f["main"], document)
    before = f["main"].read_bytes()
    with pytest.raises(ValueError, match="source/range disagree"):
        invoke(f, committed)
    assert (
        f["main"].read_bytes() == before
        and not (f["day"] / "LaboratoryDailyReport/LaboratoryDailyReport.json").exists()
    )


def test_retained_json_working_set_bound_rejects_before_original_writers(
    fixture, monkeypatch
):
    f = fixture
    committed = sidecar(f)
    before = f["main"].read_bytes()
    monkeypatch.setattr(refresh_module, "MAX_RETAINED_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="cache bound"):
        invoke(f, committed)
    assert (
        f["main"].read_bytes() == before
        and not (f["day"] / "LaboratoryDailyReport/LaboratoryDailyReport.json").exists()
    )
