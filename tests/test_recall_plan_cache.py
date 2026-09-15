from visioncortex import device_day_models as models
from visioncortex.schemas import ViewInput, ViewRole, VideoInfo


def test_plan_reuses_and_invalidates_ledger_and_rejects_corruption(
    default_config, tmp_path, monkeypatch
):
    backend = models.DeviceDayModels(default_config)
    view = ViewInput(
        view_id="test", role=ViewRole.FIRST_PERSON, video=tmp_path / "Video.mp4"
    )
    info = VideoInfo(
        path=view.video,
        width=100,
        height=100,
        fps=30,
        frame_count=300,
        duration_ms=10000,
        size_bytes=0,
    )
    ledger = tmp_path / "Ledger.jsonl"
    ledger.write_text("original")
    calls = []

    def planner(*args):
        calls.append(1)
        return [], [], {"formal_evidence_ready": True}

    monkeypatch.setattr(models, "device_scan_plan", planner)
    args = (view, {"test": ledger}, tmp_path, info, 0, 10000, 2)
    first = backend._plan(*args)
    second = backend._plan(*args)
    assert not first[3]["reused"] and second[3]["reused"] and len(calls) == 1
    ledger.write_text("changed")
    assert not backend._plan(*args)[3]["reused"] and len(calls) == 2
    for p in (tmp_path / "Plans").glob("*.json"):
        p.write_text("{broken")
    assert not backend._plan(*args)[3]["reused"] and len(calls) == 3
