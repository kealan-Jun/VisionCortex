from types import SimpleNamespace

import pytest

from visioncortex.device_day_models import DeviceDayModels


def backend():
    model = object.__new__(DeviceDayModels)
    model.config = {"performance": {"detection_fps": 20, "image_size": 640}}
    return model


def run_recovery(model, info, error=None, windows=None):
    return model._recover_short_fine(
        error or ValueError("Existing fine frame coverage evidence gate did not pass"),
        None, {}, "key", 0, "view", info, "transform", "retention",
        windows or [(0, info.duration_ms)], "previous",
    )


def test_short_coverage_recovery_uses_native_fps_and_a_separate_index():
    model = backend()
    calls = []
    model._scan = lambda *args: (calls.append(args) or ({"v": "new-ledger"}, "new-scan", {}))

    def index(*args, **kwargs):
        assert kwargs == {"index_name": "FineIndexRecovery"}
        assert args[6] == {"v": "new-ledger"}
        return "indexed", {"formal_evidence_ready": True}, ["artifact"]

    model._index_fine = index
    result = run_recovery(model, SimpleNamespace(duration_ms=566.667, fps=30))
    assert len(calls) == 1
    assert calls[0][-2:] == (30, 640)
    assert result[-1]["coverage_recovery"]["coverage_threshold_unchanged"] is True


@pytest.mark.parametrize("duration,fps", [(1001, 30), (0, 30), (500, 20), (500, 120)])
def test_recovery_does_not_expand_to_long_or_inapplicable_sources(duration, fps):
    with pytest.raises(ValueError, match="coverage evidence"):
        run_recovery(backend(), SimpleNamespace(duration_ms=duration, fps=fps))


def test_recovery_keeps_unrelated_errors_and_partial_windows_failed():
    info = SimpleNamespace(duration_ms=500, fps=30)
    with pytest.raises(ValueError, match="unrelated"):
        run_recovery(backend(), info, ValueError("unrelated"))
    with pytest.raises(ValueError, match="coverage evidence"):
        run_recovery(backend(), info, windows=[(100, 500)])


def test_failed_rescan_coverage_is_not_retried_again():
    model = backend()
    calls = []
    model._scan = lambda *args: (calls.append(args) or ({}, "new", {}))

    def index(*args, **kwargs):
        raise ValueError("still missing evidence")

    model._index_fine = index
    with pytest.raises(ValueError, match="still missing evidence"):
        run_recovery(model, SimpleNamespace(duration_ms=500, fps=30))
    assert len(calls) == 1
