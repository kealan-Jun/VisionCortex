from types import SimpleNamespace

import numpy as np
import pytest

from visioncortex.performance_stages import (
    StageTimings,
    MeasuredWriter,
    prediction_timings,
)


def test_decoder_timing_excludes_consumer_backpressure(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "visioncortex.performance_stages.time.perf_counter", lambda: clock[0]
    )

    def frames():
        for _ in range(2):
            clock[0] += 2
            yield 0, 0, np.zeros((2, 2, 3), dtype=np.uint8)

    measured = StageTimings()
    for frame in measured.frames(frames()):
        clock[0] += 100
        assert frame[2].shape == (2, 2, 3)
    assert measured.snapshot()["read_decode_supply_seconds"] == 4
    assert measured.snapshot()["decoded_pixel_bytes"] == 24


def test_failed_decode_still_records_elapsed_time(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "visioncortex.performance_stages.time.perf_counter", lambda: clock[0]
    )

    def frames():
        clock[0] += 3
        raise RuntimeError("decoder failed")
        yield

    measured = StageTimings()
    with pytest.raises(RuntimeError):
        list(measured.frames(frames()))
    assert measured.snapshot()["read_decode_supply_seconds"] == 3


def test_backend_times_include_padding_and_missing_values_stay_unknown():
    measured = StageTimings()
    prediction_timings(
        [SimpleNamespace(speed={"preprocess": 1, "inference": 2, "postprocess": 0})]
        * 4,
        measured,
    )
    prediction_timings([SimpleNamespace(speed={})], measured)
    assert measured.snapshot()["model_inference_seconds"] == 0.008
    assert measured.snapshot()["model_inference_profiled_frames"] == 4
    assert measured.snapshot()["model_postprocess_seconds"] == 0
    assert "model_inference_seconds" not in StageTimings().snapshot()


def test_ledger_writer_preserves_data_and_flushes(tmp_path):
    timings = StageTimings()
    path = tmp_path / "data"
    writer = MeasuredWriter(path.open("w"), timings)
    writer.write("data")
    writer.flush()
    writer.close()
    assert path.read_text() == "data"
    assert timings.snapshot()["ledger_write_seconds"] >= 0
