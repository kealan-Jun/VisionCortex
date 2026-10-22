"""Fractional replay origins preserve samples without exceeding provider limits."""
import math
from pathlib import Path

import pytest

from visioncortex.device_day_stt import audio_windows, usable_audio_windows
from visioncortex import speech_qwen as qwen, speech_worker


@pytest.mark.parametrize('offset', [0.123457, 123.066667, 0.000001, 1/16000])
def test_fractional_origin_never_exceeds_budget_or_loses_sample_boundaries(offset):
    origin = 1800000000000000
    start = round(offset*1000000)
    audio = {'artifact_kind': 'derived_replay', 'start_us': origin, 'duration_us': 900000000,
             'quality_evidence': [{'kind': 'synthetic_silence', 'start_us': origin,
                                   'end_us': origin+start}]}
    windows = usable_audio_windows(audio, 900, 180)
    assert windows[0][0] == start/1000000 and round(windows[-1][1]*1000000) == 900000000
    assert len(windows) == math.ceil((900-start/1000000)/180)
    assert all(0 < right-left <= 180 for left, right in windows)
    for first, second in zip(windows, windows[1:], strict=False):
        assert round(first[1]*1000000) == round(second[0]*1000000)
        for rate in (16000, 48000):
            assert round(first[1]*rate) == round(second[0]*rate)
    # Correctly represented earlier requests remain identical and cacheable.
    assert windows[:2] == [(start/1000000, start/1000000+180),
                           (start/1000000+180, start/1000000+360)]


def test_fractional_window_passes_actual_adapter_budget_without_loosening_gate(tmp_path, monkeypatch):
    source = tmp_path/'audio.opus'
    source.write_bytes(b'synthetic-audio-not-runtime-evidence')
    windows = audio_windows(900-123.066667, 180, offset=123.066667)
    left, right = windows[2]
    assert 663.066667-483.066667 > 180  # Original failure is represented exactly.
    request = {'worker_sha256': speech_worker.sha256(Path(qwen.__file__)),
               'start_seconds': left, 'end_seconds': right, 'max_audio_seconds': 180,
               'model': {'model': qwen.MODEL}, 'endpoint': 'https://dashscope.aliyuncs.com/api/v1/test',
               'source': {'folder': str(tmp_path), 'resolved_folder': str(tmp_path.resolve()),
                          'files': {source.name: speech_worker.file_record(source)}, 'audio_file': source.name}}
    calls = []
    monkeypatch.setattr(qwen.subprocess, 'run', lambda args, **kw: Path(args[-1]).write_bytes(b'synthetic-decoded'))
    def recognize(*args):
        calls.append(True)
        return {'http_status': 200, 'events': [{'message': qwen.NO_WORDS}], 'wall_seconds': 0}
    monkeypatch.setattr(qwen, 'recognize', recognize)
    assert qwen.invoke(tmp_path/'accepted', request)['status'] == 'completed'
    request['end_seconds'] = request['start_seconds']+180+1/16000
    with pytest.raises(ValueError, match='exceeds its budget'):
        qwen.invoke(tmp_path/'rejected', request)
    assert len(calls) == 1


def test_zero_origin_preserves_existing_request_json_number_types():
    import json
    assert json.dumps(audio_windows(360, 180)) == '[[0, 180], [180, 360]]'
    assert json.dumps(audio_windows(360.0, 180.0)) == '[[0.0, 180.0], [180.0, 360.0]]'
