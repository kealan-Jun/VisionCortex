"""Stream-copy contract checks at the actual subprocess port."""
import json
from types import SimpleNamespace

import pytest

from visioncortex.application.retained_video import source_stream_clip, child_segment


def test_remux_uses_editlist_and_copies_all_audio_without_lossy_fallback(monkeypatch):
    streams = [{'codec_type': 'video', 'codec_name': 'h264', 'profile': 'Constrained Baseline',
                'width': 1280, 'height': 800, 'pix_fmt': 'yuv420p', 'r_frame_rate': '30/1'},
               {'codec_type': 'audio', 'codec_name': 'aac', 'sample_rate': '48000', 'channels': 1}]
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({'streams': streams}).encode(), stderr=b'')
    monkeypatch.setattr('visioncortex.video_io._run', run)
    value = source_stream_clip('source.mp4', 'derived.mp4', 849900, 50100)
    command = commands[1]
    assert command[command.index('-c') + 1] == 'copy'
    assert command[command.index('-use_editlist') + 1] == '1'
    assert command[command.index('-ss') + 1] == '849.900000'
    assert command[command.index('-t') + 1] == '50.100000'
    assert '0:a?' in command
    assert not value['reencoded'] and value['verified_streams'] == streams


def test_parameter_drift_fails_instead_of_accepting_a_lossy_output(monkeypatch):
    calls = 0
    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        stream = {'codec_type': 'video', 'codec_name': 'h264', 'profile': 'Baseline' if calls == 1 else 'High'}
        return SimpleNamespace(returncode=0, stdout=json.dumps({'streams': [stream]}).encode(), stderr=b'')
    monkeypatch.setattr('visioncortex.video_io._run', run)
    with pytest.raises(ValueError, match='changed source stream'):
        source_stream_clip('source.mp4', 'derived.mp4', 0, 1000)


def test_export_policy_versions_identity_and_keeps_legacy_interval_readable():
    candidate = {'segment_id': 'candidate', 'source_ref': {'path': 'raw.mp4'}, 'recording_id': 'record'}
    interval = {'start_ms': 0, 'end_ms': 1000, 'start_us': 1000000, 'end_us': 2000000, 'activity': 'active'}
    old = child_segment(candidate, interval)
    new = child_segment(candidate, interval | {'export_policy': 'visioncortex-source-stream-copy/1'})
    assert new['segment_id'] != old['segment_id']
    assert child_segment(candidate, interval) == old


def test_selection_reader_streams_past_diagnostics_but_preserves_evidence_refs(tmp_path):
    from visioncortex.application.retained_video import selection_visual
    path = tmp_path/'vision.json'
    native = {'status': 'completed', 'key': 'verified-native-key', 'batches': [{'dense': [0]*10000}],
              'segments': [{'segment_id': 'candidate', 'activity_audit': {'dense': [0]*10000},
                            'activity_audit_ref': {'path': 'full-audit.json', 'sha256': 'a'*64},
                            'source_ref': {'path': 'raw.mp4'}, 'start_ms': 10, 'end_ms': 20}]}
    path.write_text(json.dumps(native))
    before = path.read_bytes()
    result = selection_visual(path)
    assert 'batches' not in result and 'activity_audit' not in result['segments'][0]
    assert result['segments'][0]['activity_audit_ref'] == native['segments'][0]['activity_audit_ref']
    assert result['segments'][0]['source_ref'] == {'path': 'raw.mp4'}
    assert path.read_bytes() == before
