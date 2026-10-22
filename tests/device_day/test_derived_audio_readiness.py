"""Declared replay clocks admit partial audio without inventing capture quality."""

import csv
from copy import deepcopy
import json

import pytest

from test_recorder_audio_native_v2 import (
    config,
    device_config,
    incoming,
    native_audio,
    ledger,
    record,
)
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_stt import usable_audio_windows, transcript_text

__all__ = ["config", "device_config"]
START = 1789005600000000


def replay(folder):
    native_audio(folder)
    quality = [
        {"kind": "synthetic_silence", "start_us": START, "end_us": START + 1000000}
    ]
    common = {
        "artifact_kind": "derived_replay",
        "schema_version": 2,
        "task_audio": True,
        "segment_id": "replay-segment",
        "segment_window_start_global_us": START,
        "segment_window_end_global_us": START + 2000000,
        "audio_valid": False,
        "global_timestamp_source": "derived_cfr_source_lineage",
        "quality_status": "partial",
        "quality_flags": ["source_clock_invalid", "synthetic_silence"],
        "packet_receipt_evidence": "unavailable_for_derived_media",
    }
    meta = (
        read_json(folder / "audio_meta.json")
        | common
        | {
            "sample_rate": 48000,
            "audio_sample_count": 96000,
            "audio_duration_us": 2000000,
            "first_audio_global_us": START,
            "last_audio_global_us": START + 1999979,
            "audio_duration_basis": "derived_decoded_sample_grid",
            "audio_timestamp_basis": "derived_decoded_sample_origin",
            "quality_evidence": quality,
        }
    )
    atomic_json(folder / "audio_meta.json", meta)
    atomic_json(
        folder / "audio_ready.json", read_json(folder / "audio_ready.json") | common
    )
    with (folder / "audio_timing.csv").open("w", newline="") as handle:
        fields = [
            "global_timestamp_us",
            "window_start_global_us",
            "window_end_global_us",
            "received_packets",
            "received_packets_basis",
            "time_basis",
            "quality_evidence_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number in range(2):
            left = START + number * 1000000
            writer.writerow(
                {
                    "global_timestamp_us": left,
                    "window_start_global_us": left,
                    "window_end_global_us": left + 1000000,
                    "received_packets": "",
                    "received_packets_basis": "unavailable_for_derived_media",
                    "time_basis": "derived_cfr_source_lineage",
                    "quality_evidence_json": json.dumps(quality if number == 0 else []),
                }
            )
    ledger(folder)


def test_partial_replay_admission_keeps_false_validity_and_only_submits_real_audio(
    config,
):
    folder = incoming(config)
    replay(folder)
    before = {p.name: p.read_bytes() for p in folder.glob("audio*")}
    value = record(config)
    assert value["stt_processable"], value["stt_readiness_reasons"]
    assert value["audio"]["capture_complete"] is False
    assert value["audio"]["time_basis"] == "derived_replay_sample_clock"
    assert value["stt_clock_evidence"]["received_packets"] is None
    assert value["stt_clock_evidence"]["source_clock_verified"] is False
    assert usable_audio_windows(value["audio"], 2.0065, 180) == [(1, 2)]
    assert before == {p.name: p.read_bytes() for p in folder.glob("audio*")}


@pytest.mark.parametrize(
    "defect", ["sample_count", "quality_missing", "schema", "source_clock", "gaps"]
)
def test_replay_does_not_waive_schema_clock_or_quality_validation(config, defect):
    folder = incoming(config)
    replay(folder)
    meta = read_json(folder / "audio_meta.json")
    if defect == "sample_count":
        meta["audio_sample_count"] += 48000
    elif defect == "quality_missing":
        meta["quality_evidence"] = []
    elif defect == "schema":
        meta["schema_version"] = 999
    elif defect == "source_clock":
        meta["global_timestamp_source"] = "unknown"
    else:
        p = folder / "audio_timing.csv"
        p.write_text(p.read_text().replace(str(START + 1000000), str(START + 1100000)))
    atomic_json(folder / "audio_meta.json", meta)
    ledger(folder)
    value = record(config)
    assert not value["stt_processable"] and value["processable"]
    assert value["stt_readiness_reasons"][0].startswith("derived_audio_")


def test_outages_keep_coherent_chunks_and_synthetic_padding_is_excluded():
    audio = {
        "artifact_kind": "derived_replay",
        "start_us": START,
        "duration_us": 10000000,
        "quality_evidence": [
            {"kind": "synthetic_silence", "start_us": START, "end_us": START + 2000000},
            {
                "kind": "source_outage",
                "start_us": START + 1000000,
                "end_us": START + 3000000,
            },
            {
                "kind": "source_outage",
                "start_us": START + 7000000,
                "end_us": START + 8000000,
            },
        ],
    }
    before = deepcopy(audio)
    assert usable_audio_windows(audio, 10.0065, 3) == [(2, 5), (5, 8), (8, 10)]
    assert audio == before


def test_entire_padding_is_not_reported_as_model_silence_detection():
    audio = {
        "artifact_kind": "derived_replay",
        "start_us": START,
        "duration_us": 1000000,
        "quality_evidence": [
            {"kind": "synthetic_silence", "start_us": START, "end_us": START + 1000000}
        ],
    }
    assert usable_audio_windows(audio, 1, 180) == []
    assert "没有向识别模型提交音频" in transcript_text(
        {"recording_id": "r"}, {"path": "audio.opus"}, [], "no_usable_audio"
    )


def test_actual_stt_owner_keeps_raw_response_and_marks_or_excludes_gap_text(tmp_path, monkeypatch):
    from pathlib import Path
    from test_speech_qwen import event
    from visioncortex import speech, speech_qwen as qwen, speech_worker
    from visioncortex.device_day_contract import DeviceDayLayout, artifact
    from visioncortex.device_day_stt import transcribe
    layout = DeviceDayLayout(tmp_path/'archive', 'fixture_cam01', START, tmp_path/'backend')
    original = layout.raw/'Fixture.opus'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'owned-synthetic-contract-audio')
    quality = [{'kind': 'synthetic_silence', 'start_us': START, 'end_us': START+2000000},
               {'kind': 'source_outage', 'start_us': START+2000000, 'end_us': START+3000000},
               {'kind': 'source_outage', 'start_us': START+4000000, 'end_us': START+4100000}]
    retention = {'recording': {'recording_id': 'fixture', 'recording_start_us': START,
                              'recording_end_us': START+5000000},
        'audio': {'status': 'provided', 'artifact_kind': 'derived_replay', 'start_us': START,
                  'duration_us': 5000000, 'quality_status': 'partial', 'quality_flags': ['source_outage'],
                  'quality_evidence': quality, 'capture_clock_verified': False},
        'sources': [{'kind': 'audio_audio', 'retained': artifact(layout.root, original)}]}
    runtime = {'provider': 'aliyun_qwen', 'worker_sha256': speech_worker.sha256(Path(qwen.__file__)),
               'max_audio_seconds': 180, 'model': {'model': qwen.MODEL}, 'endpoint': 'https://example.invalid/test'}
    monkeypatch.setattr(speech, 'probe_audio', lambda _: {'duration_seconds': 5})
    monkeypatch.setattr(speech, 'runtime_request', lambda _: runtime)
    monkeypatch.setattr(qwen.subprocess, 'run', lambda command, **kwargs: Path(command[-1]).write_bytes(b'owned-wave'))
    calls = []
    def recognize(*args):
        calls.append(True)
        return {'http_status': 200, 'events': [event('间隙推断', begin=200, end=800),
                event('有声记录', begin=1200, end=2200, number=2, cumulative='间隙推断有声记录')], 'wall_seconds': .1}
    monkeypatch.setattr(qwen, 'recognize', recognize)
    result = transcribe({'speech_recognition': {'enabled': True}}, layout, retention, 'new')
    assert len(calls) == 1
    assert result['usable_audio_windows'] == [(2, 5)]
    assert result['comments'][0]['text'] == '有声记录' and len(result['comments']) == 1
    assert result['comments'][0]['time_basis'] == 'derived_replay_sample_clock'
    assert result['comments'][0]['source_clock_verified'] is False
    assert result['comments'][0]['source_quality_ranges'] == [quality[2]]
    assert result['excluded_gap_transcripts'][0]['text'] == '间隙推断'
    transcript = layout.root/result['comments'][0]['transcript_path']
    assert len(read_json(transcript)['segments']) == 2  # Actual raw response remains intact.
