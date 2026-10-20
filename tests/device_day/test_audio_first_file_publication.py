"""Verified audio-first outputs publish through the existing file/index owner."""
from pathlib import Path

from test_independent_audio_first import audio_first, device_config, inspect, stable_video
from test_recorder_metadata import stable
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_file_index import FileIndexPublisher
from visioncortex.observed_inventory import observe

__all__ = ['audio_first', 'device_config']


def test_real_stt_port_composes_readable_views_without_claiming_video(audio_first):
    config, folder, original, models, runner = audio_first
    config['device_day']['readable_content_enabled'] = True
    (folder/'rgb.mp4').unlink()
    stable(folder)
    record = inspect(config)
    assert runner.process(record, stage='stt')['status'] == 'completed'
    observe(runner.runtime_root, {'recordings': [record]})
    publisher = FileIndexPublisher(config)
    result = publisher.tick()
    assert result['published'] == 1, result
    layout = runner.layout(record)
    canonical = read_json(layout.index)
    assert canonical['recordings'] == canonical['segments'] == canonical['understandings'] == []
    time_index = read_json(layout.comments/'TimeIndex.json')
    audio = time_index['streams']['audio'][0]
    assert audio['transcription']['sentences'][0]['text'] == '先校准仪器。'
    assert audio['transcription']['status'] == 'completed'
    assert not time_index['streams']['video'] or all(x.get('file') is None for x in time_index['streams']['video'])
    transcript = Path(config['storage']['archive_root'])/layout.name/audio['transcription']['readable_text_file']['path']
    assert '先校准仪器。' in transcript.read_text()
    report = read_json(layout.reports/'LaboratoryDailyReport.json')
    readable = report['全天记录']
    assert readable['记录范围'] == [] and readable['全天时间线'] == []
    assert readable['录音时间线'][0]['原话'] == '先校准仪器。'
    assert readable['视频留存统计']['original_video_count'] == 0
    assert models.speech_calls == 1 and models.vision_calls == models.semantic_calls == 0
    assert not runner._receipt(layout, record, 'vision').exists()
    assert not runner._receipt(layout, record, 'retention').exists()
    assert publisher.tick()['unchanged'] == 1
    transcript.unlink()
    assert publisher.tick()['published'] == 1
    assert transcript.is_file() and models.speech_calls == 1
    for name, content in original.items():
        if not name.startswith('audio'):
            (folder/name).write_bytes(content)
    stable_video(folder)
    fresh = inspect(config)
    observe(runner.runtime_root, {'recordings': [fresh]})
    assert publisher.tick()['published'] == 1
    assert read_json(layout.index)['recordings'] == []
    assert models.speech_calls == 1


def test_audio_coverage_preserves_missing_time_and_status():
    from visioncortex.device_day_narrative import day_record
    for status in ('no_audio', 'pending_publication', 'failed'):
        source = {'archive': '2026-10-20_test_cam01', 'segments': [], 'understandings': [],
            'recordings': [{'recording_id': 'missing-clock', 'audio': {'status': status},
                'transcription': {'status': 'waiting_for_capture'}}]}
        result = day_record(source)
        coverage, = result['独立录音覆盖']
        assert coverage['开始时间'] is coverage['结束时间'] is None
        assert coverage['录音状态'] == status
        assert result['录音时间线'] == []


def test_unverified_collector_audio_keeps_source_gap_without_verified_stream(device_config):
    from visioncortex.device_day_file_index import build_file_index
    sources = [{'kind': 'audio', 'original_path': '/collector/device/day/MetaData/120000/audio.opus',
                'verification_status': 'unverified', 'ownership': 'collector'}]
    verification = {'status': 'waiting_for_input_verification',
                    'reasons': ['independent_audio_session_unverified'], 'stt_binding_granted': False}
    value = build_file_index(device_config, {
        'archive': '2026-10-20_test_cam01', 'segments': [], 'understandings': [],
        'recordings': [{'recording_id': 'unverified-audio', 'start_us': 1792468800000000,
                        'end_us': 1792468860000000, 'audio': {'status': 'provided',
                            'input_verification': verification, 'collector_sources': sources}}]}, {})
    recording, = value['recordings']
    assert recording['audio_status'] == 'provided'
    assert recording['audio_verification'] == verification
    assert recording['audio_source_references'] == sources
    assert value['streams']['audio'] == []
    assert all(not row.get('file') for row in value['streams']['video'])
