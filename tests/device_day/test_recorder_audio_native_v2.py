"""Real recorder-v2 metadata shapes, owned bytes and fake inference only."""
import hashlib

import pytest

from test_device_day import FakeModels, device_config
from test_recorder_metadata import config, incoming, stable
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_content import publish_content
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_inputs import input_path, seal_input
from visioncortex.nas_recordings import scan_recordings

__all__ = ['config', 'device_config']
SESSION = 1792469228106944


def ledger(folder):
    ready = read_json(folder/'audio_ready.json')
    ready['files'] = {name: {'size': (folder/name).stat().st_size,
        'sha256': hashlib.sha256((folder/name).read_bytes()).hexdigest()}
        for name in ('audio.opus', 'audio_meta.json', 'audio_timing.csv')}
    atomic_json(folder/'audio_ready.json', ready)
    stable(folder, audio_only=True)


def native_audio(folder, start=1789005600000000):
    identity = {'recording_session_id': SESSION, 'sender_id': 'a', 'camera_id': 'cam01'}
    for name in ('meta.json', 'recording_ready.json'):
        atomic_json(folder/name, read_json(folder/name) | identity)
    (folder/'audio.opus').write_bytes(b'owned-encoded-audio-fixture-not-real-speech')
    atomic_json(folder/'audio_meta.json', identity | {'schema_version': 2, 'task_audio': True,
        'audio_valid': True, 'global_timestamp_source': 'sender_system_timestamp_us_plus_clock_offset',
        'first_audio_global_us': start+120000, 'last_audio_global_us': start+1160000,
        'audio_duration_us': 1060000, 'received_packets': 53, 'expected_packets': 53,
        'clock_sync_valid_packets': 53})
    atomic_json(folder/'audio_ready.json', identity | {'schema_version': 2, 'task_audio': True,
        'ready': True, 'audio_valid': True, 'received_packets': 53, 'expected_packets': 53})
    (folder/'audio_timing.csv').write_text('global_timestamp_us,window_start_global_us,window_end_global_us,received_packets,clock_sync_valid\n'
        f'{start},{start},{start+1000000},44,1\n'
        f'{start+1000000},{start+1000000},{start+2000000},9,1\n')
    ledger(folder)
    stable(folder)


def record(config):
    result = scan_recordings(config)
    assert result['errors'] == []
    item, = result['recordings']
    return item


class Models(FakeModels):
    def __init__(self):
        super().__init__()
        self.speech_calls = 0

    def transcribe(self, layout, retained, key):
        self.speech_calls += 1
        assert retained['audio']['status'] == 'provided'
        return {'outcome': 'no_speech', 'comments': [], 'artifacts': [], 'model_invocation': 'NOT_PROVEN'}


@pytest.mark.parametrize('text_audio_session', [False, True])
def test_native_integer_session_and_one_second_summary_clock_are_verified(config, text_audio_session):
    folder = incoming(config)
    native_audio(folder)
    if text_audio_session:
        for name in ('audio_meta.json', 'audio_ready.json'):
            atomic_json(folder/name, read_json(folder/name) | {'recording_session_id': str(SESSION)})
        ledger(folder)
    item = record(config)
    assert item['recording_session_id'] == str(SESSION)
    assert item['stt_processable'], item['stt_readiness_reasons']
    assert item['stt_clock_evidence'] == {'kind': 'recorder_task_audio_v2_window_summary',
        'rows': 2, 'received_packets': 53, 'empty_packet_windows': []}
    runner = DeviceDayRunner(config, Models())
    assert seal_input(runner, item, 'stt')['independent_audio_identity']['recording_session_id'] == str(SESSION)


@pytest.mark.parametrize('defect', ['unsynced', 'packet_count', 'window_gap', 'missing_window_field', 'unknown_schema'])
def test_native_window_clock_rejects_unverified_mapping(config, defect):
    folder = incoming(config)
    native_audio(folder)
    clock = folder/'audio_timing.csv'
    text = clock.read_text()
    if defect == 'unsynced':
        clock.write_text(text.replace(',44,1\n', ',44,0\n'))
    elif defect == 'packet_count':
        atomic_json(folder/'audio_meta.json', read_json(folder/'audio_meta.json') | {'received_packets': 52})
    elif defect == 'window_gap':
        rows = text.splitlines()
        cells = rows[-1].split(',')
        cells[:3] = [str(int(v)+1000000) for v in cells[:3]]
        clock.write_text('\n'.join([*rows[:-1], ','.join(cells)])+'\n')
    elif defect == 'missing_window_field':
        clock.write_text(text.replace('window_end_global_us', 'unknown_window_end'))
    else:
        atomic_json(folder/'audio_meta.json', read_json(folder/'audio_meta.json') | {'schema_version': 999})
    ledger(folder)
    item = record(config)
    assert item['processable'] and not item['stt_processable']
    assert item['stt_readiness_reasons'][0].startswith('independent_audio_window_clock_')


def window_clock(folder, packets, start, end):
    origin = 1789005600000000
    rows = ['global_timestamp_us,window_start_global_us,window_end_global_us,received_packets,clock_sync_valid']
    rows.extend(f'{origin+i*1000000},{origin+i*1000000},{origin+(i+1)*1000000},{count},{int(count > 0)}'
                for i, count in enumerate(packets))
    (folder/'audio_timing.csv').write_text('\n'.join(rows)+'\n')
    total = sum(packets)
    atomic_json(folder/'audio_meta.json', read_json(folder/'audio_meta.json') | {
        'first_audio_global_us': origin+start, 'last_audio_global_us': origin+end,
        'audio_duration_us': end-start, 'received_packets': total, 'clock_sync_valid_packets': total})
    atomic_json(folder/'audio_ready.json', read_json(folder/'audio_ready.json') | {'received_packets': total})
    ledger(folder)
    return origin


@pytest.mark.parametrize('packets,reason', [((0, 53), 'origin'), ((53, 0), 'end')])
def test_empty_edge_window_cannot_support_audio_endpoint(config, packets, reason):
    folder = incoming(config)
    native_audio(folder)
    window_clock(folder, packets, 120000, 1160000)
    item = record(config)
    assert item['processable'] and not item['stt_processable']
    assert item['stt_readiness_reasons'] == [f'independent_audio_window_clock_{reason}_mismatch']


def test_empty_edge_windows_are_preserved_when_endpoints_have_packets(config):
    folder = incoming(config)
    native_audio(folder)
    origin = window_clock(folder, (0, 53, 0), 1120000, 1160000)
    item = record(config)
    assert item['stt_processable'], item['stt_readiness_reasons']
    assert item['stt_clock_evidence']['empty_packet_windows'] == [
        {'start_us': origin, 'end_us': origin+1000000, 'reason': 'no_received_packets'},
        {'start_us': origin+2000000, 'end_us': origin+3000000, 'reason': 'no_received_packets'}]


def test_internal_empty_window_is_explicit_packet_coverage_gap(config):
    folder = incoming(config)
    native_audio(folder)
    origin = window_clock(folder, (44, 0, 9), 120000, 2160000)
    item = record(config)
    assert item['stt_processable'], item['stt_readiness_reasons']
    assert item['stt_clock_evidence']['empty_packet_windows'] == [
        {'start_us': origin+1000000, 'end_us': origin+2000000, 'reason': 'no_received_packets'}]


@pytest.mark.parametrize('defect', ['missing_session', 'different_device', 'bad_clock'])
def test_unverified_audio_never_blocks_visual_retention_and_late_audio_is_incremental(config, defect):
    config['device_day'].update(understanding_coverage_since_us=1, experiment_steps_since_us=1)
    folder = incoming(config)
    native_audio(folder)
    good_ready = read_json(folder/'audio_ready.json')
    good_meta = read_json(folder/'audio_meta.json')
    good_clock = (folder/'audio_timing.csv').read_text()
    if defect == 'missing_session':
        altered = dict(good_ready)
        altered.pop('recording_session_id')
        atomic_json(folder/'audio_ready.json', altered)
    elif defect == 'different_device':
        atomic_json(folder/'audio_meta.json', good_meta | {'camera_id': 'other_cam'})
    else:
        (folder/'audio_timing.csv').write_text('wrong_clock\n1\n')
    ledger(folder)
    item = record(config)
    assert item['processable'] and not item['stt_processable']
    if defect == 'missing_session':
        assert item['audio']['status'] == 'provided'
        assert item['stt_readiness_reasons'] == ['independent_audio_session_unverified']
    models = Models()
    runner = DeviceDayRunner(config, models)
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    assert runner.process(item, stage='vision')['status'] == 'completed'
    assert runner.process(item, stage='retention')['status'] == 'completed'
    assert runner.process(item, stage='understanding')['status'] == 'completed'
    assert runner.process(item, stage='stt')['status'] == 'waiting_for_capture'
    assert models.speech_calls == 0 and models.vision_calls == models.semantic_calls == 1
    retained = read_json(runner._receipt(runner.layout(item), item, 'retention'))
    assert retained['audio']['status'] == 'provided'
    assert retained['audio']['input_verification']['status'] == 'waiting_for_input_verification'
    assert retained['audio']['artifacts'] == [] and 'stt' not in retained['input_bindings']
    expected_sources = [{'kind': s['kind'], 'original_path': s['path'],
        'verification_status': 'unverified', 'ownership': 'collector'} for s in item['audio']['files']]
    assert retained['audio']['collector_sources'] == expected_sources
    public = publish_content(config, read_json(runner.layout(item).index))
    transcript = read_json(runner.layout(item).root/public['audio'][0]['content'])
    assert transcript['audio']['collector_sources'] == expected_sources
    assert transcript['status'] != 'completed'
    assert not input_path(runner, item, 'stt').exists()
    assert before == {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    semantic_path = runner._receipt(runner.layout(item), item, 'understanding')
    semantic_before = semantic_path.read_bytes()
    atomic_json(folder/'audio_meta.json', good_meta)
    atomic_json(folder/'audio_ready.json', good_ready)
    (folder/'audio_timing.csv').write_text(good_clock)
    ledger(folder)
    fresh = record(config)
    assert fresh['stt_processable']
    assert runner.process(fresh, stage='stt')['status'] == 'completed'
    assert runner.process(fresh, stage='retention')['status'] == 'completed'
    assert runner.process(fresh, stage='understanding')['status'] == 'completed'
    assert models.speech_calls == 1 and models.vision_calls == models.semantic_calls == 1
    assert semantic_path.read_bytes() == semantic_before
    final = read_json(runner._receipt(runner.layout(fresh), fresh, 'retention'))
    assert 'stt' in final['input_bindings']
    assert read_json(runner.layout(fresh).index)['recordings'][0]['transcription']['status'] == 'completed'


def test_audio_becoming_invalid_does_not_reuse_its_old_completed_transcription(config):
    config['device_day'].update(understanding_coverage_since_us=1, experiment_steps_since_us=1)
    folder = incoming(config)
    native_audio(folder)
    item = record(config)
    models = Models()
    runner = DeviceDayRunner(config, models)
    for stage in ('vision', 'stt', 'retention', 'understanding'):
        assert runner.process(item, stage=stage)['status'] == 'completed'
    prior_stt = runner._receipt(runner.layout(item), item, 'stt').read_bytes()
    (folder/'audio_timing.csv').write_text('wrong_clock\n1\n')
    ledger(folder)
    fresh = record(config)
    assert fresh['processable'] and not fresh['stt_processable']
    assert runner.process(fresh, stage='vision')['status'] == 'completed'
    assert runner.process(fresh, stage='retention')['status'] == 'completed'
    assert runner.process(fresh, stage='understanding')['status'] == 'completed'
    assert runner.process(fresh, stage='stt')['status'] == 'waiting_for_capture'
    assert models.vision_calls == models.speech_calls == models.semantic_calls == 1
    assert runner._receipt(runner.layout(fresh), fresh, 'stt').read_bytes() == prior_stt
    canonical = read_json(runner.layout(fresh).index)['recordings'][0]
    assert canonical['transcription'].get('status') != 'completed'
    assert canonical['audio']['input_verification']['status'] == 'waiting_for_input_verification'
