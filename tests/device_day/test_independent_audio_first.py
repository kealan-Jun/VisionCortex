"""Owned recorder-native audio can finish before an absent/unready video."""
from pathlib import Path
import os
import time

import pytest

from test_device_day import FakeModels, device_config
from test_recorder_metadata import incoming, audio, stable
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import artifact, atomic_json, read_json
from visioncortex.device_day_inplace import verified_stt_projection
from visioncortex.device_day_inputs import binding, input_path, processing_input, seal_input
from visioncortex.nas_recordings import scan_recordings
from visioncortex.publication_journal import PublicationJournal

__all__ = ['device_config']


@pytest.fixture
def audio_first(device_config):
    config = device_config
    config['storage']['archive_root'] = str(Path(config['collection_ingest']['source_root'])/'Archive')
    config['device_day'].update(inplace_preprocessing=True, raw_originals_read_only=True)
    folder = incoming(config)
    audio(folder)
    original = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    class SpeechModels(FakeModels):
        def __init__(self):
            super().__init__()
            self.speech_calls = 0

        def transcribe(self, layout, retained, key):
            self.speech_calls += 1
            assert all(source['kind'].startswith('audio_') for source in retained['sources'])
            origin = retained['audio']['start_us']
            target = layout.comments/'.Evidence'/'AudioFirstFixture'/f'{key}.txt'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('先校准仪器。', encoding='utf-8')
            audio_ref = next(s['retained'] for s in retained['sources'] if s['kind'] == 'audio_audio')
            return {'outcome': 'transcribed', 'comments': [{'comment_id': 'speech-test', 'text': '先校准仪器。',
                'start_us': origin, 'end_us': origin + 1000000, 'audio_ref': audio_ref,
                'source': 'machine_transcribed_speech', 'transcript_path': layout.relative(target)}],
                'artifacts': [artifact(layout.root, target)], 'transcript_file': artifact(layout.root, target),
                'model_invocation': 'NOT_PROVEN'}
    models = SpeechModels()
    runner = DeviceDayRunner(config, models)
    return config, folder, original, models, runner


def inspect(config):
    result = scan_recordings(config)
    assert result['errors'] == []
    record, = result['recordings']
    return record


def stable_video(folder):
    for path in folder.iterdir():
        if path.is_file() and not path.name.startswith('audio'):
            os.utime(path, (time.time() - 1000, time.time() - 1000))


@pytest.mark.parametrize('video_state', ['writing', 'missing_video', 'missing_clock'])
def test_ready_audio_publishes_actual_text_before_visual_and_late_video_reuses_receipt(audio_first, video_state):
    config, folder, original, models, runner = audio_first
    if video_state == 'writing':
        atomic_json(folder/'meta.json', read_json(folder/'meta.json') | {'closed': False, 'frames_publish_state': 'writing'})
        atomic_json(folder/'recording_ready.json', read_json(folder/'recording_ready.json') | {'ready': False})
    else:
        (folder/('rgb.mp4' if video_state == 'missing_video' else 'frames.csv')).unlink()
    stable(folder)
    if video_state == 'writing':
        os.utime(folder/'rgb.mp4', None)  # Audio/identity sidecars remain stable.
    record = inspect(config)
    assert record['audio']['status'] == 'provided' and record['stt_processable']
    assert not record['processable'] and record['audio']['association_issues'] == []
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    result = runner.process(record, stage='stt')
    assert result['status'] == 'completed' and result['formal_publication'] == 'pending'
    assert models.speech_calls == 1 and models.vision_calls == models.semantic_calls == 0
    layout = runner.layout(record)
    stage_path = runner._receipt(layout, record, 'stt')
    saved_bytes = stage_path.read_bytes()
    assert not runner._receipt(layout, record, 'retention').exists()
    assert not runner._receipt(layout, record, 'vision').exists()
    public = read_json(runner._receipt(layout, record, 'stt-publication'))
    assert public['status'] == 'completed' and public['retention_status'] == public['vision_status'] == 'not_granted'
    text = next(ref for ref in public['artifacts'] if ref['path'].endswith('.txt'))
    assert '先校准仪器。' in (layout.root/text['path']).read_text()
    assert verified_stt_projection(runner, record)['transcription']['comments'][0]['text'] == '先校准仪器。'
    assert before == {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    assert PublicationJournal(runner.runtime_root).pending()
    old_seal = read_json(input_path(runner, record, 'stt'))
    old_binding = binding(old_seal)
    for name, content in original.items():
        if not name.startswith('audio'):
            (folder/name).write_bytes(content)
    stable_video(folder)
    fresh = inspect(config)
    assert fresh['recording_id'] == record['recording_id'] and fresh['processable']
    assert fresh['audio']['source_signature'] == record['audio']['source_signature']
    assert runner.process(fresh, stage='stt')['status'] == 'completed'
    assert models.speech_calls == 1 and stage_path.read_bytes() == saved_bytes
    assert binding(seal_input(runner, fresh, 'stt')) == old_binding
    assert runner._key('stt', fresh, processing_input(layout, old_seal)) == read_json(stage_path)['key']
    assert runner.process(fresh, stage='retention')['status'] == 'completed'
    assert runner.process(fresh, stage='vision')['status'] == 'completed'
    assert models.speech_calls == 1
    index = read_json(layout.index)
    assert index['recordings'][0]['transcription']['comments'][0]['text'] == '先校准仪器。'


@pytest.mark.parametrize('defect', ['session', 'clock', 'not_ready', 'no_valid', 'time', 'camera'])
def test_unverified_independent_audio_stays_waiting(audio_first, defect):
    config, folder, _, models, runner = audio_first
    (folder/'rgb.mp4').unlink()
    if defect == 'clock':
        (folder/'audio_timing.csv').write_text('wrong_clock\n1\n')
    else:
        path = folder/('audio_ready.json' if defect in {'not_ready', 'no_valid'} else 'audio_meta.json')
        changes = {'session': {'recording_session_id': 'another-capture'}, 'not_ready': {'ready': False},
            'no_valid': {'audio_valid': False}, 'time': {'last_audio_global_us': 0}, 'camera': {'camera_key': 'other_cam'}}[defect]
        atomic_json(path, read_json(path) | changes)
    stable(folder)
    record = inspect(config)
    assert not record['stt_processable']
    assert runner.process(record, stage='stt')['status'] == 'waiting_for_capture'
    assert models.speech_calls == 0 and not runner._receipt(runner.layout(record), record, 'stt').exists()


def test_stt_identity_excludes_later_video_role_and_full_video_clock_window(audio_first):
    config, folder, original, models, runner = audio_first
    config['collection_ingest']['camera_role_map'] = {}
    metadata = read_json(folder/'meta.json') | {'closed': False, 'frames_publish_state': 'writing',
        'recording_window_start_global_us': 0, 'recording_window_end_global_us': 0}
    ready = read_json(folder/'recording_ready.json') | {'ready': False,
        'recording_window_start_global_us': 0, 'recording_window_end_global_us': 0}
    atomic_json(folder/'meta.json', metadata)
    atomic_json(folder/'recording_ready.json', ready)
    (folder/'frames.csv').unlink()
    stable(folder)
    record = inspect(config)
    assert record['stt_processable'] and not record.get('configured_role')
    assert runner.process(record, stage='stt')['status'] == 'completed'
    path = runner._receipt(runner.layout(record), record, 'stt')
    old = path.read_bytes()
    layout = runner.layout(record)
    public = read_json(runner._receipt(layout, record, 'stt-publication'))
    public_paths = {ref['path'] for ref in public['artifacts']}
    json_path = next(p for p in public_paths if p.endswith('.json'))
    content = read_json(layout.root/json_path)
    assert (content['start_us'], content['end_us']) == (record['audio']['start_us'], record['audio']['end_us'])
    for name, content in original.items():
        if not name.startswith('audio'):
            (folder/name).write_bytes(content)
    config['collection_ingest']['camera_role_map'] = {'a_cam01': 'first_person'}
    stable_video(folder)
    fresh = inspect(config)
    assert fresh['recording_end_us'] != record['recording_end_us']
    assert runner.process(fresh, stage='stt')['status'] == 'completed'
    assert models.speech_calls == 1 and path.read_bytes() == old
    assert {ref['path'] for ref in read_json(runner._receipt(layout, fresh, 'stt-publication'))['artifacts']} == public_paths
    assert runner.process(fresh, stage='retention')['status'] == 'completed'
    assert runner.process(fresh, stage='vision')['status'] == 'completed'
    from visioncortex.device_day_content import publish_content
    index = read_json(layout.index)
    assert index['recordings'][0]['start_us'] == fresh['recording_start_us']
    assert index['recordings'][0]['end_us'] == fresh['recording_end_us']
    assert index['recordings'][0]['transcript_span'] == verified_stt_projection(runner, fresh)['transcript_span']
    readable = publish_content(config, index)
    assert {entry[key] for entry in readable['audio'] for key in ('content', 'text')} == public_paths
    assert {p.relative_to(layout.root).as_posix() for p in layout.comments.glob('*/Transcript.*')} == public_paths
    from visioncortex.device_day_transcript_publication import publish_one
    assert publish_one(runner, fresh)['content'] == json_path
    assert models.speech_calls == 1


def test_legacy_transcript_path_keeps_its_original_video_window(audio_first):
    from visioncortex.device_day_content import publish_transcript, time_folder
    config, _, _, _, runner = audio_first
    record = inspect(config)
    layout = runner.layout(record)
    value = {'recording_id': record['recording_id'], 'start_us': record['recording_start_us'],
        'end_us': record['recording_end_us'], 'audio': record['audio'],
        'transcription': {'status': 'completed', 'outcome': 'transcribed', 'comments': [],
                          'model_invocation': 'NOT_PROVEN'}}
    published = publish_transcript(config, layout.name, value)
    assert published['content'] == 'Comment/' + time_folder(value['start_us'], value['end_us']) + '/Transcript.json'
    assert 'transcript_span' not in read_json(layout.root/published['content'])


@pytest.mark.parametrize('files', [{}, {'audio.opus': {'size': 16, 'sha256': '0'*64}}])
def test_declared_producer_receipt_is_never_bypassed_for_independent_audio(audio_first, files):
    config, folder, _, models, runner = audio_first
    (folder/'rgb.mp4').unlink()
    target = folder/'audio_ready.json'
    atomic_json(target, read_json(target) | {'files': files})
    stable(folder)
    record = inspect(config)
    assert record['stt_processable']
    result = runner.process(record, stage='stt')
    assert result['status'] == 'failed'
    assert 'producer publication receipt' in result['message']
    assert models.speech_calls == 0
    assert not runner._receipt(runner.layout(record), record, 'stt-publication').exists()
