"""Synthetic/native timestamp and fake transport contracts; no real model calls."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import numpy as np

from test_device_day_understanding import source
from visioncortex.device_day_contract import artifact, read_json
from visioncortex.device_day_coverage import request_plan, shadow_episodes, value_assessment
from visioncortex.device_day_understanding import (
    UnderstandingCheckpointYield, UnderstandingPlanPaused, frame_windows, understand,
)
from visioncortex.mllm import SCENE_TIME_REFERENCE_RULES


def segment(*, identifier='segment-fixture', activity='inactive', start=0, end=900200):
    origin = 1791936000000000
    return {'segment_id': identifier, 'recording_id': 'record-fixture', 'camera_key': 'fixture-camera',
            'activity': activity, 'start_ms': start, 'end_ms': end,
            'start_us': origin + round(start * 1000), 'end_us': origin + round(end * 1000),
            'source_ref': {'path': 'MetaData/source.avi', 'sha256': 'a'*64, 'size_bytes': 100},
            'key_frames': [], 'scene_frames': [], 'visibility': 'sampled_images_available'}


def test_long_inactive_plan_holds_without_claiming_stability_or_removing_the_interval():
    inactive = segment()
    plan = request_plan([inactive], {}, {})
    assert plan['holds'][0]['reason'] == 'anomalous_inactive_request_plan'
    assert plan['segments'][0]['estimated_requests_upper_bound'] >= 31
    assert plan['segments'][0]['end_ms'] == 900200
    assert plan['estimated_requests_upper_bound'] >= 31
    assert plan['human_semantic_recall'] == 'NOT_PROVEN'
    assert plan['segments'][0]['estimated_cost'] is None


@pytest.mark.parametrize('signal', ['action', 'waiting', 'occlusion', 'ocr', 'audio', 'binding'])
def test_protected_inactive_intervals_are_not_throttled_by_a_binary_activity_label(signal):
    row, context = segment(), {}
    if signal == 'action':
        row['key_frames'] = [{'local_ms': 1234.5}]
    elif signal == 'waiting':
        row['activity_semantics'] = 'experiment_waiting'
    elif signal == 'occlusion':
        row['visibility'] = 'dark_or_obscured'
    else:
        field = {'ocr': 'ocr_events', 'audio': 'audio_events', 'binding': 'bindings'}[signal]
        context[field] = [{'start_us': row['start_us'], 'end_us': row['end_us']}]
    assert not request_plan([row], context, {})['holds']
    assert value_assessment(row, [row], context)['protection_signals']
    assert not shadow_episodes([row], context)[0]['paid_filter_applied']


def test_waiting_between_candidates_is_protected_and_never_called_confirmed_stable():
    before = segment(identifier='before', activity='active', end=1000)
    waiting = segment(identifier='waiting', start=1000, end=901200)
    after = segment(identifier='after', activity='active', start=901200, end=902200)
    rows = [before, waiting, after]
    assert not request_plan(rows, {}, {})['holds']
    assert value_assessment(waiting, rows, {})['activity_semantics'] == 'experiment_waiting'
    assert not any(e['confirmed_stable'] for e in shadow_episodes(rows, {}))


@pytest.mark.parametrize('position,protected', [('start', True), ('inside', True), ('end', False), ('before', False)])
def test_point_context_event_uses_half_open_segment_time(position, protected):
    row = segment()
    stamp = {'start': row['start_us'], 'inside': row['start_us'] + 1,
             'end': row['end_us'], 'before': row['start_us'] - 1}[position]
    context = {'events': [{'type': 'ocr', 'start_us': stamp, 'end_us': stamp}]}
    assert bool(value_assessment(row, [row], context)['protection_signals']) is protected
    assert bool(request_plan([row], context, {})['holds']) is (not protected)


def test_shadow_episode_never_merges_unknown_identity_or_a_time_gap():
    left = segment(identifier='left', end=1000)
    right = segment(identifier='right', start=1000, end=2000)
    for row in (left, right):
        row['scene_frames'] = [{'local_ms': row['start_ms'], 'sha256': 'b'*64},
                               {'local_ms': row['end_ms'] - 1, 'sha256': 'b'*64}]
    assert len(shadow_episodes([left, right], {})) == 2
    left['capture_session_id'] = right['capture_session_id'] = 'session-fixture'
    episodes = shadow_episodes([left, right], {})
    assert len(episodes) == 1 and len(episodes[0]['source_intervals']) == 2
    assert episodes[0]['full_interval_checked'] is False
    assert episodes[0]['quality'] == 'NOT_PROVEN'
    right['start_us'] += 1
    assert len(shadow_episodes([left, right], {})) == 2


def test_48_frame_capacity_preserves_4fps_baseline_tail_and_boundary_action(tmp_path):
    video = source(tmp_path, 30)
    details = {}
    groups = list(frame_windows(video, 0, 30000, True, {}, required_times=[11800], details=details))
    all_frames = [row for _, _, frames in groups for row in frames]
    times = {round(row[1]) for row in all_frames}
    assert all(len(frames) <= 48 for _, _, frames in groups)
    assert len(groups) >= 3  # Never send only the first 48 or dilute 4 FPS over 30s.
    assert 0 in times and 29900 in times and 11800 in times
    baseline = {round((i * 250 + 99) // 100 * 100) for i in range(120)}
    assert baseline <= times
    assert all(t in times for t in range(10800, 12801, 100))
    assert any(a[1] > b[0] for a, b in zip(groups, groups[1:], strict=False))
    assert 'required_action_frame' in details[118]['selection_reasons']
    assert all(d['source_pts'] is not None for d in details.values())


def test_irregular_native_pts_are_selected_without_synthesizing_fps_grid_timestamps(tmp_path, monkeypatch):
    import cv2
    from visioncortex import source_frames
    video = tmp_path/'irregular.fixture'
    video.write_bytes(b'fake decoder boundary, not a real media quality sample')
    pts = [0, 17, 83, 257, 399, 504, 712, 1003]
    trace = SimpleNamespace(failure=None, time_base='1/1000', origin_ms=0, stat=video.stat(),
                            positions={i: [(stamp, i)] for i, stamp in enumerate(pts)})
    monkeypatch.setattr(source_frames, 'SourceFrameTrace', lambda *args: trace)
    class Capture:
        index = 0
        def isOpened(self):
            return True
        def get(self, key):
            return len(pts) if key == cv2.CAP_PROP_FRAME_COUNT else 0
        def read(self):
            if self.index == len(pts):
                return False, None
            self.index += 1
            return True, np.full((24, 32, 3), self.index, np.uint8)
        def release(self):
            pass
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: Capture())
    details = {}
    groups = list(frame_windows(video, 0, 1100, True,
        {'understanding_frames_per_request': 4, 'understanding_overlap_frames': 1, 'understanding_event_radius_ms': 0},
        required_times=[399], details=details))
    selected = {stamp for _, _, frames in groups for _, stamp, _ in frames}
    assert selected == {0, 257, 399, 504, 1003}
    assert all(len(frames) <= 4 for _, _, frames in groups)
    assert details[4]['source_pts'] == 399 and details[4]['source_time_base'] == '1/1000'


@pytest.fixture
def fake_understanding(default_config, tmp_path, monkeypatch):
    from visioncortex import mllm
    config = deepcopy(default_config)
    config['storage'].update(local_runtime_root=str(tmp_path/'runtime'), local_cache_root=str(tmp_path/'cache'))
    config['device_day'].update(understanding_requests_per_turn=1)
    config['mllm'].update(model='synthetic-scene', provider='fixture', api_protocol='chat_completions')
    root = tmp_path/'archive'
    root.mkdir()
    layout = SimpleNamespace(root=root, understanding=root/'MultimodalUnderstanding',
                             relative=lambda path: str(path.relative_to(root)))
    recorder = {'recording_id': 'record-fixture', 'camera_key': 'fixture-camera',
                'recording_start_us': 1791936000000000}
    calls = []
    class Analyzer:
        opened = 0
        def __init__(self, config):
            type(self).opened += 1
        def _call(self, prompt, metadata, images, **kwargs):
            calls.append(deepcopy(metadata))
            return {'status': 'completed', 'model': 'synthetic-scene', 'provider': 'fixture',
                    'summary': 'Synthetic replay; no real-video semantic evidence', 'activity_observed': 'uncertain',
                    'steps': [], 'frame_observations': [{'frame_id': f['frame_id'], 'text': 'synthetic frame'}
                        for f in metadata['frames']], 'uncertainties': ['Human recall NOT_PROVEN'],
                    'scene_timing_contract': {'version': 'visioncortex-scene-time-references/1',
                        'instructions': SCENE_TIME_REFERENCE_RULES, 'validation_relaxed': False},
                    'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30, 'cached_input_tokens': 0}}
        def close(self):
            pass
    monkeypatch.setattr(mllm, 'ArkAnalyzer', Analyzer)
    backend = SimpleNamespace(config=config, settings=config['device_day'])
    return backend, layout, recorder, Analyzer, calls


def test_900_second_inactive_entry_records_hold_before_decoder_or_paid_transport(fake_understanding):
    backend, layout, recording, analyzer, calls = fake_understanding
    vision = {'key': 'vision-fixture', 'segments': [segment()], 'media_coverage': {'native_frame_count': 9000}}
    with pytest.raises(UnderstandingPlanPaused) as paused:
        understand(backend, layout, recording, vision, {}, 'a'*64)
    audit = read_json(layout.root/paused.value.plan_ref)
    assert calls == [] and analyzer.opened == 0
    assert audit['status'] == 'held_for_plan_review' and audit['requests'] == []
    assert audit['source_intervals'][0]['end_ms'] == 900200
    assert audit['human_semantic_recall'] == 'NOT_PROVEN'
    assert audit['capture_coverage'] == vision['media_coverage']


def test_saved_request_quantum_resumes_without_rebilling_and_records_actual_pts(fake_understanding, tmp_path):
    from visioncortex import scene_requests
    from visioncortex.multimodal_usage import UsageLedger
    from visioncortex.sqlite_store import connection
    backend, layout, recording, _, calls = fake_understanding
    video = source(layout.root, 15)
    row = segment(activity='active', end=15000)
    row['source_ref'] = artifact(layout.root, video)
    vision = {'key': 'vision-fixture', 'segments': [row]}
    with pytest.raises(UnderstandingCheckpointYield) as checkpoint:
        understand(backend, layout, recording, vision, {}, 'a'*64)
    audit = read_json(layout.root/checkpoint.value.plan_ref)
    assert checkpoint.value.completed_new_requests == 1 and len(calls) == 1
    partial = checkpoint.value.partial_receipt
    assert partial['vision_key'] == vision['key'] and partial['content_source'] == 'video_only'
    assert partial['understandings'][0]['status'] == 'partial'
    assert len(partial['understandings'][0]['windows']) == 1 and partial['artifacts']
    assert audit['requests'][0]['status'] == 'validated_return'
    assert audit['status'] == 'scheduled_checkpoint'
    request = audit['requests'][0]
    durable_input = read_json(layout.root/request['input'])
    assert request['request_key'] == scene_requests.request_key(backend.config, durable_input['prompt'], durable_input['metadata'])
    with connection(UsageLedger(backend.config).path, readonly=True) as db:
        assert db.execute('SELECT input_key FROM calls').fetchone()[0] == request['request_key']
    final = understand(backend, layout, recording, vision, {'comments': [{'text': 'late unused speech'}]}, 'a'*64)
    assert len(calls) == 2
    assert final['context_digest'] is None and final['content_source'] == 'video_only'
    windows = final['understandings'][0]['windows']
    assert windows[0]['response_cache_reused'] and not windows[-1]['response_cache_reused']
    audit = read_json(layout.root/final['sampling_coverage_manifest'])
    assert all(r['status'] == 'validated_return' and r['result_identity'] for r in audit['requests'])
    assert all(r['coverage']['native_pts_verified'] for r in audit['requests'])
    assert audit['human_semantic_recall'] == 'NOT_PROVEN'
    assert audit['actual_request_coverage'] == 'validated_selected_frames_only'
    understand(backend, layout, recording, vision, {}, 'a'*64)
    assert len(calls) == 2  # Reused windows do not consume a paid-work quantum.


def test_explicit_zero_quantum_completes_all_bounded_requests(fake_understanding):
    backend, layout, recording, _, calls = fake_understanding
    backend.settings['understanding_requests_per_turn'] = 0
    row = segment(activity='active', end=15000)
    row['source_ref'] = artifact(layout.root, source(layout.root, 15))
    result = understand(backend, layout, recording, {'key': 'vision-fixture', 'segments': [row]}, {}, 'a'*64)
    assert len(calls) == 2 and len(result['understandings'][0]['windows']) == 2


@pytest.mark.parametrize('active_seconds', [5, 15])
def test_long_inactive_tail_cannot_block_active_windows_and_the_recording_stays_partial(fake_understanding, active_seconds):
    from visioncortex.device_day_contract import verify_artifact
    backend, layout, recording, _, calls = fake_understanding
    # Owned synthetic media has enough duration for the whole supplied CV
    # interval; the held tail must never be decoded or submitted to the model.
    video = source(layout.root, active_seconds + 901)
    ref = artifact(layout.root, video)
    active = segment(identifier='active-evidence', activity='active', end=active_seconds * 1000)
    tail = segment(identifier='inactive-tail', start=active_seconds * 1000, end=active_seconds * 1000 + 900200)
    active['source_ref'] = tail['source_ref'] = ref
    vision = {'key': 'vision-fixture', 'segments': [tail, active]}
    if active_seconds == 15:
        with pytest.raises(UnderstandingCheckpointYield) as checkpoint:
            understand(backend, layout, recording, vision, {}, 'a'*64)
        assert checkpoint.value.partial_receipt['held_segment_ids'] == ['inactive-tail']
        assert len(checkpoint.value.partial_receipt['understandings'][0]['windows']) == 1
    with pytest.raises(UnderstandingPlanPaused) as held:
        understand(backend, layout, recording, vision, {}, 'a'*64)
    partial = held.value.partial_receipt
    assert partial['held_segment_ids'] == ['inactive-tail']
    assert [meaning['segment_id'] for meaning in partial['understandings']] == ['active-evidence']
    assert len(partial['understandings'][0]['windows']) == len(calls) == (1 if active_seconds == 5 else 2)
    assert all(call['segment_id'] == 'active-evidence' for call in calls)
    assert all(verify_artifact(layout.root, item) for item in partial['artifacts'])
    audit = read_json(layout.root/held.value.plan_ref)
    assert audit['status'] == 'waiting_for_plan_review'
    assert audit['actual_request_coverage'] == 'validated_selected_frames_partial'
    assert audit['request_plan']['holds'][0]['segment_id'] == 'inactive-tail'
    with pytest.raises(UnderstandingPlanPaused):
        understand(backend, layout, recording, vision, {}, 'a'*64)
    assert len(calls) == (1 if active_seconds == 5 else 2)  # Valid active receipts remain reusable.


def test_checkpoint_resume_reuses_images_before_decoder_encoder_and_publication(fake_understanding, monkeypatch, record_property):
    import cv2
    from visioncortex import device_day_understanding as execution, device_day_window_inputs as inputs
    backend, layout, recording, _, calls = fake_understanding
    row = segment(activity='active', end=15000)
    row['source_ref'] = artifact(layout.root, source(layout.root, 15))
    vision = {'key': 'vision-fixture', 'segments': [row]}
    counts = {'decode': 0, 'encode': 0, 'image_write': 0}
    decode, encode, write = inputs.decode_selected, cv2.imencode, execution.atomic_bytes
    def counted_decode(*args):
        frames = decode(*args)
        counts['decode'] += frames.decoded_frames
        return frames
    def counted_encode(*args):
        counts['encode'] += 1
        return encode(*args)
    def counted_write(*args):
        counts['image_write'] += 1
        return write(*args)
    monkeypatch.setattr(inputs, 'decode_selected', counted_decode)
    monkeypatch.setattr(cv2, 'imencode', counted_encode)
    monkeypatch.setattr(execution, 'atomic_bytes', counted_write)
    # Count the old decoder too: the pre-fix implementation has no inputs port.
    capture = cv2.VideoCapture
    class CountedCapture:
        def __init__(self, *args):
            self.delegate = capture(*args)
        def read(self):
            result = self.delegate.read()
            counts['decode'] += int(result[0])
            return result
        def __getattr__(self, name):
            return getattr(self.delegate, name)
    monkeypatch.setattr(cv2, 'VideoCapture', CountedCapture)
    with pytest.raises(UnderstandingCheckpointYield):
        understand(backend, layout, recording, vision, {}, 'a'*64)
    first = dict(counts)
    final = understand(backend, layout, recording, vision, {}, 'a'*64)
    resume = {name: counts[name] - first[name] for name in counts}
    uncached_frames = sum(len(w['source_frames']) for w in final['understandings'][0]['windows']
                          if not w['response_cache_reused'])
    assert resume['encode'] == resume['image_write'] == uncached_frames
    assert uncached_frames <= resume['decode'] < 150  # Includes measured seek preroll, no source-origin replay.
    before = dict(counts)
    understand(backend, layout, recording, vision, {}, 'a'*64)
    fully_cached = {name: counts[name] - before[name] for name in counts}
    assert fully_cached == dict.fromkeys(counts, 0)
    # A changed stage folder also preserves exact original window inputs.
    understand(backend, layout, recording, vision, {}, 'b'*64)
    across_stage = {name: counts[name] - before[name] for name in counts}
    assert across_stage == dict.fromkeys(counts, 0)
    assert len(calls) == 2
    record_property('io_counts', {'first': first, 'resume': resume, 'fully_cached': fully_cached,
                                 'across_stage': across_stage, 'fake_model_calls': len(calls), 'real_model_calls': 0})


def test_source_bound_bounded_review_resolves_the_plan_without_claiming_filter_quality(fake_understanding, tmp_path):
    import hashlib
    import json
    from visioncortex.device_day_contract import digest
    backend, layout, recording, _, calls = fake_understanding
    row = segment(end=15000)
    row['source_ref'] = artifact(layout.root, source(layout.root, 15))
    backend.settings.update(understanding_inactive_window_limit=1, understanding_window_seconds=10)
    recording['source_signature'] = 'source-fixture'
    plan = request_plan([row], {}, backend.settings)
    assert plan['holds']
    estimate = plan['segments'][0]
    review = {'recording_id': recording['recording_id'], 'source_signature': recording['source_signature'],
              'segment_id': row['segment_id'], 'vision_key': 'vision-fixture',
              'plan_identity': digest({k: estimate[k] for k in ('segment_id', 'source_ref', 'start_ms', 'end_ms',
                  'estimated_requests_upper_bound')} | {'policy': plan['policy_version']}),
              'action': 'execute_bounded', 'paid_authorized': True, 'maximum_requests': estimate['estimated_requests_upper_bound'],
              'reviewed_by': 'synthetic authorization fixture', 'evidence': 'mock only; no production approval'}
    path = tmp_path/'review.json'
    def save():
        path.write_text(json.dumps({'schema_version': 'visioncortex-plan-reviews/1', 'entries': [review]}))
        backend.settings['understanding_plan_reviews'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save()
    vision = {'key': 'vision-fixture', 'segments': [row]}
    with pytest.raises(UnderstandingCheckpointYield):
        understand(backend, layout, recording, vision, {}, 'a'*64)
    final = understand(backend, layout, recording, vision, {}, 'a'*64)
    assert final['understandings'][0]['status'] == 'completed' and len(calls) == 2
    audit = read_json(layout.root/final['sampling_coverage_manifest'])
    assert audit['request_plan']['holds'] == []
    assert audit['human_semantic_recall'] == 'NOT_PROVEN'
    assert all(e['paid_filter_applied'] is False for e in audit['shadow_episodes'])
    review['source_signature'] = 'different-source'
    save()
    with pytest.raises(ValueError, match='identity changed'):
        understand(backend, layout, recording, vision, {}, 'a'*64)
    assert len(calls) == 2


def test_window_pixels_match_independent_sequential_decode_with_vfr_b_frames_and_nonzero_origin(tmp_path):
    import cv2
    import subprocess
    from visioncortex.device_day_window_inputs import selections, decode_selected
    from visioncortex.video_io import _ffmpeg_passthrough_arguments
    video = tmp_path/'variable.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=96x64:r=30:d=2',
        '-vf', 'select=not(eq(mod(n\\,5)\\,1)),setpts=PTS+4/TB', *_ffmpeg_passthrough_arguments(),
        '-c:v', 'libx264', '-bf', '3', str(video)], check=True, capture_output=True)
    row = segment(activity='active', start=375, end=1400)
    row['source_ref'] = artifact(tmp_path, video)
    windows = selections(video, row, {}, {'recording_id': 'record-fixture', 'source_signature': 'fixture'},
                         'visual-fixture', tmp_path/'selections.json')
    reader = cv2.VideoCapture(str(video))
    originals = []
    try:
        while True:
            ok, pixels = reader.read()
            if not ok:
                break
            originals.append(pixels)
    finally:
        reader.release()
    for window in windows:
        frames = decode_selected(video, window)
        assert frames.decoded_frames >= len(frames)
        for ordinal, _stamp, pixels, _ in frames:
            assert np.array_equal(pixels, originals[ordinal])
    assert all(f['source_pts'] * float(__import__('fractions').Fraction(f['source_time_base'])) >= 4
               for w in windows for f in w['frames'])
