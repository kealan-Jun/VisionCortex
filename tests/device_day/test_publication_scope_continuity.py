"""A one-record replay publishes the complete day without restarting models."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout
from test_readable_all_day import add_work_evidence
from test_retained_video_subintervals import exact_case
from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import artifact, atomic_json, digest, read_json
from visioncortex.device_day_content_paths import analysis_folder, private_folder
from visioncortex.device_day_file_index import FileIndexPublisher
from visioncortex.multimodal_usage import SubmittedUnknown, UsageLedger
from visioncortex.nas_recordings import scan_recordings
from visioncortex.observed_inventory import observe

__all__ = ['device_config', 'exact_case']


def test_record_scope_limits_new_exports_but_reuses_verified_sibling_material(exact_case):
    case = exact_case
    result = publish_selected(case.config, case.index, recording_id='another-recording',
                              evidence_validator=case.validator)
    assert result['materialized'] == 0 and case.calls == []
    assert len(case.candidate['retention_intervals']) == 5
    assert len(result['selected_intervals']) == 2
    assert all(not row.get('video') for row in result['selected_intervals'])
    publish_selected(case.config, case.index, recording_id=case.record['recording_id'],
                     evidence_validator=case.validator)
    published = deepcopy(next(row for row in case.candidate['retention_intervals'] if row.get('video')))
    case.candidate.pop('retention_intervals')  # The canonical renderer starts from the CV receipt.
    result = publish_selected(case.config, case.index, recording_id='another-recording',
                              evidence_validator=lambda *a: pytest.fail('No new evidence/export attempt'))
    assert result['materialized'] == 0 and len(case.calls) == 1
    assert next(row for row in case.candidate['retention_intervals'] if row.get('video')) == published


@pytest.mark.parametrize('readable', [False, True])
@pytest.mark.parametrize('later_status', ['pending', 'failed'])
def test_sibling_publication_preserves_completed_exact_video_and_whole_day_partition(
        device_config, monkeypatch, readable, later_status):
    capture(device_config)
    first, layout = item_and_layout(device_config)
    device_config['device_day'].update(derived_video_retention=True,
                                      readable_content_enabled=readable)

    class WorkModels(FakeModels):
        def understand(self, layout, recording, vision, context, key):
            if recording['recording_id'] != first['recording_id']:
                self.semantic_calls += 1
                raise ValueError('Deterministic later source failure')
            result = super().understand(layout, recording, vision, context, key)
            meaning = result['understandings'][0]
            evidence = add_work_evidence(layout, meaning, key, vision['segments'][0])
            path = layout.understanding/'ClipUnderstanding'/f'{key}.json'
            atomic_json(path, meaning)
            result['artifacts'] = [artifact(layout.root, path), *evidence]
            return result

    exports = []

    def extract(source, target, start, duration, *_args):
        exports.append((source, start, duration))
        target.write_bytes(b'owned exact interval in deterministic fixture')

    monkeypatch.setattr('visioncortex.video_io.extract_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video',
                        lambda _: SimpleNamespace(duration_ms=1000))
    models = WorkModels()
    runner = DeviceDayRunner(device_config, models)
    assert runner.process(first)['status'] == 'completed'
    previous = read_json(layout.index)
    candidate = previous['segments'][0]
    selected, = [row for row in candidate['retention_intervals'] if row.get('video')]
    saved = deepcopy(selected)
    # Simulate a surviving, verified historical whole-candidate publication
    # whose pointer is absent from the native CV/report receipts. It remains a
    # historical reference, never an additional exact selected interval.
    historical = layout.processed/'HistoricalWhole.mp4'
    historical.write_bytes(b'owned historical whole-candidate fixture')
    whole_video = artifact(layout.root, historical)
    whole_receipt = private_folder(layout, 'ProcessedClips', 'Retention', candidate['segment_id'], 'Historical.json')
    atomic_json(whole_receipt, {key: candidate[key] for key in (
        'source_ref', 'start_us', 'end_us')} | {'source_start_ms': candidate['start_ms'],
        'source_end_ms': candidate['end_ms'], 'video': whole_video})
    whole_pointer = whole_receipt.with_name('Published.json')
    atomic_json(whole_pointer, {'receipt': artifact(layout.root, whole_receipt)})
    protected = {layout.root/selected[key]['path']: (layout.root/selected[key]['path']).read_bytes()
                 for key in ('video', 'receipt')}
    protected.update({path: path.read_bytes() for path in (historical, whole_receipt, whole_pointer)})
    if readable:
        path = layout.root/selected['understanding_path']
        protected[path] = path.read_bytes()
        for entry in previous['readable_content']['audio']:
            for field in ('content', 'text'):
                path = layout.root/entry[field]
                protected[path] = path.read_bytes()
    original_receipts = {stage: runner._receipt(layout, first, stage).read_bytes()
                         for stage in ('retention', 'vision', 'stt', 'understanding', 'report')}

    capture(device_config, start=first['recording_end_us'], label='101500')
    second, = [row for row in scan_recordings(device_config)['recordings']
               if row['recording_id'] != first['recording_id']]
    for stage in ('retention', 'vision', 'stt'):
        assert runner.process(second, stage=stage)['status'] == 'completed'
    if later_status == 'failed':
        assert runner.process(second, stage='understanding')['status'] == 'failed'
    calls = models.vision_calls, models.semantic_calls, len(exports)
    monkeypatch.setattr(runner, '_backend', lambda: pytest.fail('Publication must not start models'))
    assert runner.refresh_index(layout, recording_id=second['recording_id'])

    current = read_json(layout.index)
    assert len(current['recordings']) == len(current['segments']) == 2
    parent = next(row for row in current['segments'] if row['recording_id'] == first['recording_id'])
    assert parent['video'] == whole_video
    assert parent['historical_scope'] == {'source_start_ms': candidate['start_ms'],
        'source_end_ms': candidate['end_ms'], 'video': whole_video,
        'selection_status': 'historical_retained_not_revalidated_for_exact_selection'}
    retained, = [row for row in parent['retention_intervals'] if row.get('video')]
    assert retained == saved
    assert retained['video'] != whole_video
    assert all(row.get('retention_intervals') for row in current['segments'])
    time_index = read_json(layout.comments/'TimeIndex.json')
    intervals = [row for row in time_index['streams']['video']
                 if row.get('record_scope') == 'retention_interval']
    assert {row['recording_id'] for row in intervals} == {first['recording_id'], second['recording_id']}
    published, = [row for row in intervals if row.get('video_value') == 'selected']
    assert published['file']['path'] == selected['video']['path']
    assert published['source_start_ms'] == selected['start_ms']
    assert published['source_end_ms'] == selected['end_ms']
    if readable:
        meaning, = [row for row in time_index['streams']['multimodal']
                    if row['segment_id'] == selected['segment_id']]
        assert meaning['content_file']['path'] == selected['understanding_path']
        assert {row['recording_id'] for row in time_index['readable_content']['audio']} == {
            first['recording_id'], second['recording_id']}
        assert next(row for row in time_index['readable_content']['audio']
                    if row['recording_id'] == first['recording_id']) == previous['readable_content']['audio'][0]
        assert all((layout.root/row['content']).is_file()
                   for row in time_index['readable_content']['multimodal'])
    report = read_json(layout.reports/'LaboratoryDailyReport.json')
    assert report['视频留存统计']['selected_interval_count'] == 1
    assert report['视频留存统计']['actual_video_duration_seconds'] == 1
    assert {path: path.read_bytes() for path in protected} == protected
    assert {stage: runner._receipt(layout, first, stage).read_bytes()
            for stage in original_receipts} == original_receipts
    assert (models.vision_calls, models.semantic_calls, len(exports)) == calls


def _native_window_continuity(
        device_config, monkeypatch, later_status, *, initial_video=True):
    capture(device_config)
    first, layout = item_and_layout(device_config)
    device_config['device_day'].update(derived_video_retention=initial_video, readable_content_enabled=True,
                                      understanding_coverage_since_us=1)
    ledger = UsageLedger(device_config)

    class WindowModels(FakeModels):
        def vision(self, *args):
            result = super().vision(*args)
            result['batches'] = [{'dense_cv_evidence': 'unchanged source diagnostics' * 100}]
            return result

        def understand(self, layout, recording, vision, context, key):
            result = super().understand(layout, recording, vision, context, key)
            meaning = result['understandings'][0]
            add_work_evidence(layout, meaning, key, vision['segments'][0])
            window = meaning['windows'][0]
            request = read_json(layout.root/window['input'])
            response = read_json(layout.root/window['model_receipt'])
            native = analysis_folder(layout, vision['segments'][0], key)/'00-00-00_00-00-30'
            frames = []
            for frame in request['metadata']['frames']:
                path = native/'SceneFrames'/(frame['frame_id']+'.jpg')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((layout.root/frame['path']).read_bytes())
                frames.append(frame | artifact(layout.root, path))
            request['metadata']['frames'] = frames
            request['metadata']['end_ms'] = 30000
            request['input_key'] = digest([request['metadata'], request['prompt']])
            response['input_key'] = request['input_key']
            atomic_json(native/'Input.json', request)
            atomic_json(native/'Result.json', response)
            window.update(end_ms=30000, source_frames=frames, input=layout.relative(native/'Input.json'),
                          model_receipt=layout.relative(native/'Result.json'))
            meaning_path = native.parent/'Observation.json'
            partial = recording['recording_id'] == first['recording_id']
            if partial:
                meaning['status'] = 'partial'
            atomic_json(meaning_path, meaning)
            result['artifacts'] = [artifact(layout.root, meaning_path),
                                   *[artifact(layout.root, layout.root/frame['path']) for frame in frames],
                                   artifact(layout.root, native/'Input.json'),
                                   artifact(layout.root, native/'Result.json')]
            if not partial:
                return result
            result['partial_observations'] = True
            if later_status == 'submitted_unknown':
                claim = ledger.reserve(key)
                ledger.submitted_unknown(claim)
                error = SubmittedUnknown('Deterministic later window has no known response; never resend')
            else:
                error = ValueError('Deterministic failure after the first returned window')
            error.partial_receipt = result
            raise error

    exports = []

    def extract(source, target, start, duration, *_args):
        exports.append((source, start, duration))
        target.write_bytes(b'owned native-window exact selected fixture')

    monkeypatch.setattr('visioncortex.video_io.extract_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video',
                        lambda _: SimpleNamespace(duration_ms=1000))
    models = WindowModels()
    original_runner = DeviceDayRunner(device_config, models)
    for stage in ('retention', 'vision', 'stt'):
        assert original_runner.process(first, stage=stage)['status'] == 'completed'
    result = original_runner.run_once({'recordings': [first]}, stage='understanding')['results'][0]
    assert result['status'] == later_status
    native_path = original_runner._receipt(layout, first, 'understanding')
    native = read_json(native_path)
    first_candidate = read_json(layout.index)['segments'][0]
    protected = {native_path: native_path.read_bytes()}
    for ref in [*native['artifacts'], first_candidate['source_ref']]:
        path = layout.root/ref['path']
        protected[path] = path.read_bytes()

    # A changed model recipe cannot accept this parent as a new completion.
    # Its original returned window remains eligible only for the existing
    # source/request/response/frame-verified readable observation path.
    current_config = deepcopy(device_config)
    current_config['device_day']['derived_video_retention'] = True
    current_config['mllm']['temperature'] = (current_config['mllm'].get('temperature') or 0) + .125
    current = DeviceDayRunner(current_config, models)
    expected = current._key('understanding', first, {
        'vision': read_json(current._receipt(layout, first, 'vision'))})
    assert native['key'] != expected and not current._accepts_receipt(native, expected)
    capture(current_config, start=first['recording_end_us'], label='101500')
    second, = [row for row in scan_recordings(current_config)['recordings']
               if row['recording_id'] != first['recording_id']]
    assert current.process(second)['status'] == 'completed'
    dense_processing = {row['recording_id']: row['processing'] for row in read_json(layout.index)['recordings']}
    monkeypatch.setattr(current, '_backend', lambda: pytest.fail('Publication must not start models'))
    if not initial_video:
        parent = next(row for row in read_json(layout.index)['segments']
                      if row['recording_id'] == first['recording_id'])
        assert not any(row.get('video') for row in parent['retention_intervals'])
        assert len(exports) == 1  # Only the other recording has an actual MP4.
        assert current.refresh_index(layout, recording_id=first['recording_id'])
        assert len(exports) == 2
        assert exports[-1][0] == layout.root/first_candidate['source_ref']['path']
        assert exports[-1][1:] == (0, 1000)
    observe(current.runtime_root, {'recordings': [first, second]})
    publisher = FileIndexPublisher(current_config)
    outcome = publisher.tick(archives=[layout.name])
    assert not outcome['errors'], outcome
    calls = models.vision_calls, models.semantic_calls, len(exports)
    monkeypatch.setattr('visioncortex.video_io.extract_clip', lambda *a, **k: pytest.fail('No repeated export'))

    def queues():
        values = {}
        for stage, queue in current.queues.items():
            with queue.connect() as db:
                values[stage] = [dict(row) for row in db.execute('SELECT * FROM recordings ORDER BY recording_id')]
        return values

    queue_before = queues()
    unknown_row, = [row for row in queue_before['understanding'] if row['recording_id'] == first['recording_id']]
    assert unknown_row['status'] == ('blocked' if later_status == 'submitted_unknown' else 'failed')

    def visible_scope():
        canonical = read_json(layout.index)
        assert len(canonical['recordings']) == len(canonical['segments']) == 2
        assert {row['recording_id']: row['processing'] for row in canonical['recordings']} == dense_processing
        assert all(row['segment_id'] != first_candidate['segment_id'] for row in canonical['understandings'])
        record = next(row for row in canonical['recordings'] if row['recording_id'] == first['recording_id'])
        assert record['stages']['understanding']['status'] == later_status
        assert record['stages']['understanding']['key'] == native['key']
        times = read_json(layout.comments/'TimeIndex.json')
        selected = [row for row in times['streams']['video'] if row.get('video_value') == 'selected']
        assert {row['recording_id'] for row in selected} == {first['recording_id'], second['recording_id']}
        video = next(row for row in selected if row['recording_id'] == first['recording_id'])
        meaning, = [row for row in times['streams']['multimodal'] if row['segment_id'] == video['segment_id']]
        assert meaning['status'] == 'partial'
        content = layout.root/meaning['content_file']['path']
        value = read_json(content)
        assert value['status'] == value['understanding']['status'] == 'partial'
        assert value['source_video'] == first_candidate['source_ref']
        assert (value['source_start_ms'], value['source_end_ms']) == (0, 1000)
        assert (value['start_us'], value['end_us']) == (video['start_us'], video['end_us'])
        assert value['understanding']['windows'][0]['model_receipt'] == native['understandings'][0]['windows'][0]['model_receipt']
        assert value['understanding']['windows'][0]['input'] == native['understandings'][0]['windows'][0]['input']
        assert (video['source_start_ms'], video['source_end_ms']) == (0, 1000)
        report = read_json(layout.reports/'LaboratoryDailyReport.json')
        entry, = [row for row in report['entries'] if row['segment_id'] == first_candidate['segment_id']]
        assert entry['understanding_status'] == 'partial'
        assert report['视频留存统计']['selected_interval_count'] == 2
        parts = [row for row in report['视频留存统计']['partitions'] if row['recording_id'] == first['recording_id']]
        chosen, = [row for row in parts if row['video_value'] == 'selected']
        assert (chosen['start_ms'], chosen['end_ms']) == (0, 1000)
        assert sum(row['duration_ms'] for row in parts if row['video_value'] == 'pending') == 899000
        return video, meaning, content.read_bytes(), (layout.root/video['file']['path']).read_bytes()

    expected = visible_scope()
    for _ in range(2):
        assert current.refresh_index(layout, recording_id=second['recording_id'])
        assert visible_scope() == expected
        outcome = publisher.tick(archives=[layout.name])
        assert not outcome['errors'], outcome
        assert visible_scope() == expected
    assert {path: path.read_bytes() for path in protected} == protected
    assert queues() == queue_before
    assert (models.vision_calls, models.semantic_calls, len(exports)) == calls


@pytest.mark.parametrize('later_status', ['submitted_unknown', 'failed'])
def test_readable_native_window_survives_alternating_publishers_without_promoting_parent(
        device_config, monkeypatch, later_status):
    _native_window_continuity(device_config, monkeypatch, later_status)


def test_unknown_parent_first_publication_exports_only_its_actual_completed_native_window(device_config, monkeypatch):
    _native_window_continuity(
        device_config, monkeypatch, 'submitted_unknown', initial_video=False)
