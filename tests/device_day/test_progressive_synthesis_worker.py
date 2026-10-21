from copy import deepcopy
from threading import Event

from visioncortex.device_day_contract import DeviceDayLayout, atomic_json, digest, read_json
from visioncortex.device_day_synthesis import evidence_bundle
from visioncortex.device_day_synthesis_worker import DaySynthesisWorker


def setup(tmp_path, monkeypatch):
    import visioncortex.device_day_synthesis as synthesis
    import visioncortex.device_day_reports as reports
    config = {'runtime': {'local_only': True}, 'device_day': {'day_synthesis': {'enabled': True}},
              'storage': {key: str(tmp_path / key) for key in
                          ('local_runtime_root', 'local_cache_root', 'archive_root', 'local_staging_root')}}
    layout = DeviceDayLayout(tmp_path / 'archive_root', 'camera', 1789488000000000)
    source = {'archive': layout.name, 'recordings': [], 'segments': [], 'understandings': [],
              'context': {'events': [{'type': 'ocr_observation', 'start_us': 1789488000000000,
                  'end_us': 1789488000000000, 'capture_device_id': 'camera', 'payload': {'value': None}}]}}
    atomic_json(layout.index, source)
    worker = DaySynthesisWorker()
    monkeypatch.setattr(worker, 'candidates', lambda _: [layout])
    monkeypatch.setattr(synthesis, 'current_synthesis', lambda *_: {})
    monkeypatch.setattr(reports, 'render_day', lambda *_: None)
    class Analyzer:
        def close(self):
            pass
    return worker, config, layout, source, Analyzer


def test_progressive_update_publishes_once_then_waits_without_rebilling(tmp_path, monkeypatch):
    worker, config, layout, source, analyzer = setup(tmp_path, monkeypatch)
    calls = []
    def synthesize(config, layout, index, analyzer, **kwargs):
        calls.append(deepcopy(index))
        return {'context_archives': [layout.name], 'input_digest': digest(evidence_bundle(index)),
                'status': 'completed_for_available_inputs', 'generated_at': 'saved', 'findings': []}
    monkeypatch.setattr('visioncortex.device_day_synthesis.synthesize', synthesize)
    assert worker.tick(config, Event(), analyzer_factory=lambda _: analyzer())['status'] == 'processed'
    saved = read_json(layout.reports / 'LaboratoryDailyReport.json')['全天综合理解']
    assert saved['context_archives'] == [source['archive']]
    assert worker.tick(config, Event(), analyzer_factory=lambda _: analyzer())['status'] == 'waiting'
    assert len(calls) == 1


def test_evidence_change_during_call_keeps_asof_and_renders_latest(tmp_path, monkeypatch):
    worker, config, layout, source, analyzer = setup(tmp_path, monkeypatch)
    latest = deepcopy(source)
    latest['context']['events'][0]['payload']['value'] = 12
    rendered = []
    def synthesize(config, layout, index, analyzer, **kwargs):
        atomic_json(layout.index, latest)
        return {'context_archives': [layout.name], 'status': 'completed_for_available_inputs',
                'input_digest': digest(evidence_bundle(index)), 'findings': []}
    monkeypatch.setattr('visioncortex.device_day_synthesis.synthesize', synthesize)
    monkeypatch.setattr('visioncortex.device_day_reports.render_day', lambda layout, index: rendered.append(index))
    assert worker.tick(config, Event(), analyzer_factory=lambda _: analyzer())['status'] == 'processed'
    saved = read_json(layout.reports / 'LaboratoryDailyReport.json')['全天综合理解']
    assert saved['input_digest'] == digest(evidence_bundle(source))
    assert rendered == [latest]


def test_unresolved_paid_request_is_retained_as_waiting(tmp_path, monkeypatch):
    from visioncortex.multimodal_usage import SubmittedUnknown
    worker, config, layout, _, analyzer = setup(tmp_path, monkeypatch)
    def synthesize(*args, **kwargs):
        raise SubmittedUnknown('existing paid request has no receipt')
    monkeypatch.setattr('visioncortex.device_day_synthesis.synthesize', synthesize)
    result = worker.tick(config, Event(), analyzer_factory=lambda _: analyzer())
    assert result['status'] == 'waiting_for_provider_receipt'
    assert not (layout.reports / 'LaboratoryDailyReport.json').exists()


def test_cross_camera_output_is_rejected(tmp_path, monkeypatch):
    worker, config, layout, _, analyzer = setup(tmp_path, monkeypatch)
    monkeypatch.setattr('visioncortex.device_day_synthesis.synthesize',
                        lambda *args, **kwargs: {'context_archives': ['another-camera']})
    assert worker.tick(config, Event(), analyzer_factory=lambda _: analyzer())['status'] == 'failed'
    assert not (layout.reports / 'LaboratoryDailyReport.json').exists()


def scheduled_days(tmp_path, monkeypatch):
    import time
    from visioncortex.device_day_contract import archive_name
    from visioncortex.observed_inventory import observe
    root = tmp_path / 'runtime' / 'device-day'
    now = time.time()
    records = [{'recording_id': str(i), 'camera_key': f'camera{i}',
                'recording_start_us': round((now - i * 86400) * 1e6)} for i in range(4)]
    observe(root, {'recordings': records})
    names = [archive_name(r['camera_key'], r['recording_start_us']) for r in records]
    config = {'device_day': {}, 'storage': {'local_runtime_root': str(root.parent),
              'archive_root': str(tmp_path / 'archive'), 'local_cache_root': str(tmp_path / 'cache')}}
    worker = DaySynthesisWorker()
    worker.checked = {name: now - (i + 1) * 1000 for i, name in enumerate(names)}
    monkeypatch.setattr('visioncortex.device_day_publication_worker.local_change_hints',
                        lambda *args: dict.fromkeys(names, 'saved-change'))
    return worker, config, names, now


def test_changed_current_day_overtakes_older_cursor_and_history_keeps_a_turn(tmp_path, monkeypatch):
    worker, config, names, now = scheduled_days(tmp_path, monkeypatch)
    for turn in range(1, 4):
        first = worker.candidates(config)[0]
        assert first.name == (names[3] if turn == 3 else names[0])
        assert worker.lane == ('historical_fair_share' if turn == 3 else 'changed_or_live')
        worker.checked[first.name] = now


def test_recent_backoff_does_not_hide_other_days_or_reset_unknown_receipts(tmp_path, monkeypatch):
    worker, config, names, now = scheduled_days(tmp_path, monkeypatch)
    unknown = {'attempted_at': now, 'status': 'waiting_for_provider_receipt', 'error_type': 'SubmittedUnknown'}
    worker.state['days'][names[0]] = deepcopy(unknown)
    assert names[0] not in [r.name for r in worker.candidates(config)]
    assert worker.state['days'][names[0]] == unknown


def test_inventory_grouping_bounds_nas_layout_resolution_and_respects_scope(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from visioncortex.observed_inventory import observe
    root = tmp_path / 'runtime' / 'device-day'
    stamp = 1_800_000_000_000_000
    records = [{'recording_id': f'{camera}-{i}', 'camera_key': f'camera{camera}',
                'recording_start_us': stamp + i * 1_000_000}
               for camera in range(25) for i in range(10)]
    records.append({'recording_id': 'out-of-scope', 'camera_key': 'old', 'recording_start_us': stamp - 86400_000_000})
    observe(root, {'recordings': records})
    calls = []
    def layout(root, camera, start, backend):
        calls.append(camera)
        return SimpleNamespace(name=camera)
    monkeypatch.setattr('visioncortex.device_day_synthesis_worker.DeviceDayLayout', layout)
    config = {'device_day': {'process_since_us': stamp}, 'storage': {
        'local_runtime_root': str(root.parent), 'archive_root': 'unused', 'local_cache_root': 'unused'}}
    assert len(DaySynthesisWorker().candidates(config)) == 16
    assert len(calls) == len(set(calls)) == 16
    assert 'old' not in calls


def test_restart_keeps_day_rotation_without_rebilling_unchanged_evidence(tmp_path, monkeypatch):
    worker, config, layout, _, analyzer = setup(tmp_path, monkeypatch)
    state_path = tmp_path / 'local_runtime_root' / 'device-day' / 'DaySynthesis' / 'State.json'
    atomic_json(state_path, {'days': {}, 'schedule': {'checked': {'historical': 123}, 'turn': 2}})
    calls = []
    def synthesize(config, layout, index, analyzer, **kwargs):
        calls.append(index)
        return {'context_archives': [layout.name], 'input_digest': digest(evidence_bundle(index)),
                'status': 'completed_for_available_inputs', 'findings': []}
    monkeypatch.setattr('visioncortex.device_day_synthesis.synthesize', synthesize)
    assert worker.tick(config, Event(), analyzer_factory=lambda _: analyzer())['status'] == 'processed'
    restarted = DaySynthesisWorker()
    monkeypatch.setattr(restarted, 'candidates', lambda _: [layout])
    assert restarted.tick(config, Event(), analyzer_factory=lambda _: analyzer())['status'] == 'waiting'
    assert len(calls) == 1
    assert read_json(state_path)['schedule']['checked']['historical'] == 123
