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
