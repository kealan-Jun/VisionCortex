"""Overview uses exact owning publication receipts; fixtures prove file contracts only."""
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_retained_video_subintervals import exact_case
from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_content import publish_content
from visioncortex.device_day_content_paths import clip_name
from visioncortex.device_day_contract import artifact, atomic_json
from visioncortex.device_day_overview import ArchiveOverview, assemble, render, summarize
from visioncortex.device_day_progress import snapshot
from visioncortex.device_day_publication import reconcile_outputs

__all__ = ['exact_case']


class Anchors(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links, self.current = [], None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.current = dict(attrs) | {'text': ''}

    def handle_data(self, data):
        if self.current is not None:
            self.current['text'] += data

    def handle_endtag(self, tag):
        if tag == 'a' and self.current is not None:
            self.links.append(self.current)
            self.current = None


@pytest.fixture
def published(exact_case):
    case = exact_case
    case.config['storage']['local_runtime_root'] = str(case.layout.root.parent/'runtime')
    case.config['collection_ingest'] = {'source_root': str(case.layout.root)}
    case.candidate['json_path'] = ('ProcessedClips/Clips/' + clip_name(case.candidate)
                                   + '/ExperimentActivity.json')
    case.record['sources'] = [{'kind': 'video', 'retained': case.candidate['source_ref']}]
    historical = case.layout.processed/'HistoricalWhole.mp4'
    historical.parent.mkdir(parents=True, exist_ok=True)
    historical.write_bytes(b'historical whole candidate, never an exact selected export')
    case.candidate['video'] = artifact(case.layout.root, historical)
    result = publish_selected(case.config, case.index, maximum_new_videos=2,
                              evidence_validator=case.validator)
    assert result['materialized'] == 2
    case.index['readable_content'] = publish_content(case.config, case.index)
    case.rows = case.candidate['retention_intervals']
    case.selected = [row for row in case.rows if row['video_value'] == 'selected']
    return case


def page(case, *, root=True):
    summary = summarize(case.layout.name, case.index, case.config,
                        root=case.layout.root if root else None,
                        report_exists=True, understanding_exists=True)
    value = assemble(snapshot(case.config), [summary], [])
    return summary, value, render(value)


def test_only_two_real_exact_experiment_and_nonexperiment_exports_are_current(published):
    case = published
    before = case.source.read_bytes()
    summary, data, text = page(case)
    assert summary['active_count'] == 1 and summary['segment_count'] == 1
    assert summary['selected_count'] == summary['available_selected_count'] == 2
    assert summary['selected_video_seconds'] == 45
    assert summary['pending_value_count'] == 3 and summary['confirmed_no_value_count'] == 0
    assert data['days'][0]['activity_seconds'] == 45
    links = Anchors(text).links
    current = [link for link in links if link['text'] == '打开选中视频']
    assert len(current) == 2
    assert {link['data-path'] for link in current} == {row['video']['path'] for row in case.selected}
    assert all('data-watch' not in link for link in links)  # direct files, no candidate/player fallback
    assert any(link.get('data-path') == 'ProcessedClips/HistoricalWhole.mp4'
               and '历史候选视频' in link['text'] for link in links)
    assert '实验 · 选中 · 处理完成' in text and '非实验 · 选中 · 处理完成' in text
    assert '待判定 · 等待价值判断' in text and '原片时间映射（追溯）' in text
    assert len([link for link in links if link['text'] == 'Understanding.json']) == 5
    assert any(link.get('data-path') == 'LaboratoryDailyReport/LaboratoryDailyReport.json' for link in links)
    assert not any(link.get('data-path', '').endswith('.html') for link in links)
    assert case.source.read_bytes() == before


def test_without_actual_archive_root_plan_never_becomes_current_video(published):
    summary, _, text = page(published, root=False)
    assert summary['selected_count'] == 2 and summary['available_selected_count'] == 0
    assert not summary['current_files_checked'] and summary['selected_video_seconds'] == 0
    assert all(not row['video'] and row['actual_video_verification'] == 'NOT_PROVEN' for row in summary['intervals'])
    assert not any(link['text'] == '打开选中视频' for link in Anchors(text).links)
    assert '待判定' in text and '预处理候选 JSON' in text


@pytest.mark.parametrize('state', ['queued', 'failed', 'partial', 'submitted_unknown', 'waiting_for_evidence'])
def test_present_bytes_do_not_forge_processing_completion(published, state):
    published.selected[0]['processing_status'] = state
    summary, _, text = page(published)
    assert summary['selected_count'] == 2 and summary['available_selected_count'] == 1
    assert len([link for link in Anchors(text).links if link['text'] == '打开选中视频']) == 1
    row = next(row for row in summary['intervals'] if row['segment_id'] == published.selected[0]['segment_id'])
    assert row['processing_status'] == state and row['video_value'] == 'selected'
    assert not row['video'] and row['verification_reason'] == 'processing_not_completed'


@pytest.mark.parametrize('bad', ['missing_video', 'corrupt_video', 'corrupt_receipt', 'raw_path', 'historical_path'])
def test_bad_or_mislabeled_artifact_never_becomes_selected_player(published, bad, monkeypatch):
    case = published
    row = case.selected[0]
    if bad == 'missing_video':
        (case.layout.root/row['video']['path']).unlink()  # owned output only
    elif bad == 'corrupt_video':
        path = case.layout.root/row['video']['path']
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif bad == 'corrupt_receipt':
        (case.layout.root/row['receipt']['path']).write_bytes(b'corrupt owned receipt')
    else:
        row['video'] = case.candidate['source_ref'] if bad == 'raw_path' else case.candidate['video']
    if bad == 'raw_path':
        from visioncortex import device_day_verification
        original_hash = device_day_verification.file_hash
        def no_original_hash(path):
            assert path != case.source, 'Overview must never fully read the original media'
            return original_hash(path)
        monkeypatch.setattr(device_day_verification, 'file_hash', no_original_hash)
    summary, _, text = page(case)
    assert summary['available_selected_count'] == 1
    current = [link for link in Anchors(text).links if link['text'] == '打开选中视频']
    assert len(current) == 1 and current[0]['data-path'] == case.selected[1]['video']['path']
    assert all(not link.get('data-path', '').startswith(('MetaData/', 'MetaVideo/')) for link in current)


def test_cache_rechecks_changed_output_without_reduced_canonical_write(published):
    case = published
    atomic_json(case.layout.index, case.index)
    original_index = case.layout.index.read_bytes()
    runner = SimpleNamespace(config=case.config, archive_root=case.layout.root.parent,
        runtime_root=case.layout.root.parent/'runtime/device-day')
    worker = ArchiveOverview()
    assert worker.publish(runner)['selected_videos'] == 2
    (case.layout.root/case.selected[0]['video']['path']).unlink()  # owned output only
    assert worker.publish(runner)['selected_videos'] == 1
    assert case.layout.index.read_bytes() == original_index
    assert case.candidate['video']['path'] == 'ProcessedClips/HistoricalWhole.mp4'


def test_oversized_canonical_defers_before_open_without_reusing_a_good_selected_count(published, monkeypatch):
    case = published
    atomic_json(case.layout.index, case.index)
    runner = SimpleNamespace(config=case.config, archive_root=case.layout.root.parent,
        runtime_root=case.layout.root.parent/'runtime/device-day')
    worker = ArchiveOverview()
    assert worker.publish(runner)['selected_videos'] == 2
    with case.layout.index.open('r+b') as handle:
        handle.truncate(worker._published_views.source_limit + 1)  # owned sparse fixture; no bulk read
    original = Path.open
    before = case.layout.index.stat()
    def refuse_source_open(path, *args, **kwargs):
        if path == case.layout.index:
            pytest.fail('Oversized canonical bytes must not be opened by the compatibility overview')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', refuse_source_open)
    result = worker.publish(runner)
    after = case.layout.index.stat()
    assert result['status'] == 'partial' and result['selected_videos'] == 0
    assert result['index_repair'] is None and case.layout.name not in worker.cache
    assert result['errors'][0]['reason'] == 'ReadModelDeferred'
    assert result['errors'][0]['read_status'] == 'deferred'
    assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    assert '兼容总览读取已延后' in (runner.runtime_root/'ArchiveOverview.html').read_text()


def test_projection_resident_limit_defers_without_rewriting_canonical(published):
    case = published
    atomic_json(case.layout.index, case.index)
    before = case.layout.index.read_bytes()
    runner = SimpleNamespace(config=case.config, archive_root=case.layout.root.parent,
        runtime_root=case.layout.root.parent/'runtime/device-day')
    worker = ArchiveOverview()
    worker._published_views.resident_limit = 128
    result = worker.publish(runner)
    assert result['status'] == 'partial' and result['selected_videos'] == 0
    assert result['errors'][0]['reason'] == 'ProjectionDeferred'
    assert result['errors'][0]['read_status'] == 'deferred'
    assert case.layout.index.read_bytes() == before


def test_dense_canonical_diagnostics_are_omitted_before_summary_hydration(published):
    case = published
    dense = ['bounded owned synthetic diagnostic payload' * 20] * 1024
    case.index['recordings'][0]['processing'] = {key: dense for key in ('batches', 'scan_reports', 'audit_artifacts')}
    case.index['segments'][0]['activity_audit'] = dense
    atomic_json(case.layout.index, case.index)
    before = case.layout.index.read_bytes()
    runner = SimpleNamespace(config=case.config, archive_root=case.layout.root.parent,
        runtime_root=case.layout.root.parent/'runtime/device-day')
    worker = ArchiveOverview()
    worker._published_views.resident_limit = 512 * 1024
    result = worker.publish(runner)
    assert result['status'] == 'completed' and result['selected_videos'] == 2
    assert not result['errors'] and case.layout.index.read_bytes() == before
    assert not worker._published_views.cache  # the overview retains summaries, not another device/day index


def test_file_changed_after_byte_verification_cannot_enter_good_cache(published, monkeypatch):
    case = published
    atomic_json(case.layout.index, case.index)
    runner = SimpleNamespace(config=case.config, archive_root=case.layout.root.parent,
        runtime_root=case.layout.root.parent/'runtime/device-day')
    import visioncortex.device_day_overview as overview
    original = overview.summarize
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result['available_selected_count'] == 2
        path = case.layout.root/case.selected[0]['video']['path']
        path.write_bytes(b'owned changed output after successful verification')
        return result
    monkeypatch.setattr(overview, 'summarize', changing)
    worker = ArchiveOverview()
    result = worker.publish(runner)
    assert result['selected_videos'] == 0 and result['errors'][0]['reason'] == 'ValueError'
    assert case.layout.name not in worker.cache
    monkeypatch.setattr(overview, 'summarize', original)
    assert worker.publish(runner)['selected_videos'] == 1


def test_historical_snapshot_does_not_claim_current_selected_files(published):
    case = published
    atomic_json(case.layout.index, case.index)
    runner = SimpleNamespace(config=case.config, archive_root=case.layout.root.parent,
        runtime_root=case.layout.root.parent/'runtime/device-day')
    worker = ArchiveOverview()
    assert worker.publish(runner)['selected_videos'] == 2
    case.config['device_day']['process_since_us'] = 1789637306000000
    case.layout.index.write_text('historical index deliberately unavailable')
    result = worker.publish(runner)
    assert result['selected_videos'] == 0 and not result['errors']
    text = (runner.runtime_root/'ArchiveOverview.html').read_text()
    assert not any(link['text'] == '打开选中视频' for link in Anchors(text).links)
    assert any('历史候选视频' in link['text'] for link in Anchors(text).links)


def test_partial_understanding_is_not_reported_as_completed_candidate(published):
    case = published
    case.index['understandings'] = [{'segment_id': case.candidate['segment_id'],
                                     'status': 'partial', 'windows': [{'summary': 'actual partial'}]}]
    assert page(case)[0]['understood_segments'] == 0


def test_compatibility_reconcile_never_reads_or_moves_old_outputs(published):
    case = published
    before = {path: path.read_bytes() for path in case.layout.root.rglob('*') if path.is_file()}
    runner = SimpleNamespace(archive_root=case.layout.root.parent)
    assert reconcile_outputs(runner)['relocated'] == []
    assert reconcile_outputs(runner)['automatic_relocation'] is False
    assert {path: path.read_bytes() for path in case.layout.root.rglob('*') if path.is_file()} == before
    # Old whole material remains reachable but cannot become a third current export.
    assert page(case)[0]['available_selected_count'] == 2
    old = deepcopy(case.candidate['video'])
    assert (case.layout.root/old['path']).is_file()
