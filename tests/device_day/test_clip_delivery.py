"""Display consolidation preserves exact native evidence and source clocks."""
from copy import deepcopy

import pytest

from test_retained_video_subintervals import exact_case
from visioncortex.application import clip_delivery, retained_video
from visioncortex.device_day_content import publish_content
from visioncortex.device_day_content_paths import clip_name, aliases, private_folder
from visioncortex.device_day_contract import artifact, atomic_json, safe_child, verify_artifact
from visioncortex.device_day_file_index import build_file_index
from visioncortex.device_day_narrative import retention_summary

__all__ = ['exact_case']


def public_name(case):
    return clip_delivery.original_units(case.layout, case.index)['MetaData/capture']['name']


def test_native_photo_root_never_becomes_a_video_collection(exact_case):
    case = exact_case
    photo = case.layout.root/'MetaData'/'Photos'/'12-00-00'/'20261029_120000.jpg'
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b'original photo bytes')
    units = clip_delivery.original_units(case.layout, case.index)
    assert 'MetaData/Photos' not in units
    old = 'TimePending_TimePending_NoExperimentActivity_oldphotos'
    header = {'schema_version': clip_delivery.VERSION, 'source_unit': 'MetaData/Photos',
              'source_folder': old, 'original_source': {'path': 'MetaData/Photos/rgb.mp4'},
              'interval_count': 0, 'available_video_count': 0, 'recording_ids': [], 'intervals': []}
    for parent, filename in [(case.layout.processed/'Clips', 'Index.json'),
                             (case.layout.understanding, 'Understanding.json')]:
        atomic_json(parent/old/filename, header)
    result = clip_delivery.retire_native(case.layout, case.index,
        {unit['name'] for unit in units.values()}, source_units=units)
    assert result['retired'] == 2
    assert photo.read_bytes() == b'original photo bytes'
    assert not (case.layout.processed/'Clips'/old).exists()
    assert not (case.layout.understanding/old).exists()


def test_cv_activity_never_names_an_experiment_and_no_value_has_no_video(monkeypatch):
    candidate = {'segment_id': 'candidate'}
    record = {'stages': {'understanding': {'status': 'completed'}}}
    interval = {'video_value': 'selected', 'start_ms': 0, 'end_ms': 1000,
                'reason': 'existing_fine_audited_activity_process'}
    monkeypatch.setattr('visioncortex.media.retained_intervals.observed_categories', lambda *args: [])
    assert clip_delivery.category(candidate, None, record, interval) == 'ActivityPendingClassification'
    assert clip_delivery.category(candidate, None, record, interval | {'video_value': 'pending'}) == 'PendingValueReview'
    assert clip_delivery.category(candidate, None, record, interval | {
        'video_value': 'confirmed_no_value'}) == 'NoValueNoExperimentActivity'
    monkeypatch.setattr('visioncortex.media.retained_intervals.observed_categories', lambda *args: ['non_experiment'])
    assert clip_delivery.category(candidate, {}, record, interval) == 'ValuableNoExperimentActivity'
    assert clip_delivery.category(candidate, {'status': 'partial'}, record, interval) == 'ActivityPendingClassification'
    monkeypatch.setattr('visioncortex.media.retained_intervals.observed_categories', lambda *args: ['experiment'])
    assert clip_delivery.category(candidate, {}, record, interval) == 'ExperimentActivity'


@pytest.mark.parametrize('change,expected', [
    ({}, 'ExperimentActivity'),
    ({'video_value': 'pending'}, 'PendingValueReview'),
    ({'video_value': 'confirmed_no_value'}, 'NoValueNoExperimentActivity'),
    ({'selection_owner': 'motion_only'}, 'ActivityPendingClassification'),
    ({'reason': 'motion_detected'}, 'ActivityPendingClassification'),
    ({'activity_category': 'non_experiment'}, 'ActivityPendingClassification'),
])
def test_audited_experiment_selection_survives_paused_semantics(monkeypatch, change, expected):
    candidate = {'activity': 'active',
        'classification_basis': 'existing_fine_audit_and_device_boundaries_not_ground_truth',
        'activity_audit_ref': {'path': 'ActivityAudit.json', 'size_bytes': 100,
                               'sha256': 'a'*64, 'storage_root': 'local_cache_root'}}
    interval = {'video_value': 'selected', 'start_ms': 0, 'end_ms': 219300,
        'reason': 'existing_fine_audited_activity_process',
        'selection_owner': 'audited_device_activity', 'activity_category': 'experiment'}
    monkeypatch.setattr('visioncortex.media.retained_intervals.observed_categories', lambda *args: [])
    record = {'stages': {'understanding': {'status': 'paused_by_user'}}}
    assert clip_delivery.category(candidate, None, record, interval | change) == expected
    # Neither an unaudited activity label nor an invalid audit reference suffices.
    for replacement in ({}, candidate | {'activity_audit_ref': {}},
                        candidate | {'classification_basis': 'motion_only'}):
        assert clip_delivery.category(replacement, None, record, interval) == 'ActivityPendingClassification'
    monkeypatch.setattr('visioncortex.media.retained_intervals.observed_categories', lambda *args: ['non_experiment'])
    assert clip_delivery.category(candidate, {'status': 'completed'}, record, interval) == 'ValuableNoExperimentActivity'
    assert clip_delivery.category(candidate, {'status': 'partial'}, record, interval) == 'ActivityPendingClassification'


def test_audited_selected_process_publishes_matching_experiment_groups_without_semantics(exact_case):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    case.candidate.update(classification_basis='existing_fine_audit_and_device_boundaries_not_ground_truth',
        activity_audit_ref={'path': 'ActivityAudit.json', 'size_bytes': 100,
                            'sha256': 'a'*64, 'storage_root': 'local_cache_root'})
    case.intervals[1].update(reason='existing_fine_audited_activity_process',
        selection_owner='audited_device_activity', physical_action_confirmed=False)
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    result = publish_content(case.config, case.index)
    chosen, = [row for row in result['multimodal'] if row['display_category'] == 'ExperimentActivity']
    assert chosen['video'] and case.index['understandings'] == []
    name = public_name(case)
    assert '_ExperimentActivity_' in name
    for parent in (case.layout.processed/'Clips', case.layout.understanding):
        assert [p.name for p in parent.iterdir() if p.is_dir() and not p.name.startswith('.')] == [name]
    assert case.intervals[1]['physical_action_confirmed'] is False


def test_multiple_intervals_share_original_folder_and_native_receipts_remain_readable(exact_case):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    case.config['collection_ingest'] = {'source_root': str(case.layout.root)}
    candidate_path = case.layout.processed/'Clips'/clip_name(case.candidate)/'ExperimentActivity.json'
    case.candidate['json_path'] = case.layout.relative(candidate_path)
    atomic_json(candidate_path, case.candidate)
    native = artifact(case.layout.root, candidate_path)
    selected = retained_video.publish_selected(case.config, case.index, maximum_new_videos=2,
                                               evidence_validator=case.validator)['selected_intervals']
    old_videos = [deepcopy(row['video']) for row in selected]
    # Simulate a previous failed export at exactly the same source extent.
    old = case.layout.processed/'Clips'/'00-02-11_00-02-36_ExperimentActivity_failed'
    atomic_json(old/'ExperimentActivity.json', selected[0] | {'start_ms': float(selected[0]['start_ms'])})
    atomic_json(old/'Retention.json', {'processing_status': 'failed', 'error': 'old failure'})
    original = deepcopy(case.index)
    result = publish_content(case.config, case.index)
    assert case.index == original
    assert result['clip_delivery']['retired'] == 2
    public = list((case.layout.processed/'Clips').iterdir())
    assert [folder.name for folder in public] == [public_name(case)]
    assert len(list(public[0].glob('*.json'))) == 7  # five intervals and two original-level indexes
    assert [folder.name for folder in case.layout.understanding.iterdir() if folder.is_dir() and not folder.name.startswith('.')] == [public_name(case)]
    assert len(list((case.layout.processed/'Clips').glob('*/*.mp4'))) == 2
    assert verify_artifact(case.layout.root, native)
    assert '.Evidence/LegacyPresentation' in str(safe_child(case.layout.root, native['path']))
    for reference in old_videos:
        assert verify_artifact(case.layout.root, reference)
        assert 'NoExperimentActivity' in safe_child(case.layout.root, reference['path']).name
        assert safe_child(case.layout.root, reference['path']).parent.name == public_name(case)
    case.index['readable_content'] = result
    timeline = build_file_index(case.config, case.index, {})
    videos = [row for row in timeline['streams']['video'] if row['kind'] == 'video_segment']
    assert len(videos) == 5 and sum(bool(row['file']) for row in videos) == 2
    assert all(row['metadata_file']['path'].endswith('.json') and '/Clips/' in row['metadata_file']['path'] for row in videos)
    summary = retention_summary(case.index, case.layout.root)
    assert summary['evidence_status'] == 'PROVEN'
    # Repeated publication is idempotent and never calls the extractor again.
    again = publish_content(case.config, case.index)
    assert not again['clip_delivery']['retired']
    assert again['multimodal'] == result['multimodal']
    resumed = retained_video.publish_selected(case.config, case.index, maximum_new_videos=2,
                                              evidence_validator=case.validator)
    assert resumed['materialized'] == 0 and len(case.calls) == 2
    assert case.source.read_bytes().startswith(b'sealed original')


def test_display_identity_survives_policy_ids_but_never_merges_different_sources(exact_case):
    case = exact_case
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    first = clip_delivery.views(case.index)
    changed = deepcopy(case.index)
    for row in changed['segments'][0]['retention_intervals']:
        row['segment_id'] = 'different-export-policy-id'
        row['export_policy'] = 'later-export-recipe'
    assert [row['key'] for row in clip_delivery.views(changed)] == [row['key'] for row in first]
    changed['segments'][0]['source_ref']['sha256'] = 'b'*64
    assert {row['key'] for row in clip_delivery.views(changed)}.isdisjoint(row['key'] for row in first)


def test_relocation_crash_replays_alias_without_changing_native_json(exact_case, monkeypatch):
    case = exact_case
    folder = case.layout.processed/'Clips'/clip_name(case.candidate)
    atomic_json(folder/'ExperimentActivity.json', case.candidate)
    reference = artifact(case.layout.root, folder/'ExperimentActivity.json')
    real = clip_delivery._mapping
    monkeypatch.setattr(clip_delivery, '_mapping', lambda *args: (_ for _ in ()).throw(OSError('crash after rename')))
    with pytest.raises(OSError, match='crash after rename'):
        clip_delivery._retire(case.layout, folder)
    assert not folder.exists()
    monkeypatch.setattr(clip_delivery, '_mapping', real)
    assert not clip_delivery._retire(case.layout, folder)
    assert verify_artifact(case.layout.root, reference)
    assert aliases(case.layout.root)[case.layout.relative(folder)].startswith('ProcessedClips/.Evidence/')


def test_no_value_never_adopts_a_historical_video(exact_case):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    for row in case.intervals:
        row['video_value'] = 'confirmed_no_value'
    retained_video.publish_selected(case.config, case.index, evidence_validator=case.validator)
    old_video = case.layout.processed/'Historical.mp4'
    old_video.parent.mkdir(parents=True, exist_ok=True)
    old_video.write_bytes(b'old derived file')
    case.candidate['video'] = artifact(case.layout.root, old_video)
    result = publish_content(case.config, case.index)
    assert not case.calls and not list((case.layout.processed/'Clips').glob('*/*.mp4'))
    assert all('NoValueNoExperimentActivity' in row['content'] and not row['video'] for row in result['multimodal'])


def test_asset_collision_does_not_overwrite_or_remove_evidence(exact_case):
    case = exact_case
    row = retained_video.publish_selected(case.config, case.index, evidence_validator=case.validator)['selected_intervals'][0]
    target = case.layout.processed/'Clips'/'public-name'/'Video.mp4'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'unrelated bytes')
    with pytest.raises(ValueError, match='collision'):
        clip_delivery.publish_asset(case.layout, row['video'], 'public-name')
    assert target.read_bytes() == b'unrelated bytes' and verify_artifact(case.layout.root, row['video'])


def test_native_clip_name_survives_explicit_hidden_relocation(exact_case):
    case = exact_case
    name = clip_name(case.candidate)
    relocated = case.candidate | {'json_path': 'ProcessedClips/.Evidence/LegacyPresentation/'+name+'/ExperimentActivity.json'}
    assert clip_name(relocated) == name
    assert private_folder(case.layout, 'ProcessedClips', 'LegacyPresentation', name).parent.name == 'LegacyPresentation'


def test_category_updates_retire_interval_files_inside_same_original_folder(exact_case, monkeypatch):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    monkeypatch.setattr('visioncortex.media.retained_intervals.observed_categories', lambda *args: ['non_experiment'])
    first = publish_content(case.config, case.index)
    saved = [row for row in first['multimodal'] if row.get('video')]
    assert all(row['display_category'] == 'ValuableNoExperimentActivity' for row in saved)
    references = [artifact(case.layout.root, case.layout.root/row['content']) for row in saved]
    for category in ('experiment', 'non_experiment'):
        monkeypatch.setattr('visioncortex.media.retained_intervals.observed_categories', lambda *args: [category])
        result = publish_content(case.config, case.index)
        assert [folder.name for folder in (case.layout.processed/'Clips').iterdir()] == [public_name(case)]
        assert len(list((case.layout.processed/'Clips').glob('*/*.mp4'))) == 2
        assert len({row['source_interval_key'] for row in result['multimodal']}) == 5
        assert all(verify_artifact(case.layout.root, row['video']) for row in result['multimodal'] if row.get('video'))
        if category == 'experiment':
            assert all(verify_artifact(case.layout.root, reference) for reference in references)
    assert len(case.calls) == 2 and case.source.read_bytes().startswith(b'sealed original')


def test_delivery_budget_is_bounded_and_retries_without_another_export(exact_case):
    case = exact_case
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    from visioncortex.device_day_content import segment_documents, visual_entry, _write
    def publish():
        return clip_delivery.publish(case.config, case.index, [], document_builder=segment_documents,
            writer=_write, entry_builder=visual_entry, maximum_new_videos=1)
    first, progress = publish()
    assert progress['pending'] and progress['videos_published'] == 1
    assert sum(bool(row['video']) for row in first) == 1
    second, progress = publish()
    assert not progress['pending'] and progress['videos_published'] == 1
    assert sum(bool(row['video']) for row in second) == 2 and len(case.calls) == 2


def test_raw_media_is_never_a_relocation_or_reclaim_target(exact_case):
    case = exact_case
    with pytest.raises(ValueError, match='derived clip folders'):
        clip_delivery._retire(case.layout, case.source.parent)
    assert clip_delivery.publish_asset(case.layout, artifact(case.layout.root, case.source), 'wrong') is None
    assert case.source.read_bytes().startswith(b'sealed original')


def test_two_originals_each_own_multiple_intervals_and_understanding_files(exact_case, monkeypatch):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    second_source = case.source.parent.with_name('capture_001')/'rgb.mp4'
    second_source.parent.mkdir()
    second_source.write_bytes(b'independent sealed original')
    shift = 1000000000
    second = deepcopy(case.candidate) | {'segment_id': 'candidate-second', 'recording_id': 'recording-second',
        'source_ref': artifact(case.layout.root, second_source),
        'start_us': case.candidate['start_us']+shift, 'end_us': case.candidate['end_us']+shift}
    record = deepcopy(case.record) | {'recording_id': second['recording_id'],
        'start_us': second['start_us'], 'end_us': second['end_us'],
        'sources': [{'kind': 'video', 'retained': second['source_ref']}]}
    case.record['sources'] = [{'kind': 'video', 'retained': case.candidate['source_ref']}]
    group = deepcopy(case.grouped) | {'candidate_segment_id': second['segment_id'], 'source_ref': second['source_ref']}
    for interval in group['intervals']:
        interval['start_us'] += shift
        interval['end_us'] += shift
    groups = {case.candidate['segment_id']: case.grouped, second['segment_id']: group}
    monkeypatch.setattr(retained_video, 'partition', lambda candidate, *args: deepcopy(groups[candidate['segment_id']]))
    case.index['segments'].append(second)
    case.index['recordings'].append(record)
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=4, evidence_validator=case.validator)
    result = publish_content(case.config, case.index)
    expected = {u['name'] for u in clip_delivery.original_units(case.layout, case.index).values()}
    assert len(expected) == 2
    assert any(n.startswith('08-00-01_08-15-01_') for n in expected)
    for root in (case.layout.processed/'Clips', case.layout.understanding):
        assert {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')} == expected
    from visioncortex.device_day_contract import read_json
    for unit in expected:
        folder = case.layout.processed/'Clips'/unit
        saved = read_json(folder/'Index.json')
        assert saved['interval_count'] == 5 and saved['available_video_count'] == 2
        assert len(list(folder.glob('*.mp4'))) == 2
        meaning = read_json(case.layout.understanding/unit/'Understanding.json')
        assert len(meaning['片段记录']) == 5
        assert {r['content'] for r in saved['intervals']} == {
            row['详细记录'] for row in meaning['片段记录']}
        assert all(row['metadata'].startswith('ProcessedClips/Clips/'+unit+'/') for row in saved['intervals'])
    assert len(result['multimodal']) == 10 and len(case.calls) == 4


def test_unready_original_gets_matching_status_folder_without_model_or_queue(exact_case):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    pending = case.source.parent.with_name('capture_pending')
    pending.mkdir()
    case.index.update(recordings=[], segments=[])
    result = publish_content(case.config, case.index)
    from visioncortex.device_day_contract import read_json
    for unit in (u['name'] for u in clip_delivery.original_units(case.layout, case.index).values()):
        processed = read_json(case.layout.processed/'Clips'/unit/'Index.json')
        meaning = read_json(case.layout.understanding/unit/'Understanding.json')
        assert processed['processing_status'] == 'waiting_for_source_finalization_or_processing'
        assert processed['intervals'] == meaning['intervals'] == []
        assert not meaning['model_invoked_by_publication']
    assert result['multimodal'] == [] and case.calls == [] and not (pending/'rgb.mp4').exists()


def test_interval_file_rename_crash_preserves_understanding_reference(exact_case, monkeypatch):
    case = exact_case
    relative = 'MultimodalUnderstanding/capture/00-00-01_00-00-02_ExperimentActivity_example.json'
    atomic_json(case.layout.root/relative, {'original': 'actual saved observation'})
    reference = artifact(case.layout.root, case.layout.root/relative)
    real = clip_delivery._mapping
    monkeypatch.setattr(clip_delivery, '_mapping', lambda *args: (_ for _ in ()).throw(OSError('file rename crash')))
    with pytest.raises(OSError, match='file rename crash'):
        clip_delivery._retire_file(case.layout, relative)
    monkeypatch.setattr(clip_delivery, '_mapping', real)
    clip_delivery._recover_files(case.layout)
    assert verify_artifact(case.layout.root, reference)
    assert safe_child(case.layout.root, relative).is_relative_to(case.layout.understanding/'.Evidence')


def test_output_revision_checks_original_indexes_and_actual_interval_files(exact_case):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    case.config['storage']['local_runtime_root'] = str(case.layout.root/'runtime')
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    result = publish_content(case.config, case.index)
    from visioncortex.device_day_file_index import FileIndexPublisher
    publisher = FileIndexPublisher(case.config)
    revision = publisher._output_revision(case.layout.name, case.index)
    paths = {path for path, _ in revision}
    assert str(case.layout.understanding/public_name(case)/'Understanding.json') in paths
    for row in result['multimodal']:
        assert str(case.layout.root/row['content']) in paths
        assert str(case.layout.root/row['metadata']) in paths
    saved = next(row for row in result['multimodal'] if row['video'])
    (case.layout.root/saved['video']['path']).unlink()
    assert publisher._output_revision(case.layout.name, case.index) != revision


def test_grouped_producers_never_create_visible_per_interval_folders(exact_case):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    from visioncortex.device_day_content_paths import producer_clip_folder
    folder = producer_clip_folder(case.layout, clip_name(case.candidate), private=True)
    atomic_json(folder/'ExperimentActivity.json', case.candidate)
    assert clip_name(case.candidate | {'json_path': case.layout.relative(folder/'ExperimentActivity.json')}) == clip_name(case.candidate)
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    assert not (case.layout.processed/'Clips').exists()
    assert aliases(case.layout.root) == {}  # parallel producers do not race on publication alias state
    assert len(list((case.layout.processed/'.Evidence/NativeClips').glob('*/*.mp4'))) == 2
    result = publish_content(case.config, case.index)
    assert [path.name for path in (case.layout.processed/'Clips').iterdir()] == [public_name(case)]
    assert sum(bool(row['video']) for row in result['multimodal']) == 2


def test_full_original_range_names_both_groups_not_selected_subintervals(exact_case):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    # Actual nine-second closed source, despite a nominal 15-minute endpoint.
    start = 1792650335785180
    end = 1792650344802068
    atomic_json(case.source.parent/'meta.json', {
        'recording_window_start_global_us': start, 'recording_window_end_global_us': end,
        'recording_quality_window_start_global_us': start, 'recording_quality_window_end_global_us': end,
        'segment_window_end_global_us': start+900000000})
    unit = clip_delivery.original_units(case.layout, case.index)['MetaData/capture']
    assert unit['name'].startswith('14-25-35_14-25-44_')
    assert unit['capture_name'] == 'capture'
    assert (unit['start_us'], unit['end_us']) == (start, end)
    publish_content(case.config, case.index)
    assert (case.layout.processed/'Clips'/unit['name']/'Index.json').is_file()
    assert (case.layout.understanding/unit['name']/'Understanding.json').is_file()
    assert case.source.parent.name == 'capture'


def test_later_original_never_uses_the_session_start_or_nominal_segment_end(exact_case):
    case = exact_case
    start, end = 1792638961324222, 1792639444784727
    atomic_json(case.source.parent/'meta.json', {
        'recording_window_start_global_us': start-900000000, 'recording_window_end_global_us': end,
        'recording_quality_window_start_global_us': start, 'recording_quality_window_end_global_us': end,
        'segment_window_start_global_us': start, 'segment_window_end_global_us': start+900000000})
    unit = clip_delivery.original_units(case.layout, case.index)['MetaData/capture']
    assert unit['name'].startswith('11-16-01_11-24-04_')
    assert (unit['start_us'], unit['end_us']) == (start, end)


def test_numbered_group_rename_replays_crash_and_preserves_video_bytes(exact_case, monkeypatch):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    result = publish_content(case.config, case.index)
    references = [row['video'] for row in result['multimodal'] if row['video']]
    name = public_name(case)
    # Reproduce the deployed raw-number presentation, with authentic headers.
    for parent in (case.layout.processed/'Clips', case.layout.understanding):
        (parent/name).rename(parent/'capture')
    raw_refs = [r | {'path': r['path'].replace('/'+name+'/', '/capture/')} for r in references]
    real = clip_delivery._publish_aliases
    monkeypatch.setattr(clip_delivery, '_publish_aliases', lambda *a: (_ for _ in ()).throw(OSError('group crash')))
    with pytest.raises(OSError, match='group crash'):
        clip_delivery.rename_original_groups(case.layout, clip_delivery.original_units(case.layout, case.index))
    monkeypatch.setattr(clip_delivery, '_publish_aliases', real)
    again = publish_content(case.config, case.index)
    for parent in (case.layout.processed/'Clips', case.layout.understanding):
        assert [p.name for p in parent.iterdir() if p.is_dir() and not p.name.startswith('.')] == [name]
    assert all(verify_artifact(case.layout.root, r) for r in raw_refs+references)
    assert sum(bool(row['video']) for row in again['multimodal']) == 2
    assert len(case.calls) == 2


def test_group_rename_refuses_unowned_destination(exact_case):
    case = exact_case
    units = clip_delivery.original_units(case.layout, case.index)
    unit = units['MetaData/capture']
    old = case.layout.processed/'Clips'/'capture'
    atomic_json(old/'Index.json', {'schema_version': clip_delivery.VERSION, 'source_unit': unit['key']})
    target = old.parent/unit['name']
    atomic_json(target/'Foreign.json', {'untouched': True})
    with pytest.raises(ValueError, match='destination changed'):
        clip_delivery.rename_original_groups(case.layout, units)
    assert (old/'Index.json').is_file() and (target/'Foreign.json').is_file()


def test_processing_states_stay_in_json_not_in_public_filenames(exact_case):
    case = exact_case
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    result = publish_content(case.config, case.index)
    assert 'NoExperimentActivity' in public_name(case)
    assert all('Pending' not in row['content'] for row in result['multimodal'])
    assert any(row['display_category'] == 'ActivityPendingClassification' for row in result['multimodal'])
    assert sum(bool(row['video']) for row in result['multimodal']) == 2


def test_recovered_file_uses_its_own_recorded_rgb_extent_not_shared_window(exact_case):
    case = exact_case
    atomic_json(case.source.parent/'meta.json', {
        'recording_session_id': 1792640296154871, 'global_segment_index': 0,
        'recording_quality_window_start_global_us': 1792640297154844,
        'recording_quality_window_end_global_us': 1792640404193809,
        'recording_window_first_valid_rgb_global_us': 1792640321176943,
        'recording_window_last_valid_rgb_global_us': 1792640374098139})
    unit = clip_delivery.original_units(case.layout, case.index)['MetaData/capture']
    assert unit['name'].startswith('11-38-41_11-39-34_')
    assert unit['time_basis'] == 'recorded_rgb_frame_extent'
    assert unit['capture_segment_index'] == 0


def test_retirement_does_not_follow_a_reclaimed_history_alias_back_to_source(exact_case):
    from visioncortex.device_day_contract import artifact, read_json

    case = exact_case
    source = case.layout.processed/'Clips'/'OwnedGroup'/'OwnedInterval.mp4'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b'previous-verified-derived-video')
    ref = artifact(case.layout.root, source)
    hidden = f"ProcessedClips/.Evidence/LegacyPresentation/OwnedGroup/{ref['sha256']}/OwnedInterval.mp4"
    clip_delivery._mapping(case.layout, 'ProcessedClips', {hidden: ref['path']})
    clip_delivery._retire_file(case.layout, ref['path'], ref)
    assert not source.exists()
    assert (case.layout.root/hidden).read_bytes() == b'previous-verified-derived-video'
    assert verify_artifact(case.layout.root, ref)
    journals = list((case.layout.processed/'.Evidence/ClipFileRelocations').glob('*.json'))
    assert all(read_json(p)['from'] != read_json(p)['to'] for p in journals)


@pytest.mark.parametrize('status', ['running', 'paused_for_live', 'scheduled_checkpoint'])
def test_reanalysis_cannot_replace_published_video_and_understanding_with_empty_groups(exact_case, status):
    from visioncortex.receipt_projection import ProjectionGenerationChanged
    case = exact_case
    case.layout.backend_root = case.layout.root.parent/'Backend'
    case.config['storage']['local_cache_root'] = str(case.layout.backend_root)
    case.config['device_day']['clip_delivery'] = 'source_recording_v2'
    retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    publish_content(case.config, case.index)
    parents = (case.layout.processed/'Clips'/public_name(case), case.layout.understanding/public_name(case))
    before = {p: p.read_bytes() for folder in parents for p in folder.iterdir() if p.is_file()}
    record = case.index['recordings'][0]
    record['sources'] = [{'kind': 'video', 'retained': case.candidate['source_ref']}]
    case.index.update(segments=[], understandings=[])
    atomic_json(case.layout.receipts/record['recording_id']/'vision.json',
                {'stage': 'vision', 'status': status, 'recording_id': record['recording_id'], 'key': 'new-key'})
    with pytest.raises(ProjectionGenerationChanged):
        publish_content(case.config, case.index)
    assert before and all(p.is_file() and p.read_bytes() == content for p, content in before.items())
    # A genuinely completed empty new result may replace the previous view.
    atomic_json(case.layout.receipts/record['recording_id']/'vision.json',
                {'stage': 'vision', 'status': 'completed', 'recording_id': record['recording_id'], 'key': 'new-key'})
    publish_content(case.config, case.index)
