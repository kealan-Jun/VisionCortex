"""Native publication must not promote an unmarked historical CV candidate."""
from copy import deepcopy
from pathlib import Path

import pytest

from test_frozen_archive_layout import fixture as content_fixture
from test_retained_video_subintervals import exact_case
from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_content_paths import aliases, clip_name, private_folder, publish_selected_videos
from visioncortex.device_day_contract import DeviceDayLayout, artifact, atomic_json, read_json
from visioncortex.device_day_file_index import FileIndexPublisher
from visioncortex.observed_inventory import observe

__all__ = ['exact_case']


def legacy_candidate(case):
    parent = case.candidate
    parent['json_path'] = 'ProcessedClips/Clips/'+clip_name(parent)+'/ExperimentActivity.json'
    old = case.layout.processed/'.Evidence/Historical/Activity.mp4'
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b'previous whole CV candidate, never an exact selected child')
    parent['video'] = artifact(case.layout.root, old)
    parent['derived_video_disposition'] = {'decision': 'retain_derivative', 'reason': 'historical selection policy'}
    target = case.layout.root/parent['json_path'].replace('.json', '.mp4')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'different existing immutable generation')
    # Both versions have their own valid complete reference, as in the real
    # collision. Neither authorizes adopting or replacing the other version.
    atomic_json(old.with_suffix('.json'), {'segment_id': parent['segment_id'], 'video': parent['video']})
    atomic_json(target.with_suffix('.json'), {'segment_id': parent['segment_id'],
                                            'video': artifact(case.layout.root, target)})
    return old, target


def test_native_cold_ticks_leave_colliding_whole_candidate_as_history(exact_case, monkeypatch):
    case = exact_case
    # The small unit fixture starts at epoch+1s; a native day needs a positive
    # Shanghai midnight as well as a positive capture timestamp.
    delta = 1792376825000000 - case.candidate['start_us']
    original_root = case.layout.root
    layout = DeviceDayLayout(original_root.parent, case.layout.name[11:], case.candidate['start_us']+delta)
    relative_source = case.source.relative_to(original_root)
    original_root.rename(layout.root)  # owned temporary fixture only
    case.layout, case.source = layout, layout.root/relative_source
    case.index['archive'] = layout.name
    for row in (case.candidate, case.record, *case.intervals):
        for key in ('start_us', 'end_us'):
            row[key] += delta
    # Prepare two exact receipts through their original owner, then recreate an
    # older canonical generation with no exact/historical markers at all.
    publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    case.candidate.pop('retention_intervals')
    old, target = legacy_candidate(case)
    case.candidate['activity_label'] = '实验活动'
    case.record['processing'] = {'clock_mapping': {'origin_us': case.candidate['start_us']},
                                 'batches': [{'dense': 'preserve canonical bytes'}]}
    case.candidate['activity_audit'] = {'dense': ['preserve the complete CV evidence']}
    strong_result = case.layout.understanding/'.Evidence/Analysis/strong/Result.json'
    atomic_json(strong_result, {'model_result': {'status': 'completed', 'summary': 'owned existing strong evidence'}})
    case.index['understandings'] = [{'segment_id': case.candidate['segment_id'], 'status': 'completed',
        'windows': [{'summary': 'Retain the complete nested strong observation',
                     'model_receipt': strong_result.relative_to(case.layout.root).as_posix(),
                     'uncertainties': ['Physical action remains unproven in this fixture']}]}]
    before = deepcopy(case.index)
    original_bytes = {path: path.read_bytes() for path in
                      (old, old.with_suffix('.json'), target, target.with_suffix('.json'), case.source, strong_result)}
    case.config['storage']['local_runtime_root'] = str(case.layout.root/'OwnedRuntime')
    case.config['collection_ingest'] = {'source_root': str(case.layout.root)}
    case.config['device_day']['readable_content_enabled'] = True
    atomic_json(case.layout.index, before)
    observe(Path(case.config['storage']['local_runtime_root'])/'device-day', {'recordings': [{
        'recording_id': case.record['recording_id'], 'camera_key': case.layout.name[11:],
        'recording_start_us': case.candidate['start_us'], 'recording_end_us': case.candidate['end_us']}]})
    monkeypatch.setattr('visioncortex.video_io.extract_clip', lambda *_a, **_k: pytest.fail('No repeated/full export'))
    monkeypatch.setattr('visioncortex.device_day.DeviceDayRunner._backend',
                        lambda *_a, **_k: pytest.fail('No model backend execution'))
    for _ in range(2):
        publisher = FileIndexPublisher(case.config)
        outcome = publisher.tick(archives=[case.layout.name])
        assert outcome['published'] == 1, outcome
        assert all(row['operation'] == 'canonical_video_publication' for row in outcome['errors']), outcome
        canonical = read_json(case.layout.index)
        canonical.pop('time_index')
        assert canonical == before
        entries = read_json(case.layout.comments/'TimeIndex.json')['streams']['video']
        selected = [row for row in entries if row.get('video_value') == 'selected']
        assert [(row['source_start_ms'], row['source_end_ms']) for row in selected] == [(130000, 155000), (540000, 560000)]
        assert all(row['record_scope'] == 'retention_interval' for row in selected)
        assert all(row['file']['path'] != before['segments'][0]['video']['path'] for row in selected)
        daily = read_json(case.layout.reports/'LaboratoryDailyReport.json')
        assert daily['视频留存统计']['selected_interval_count'] == 2
        assert daily['视频留存统计']['actual_video_duration_seconds'] == 45
        assert {path: path.read_bytes() for path in original_bytes} == original_bytes
        assert before['segments'][0]['video']['path'] not in aliases(case.layout.root)
    assert len(case.calls) == 2  # Both exact outputs were prepared before the ticks.


def test_exact_mode_skips_unmarked_parent_before_paths_or_byte_reads(exact_case, monkeypatch):
    case = exact_case
    old, target = legacy_candidate(case)
    case.candidate['json_path'] = 'ProcessedClips/legacy-invalid-path/Activity.json'
    before, original = deepcopy(case.candidate), {path: path.read_bytes() for path in (old, target, case.source)}
    monkeypatch.setattr('visioncortex.device_day_content_paths.clip_name', lambda *_: pytest.fail('Parent path inspected'))
    progress = {}
    assert publish_selected_videos(case.layout, [case.candidate], exact_intervals_only=True, progress=progress) == []
    assert progress == {'pending': 0} and case.candidate == before
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize('change', [
    {'video_value': None}, {'video_value': 'pending'}, {'video_value': 'confirmed_no_value'},
    {'candidate_segment_id': None}, {'candidate_segment_id': 'same'},
    {'derived_video_disposition': {}},
    {'derived_video_disposition': {'selection_policy': 'unknown', 'decision': 'retain_derivative'}},
    {'derived_video_disposition': {'selection_policy': 'visioncortex-exact-value-intervals/1',
                                 'decision': 'pending_value_review'}},
])
def test_ineligible_flat_child_cannot_copy_or_lookup_a_receipt(exact_case, monkeypatch, change):
    row = {'segment_id': 'same', 'candidate_segment_id': 'parent', 'video_value': 'selected',
           'derived_video_disposition': {'selection_policy': 'visioncortex-exact-value-intervals/1',
                                        'decision': 'retain_derivative'}} | change
    monkeypatch.setattr('visioncortex.application.retained_video.published_material',
                        lambda *_: pytest.fail('Ineligible selection consulted publication receipt'))
    progress = {}
    assert publish_selected_videos(exact_case.layout, [row], exact_intervals_only=True, progress=progress) == []
    assert progress == {'pending': 0}


def legacy_exact_child(case):
    result = publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    row = result['selected_intervals'][0]
    target = case.layout.root/row['video']['path']
    legacy = case.layout.processed/'.Evidence/LegacyPresentation/ExactHistory'/row['segment_id']/'Activity.mp4'
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(target.read_bytes())
    reference = artifact(case.layout.root, legacy)
    manifest = case.layout.root/row['receipt']['path']
    value = read_json(manifest)
    value['video'] = reference
    atomic_json(manifest, value)
    pointer = private_folder(case.layout, 'ProcessedClips', 'Retention', row['segment_id'], 'Published.json')
    atomic_json(pointer, {'receipt': artifact(case.layout.root, manifest)})
    row.update(video=reference, derived_video_disposition=value)
    return row, legacy, target


def test_exact_child_reuses_original_receipt_byte_verifier_and_is_idempotent(exact_case):
    case = exact_case
    row, legacy, target = legacy_exact_child(case)
    original = target.read_bytes()
    assert len(publish_selected_videos(case.layout, [row], exact_intervals_only=True)) == 1
    assert target.read_bytes() == legacy.read_bytes() == original
    assert publish_selected_videos(case.layout, [row], exact_intervals_only=True) == []
    assert len(case.calls) == 2


def test_verified_exact_child_still_refuses_a_different_immutable_target(exact_case):
    row, legacy, target = legacy_exact_child(exact_case)
    target.write_bytes(b'a different legitimate historical generation')
    before = legacy.read_bytes(), target.read_bytes()
    with pytest.raises(ValueError, match='Immutable retained source collision'):
        publish_selected_videos(exact_case.layout, [row], exact_intervals_only=True)
    assert (legacy.read_bytes(), target.read_bytes()) == before


@pytest.mark.parametrize('change', [
    {'candidate_segment_id': 'another-parent'}, {'segment_id': 'another-child'},
    {'derived_video_disposition': {'selection_policy': 'visioncortex-exact-value-intervals/1',
                                 'decision': 'retain_derivative'}},
])
def test_claimed_exact_child_needs_a_matching_complete_publication_receipt(exact_case, change):
    row, legacy, target = legacy_exact_child(exact_case)
    row = deepcopy(row) | change
    before = legacy.read_bytes(), target.read_bytes()
    with pytest.raises(ValueError, match='matching verified publication receipt'):
        publish_selected_videos(exact_case.layout, [row], exact_intervals_only=True)
    assert (legacy.read_bytes(), target.read_bytes()) == before


@pytest.mark.parametrize(('function', 'operation'), [
    ('migrate_display_layout', 'frozen_layout_migration'),
    ('publish_selected_videos', 'legacy_selected_video_publication'),
])
def test_native_failures_identify_the_actual_owner_and_keep_bounded_reason(tmp_path, monkeypatch, function, operation):
    config, index = content_fixture(tmp_path)
    config['storage']['local_runtime_root'] = str(tmp_path/'OwnedRuntime')
    config['device_day']['readable_content_enabled'] = True
    atomic_json(tmp_path/index['archive']/'ProcessedClips/Index.json', index)
    observe(tmp_path/'OwnedRuntime/device-day', {'recordings': [{
        'recording_id': 'recording', 'camera_key': index['archive'][11:],
        'recording_start_us': index['segments'][0]['start_us']}]})
    def failed(*_args, **kwargs):
        if function == 'publish_selected_videos':
            assert kwargs['exact_intervals_only'] is True
        raise ValueError('owned immutable conflict: '+'x'*800)
    monkeypatch.setattr('visioncortex.device_day_content_paths.'+function, failed)
    publisher = FileIndexPublisher(config)
    outcome = publisher.tick(archives=[index['archive']])
    assert outcome['published'] == 0 and outcome['status'] == 'partial'
    assert outcome['errors'] == [{'archive': index['archive'], 'operation': operation,
        'error_type': 'ValueError', 'reason': ('owned immutable conflict: '+'x'*800)[:500],
        'automatic_action': 'preserve_existing_bytes_and_retry_verified_publication'}]
    assert index['archive'] not in publisher.versions
