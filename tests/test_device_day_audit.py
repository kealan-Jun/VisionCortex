from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_device_day import device_config  # noqa: F401
from visioncortex.device_day_audit import AuditReader, compact_index_visual, persist
from visioncortex.device_day_contract import DeviceDayLayout, digest, read_json
from visioncortex.device_day_timeline import build_timeline, day_bounds


DAY = '2026-09-14'


def fixture(tmp_path):
    layout = DeviceDayLayout(tmp_path/'archive', 'camera', day_bounds(DAY)[0], tmp_path/'cache')
    audit = {'scope': 'single_device_activity_not_cross_view_experiment_confirmation',
             'events': [{'event_id': 'one', 'candidates': [{
                 'candidate_id': 'candidate', 'local_start_ms': 0, 'local_end_ms': 1000,
                 'evidence': [{'frame': i, 'details': 'unchanged-witness' * 50} for i in range(100)]}]}],
             'selected_key_events': [{'event_id': 'selected', 'witness': {'frame': 5}}],
             'key_selection_summary': {'selected_events': 1}, 'rejected': [{'reason': 'not accepted'}],
             'physical_action_confirmed': False, 'cross_camera_alignment_verified': False}
    return layout, {'storage': {'local_cache_root': str(layout.backend_root)}}, audit


def test_full_audit_is_stored_once_and_segments_keep_exact_selected_witness(tmp_path):
    layout, config, audit = fixture(tmp_path)
    original = deepcopy(audit)
    summary, ref = persist(layout, 'recording', 'key', 0, audit)
    batch = {'activity_audit': summary, 'activity_audit_ref': ref}
    segments = [dict(batch) for _ in range(10)]
    files = list(layout.receipts.rglob('ActivityAudit-*.json'))
    assert len(files) == 1
    assert read_json(files[0]) == audit == original
    assert summary['selected_key_events'] == audit['selected_key_events']
    assert summary['event_count'] == 1 and summary['rejected_count'] == 1
    assert summary['physical_action_confirmed'] is False
    assert 'events' not in summary
    reader = AuditReader(config)
    assert all(reader.read(row) == audit for row in [batch, *segments])
    assert digest(reader.read(batch)) == digest(audit)
    assert len(json.dumps([batch, *segments])) < len(json.dumps([{'activity_audit': audit}] * 11)) / 10


def test_reference_and_legacy_audits_produce_same_timeline_candidates(tmp_path):
    layout, config, audit = fixture(tmp_path)
    summary, ref = persist(layout, 'recording', 'key', 0, audit)
    start = day_bounds(DAY)[0]
    segment = {'recording_id': 'recording', 'segment_id': 'segment', 'start_us': start,
               'end_us': start + 1000000, 'start_ms': 0, 'end_ms': 1000, 'activity': 'active',
               'key_frames': [], 'scene_frames': [], 'activity_audit': audit}
    old = build_timeline(DAY, [(layout.name, {'segments': [segment]})])
    new = build_timeline(DAY, [(layout.name, {'segments': [segment | {
        'activity_audit': summary, 'activity_audit_ref': ref}]})], config=config)
    assert old['entries'] == new['entries']
    assert old['periods'] == new['periods']


def test_missing_or_modified_audit_never_falls_back_to_summary(tmp_path):
    layout, config, audit = fixture(tmp_path)
    summary, ref = persist(layout, 'recording', 'key', 0, audit)
    reader = AuditReader(config)
    item = {'activity_audit': summary, 'activity_audit_ref': ref}
    assert reader.read(item) == audit
    path = Path(config['storage']['local_cache_root']) / ref['path']
    path.write_bytes(b'{}')
    with pytest.raises(ValueError, match='content verification'):
        reader.read(item)
    with pytest.raises(ValueError, match='reference is missing'):
        reader.read({'activity_audit': summary})
    with pytest.raises(ValueError, match='storage root'):
        AuditReader({}).read(item)
    path.unlink()
    with pytest.raises(OSError):
        reader.read(item)


def test_repeated_reference_reuses_one_decode_and_reads_legacy_inline(tmp_path, monkeypatch):
    from visioncortex import device_day_audit as module
    layout, config, audit = fixture(tmp_path)
    summary, ref = persist(layout, 'recording', 'key', 0, audit)
    item = {'activity_audit': summary, 'activity_audit_ref': ref}
    original = module.read_json
    calls = []
    def read(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(module, 'read_json', read)
    reader = AuditReader(config)
    assert reader.read(item) == reader.read(item) == audit
    assert len(calls) == 1
    assert reader.read({'activity_audit': audit}) is audit


def test_historical_index_compaction_preserves_original_receipt_and_one_full_audit(tmp_path):
    layout, config, audit = fixture(tmp_path)
    original = {'status': 'completed', 'key': 'original-execution-key',
                'batches': [{'activity_audit': audit}],
                'segments': [{'segment_id': str(i), 'activity_audit': deepcopy(audit)} for i in range(5)],
                'audit_artifacts': [{'path': 'existing-original-proof'}]}
    before = digest(original)
    projected = compact_index_visual(layout, original, recording_id='recording')
    assert digest(original) == before
    assert all('activity_audit_ref' not in s for s in original['segments'])
    assert len(list(layout.receipts.rglob('ActivityAudit-*.json'))) == 1
    reader = AuditReader(config)
    assert all(reader.read(row) == audit for row in [*projected['batches'], *projected['segments']])
    assert projected['key'] == original['key']
    assert projected['audit_artifacts'][0] == original['audit_artifacts'][0]
    assert len(projected['audit_artifacts']) == 2
    assert compact_index_visual(layout, projected, recording_id='recording') == projected


def test_different_historical_audits_never_share_reference_by_matching_counts(tmp_path):
    layout, config, audit = fixture(tmp_path)
    other = deepcopy(audit)
    other['events'][0]['candidates'][0]['evidence'][0]['details'] = 'different-observed-evidence'
    original = {'status': 'completed', 'key': 'key', 'batches': [{'activity_audit': audit}],
                'segments': [{'activity_audit': other}]}
    projected = compact_index_visual(layout, original, recording_id='recording')
    reader = AuditReader(config)
    assert reader.read(projected['batches'][0]) == audit
    assert reader.read(projected['segments'][0]) == other
    assert len(list(layout.receipts.rglob('ActivityAudit-*.json'))) == 2


def test_verified_content_addressed_audit_is_not_rewritten_per_refresh(tmp_path, monkeypatch):
    from visioncortex import device_day_audit as module
    layout, config, audit = fixture(tmp_path)
    audit['status'] = 'coarse_no_candidates'
    original = module.atomic_json
    calls = []
    def write(path, value):
        calls.append(path)
        original(path, value)
    monkeypatch.setattr(module, 'atomic_json', write)
    first = persist(layout, 'recording', 'key', 0, audit)
    second = persist(layout, 'recording', 'key', 0, deepcopy(audit))
    assert first == second and len(calls) == 1
    assert first[0]['status'] == audit['status']
    # Stat/content changes revoke reuse; only the complete authoritative audit
    # supplied to persist can restore this generated, content-addressed file.
    calls[0].write_text('{}')
    restored = persist(layout, 'recording', 'key', 0, audit)
    assert len(calls) == 2
    assert AuditReader(config).read({'activity_audit_ref': restored[1]}) == audit


def test_index_projection_keeps_authoritative_receipt_and_completed_semantic_cache(device_config, tmp_path):  # noqa: F811
    from test_device_day import FakeModels, capture, item_and_layout
    from visioncortex.device_day import DeviceDayRunner
    capture(device_config)
    record, layout = item_and_layout(device_config)
    _, _, audit = fixture(tmp_path/'audit-fixture')
    backend = FakeModels()
    original = backend.vision
    def vision(*args):
        result = original(*args)
        result['batches'] = [{'activity_audit': audit}]
        for segment in result['segments']:
            segment['activity_audit'] = deepcopy(audit)
        return result
    backend.vision = vision
    runner = DeviceDayRunner(device_config, backend)
    assert runner.process(record)['status'] == 'completed'
    receipt = layout.receipts/record['recording_id']/'vision.json'
    receipt_bytes = receipt.read_bytes()
    published = read_json(layout.index)
    assert published['understandings']
    assert published['segments'][0]['activity_audit_ref']
    assert published['recordings'][0]['processing']['batches'][0]['activity_audit_ref']
    audit_path = layout.backend_root/published['segments'][0]['activity_audit_ref']['path']
    before = audit_path.stat().st_mtime_ns
    assert runner.refresh_index(layout, recording_id=record['recording_id'])
    assert receipt.read_bytes() == receipt_bytes
    assert audit_path.stat().st_mtime_ns == before
    assert read_json(layout.index)['understandings'] == published['understandings']
    assert runner.process(record)['status'] == 'completed'
    assert backend.vision_calls == backend.semantic_calls == 1
