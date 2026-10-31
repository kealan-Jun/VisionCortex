"""Local candidate/native schema publication; generated fixtures, no model calls."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from visioncortex.lab.archive_layout import contract_bytes
from visioncortex.lab.domain import Conflict
from visioncortex.lab.timeline_native_artifacts import artifact_documents
from visioncortex.lab.timeline_publication import TimelinePublication

from .test_person_report import scene, media, report
from .test_operator_scan import scan, rows

__all__ = ['scene']


def submit(s, source, kind, events):
    job = s['runtime'].tasks.claim('artifact-claim:' + source['media_id'], kind, 'synthetic-supplied-no-inference/1')
    assert job['request']['primary_media_id'] == source['media_id']
    result = dict(schema_version='vc-lab-timeline-model-result/1', input_fingerprint=job['request']['input_fingerprint'],
                  covered_window_indices=list(range(len(job['request']['windows']))), events=events)
    return s['runtime'].tasks.submit('artifact-submit:' + source['media_id'], job['task_id'],
        job['lease_token'], 'synthetic-supplied-no-inference/1', result)


@pytest.fixture
def prepared(scene):
    s = scene
    basis = dict(basis='generated source clock', uncertainty_us=0, clock_sync_valid=True)
    photo = media(s, 'artifact-photo', kind='photo', basis=basis)
    ocr = submit(s, photo, 'ocr', [dict(kind='reading', text='550 990',
        spans=[dict(media_id=photo['media_id'], start_seconds='0', end_seconds='0')],
        payload=dict(raw_tokens=[dict(text='550 990', score='.9')], fields=[], field_mapping='unmapped_raw_ocr'))])
    second = media(s, 'artifact-other-camera', kind='photo', camera=1, basis=basis)
    other = submit(s, second, 'ocr', [dict(kind='reading', text='other camera candidate',
        spans=[dict(media_id=second['media_id'], start_seconds='0', end_seconds='0')], payload=dict(raw_tokens=[], fields=[]))])
    audio = media(s, 'artifact-audio', kind='audio', basis=basis)
    speech = submit(s, audio, 'transcription', [dict(kind='speech', text='原话候选，不作为确认指令。',
        spans=[dict(media_id=audio['media_id'], start_seconds='.25', end_seconds='1.25')],
        payload=dict(time_basis='whole_input_only_alignment_unavailable',
            asr_alignment=dict(selected_unit='whole_input_alignment_unavailable'),
            comment_candidate=dict(status='pending_interpretation', speaker_id=None)))])
    declaration = scan(s, 'artifact-explicit', run_id=s['run'])
    snapshot = report(s)
    archive = s['settings'].incoming
    (archive / '.System').mkdir()
    (archive / '.System/ArchiveContract.json').write_bytes(contract_bytes())
    settings = replace(s['settings'], source_root=archive, unified_archive_required=True, timeline_publish=True)
    publication = TimelinePublication(s['runtime'], settings)
    return dict(s=s, publication=publication, snapshot=snapshot, archive=archive,
                ocr=ocr, speech=speech, other=other, declaration=declaration)


def test_new_completed_timeline_tasks_publish_original_comment_schemas_and_shallow_bindings(prepared, tmp_path):
    p = prepared
    root = p['archive']
    original = root / '2026-10-29_fixture-camera-0/Comment/OCR/VoicePhotoReadings/producer-owned/Result.json'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'original producer bytes - never replace')
    result = p['publication'].tick()
    assert result['revision'] == p['snapshot']['revision']
    binding_path = root / '2026-10-29_fixture-camera-0/Comment/Bindings.json'
    value = json.loads(binding_path.read_text())
    assert value['schema_version'] == 'vc-lab-camera-bindings/1'
    assert value['revision'] == 1 and value['snapshot_revision'] == p['snapshot']['revision']
    assert original.read_bytes() == b'original producer bytes - never replace'
    assert value['material_quantity_written'] is False
    assert any(o['operator_scan_snapshot'] == p['declaration']['snapshot'] for o in value['observations'])
    assert {o['activity_id'] for o in value['observations']} >= set(p['ocr']['activity_ids'] + p['speech']['activity_ids'])
    assert not {o['activity_id'] for o in value['observations']} & set(p['other']['activity_ids'])
    other = json.loads((root / '2026-10-29_fixture-camera-1/Comment/Bindings.json').read_text())
    assert not {o['activity_id'] for o in other['observations']} & set(p['ocr']['activity_ids'] + p['speech']['activity_ids'])
    refs = value['comment_artifacts']
    reading_ref = next(r for r in refs if r['source_type'] == 'ocr')
    readings = json.loads((root / reading_ref['readings']).read_text())
    assert readings['schema'] == 'field-numeric-observations/1'
    assert readings['readings'][0]['display_value'] == '550 990'
    assert readings['readings'][0]['value'] is readings['readings'][0]['unit'] is None
    evidence = json.loads((root / reading_ref['evidence']).read_text())
    assert evidence['schema'] == 'field-recognition-event/2' and evidence['measurement_files'] == []
    assert evidence['model']['actual_task']['status'] == 'completed'
    assert not evidence['six_field_measurements_generated']
    assert (root / reading_ref['unbound_readings']).exists()
    transcript_ref = next(r for r in refs if r['source_type'] == 'speech')
    transcript = json.loads((root / transcript_ref['transcript']).read_text())
    assert transcript['schema_version'] == 'visioncortex-readable-transcript/1'
    assert transcript['alignment_status'] == 'alignment_unknown'
    assert transcript['sentences'][0]['source_start_seconds'] == '0.250000'
    assert transcript['speaker_attribution'] == 'unknown_not_inferred'
    assert '原话候选' in (root / transcript_ref['text']).read_text()
    assert binding_path.read_bytes() == (root / value['immutable_history']['relative_path']).read_bytes()
    assert rows(p['s'], 'quantity_ledger_entry') == rows(p['s'], 'fact_revision') == []
    (tmp_path / 'CommentPublicationProof.json').write_text(json.dumps(dict(result=result, bindings=value,
        original_producer_sha256=hashlib.sha256(original.read_bytes()).hexdigest(), actual_model_calls=0), indent=2) + '\n')


def revised(p):
    first = p['publication'].tick()
    path = p['archive'] / '2026-10-29_fixture-camera-0/Comment/Bindings.json'
    old = path.read_bytes()
    scan(p['s'], 'artifact-late-B', value=p['s']['devices'][1], when='09:00:02')
    snapshot = report(p['s'])
    second = p['publication'].tick()
    assert second['revision'] == snapshot['revision'] > first['revision']
    current = json.loads(path.read_text())
    assert current['revision'] == 2
    return path, old, path.read_bytes(), current


def test_publisher_release_does_not_clone_unchanged_completed_observation(prepared):
    from visioncortex.lab.timeline_native_artifacts import artifact_documents
    from visioncortex.lab.rules import Rules

    p = prepared
    p['publication'].tick()
    bindings = json.loads((p['archive'] / '2026-10-29_fixture-camera-0/Comment/Bindings.json').read_text())
    reference = next(r for r in bindings['comment_artifacts'] if r['source_type'] == 'ocr')
    value = json.loads((p['archive'] / reference['readings']).read_text())
    with p['s']['core'].db.transaction(readonly=True) as conn:
        task = conn.execute('SELECT request,result FROM hj_timeline_task WHERE customer_id=%s AND id=%s',
                            (p['s']['core'].customer, value['model']['task_id'])).fetchone()
    # Reuse the exact report activity/source snapshot used by publication.
    activities = p['snapshot']['document']['unassigned_activities']
    activity = next(a for a in activities if a['id'] == reference['activity_id'])
    args = dict(archive='2026-10-29_fixture-camera-0', camera='fixture-camera-0',
                activities=[activity], media_sources=value['source_refs'])
    original = artifact_documents(p['s']['core'], **args, rules=task['request']['processing_rules'])
    changed = artifact_documents(p['s']['core'], **args, rules=dict(Rules().identity(), revision='fixture-later-publisher', sha256='f'*64))
    assert changed == original
    assert reference['readings'] in changed[0]


@pytest.mark.parametrize('case', ['old_valid_revision', 'changed_same_revision'])
def test_new_latest_revision_guard_rejects_rollback_and_hash_tampering(prepared, case):
    p = prepared
    path, old, fresh, value = revised(p)
    if case == 'old_valid_revision':
        path.write_bytes(old)  # valid immutable old bytes are still a rollback
    else:
        value['material_quantity_written'] = True
        path.write_text(json.dumps(value))
    scan(p['s'], 'artifact-next', when='09:00:03')
    report(p['s'])
    with pytest.raises(Conflict, match='rolled back|immutable history'):
        p['publication'].tick()
    assert (p['archive'] / json.loads(fresh)['immutable_history']['relative_path']).read_bytes() == fresh


def test_new_older_snapshot_cannot_replace_latest_and_history_is_retained(prepared):
    from visioncortex.lab.components import pipeline_module
    from visioncortex.lab.storage import publication_config

    p = prepared
    path, old, fresh, value = revised(p)
    content = {k:v for k,v in json.loads(old).items() if k not in {'revision','view_fingerprint','immutable_history'}}
    guard = pipeline_module('device_day_write_guard')
    with p['s']['core'].db.transaction() as conn, guard.storage_guard(publication_config(p['publication'].settings)):
        receipts = p['publication']._write_latest(conn, p['archive'], path.relative_to(p['archive']).as_posix(), content, guard)
    assert receipts[0]['status'] == 'superseded_snapshot_not_written'
    assert path.read_bytes() == fresh
    assert (p['archive'] / json.loads(old)['immutable_history']['relative_path']).read_bytes() == old


def test_new_foreign_producer_bindings_is_never_overwritten(prepared):
    p = prepared
    path = p['archive'] / '2026-10-29_fixture-camera-0/Comment/Bindings.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"schema":"producer-owned-original"}')
    with pytest.raises(Conflict, match='foreign'):
        p['publication'].tick()
    assert path.read_bytes() == b'{"schema":"producer-owned-original"}'


def test_new_generic_unknown_time_activities_publish_by_received_partition_without_dating_source(scene, tmp_path):
    from visioncortex.lab.timeline_policy import ZONE

    s = scene
    photo = media(s, 'undated-photo-no-day-directory', kind='photo', captured=False)
    ocr = submit(s, photo, 'ocr', [dict(kind='reading', text='unknown-clock candidate',
        spans=[dict(media_id=photo['media_id'], start_seconds='0', end_seconds='0')],
        payload=dict(raw_tokens=[dict(text='24.2')], fields=[]))])
    audio = media(s, 'undated-audio-no-day-directory', kind='audio', captured=False)
    speech = submit(s, audio, 'transcription', [dict(kind='speech', text='无可靠全局时钟的原话候选。',
        spans=[dict(media_id=audio['media_id'], start_seconds='.25', end_seconds='1.25')],
        payload=dict(asr_alignment=dict(selected_unit='whole_input_alignment_unavailable')))])
    with s['core'].db.transaction(readonly=True) as conn:
        received = conn.execute('SELECT received_at FROM capture WHERE customer_id=%s AND id=%s',
            (s['core'].customer, photo['capture_id'])).fetchone()['received_at']
    day = str(received.astimezone(ZONE).date())
    snapshot = s['runtime'].views.snapshot(day)
    assert {a['id'] for a in snapshot['document']['unknown_time_activities']['activities']} >= set(ocr['activity_ids'] + speech['activity_ids'])
    archive = s['settings'].incoming
    (archive / '.System').mkdir()
    (archive / '.System/ArchiveContract.json').write_bytes(contract_bytes())
    settings = replace(s['settings'], source_root=archive, unified_archive_required=True, timeline_publish=True)
    published = TimelinePublication(s['runtime'], settings).tick()
    bindings = json.loads((archive / f'{day}_fixture-camera-0/Comment/Bindings.json').read_text())
    assert {r['activity_id'] for r in bindings['comment_artifacts']} == set(ocr['activity_ids'] + speech['activity_ids'])
    for source in bindings['media_sources'].values():
        assert source['archive_day_basis'] == 'capture_received_at_partition_only'
        assert source['capture_time_status'] == 'unknown' and source['source_reported_at'] is None
        assert source['receipt_day_is_not_physical_event_time']
    for ref in bindings['comment_artifacts']:
        path = ref.get('evidence') or ref['transcript']
        document = json.loads((archive / path).read_text())
        assert document['event_at'] is None and document['archive_day_is_not_physical_event_time']
        assert 'UnknownTime_' in path
        if ref['source_type'] == 'speech':
            assert document['start_us'] is document['sentences'][0]['start_us'] is None
            assert document['sentences'][0]['source_start_seconds'] == '0.250000'
    assert rows(s, 'quantity_ledger_entry') == rows(s, 'fact_revision') == []
    (tmp_path / 'UnknownClockPublicationProof.json').write_text(json.dumps(dict(
        received_day_is_archive_partition_only=day, publication=published, bindings=bindings,
        actual_model_calls=0), indent=2) + '\n')


class NoTaskDB:
    @contextmanager
    def transaction(self, **kwargs):
        yield SimpleNamespace(execute=lambda *a: (_ for _ in ()).throw(AssertionError('No task should be queried')))


def test_new_unknown_source_clock_keeps_relative_transcript_and_never_invents_unix_word_alignment():
    core = SimpleNamespace(db=NoTaskDB(), customer=uuid4())
    aid, mid = str(uuid4()), str(uuid4())
    activity = dict(id=aid, source_type='speech', kind='speech', text_content='原话，不推定业务完成。',
        start_at=None, end_at=None, model_revision='supplied-candidate', payload=dict(asr_alignment=dict(selected_unit='whole_input_alignment_unavailable')),
        evidence=[dict(id=str(uuid4()), media_id=mid, kind='audio', start_seconds='.0065', end_seconds='107.4')])
    docs, refs = artifact_documents(core, archive='2026-10-29_fixture-camera', camera='fixture-camera',
        activities=[activity], media_sources={mid:dict(source_sha256='a'*64, time_basis=dict(basis='filename', clock_sync_valid=False),
        source_reported_at='2026-10-29T09:00:00+08:00')}, rules=dict(revision='fixture', sha256='b'*64))
    transcript = docs[refs[0]['transcript']]
    assert transcript['start_us'] is transcript['end_us'] is None
    assert transcript['sentences'][0]['source_start_seconds'] == '.0065'
    assert transcript['sentences'][0]['start_us'] is None and transcript['alignment_status'] == 'alignment_unknown'
    assert not transcript['word_alignment_is_forced_alignment']
    assert not any('Result.json' in k or 'Measurements' in k for k in docs)
