import hashlib
import json

import pytest

from visioncortex.comment_layout import current_audio_references, publish_legacy_index
from visioncortex.device_day_contract import artifact, atomic_json, digest, safe_child
from visioncortex.device_day_verification import ArtifactVerifier


GROUP = '10-00-00.000000_10-00-05.000000'


def test_current_links_change_but_speech_identity_and_times_do_not():
    old = f'Comment/{GROUP}/Transcript.txt'
    new = f'Comment/Audio/{GROUP}/Transcript.txt'
    ref = dict(sha256='new', size_bytes=12, byte_size=12)
    value = dict(entries=[dict(recording_id='recording', start_us=1, end_us=2, text=old)],
                 streams=dict(audio=[dict(transcription=dict(sentences=[dict(text=old, transcript_file=dict(path=old, sha256='old', size_bytes=5))]))]),
                 readable_content=dict(audio=[dict(text=old)], multimodal=[], digest='old'))
    got = current_audio_references(value, [GROUP], {new: ref})
    assert got['entries'][0] == dict(recording_id='recording', start_us=1, end_us=2, text=new)
    sentence = got['streams']['audio'][0]['transcription']['sentences'][0]
    assert sentence['text'] == old
    assert sentence['transcript_file'] == dict(path=new, sha256='new', size_bytes=12)
    assert got['readable_content']['digest'] == digest([[dict(text=new)], []])
    assert value['entries'][0]['text'] == old
    with pytest.raises(ValueError):
        current_audio_references(value, ['../bad'], {})


def test_old_receipt_alias_reads_original_bytes_after_directory_removal(tmp_path):
    folder = tmp_path/'Comment'/GROUP
    folder.mkdir(parents=True)
    original = folder/'Transcript.txt'
    original.write_bytes(b'old presentation')
    receipt = artifact(tmp_path, original)
    audio = tmp_path/'Comment/Audio'/GROUP
    audio.mkdir(parents=True)
    (audio/'Transcript.txt').write_bytes(b'new readable presentation')
    (audio/'Transcript.original.txt').write_bytes(original.read_bytes())
    atomic_json(tmp_path/'ProcessedClips/.Evidence/CommentLegacyPaths.json',
                dict(schema_version='visioncortex-content-aliases/1', aliases={
                    f'Comment/{GROUP}': f'Comment/Audio/{GROUP}',
                    receipt['path']: f'Comment/Audio/{GROUP}/Transcript.original.txt'}))
    original.unlink();folder.rmdir()
    assert safe_child(tmp_path, receipt['path']).read_bytes() == b'old presentation'
    assert ArtifactVerifier().verify(tmp_path, receipt)
    (audio/'Transcript.original.txt').write_bytes(b'tampered')
    assert not ArtifactVerifier().verify(tmp_path, receipt)


def test_compatibility_index_refresh_is_idempotent_and_does_not_create_time_folder(tmp_path):
    value = dict(schema_version='example/1', entries=[dict(content=f'Comment/Audio/{GROUP}/Transcript.json')])
    publish_legacy_index(tmp_path, 'Index.json', value)
    path = tmp_path/'Comment/Index.json'
    stamp = path.stat().st_mtime_ns
    publish_legacy_index(tmp_path, 'Index.json', value)
    assert path.stat().st_mtime_ns == stamp
    assert json.loads(path.read_bytes()) == value
    assert not (tmp_path/'Comment'/GROUP).exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        publish_legacy_index(tmp_path, 'Transcript.txt', value)


def test_frozen_publication_uses_its_own_injected_io_and_keeps_old_entry(tmp_path, monkeypatch):
    from visioncortex.lab.native_comment_layout import module
    owner = module('device_day_file_index')
    archive = '2026-09-17_camera_cam01'
    value = dict(archive=archive, schema_version='visioncortex-file-time-index/2')
    monkeypatch.setattr(owner, 'build_file_index', lambda *args: value)
    monkeypatch.setattr(owner, 'verified_field_events_reference', lambda *args: None)
    writes = []
    def write(path, data):
        writes.append(path)
        atomic_json(path, data)
    monkeypatch.setattr(owner, 'atomic_json', write)
    index = dict(archive=archive)
    owner.publish_file_index(dict(storage=dict(archive_root=str(tmp_path))), index, {})
    assert writes == [tmp_path/archive/'Comment/Audio/TimeIndex.json',
                      tmp_path/archive/'Comment/TimeIndex.json']


def test_readable_field_summary_preserves_canonical_evidence_verification(tmp_path):
    from visioncortex.comment_evidence import backend_path
    from visioncortex.comment_readability import field_context_document
    from visioncortex.device_day_file_index import verified_field_events_reference
    root = tmp_path/'2026-09-17_camera_cam01'
    backend = tmp_path/'backend'
    canonical = dict(schema_version='visioncortex-field-archive-context/1', archive=root.name,
                     source='field_recognition', model_invoked=False, physical_action_verified=False,
                     status='not_received', events=[], errors=[], source_references=[],
                     association_basis='same_camera_actual_interval_overlap_not_action_confirmation')
    public = root/'Comment/OCR/FieldEvents.json'
    atomic_json(public, field_context_document(canonical))
    saved = backend_path(root, 'Comment/.Evidence/FieldRecognition/'+digest(canonical)+'.json', backend)
    atomic_json(saved, canonical)
    old = saved.read_bytes()
    config = dict(storage=dict(local_cache_root=str(backend)), device_day=dict(field_recognition=dict(archive_context_enabled=True)))
    index = dict(archive=root.name, context={})
    reference = verified_field_events_reference(config, index, root)
    assert reference['path'] == 'Comment/OCR/FieldEvents.json'
    assert reference['sha256'] == hashlib.sha256(public.read_bytes()).hexdigest()
    assert saved.read_bytes() == old
    changed = field_context_document(canonical)
    changed['events'] = [dict(event_id='unrelated')]
    atomic_json(public, changed)
    assert verified_field_events_reference(config, index, root) is None
