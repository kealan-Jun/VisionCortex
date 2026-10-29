import pytest

from visioncortex.comment_evidence import backend_path, resolve
from visioncortex.comment_evidence_migration import migrate
from visioncortex.device_day_contract import DeviceDayLayout, artifact, atomic_json
from visioncortex.device_day_verification import verify_artifact_cached


def layout(tmp_path):
    return DeviceDayLayout(tmp_path/'Archives', 'camera', 1793232000000000, tmp_path/'Backend')


def test_migration_preserves_public_data_receipts_aliases_and_exact_bytes(tmp_path):
    day = layout(tmp_path)
    source = day.comments/'.Evidence/Recognition/r/key/Transcript.json'
    atomic_json(source, {'text': '实际转写'})
    original = source.read_bytes()
    ref = artifact(day.root, source)
    public = day.comments/'12-00-00_12-01-00/Transcript.txt'
    public.parent.mkdir(parents=True)
    public.write_text('实际转写')
    raw = day.raw/'r/audio.opus'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'original audio')
    receipt = day.receipts/'r/stt.json'
    atomic_json(receipt, {'status': 'completed', 'artifacts': [ref]})
    before = {p: p.read_bytes() for p in (public, raw, receipt)}
    atomic_json(day.comments/'.Evidence/LegacyPaths.json', {'aliases': {'Comment/Stt/r': 'Comment/.Evidence/Recognition/r'}})
    result = migrate(day)
    assert result[0]['status'] == 'completed' and result[0]['model_calls'] == 0
    assert not (day.comments/'.Evidence').exists()
    assert resolve(day.root, ref['path'], day.backend_root).read_bytes() == original
    assert resolve(day.root, 'Comment/Stt/r/key/Transcript.json', day.backend_root).read_bytes() == original
    assert verify_artifact_cached(day.root, ref, backend_root=day.backend_root)
    assert before == {p: p.read_bytes() for p in before}
    assert migrate(day) == result
    target = backend_path(day.root, ref['path'], day.backend_root)
    target.write_bytes(b'corrupt')
    assert not verify_artifact_cached(day.root, ref, backend_root=day.backend_root)
    target.unlink()
    assert not verify_artifact_cached(day.root, ref, backend_root=day.backend_root)


@pytest.mark.parametrize('path', ['Comment/.Evidence/../../outside', 'Comment/.Evidence/a/../b',
    '/Comment/.Evidence/a', 'MetaData/a', 'Comment/.Evidence/a:stream'])
def test_backend_resolution_does_not_grant_other_paths(tmp_path, path):
    day = layout(tmp_path)
    with pytest.raises(ValueError):
        backend_path(day.root, path, day.backend_root)


def test_conflicting_or_linked_destination_does_not_retire_source(tmp_path):
    day = layout(tmp_path)
    source = day.comments/'.Evidence/Recognition/r/Result.json'
    atomic_json(source, {'real': True})
    target = backend_path(day.root, day.relative(source), day.backend_root)
    target.parent.mkdir(parents=True)
    target.write_bytes(b'different')
    with pytest.raises(ValueError, match='collision'):
        migrate(day)
    assert source.is_file()
    target.unlink()
    target.symlink_to(source)
    with pytest.raises(ValueError):
        migrate(day)
    assert source.is_file()


def test_partial_copy_keeps_old_unmoved_refs_until_commit(tmp_path):
    day = layout(tmp_path)
    source = day.comments/'.Evidence/Recognition/old/Result.json'
    atomic_json(source, {'real': True})
    ref = artifact(day.root, source)
    atomic_json(backend_path(day.root, 'Comment/.Evidence/Recognition/new/Result.json', day.backend_root), {'new': True})
    assert verify_artifact_cached(day.root, ref, backend_root=day.backend_root)
