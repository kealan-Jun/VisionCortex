"""Synthetic private storage bindings; no site share or marker is embedded."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from visioncortex.lab import storage
from visioncortex.lab.archive_layout import _immutable_write


def mounted(monkeypatch, tmp_path, *, wrong=None, drift=False):
    bindings = []
    for name in ('host', 'archive'):
        mount = tmp_path / ('mount-' + name)
        root = mount / 'fixture-project'
        root.mkdir(parents=True)
        marker = mount / 'marker.json'
        marker.write_text(json.dumps({'id': 'fixture-' + name}))
        bindings.append(dict(mount_path=str(mount), share='//fixture.invalid/' + name,
                             project_root=str(root), marker_path=str(marker),
                             marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest()))
    manifest = tmp_path / 'private-storage.json'
    manifest.write_text(json.dumps(dict(schema_version='vc-lab-storage/1', bindings=bindings)))
    monkeypatch.setenv('VC_LAB_STORAGE_MANIFEST', str(manifest))
    reads = []
    native_read = Path.read_bytes

    def read(path):
        reads.append(path)
        return native_read(path)

    def info(path):
        binding = next(b for b in bindings if Path(path).is_relative_to(Path(b['mount_path'])))
        return [dict(target=binding['mount_path'], source=binding['share'],
                     fstype='ext4' if str(path) == wrong else 'cifs')]

    monkeypatch.setattr(storage, '_mount_info', info)
    monkeypatch.setattr(Path, 'read_bytes', read)
    if drift:
        Path(bindings[1]['marker_path']).write_text('{"id":"drifted"}')
    return bindings, reads


def test_visible_root_verifies_both_shares_and_project_marker(monkeypatch, tmp_path):
    bindings, reads = mounted(monkeypatch, tmp_path)
    root = Path(bindings[1]['project_root']) / '2026-10-26_fixture-camera'
    digest = bindings[0]['marker_sha256']
    assert storage.verify_root(root, digest) == root
    config = storage.storage_binding(root, digest)
    assert config['runtime']['local_only'] is False
    checks = config['device_day']['publication_storage_checks']
    assert [c['source'] for c in checks] == [b['share'] for b in bindings]
    assert checks == config['device_day']['recovery']['storage_checks']
    assert {Path(b['marker_path']) for b in bindings} <= set(reads)


@pytest.mark.parametrize('index', [0, 1])
def test_either_wrong_mount_blocks_before_marker_reads(monkeypatch, tmp_path, index):
    bindings, reads = mounted(monkeypatch, tmp_path)
    real_info = storage._mount_info
    def wrong_info(path):
        result = real_info(path)
        if str(path) == bindings[index]['mount_path']:
            result[0]['fstype'] = 'ext4'
        return result
    monkeypatch.setattr(storage, '_mount_info', wrong_info)
    with pytest.raises(OSError, match='mount identity mismatch'):
        storage.verify_root(Path(bindings[1]['project_root']))
    assert reads == []


def test_visible_root_does_not_allow_other_projects_or_marker_drift(monkeypatch, tmp_path):
    bindings, _ = mounted(monkeypatch, tmp_path, drift=True)
    with pytest.raises(OSError, match='marker SHA mismatch'):
        storage.verify_root(Path(bindings[1]['project_root']))
    # Restore the actual marker, then verify an unrelated path remains excluded.
    Path(bindings[1]['marker_path']).write_text(json.dumps({'id': 'fixture-archive'}))
    with pytest.raises(OSError, match='outside explicitly configured project root'):
        storage.verify_root(Path(bindings[1]['mount_path']) / 'OtherProject')


def test_nas_initialization_never_falls_back_to_local_write(monkeypatch):
    monkeypatch.delenv('VC_LAB_STORAGE_MANIFEST', raising=False)
    root = Path('/mnt/fixture-unmounted/project')
    with pytest.raises(OSError):
        _immutable_write(root / 'README.txt', b'fixture', SimpleNamespace(guarded_atomic=lambda *a: False))
