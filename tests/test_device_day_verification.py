import os
from concurrent.futures import ThreadPoolExecutor

from visioncortex.device_day_contract import artifact, atomic_json, DeviceDayLayout
from visioncortex.device_day_models import DeviceDayModels
from visioncortex import device_day_verification as module


def test_concurrent_artifact_checks_hash_once_and_reject_mutation(tmp_path, monkeypatch):
    p = tmp_path / 'Evidence.jsonl'
    p.write_bytes(b'original')
    ref = artifact(tmp_path, p)
    verifier = module.ArtifactVerifier()
    calls = []
    original = module.file_hash
    def hashed(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(module, 'file_hash', hashed)
    with ThreadPoolExecutor(max_workers=9) as workers:
        assert all(workers.map(lambda _: verifier.verify(tmp_path, ref), range(18)))
    assert len(calls) == 1
    before = p.stat()
    p.write_bytes(b'modified')
    os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not verifier.verify(tmp_path, ref)
    assert len(calls) == 2
    p.unlink()
    assert not verifier.verify(tmp_path, ref)


def test_wrong_digest_symlinks_and_eviction(tmp_path):
    verifier = module.ArtifactVerifier(maximum=1)
    for name in ['A', 'B']:
        p = tmp_path / name
        p.write_bytes(name.encode())
        ref = artifact(tmp_path, p)
        assert verifier.verify(tmp_path, ref)
        assert not verifier.verify(tmp_path, {**ref, 'sha256': 'incorrect'})
    assert len(verifier._verified) == 1
    link = tmp_path / 'Link'
    link.symlink_to(tmp_path / 'A')
    assert not verifier.verify(tmp_path, {**ref, 'path': 'Link'})


def test_mutation_while_hashing_is_not_cached(tmp_path, monkeypatch):
    p = tmp_path / 'Evidence'
    p.write_bytes(b'abcd')
    ref = artifact(tmp_path, p)
    verifier = module.ArtifactVerifier()
    def mutate(path):
        path.write_bytes(b'efgh')
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
        return ref['sha256']
    monkeypatch.setattr(module, 'file_hash', mutate)
    assert not verifier.verify(tmp_path, ref)
    assert not verifier._verified


def test_scan_publication_references_sealed_bytes_without_copy(default_config, tmp_path):
    cache = tmp_path / 'Cache'
    scan = cache / 'device-day-scans' / 'Identity'
    scan.mkdir(parents=True)
    ledger = scan / 'camera.detections.jsonl'
    ledger.write_text('{"fixture": true}\n')
    runtime = scan / 'runtime_camera.json'
    atomic_json(runtime, {'fixture': True})
    refs = [artifact(scan, ledger), artifact(scan, runtime)]
    atomic_json(scan / 'scan-receipt.json', {'status': 'completed', 'artifacts': refs})
    layout = DeviceDayLayout(tmp_path / 'Archive', 'camera', 1789000000000000, cache)
    model = DeviceDayModels(default_config)
    reports, references = model._scan_artifacts(layout, scan)
    assert reports == [{'fixture': True}]
    assert len(references) == 3
    assert all((cache / ref['path']).is_file() for ref in references)
    assert references[0]['sha256'] == refs[0]['sha256']
    assert not layout.receipts.exists()
    ledger.write_text('tampered')
    import pytest
    with pytest.raises(ValueError, match='changed'):
        model._scan_artifacts(layout, scan)


def test_storage_capacity_does_not_limit_dynamic_yolo_lanes():
    from visioncortex.device_day import stage_worker_capacity
    settings = {"camera_lanes": True, "retention_io_workers": 2}
    for cameras in [1, 6, 9, 12]:
        assert stage_worker_capacity(settings, "retention", cameras) == min(cameras, 2)
        for stage in ["vision", "stt", "understanding", "report"]:
            assert stage_worker_capacity(settings, stage, cameras) == cameras
    assert stage_worker_capacity({"camera_lanes": True}, "retention", 12) == 12
