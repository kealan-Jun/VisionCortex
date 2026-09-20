import hashlib
from pathlib import Path
from types import SimpleNamespace
from bisect import bisect_left

from visioncortex.device_day_models import check_coverage
from visioncortex.device_day_runtime_identity import compatible_runtime_hash


def test_native_probe_deadline_only_compatibility_is_exact(tmp_path):
    from visioncortex import source_frames
    path = Path(source_frames.__file__)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compatible_runtime_hash(path, actual) == '3bf96e4268b9b164d75463b1b340037fc256315be94f6a1a2711da57712cb2fd'
    changed = hashlib.sha256(path.read_bytes() + b'\n# unrelated change').hexdigest()
    assert compatible_runtime_hash(path, changed) == changed
    assert compatible_runtime_hash(tmp_path/path.name, actual) == actual


def test_runtime_compatibility_never_accepts_unknown_code_or_external_name(tmp_path):
    import visioncortex.detection as detection
    path = Path(detection.__file__)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    # The new cancellation-aware implementation has its own execution identity.
    assert compatible_runtime_hash(path, actual) == actual
    changed = hashlib.sha256(path.read_bytes()+b'\n# another edit\n').hexdigest()
    assert compatible_runtime_hash(path, changed) == changed
    assert compatible_runtime_hash(tmp_path/path.name, actual) == actual


def test_indexed_range_coverage_preserves_overlaps_duplicates_and_boundary_samples():
    frames = [SimpleNamespace(local_ms=n) for n in [1000,0,500,500,1500,2000,2500]]
    ordered = sorted(frames,key=lambda f:f.local_ms)
    times = [f.local_ms for f in ordered]
    for a,b in [(0,1000),(500,2000),(1500,3000)]:
        subset = ordered[bisect_left(times,a):bisect_left(times,b)]
        assert check_coverage(subset,a,b,2) == check_coverage(frames,a,b,2)


def test_changed_scheduler_invalidates_runtime_identity():
    import visioncortex.scan_scheduler as scheduler
    path = Path(scheduler.__file__)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compatible_runtime_hash(path, "74961086aa700c28df181570012056be6e72ae2fcbca018e098a699dd790a209") is None
    assert compatible_runtime_hash(path, actual) == actual
    changed = hashlib.sha256(path.read_bytes()+b"# changed").hexdigest()
    assert compatible_runtime_hash(path, changed) == changed


def test_shared_scheduler_honors_phase_lanes_without_fixed_camera_count(tmp_path):
    from visioncortex.scan_scheduler import scan_views_concurrently
    from visioncortex.schemas import ViewRole
    config = {"performance": {"source_workers": 20, "coarse_decode_lanes": ["cpu"],
                              "fine_decode_lanes": ["cuda"], "ffmpeg_hwaccel": True}}
    for count in [1, 3, 11]:
        views = [SimpleNamespace(view_id=str(i), role=ViewRole.FIRST_PERSON, segments=[]) for i in range(count)]
        for phase, backend in [("coarse", "cpu"), ("fine", "cuda")]:
            calls = []
            def scan(group, *args, **kwargs):
                calls.append(kwargs["decode_backends"])
                return {v.view_id: tmp_path / v.view_id for v in group}
            result = scan_views_concurrently(config, views, {}, {}, tmp_path, phase=phase, scanner=scan)
            assert len(result) == count
            assert calls == [{str(i): backend for i in range(count)}]


def test_legacy_entry_delegates_to_shared_scheduler(monkeypatch, tmp_path):
    from visioncortex.pipeline import EvidencePipeline
    import visioncortex.scan_scheduler as scheduler
    calls = []
    def scan(*args, **kwargs):
        calls.append((args, kwargs))
        return {"view": "ledger"}
    monkeypatch.setattr(scheduler, "scan_views_concurrently", scan)
    owner = SimpleNamespace(config={"performance": {}}, _view_runtime={}, _scan_progress=lambda *args: None)
    result = EvidencePipeline._scan_all_views_concurrently(owner, SimpleNamespace(views=[]), {}, {}, tmp_path, phase="coarse")
    assert result == {"view": "ledger"}
    assert calls[0][1]["phase"] == "coarse"


def test_unknown_performance_change_is_not_aliased():
    from visioncortex.device_day_runtime_identity import compatible_performance
    config = {"cpu_decode_threads": 2, "coarse_decode_lanes": ["cpu"], "detection_fps": 1}
    assert compatible_performance(config) == config


def test_coarse_gpu_capacity_is_shared_and_released_on_failure(monkeypatch, tmp_path):
    import visioncortex.scan_scheduler as scheduler
    from threading import Event
    from concurrent.futures import ThreadPoolExecutor
    from visioncortex.schemas import ViewRole
    monkeypatch.setattr(scheduler, '_COARSE_CUDA_USERS', 0)
    started, release = Event(), Event()
    calls = {}
    config = {"performance": {"coarse_cuda_max_concurrent_sources": 1,
                              "source_workers": 4, "coarse_decode_lanes": ["cpu"]}}
    def scan(group, *args, **kwargs):
        view = group[0]
        calls[view.view_id] = kwargs['decode_backends'][view.view_id]
        if view.view_id == 'a':
            started.set()
            assert release.wait(3)
            raise RuntimeError('decode failed')
        return {view.view_id: tmp_path / view.view_id}
    def run(name):
        view = SimpleNamespace(view_id=name, role=ViewRole.FIRST_PERSON, segments=[])
        return scheduler.scan_views_concurrently(config, [view], {}, {}, tmp_path / name,
                                                phase='coarse', scanner=scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, 'a')
        assert started.wait(3)
        assert run('b')
        release.set()
        import pytest
        with pytest.raises(RuntimeError, match='decode failed'):
            first.result()
    assert run('c')
    assert calls == {'a': 'cuda', 'b': 'cpu', 'c': 'cuda'}
    assert scheduler._COARSE_CUDA_USERS == 0


def test_phase_engine_selection_is_shared_and_does_not_mutate_fine(monkeypatch, tmp_path):
    from visioncortex import scan_scheduler as scheduler
    config = {'models': {'first_person_engine': 'reviewed.engine',
                         'first_person_coarse_engine': 'coarse32.engine'},
              'performance': {'inference_batch_wait_ms': 25, 'coarse_inference_batch_wait_ms': 500}}
    received = []
    monkeypatch.setattr(scheduler, '_scan_views_concurrently', lambda c, *a, **k: received.append(c))
    for phase in ('coarse', 'fine'):
        scheduler.scan_views_concurrently(config, [], {}, {}, tmp_path, phase=phase)
    assert received[0]['models']['first_person_engine'] == 'coarse32.engine'
    assert received[0]['performance']['inference_batch_wait_ms'] == 500
    assert received[1]['models']['first_person_engine'] == 'reviewed.engine'
    assert config['performance']['inference_batch_wait_ms'] == 25
