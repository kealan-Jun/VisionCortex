"""Standard split input reconciliation uses existing stores and no model owner."""
from pathlib import Path
import threading

import pytest

from test_device_day import capture, device_config  # noqa: F401
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.nas_recordings import scan_recordings
from visioncortex.runtime_services import serve_availability


class OneTick(threading.Event):
    def wait(self, timeout=None):
        self.set()
        return True


def prepare(settings, monkeypatch):
    settings['runtime']['local_only'] = True
    from visioncortex import build_identity, config, device_day
    monkeypatch.setattr(config, 'load_config', lambda _: settings)
    monkeypatch.setattr(build_identity, 'identity', lambda: {'commit': 'a' * 40})
    monkeypatch.setattr(device_day, 'DeviceDayRunner', lambda *a, **kw: pytest.fail('Reconciler started model owner'))
    return Path(settings['storage']['local_runtime_root']) / 'device-day'


def test_standard_service_reconciles_source_and_synchronizes_existing_queue(device_config, monkeypatch):  # noqa: F811
    capture(device_config)
    record, = scan_recordings(device_config)['recordings']
    root = prepare(device_config, monkeypatch)
    queue = DeviceDayQueue(root / 'queue-retention.sqlite3')
    queue.enqueue(record, 'existing-source-revision')
    original = Path(record['video_path'])
    before = original.read_bytes(), original.stat().st_ino
    serve_availability(Path('fixture.yaml'), OneTick(), interval=0)
    state = read_json(root / 'InputReconciliation/Service.json')
    assert state['status'] == 'stopped' and state['ticks'] == 1 and state['ready'] == 1
    assert state['model_invoked'] is state['capture_modified'] is False
    assert state['build']['commit'] == state['source_sha'] == 'a' * 40
    with queue.connect() as db:
        row = db.execute('SELECT * FROM recordings').fetchone()
    assert row['input_status'] == 'ready' and row['revision'] == 'existing-source-revision'
    assert (original.read_bytes(), original.stat().st_ino) == before


def test_storage_failure_is_visible_and_does_not_reconcile(device_config, monkeypatch):  # noqa: F811
    root = prepare(device_config, monkeypatch)
    from visioncortex import device_day_recovery_worker, input_availability, runtime_services
    monkeypatch.setattr(device_day_recovery_worker, 'verify_storage', lambda _: (_ for _ in ()).throw(OSError('fixture')))
    monkeypatch.setattr(input_availability.Reconciler, 'tick', lambda _: pytest.fail('Unverified storage was read'))
    original, published = runtime_services.atomic_json, []
    def publish(path, value):
        published.append(dict(value))
        original(path, value)
    monkeypatch.setattr(runtime_services, 'atomic_json', publish)
    serve_availability(Path('fixture.yaml'), OneTick(), interval=0)
    assert any(row['status'] == 'failed' and row['error_type'] == 'OSError' for row in published)
    assert read_json(root / 'InputReconciliation/Service.json')['ticks'] == 0


def test_standard_cli_selects_input_availability_owner(monkeypatch):
    import signal
    from visioncortex import runtime_services
    called = []
    monkeypatch.setattr(signal, 'signal', lambda *a: None)
    monkeypatch.setattr(runtime_services, 'serve_availability', lambda path, stop: called.append(path))
    runtime_services.main(['--input-availability', '--config', 'fixture.yaml'])
    assert called == [Path('fixture.yaml')]
