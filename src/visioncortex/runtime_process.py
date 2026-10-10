"""Process ownership and local operational status for HTTP and worker roles."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time


def role(config):
    from .runtime_options import RuntimeOptions
    selected = os.environ.get('VISIONCORTEX_RUNTIME_ROLE') or (config.get('runtime') or {}).get('role', 'combined')
    return RuntimeOptions(role=selected).role


@contextmanager
def worker_owner(config):
    from .device_day import exclusive
    from .device_day_contract import atomic_json
    from .build_identity import identity
    root = Path(config['storage']['local_runtime_root']) / 'state'
    root.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    state = {'pid': os.getpid(), 'role': role(config), 'build': identity()}
    def heartbeat():
        while not stopped.is_set():
            atomic_json(root / 'WorkerStatus.json', state | {'status': 'running', 'at': time.time()})
            if stopped.wait(2):
                break
    # Multiple web processes are fine; discovery/recovery has one owner.
    with exclusive(root / 'Worker.lock'):
        thread = threading.Thread(target=heartbeat, name='worker-heartbeat', daemon=True)
        thread.start()
        try:
            from .runtime_services import services
            with services(config):
                yield
        finally:
            stopped.set()
            thread.join(timeout=5)
            atomic_json(root / 'WorkerStatus.json', state | {'status': 'stopped', 'at': time.time()})


def worker_status(config):
    path = Path(config['storage']['local_runtime_root']) / 'state' / 'WorkerStatus.json'
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {'status': 'not_started'}
    except (OSError, ValueError):
        return {'status': 'status_unavailable'}
    if data.get('status') == 'running' and time.time()-data.get('at', 0) > 15:
        return data | {'status': 'heartbeat_expired'}
    return data
