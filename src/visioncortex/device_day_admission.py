"""Soft admission limits; existing leases and model executions are untouched."""
import math
from pathlib import Path

import psutil

from .device_day_contract import STAGES


def validate(settings):
    limits = settings.get('stage_worker_limits', {})
    if (not isinstance(limits, dict) or set(limits) - set(STAGES)
            or any(type(value) is not int or value < 1 for value in limits.values())):
        raise ValueError('device_day.stage_worker_limits must map known stages to positive integers')
    reserve = settings.get('admission_min_available_gib', 0)
    if (type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0):
        raise ValueError('device_day.admission_min_available_gib must be a finite nonnegative number')
    storage = settings.get('local_storage_reserves', [])
    if (not isinstance(storage, list) or any(
            not isinstance(item, dict) or set(item) != {'path', 'min_available_gib'}
            or not isinstance(item['path'], str) or not item['path'] or not Path(item['path']).is_absolute()
            or type(item['min_available_gib']) not in (int, float)
            or not math.isfinite(item['min_available_gib']) or item['min_available_gib'] < 0
            for item in storage)):
        raise ValueError('device_day.local_storage_reserves must name absolute local paths and finite nonnegative reserves')
    if len({item['path'] for item in storage}) != len(storage):
        raise ValueError('device_day.local_storage_reserves must not repeat a path')


def admission_status(config, stage):
    """Return a wait reason for new heavy work, or None when admission is open.

    Local filesystem reserves apply to every stage; the existing host-memory
    reserve applies to heavy stages. Neither is an allocation guarantee or
    cancellation signal. Existing executions retain their leases and finish.
    """
    settings = config.get('device_day') or {}
    validate(settings)
    from .local_storage import require_local_path
    # Prove every destination local before statvfs/disk_usage can touch any
    # configured path. A network path is a configuration error, never a NAS
    # probe or a reason to create a fallback directory.
    storage = [(require_local_path(Path(item['path'])), math.ceil(item['min_available_gib'] * 1024**3))
               for item in settings.get('local_storage_reserves', [])]
    for path, minimum in storage:
        try:
            available = psutil.disk_usage(str(path)).free
        except OSError:
            return {'status': 'waiting_for_storage', 'stage': stage,
                    'path': str(path), 'reason': 'local_capacity_unavailable', 'reserve_bytes': minimum}
        if available < minimum:
            return {'status': 'waiting_for_storage', 'stage': stage,
                    'path': str(path), 'available_bytes': available, 'reserve_bytes': minimum}
    reserve = settings.get('admission_min_available_gib', 0)
    if not reserve or stage not in {'vision', 'understanding', 'report'}:
        return None
    available = psutil.virtual_memory().available
    minimum = math.ceil(reserve * 1024**3)
    if available < minimum:
        return {'status': 'waiting_for_memory', 'stage': stage,
                'available_bytes': available, 'reserve_bytes': minimum}
    return None
