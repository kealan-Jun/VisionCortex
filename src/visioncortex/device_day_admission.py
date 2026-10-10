"""Soft admission limits; existing leases and model executions are untouched."""
import math

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


def admission_status(config, stage):
    """Return a wait reason for new heavy work, or None when admission is open.

    This is a soft host-memory reserve, not an allocation guarantee. It never
    cancels work and leaves speech and archive copying available to progress.
    """
    settings = config.get('device_day') or {}
    validate(settings)
    reserve = settings.get('admission_min_available_gib', 0)
    if not reserve or stage not in {'vision', 'understanding', 'report'}:
        return None
    available = psutil.virtual_memory().available
    minimum = math.ceil(reserve * 1024**3)
    if available < minimum:
        return {'status': 'waiting_for_memory', 'stage': stage,
                'available_bytes': available, 'reserve_bytes': minimum}
    return None
