"""Wall-clock admission gate; never interrupts running work or rewrites receipts."""
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

NIGHT_STAGES = frozenset({'understanding', 'report'})


def paused_stages(config):
    from .device_day_contract import STAGES
    settings = config.get('device_day') or {}
    paused = settings.get('paused_stages', [])
    if not isinstance(paused, list) or any(stage not in STAGES for stage in paused):
        raise ValueError('device_day.paused_stages must list known processing stages')
    return set(paused) | (set(STAGES) - {'retention', 'vision'}
                          if settings.get('preprocessing_only') else set())


def night_schedule(config, now=None):
    settings = (config.get('device_day') or {}).get('night_processing') or {}
    zone = ZoneInfo(settings.get('timezone', 'Asia/Shanghai'))
    current = datetime.now(zone) if now is None else now.astimezone(zone)
    start = time.fromisoformat(settings.get('start', '20:30'))
    end = time.fromisoformat(settings.get('end', '08:00'))
    if start == end:
        raise ValueError('Night processing start and end must differ')
    enabled = bool(settings.get('enabled', False))
    local = current.time().replace(tzinfo=None)
    admitted = (start <= local < end) if start < end else (local >= start or local < end)
    next_start = datetime.combine(current.date(), start, zone)
    if next_start <= current:
        next_start += timedelta(days=1)
    return {'enabled': enabled, 'open': not enabled or admitted,
            'start': start.strftime('%H:%M'), 'end': end.strftime('%H:%M'),
            'timezone': str(zone), 'next_start': next_start.isoformat(),
            'stages': sorted(NIGHT_STAGES), 'paused_stages': sorted(paused_stages(config)),
            'running_jobs': 'finish_without_interruption'}


def stage_admitted(config, stage, now=None):
    return stage not in paused_stages(config) and (
        stage not in NIGHT_STAGES or night_schedule(config, now)['open'])
