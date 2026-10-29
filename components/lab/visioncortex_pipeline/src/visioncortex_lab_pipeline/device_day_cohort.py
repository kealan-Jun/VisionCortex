"""Source-bound admission recovery; never authorize the historical paid queue."""
import hashlib
import json
import math
from pathlib import Path

from .device_day_contract import STAGES


def recovery_entries(settings):
    option = settings.get('admission_restore')
    if option is None:
        return {}
    if (not isinstance(option, dict) or set(option) != {'path', 'sha256'}
            or not isinstance(option['path'], str) or not Path(option['path']).is_absolute()):
        raise ValueError('admission_restore requires an absolute local path and sha256')
    from .local_storage import require_local_path
    path = require_local_path(Path(option['path']))
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Admission restore manifest exceeds 1 MiB')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != option['sha256']:
        raise ValueError('Admission restore manifest checksum mismatch')
    manifest = json.loads(data)
    if (manifest.get('schema_version') != 'visioncortex-admission-restore/1'
            or manifest.get('historical_paid_authorized') is not False):
        raise ValueError('Admission recovery must keep historical paid authorization separate')
    result = {}
    for item in manifest.get('entries', []):
        if (item.get('stage') not in STAGES or item.get('processing_cohort') != 'current_recovery'
                or not all(isinstance(item.get(k), str) and item[k]
                           for k in ('recording_id', 'source_signature', 'admission_reason'))):
            raise ValueError('Invalid source-bound admission recovery entry')
        identity = (item['stage'], item['recording_id'])
        if identity in result:
            raise ValueError('Duplicate admission recovery entry')
        result[identity] = item
    return result


def scheduling_options(settings):
    option = settings.get('scheduling') or {}
    if not isinstance(option, dict):
        raise ValueError('device_day.scheduling must be a mapping')
    enabled = option.get('enabled', False)
    age = option.get('max_ready_wait_seconds', 900)
    horizon = option.get('capacity_horizon_seconds', 3600)
    shares = option.get('resource_shares', {'experiment_candidate': 4, 'support_activity': 2,
                                         'unresolved': 2, 'historical_backfill': 1})
    if (type(enabled) is not bool or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                                        for v in (age, horizon))
            or not isinstance(shares, dict) or not shares
            or any(not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for k, v in shares.items())):
        raise ValueError('Invalid cost scheduling shares or waiting limit')
    return {'enabled': enabled, 'max_ready_wait_seconds': float(age),
            'capacity_horizon_seconds': float(horizon), 'resource_shares': shares}
