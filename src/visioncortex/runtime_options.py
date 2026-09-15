"""Validated deployment controls; no inference/storage initialization."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


class RuntimeOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['combined', 'web', 'worker'] = 'combined'
    resource_limits: dict[str, StrictInt] = Field(default_factory=dict)
    admission_timeout_seconds: float = Field(default=300, gt=0, le=3600, allow_inf_nan=False)
    local_only: StrictBool = False
    provider_circuit_enabled: StrictBool = False


def validate(config):
    options = RuntimeOptions.model_validate(config.get('runtime') or {})
    if any(not 1 <= value <= 256 for value in options.resource_limits.values()):
        raise ValueError('Runtime resource limits must be integers in [1,256]')
    if any(key not in {'vision', 'storage', 'cpu', 'cloud', 'stt'} for key in options.resource_limits):
        raise ValueError('Unknown runtime resource class')
    if options.local_only:
        import os
        from pathlib import Path
        storage = config['storage']
        root = Path(os.path.abspath(storage['local_runtime_root']))
        if storage.get('sync_to_nas') or config.get('device_day', {}).get('enabled') or config.get('collection_ingest', {}).get('enabled'):
            raise ValueError('Local-only runtime cannot enable NAS producers')
        for name in ('archive_root', 'cache_root', 'staging_root', 'input_root',
                     'local_cache_root', 'local_staging_root', 'local_input_root'):
            value = storage.get(name)
            if value and not Path(os.path.abspath(value)).is_relative_to(root):
                raise ValueError(f'Local-only {name} must remain under local_runtime_root')
        if str(root).startswith(('//', '/mnt/', '/media/', '/run/user/')):
            raise ValueError('Local-only runtime root must not point to a mounted share')
    return options
