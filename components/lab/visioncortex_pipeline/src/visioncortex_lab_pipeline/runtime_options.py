"""Validated deployment controls; no inference/storage initialization."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator


class ResultCallback(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(pattern=r'^[A-Za-z0-9_.-]{1,100}$')
    url: str
    secret_env: str = Field(pattern=r'^[A-Z][A-Z0-9_]{1,100}$')

    @field_validator('url')
    @classmethod
    def https_recipient(cls, value):
        from urllib.parse import urlsplit
        url = urlsplit(value)
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.fragment:
            raise ValueError('Result callbacks require configured HTTPS recipients without URL credentials')
        return value


class MachineAdmission(BaseModel):
    """One active run, admitted against whole-device memory and native slots."""
    model_config = ConfigDict(extra='forbid')
    gpu_uuid: str = Field(pattern=r'^GPU-[A-Za-z0-9-]+$')
    gpu_budget_mib: int = Field(default=8192, ge=512)
    gpu_reserve_mib: int = Field(default=6144, ge=2048)
    max_gpu_used_fraction: float = Field(default=.75, gt=0, le=.9, allow_inf_nan=False)
    minimum_available_ram_mib: int = Field(default=8192, ge=1024)
    inference_batch_size: int = Field(default=4, ge=1, le=16)
    cache_limit_bytes: int = Field(default=2147483648, ge=1048576)


class RuntimeOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['combined', 'web', 'worker'] = 'combined'
    resource_limits: dict[str, StrictInt] = Field(default_factory=dict)
    resource_coordinator_root: str | None = None
    machine_admission: MachineAdmission | None = None
    admission_timeout_seconds: float = Field(default=300, gt=0, le=3600, allow_inf_nan=False)
    local_only: StrictBool = False
    provider_circuit_enabled: StrictBool = False
    result_callbacks: list[ResultCallback] = Field(default_factory=list, max_length=32)


def validate(config):
    options = RuntimeOptions.model_validate(config.get('runtime') or {})
    if len({s.name for s in options.result_callbacks}) != len(options.result_callbacks):
        raise ValueError('Result callback names must be unique')
    if any(not 1 <= value <= 256 for value in options.resource_limits.values()):
        raise ValueError('Runtime resource limits must be integers in [1,256]')
    if any(key not in {'vision', 'storage', 'cpu', 'cloud', 'stt'} for key in options.resource_limits):
        raise ValueError('Unknown runtime resource class')
    if options.machine_admission and (not options.resource_coordinator_root or 'vision' not in options.resource_limits):
        raise ValueError('Machine admission requires an explicit shared coordinator and vision capacity')
    if options.resource_coordinator_root is not None:
        from pathlib import Path
        from .local_storage import require_local_path
        coordinator = Path(options.resource_coordinator_root)
        if not coordinator.is_absolute():
            raise ValueError('Resource coordinator root must be absolute and local')
        require_local_path(coordinator)
    if options.local_only:
        import os
        from pathlib import Path, PureWindowsPath
        storage = config['storage']
        raw_root = str(storage['local_runtime_root'])
        if PureWindowsPath(raw_root).drive.startswith('\\\\'):
            raise ValueError('Local-only runtime root must not point to a mounted share')
        root = Path(os.path.abspath(storage['local_runtime_root']))
        if options.resource_coordinator_root and not Path(options.resource_coordinator_root).resolve().is_relative_to(root.resolve()):
            raise ValueError('Local-only resource coordinator must remain under local_runtime_root')
        if storage.get('sync_to_nas') or config.get('device_day', {}).get('enabled') or config.get('collection_ingest', {}).get('enabled'):
            raise ValueError('Local-only runtime cannot enable NAS producers')
        for name in ('archive_root', 'cache_root', 'staging_root', 'input_root',
                     'local_cache_root', 'local_staging_root', 'local_input_root'):
            value = storage.get(name)
            if value and not Path(os.path.abspath(value)).is_relative_to(root):
                raise ValueError(f'Local-only {name} must remain under local_runtime_root')
        if str(root).startswith(('\\\\', '//', '/mnt/', '/media/', '/run/user/')):
            raise ValueError('Local-only runtime root must not point to a mounted share')
    return options
