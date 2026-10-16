"""Bounded machine-run admission; never claims, retries or stops NAS work."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import shutil

import psutil

from .device_day_contract import atomic_json
from .runtime_options import MachineAdmission


def policy(config):
    value = (config.get('runtime') or {}).get('machine_admission')
    return MachineAdmission.model_validate(value) if value is not None else None


def sample_gpu(gpu_uuid):
    result = subprocess.run(
        ['nvidia-smi', '-i', gpu_uuid, '--query-gpu=memory.total,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=5,
    )
    total, used = (float(x.strip()) for x in result.stdout.strip().split(','))
    if not 0 <= used <= total or total <= 0:
        raise ValueError('Invalid GPU telemetry')
    return total, used


def status(config, *, gpu_sample=sample_gpu, available_ram=None):
    selected = policy(config)
    if selected is None:
        return {'ready': True, 'reason': 'legacy_worker_policy'}
    state = Path(config['storage']['local_runtime_root']) / 'state'
    if (state / 'MachineAdmissionHold.json').exists():
        return {'ready': False, 'reason': 'MACHINE_ADMISSION_HOLD'}
    try:
        from .local_storage import require_local_path
        root = require_local_path(Path(config['runtime']['resource_coordinator_root']))
        cache_value = config['storage'].get('local_cache_root')
        if cache_value:
            cache = require_local_path(Path(cache_value))
            cache_bytes = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file() and not p.is_symlink())
            if cache_bytes > selected.cache_limit_bytes:
                return {'ready': False, 'reason': 'WAITING_CACHE_SPACE'}
        runtime_root = state.parent
        while not runtime_root.exists():
            runtime_root = runtime_root.parent
        if shutil.disk_usage('/').free < 50 * 1024**3 or shutil.disk_usage(runtime_root).free < 80 * 1024**3:
            return {'ready': False, 'reason': 'WAITING_LOCAL_STORAGE'}
        database = root / 'state/resources.sqlite3'
        # Read only: a missing native coordinator cannot turn into a second ledger.
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=1) as db:
            capacities = dict(db.execute('SELECT name,capacity FROM resources'))
            for name, capacity in config['runtime']['resource_limits'].items():
                if capacities.get(name) != capacity:
                    return {'ready': False, 'reason': 'RESOURCE_CAPACITY_MISMATCH', 'resource': name}
            used_slots = db.execute("SELECT COALESCE(SUM(units),0) FROM leases "
                                    "WHERE resource='vision' AND state='running' AND expires>?",
                                    (time.time(),)).fetchone()[0]
            if used_slots >= capacities.get('vision', 0):
                return {'ready': False, 'reason': 'WAITING_NATIVE_VISION'}
        total, used = gpu_sample(selected.gpu_uuid)
        enough = (used + selected.gpu_budget_mib <= total * selected.max_gpu_used_fraction
                  and total - used >= selected.gpu_budget_mib + selected.gpu_reserve_mib)
        observation = {'gpu_total_mib': total, 'gpu_used_mib': used,
                       'gpu_budget_mib': selected.gpu_budget_mib,
                       'gpu_reserve_mib': selected.gpu_reserve_mib}
        if not enough:
            return observation | {'ready': False, 'reason': 'WAITING_GPU_MEMORY'}
        ram = psutil.virtual_memory().available / 1024**2 if available_ram is None else available_ram
        if ram < selected.minimum_available_ram_mib:
            return observation | {'ready': False, 'reason': 'WAITING_HOST_MEMORY'}
        return observation | {'ready': True, 'reason': 'admitted'}
    except (OSError, ValueError, KeyError, sqlite3.Error, subprocess.SubprocessError):
        return {'ready': False, 'reason': 'ADMISSION_UNAVAILABLE'}


def may_claim(config):
    if policy(config) is None:
        return True
    observed = status(config)
    atomic_json(Path(config['storage']['local_runtime_root']) / 'state/MachineAdmission.json',
                observed | {'observed_at': time.time()})
    return observed['ready']


def execution_settings(sealed_settings, runtime):
    """Keep source/model/provider settings; record bounded worker scheduling policy."""
    settings = copy.deepcopy(sealed_settings)
    settings['runtime'] = copy.deepcopy(runtime)
    selected = policy(settings)
    if selected is None:
        return settings
    performance = settings.setdefault('performance', {})
    caps = {'batch_size': selected.inference_batch_size,
            'coarse_batch_size': selected.inference_batch_size,
            'motion_probe_batch_size': selected.inference_batch_size,
            'fine_batch_size': selected.inference_batch_size,
            'workers': 1, 'yolo_inference_workers': 1, 'fine_inference_workers_per_role': 1,
            'source_workers': 2, 'decode_workers': 2,
            'fine_first_person_decode_workers': 2, 'fine_third_person_decode_workers': 2}
    for name, ceiling in caps.items():
        performance[name] = min(int(performance.get(name, ceiling)), ceiling)
    performance['concurrent_role_scanners'] = False
    mllm = settings.setdefault('mllm', {})
    for name in ('workers', 'group_workers'):
        mllm[name] = min(int(mllm.get(name, 1)), 1)
    return settings


def record_execution(job, settings):
    if policy(settings) is None:
        return
    from .build_identity import identity
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
    atomic_json(Path(job.payload['nas_root']) / 'JSON-Config-Files/MachineExecution.json', {
        'schema_version': 'visioncortex-machine-execution/1', 'run_id': job.run_id,
        'attempt': job.attempts, 'reclaimed': job.reclaimed, 'build': identity(),
        'accepted_settings_sha256': digest(job.payload['settings']),
        'effective_settings_sha256': digest(settings),
        'manifest_sha256': digest(job.payload['manifest']),
        'resource_policy': settings['runtime'], 'created_at': time.time(),
    })


def hold_after_failure(config, error):
    if policy(config) is None or not error:
        return
    if any(value in str(error).lower() for value in ('out of memory', 'outofmemory', 'cuda_error_out_of_memory')):
        atomic_json(Path(config['storage']['local_runtime_root']) / 'state/MachineAdmissionHold.json', {
            'reason': 'GPU_OUT_OF_MEMORY', 'created_at': time.time(),
            'action': 'New machine claims paused; inspect budget before removing this owned hold.',
        })
