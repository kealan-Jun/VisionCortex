"""Bounded machine-run admission; never claims, retries or stops NAS work."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
import shutil
import uuid

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
    if mllm.get('credential_ref'):
        from .ai_settings import reverified_job_mllm
        # Only an actually reverified revision with the same connection AND key
        # may replace stale transport metadata. No provider/model rotation.
        mllm = settings['mllm'] = reverified_job_mllm(settings)
    for name in ('workers', 'group_workers'):
        mllm[name] = min(int(mllm.get(name, 1)), 1)
    return settings


def record_execution(job, settings):
    if policy(settings) is None:
        return
    from .build_identity import identity
    recovery = prepare_run_owner(job, settings)
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
    receipt = {
        'schema_version': 'visioncortex-machine-execution/1', 'run_id': job.run_id,
        'attempt': job.attempts, 'reclaimed': job.reclaimed, 'build': identity(),
        'accepted_settings_sha256': digest(job.payload['settings']),
        'effective_settings_sha256': digest(settings),
        'manifest_sha256': digest(job.payload['manifest']),
        'resource_policy': settings['runtime'], 'created_at': time.time(),
        'run_lock_recovery': recovery,
    }
    directory = Path(job.payload['nas_root']) / 'JSON-Config-Files'
    atomic_json(directory / f'MachineExecutionAttempt{job.attempts}.json', receipt)
    atomic_json(directory / 'MachineExecution.json', receipt)


def current_process_owner():
    return {'pid': os.getpid(), 'process_created_at': psutil.Process().create_time(),
            'host_id': hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest()}


def owner_alive(owner):
    try:
        process = psutil.Process(owner['pid'])
        return (process.status() != psutil.STATUS_ZOMBIE
                and abs(process.create_time() - owner['process_created_at']) < .01)
    except psutil.NoSuchProcess:
        return False


def prepare_run_owner(job, settings):
    """Recover only this local queue's same-host, provably dead run lock."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', job.run_id):
        raise ValueError('Unsafe machine run identity')
    root = Path(job.payload['nas_root'])
    local = Path(settings['storage']['local_runtime_root']) / 'state/MachineRunOwners' / (job.run_id + '.json')
    current = current_process_owner()
    lock = root / 'run.lock'
    recovery = None
    if lock.exists() or lock.is_symlink():
        if lock.is_symlink() or not lock.is_file() or not local.is_file():
            raise RuntimeError('Run lock has no verified local owner; recovery refused')
        previous = json.loads(local.read_text())
        before = lock.stat()
        contents = json.loads(lock.read_text())
        if (previous.get('host_id') != current['host_id']
                or previous.get('run_id') != job.run_id
                or previous.get('nas_root') != str(root)
                or contents.get('pid') != previous.get('pid')
                or owner_alive(previous)):
            raise RuntimeError('Run lock owner is live or does not match this local task')
        after = lock.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError('Run lock changed during recovery')
        relative = 'JSON-Config-Files/RecoveredRunLock' + uuid.uuid4().hex + '.json'
        os.replace(lock, root / relative)
        recovery = {'previous_pid': previous['pid'], 'retained_lock': relative,
                    'reason': 'verified_same_host_process_terminated'}
    atomic_json(local, current | {'run_id': job.run_id, 'nas_root': str(root),
                                 'attempt': job.attempts, 'recorded_at': time.time()})
    return recovery


def hold_after_failure(config, error):
    if policy(config) is None or not error:
        return
    if any(value in str(error).lower() for value in ('out of memory', 'outofmemory', 'cuda_error_out_of_memory')):
        atomic_json(Path(config['storage']['local_runtime_root']) / 'state/MachineAdmissionHold.json', {
            'reason': 'GPU_OUT_OF_MEMORY', 'created_at': time.time(),
            'action': 'New machine claims paused; inspect budget before removing this owned hold.',
        })
