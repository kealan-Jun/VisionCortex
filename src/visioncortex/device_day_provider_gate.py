"""Persistent cloud circuit; local preprocessing never depends on this gate."""
from __future__ import annotations

import time
from pathlib import Path

from .device_day_contract import atomic_json, read_json


class ProviderGate:
    def __init__(self, config):
        self.config = config
        self.path = Path(config['storage']['local_runtime_root']) / 'device-day' / 'provider-blocks' / 'Aliyun.json'

    def state(self, stage='understanding'):
        if self.config.get('runtime', {}).get('provider_circuit_enabled'):
            from .provider_control import Circuit
            binding = self.config.get('mllm', {})
            if stage == 'stt':
                binding = self.config.get('speech_recognition', {}).get('connection') or binding
            rows = Circuit(self.config['storage']['local_runtime_root'], binding, stage).state()
            active = [row for row in rows if max(row['retry_at'], row['probe_until']) > time.time()]
            if active:
                return {'active': True, 'reason': active[0]['reason'], 'stage': stage,
                        'message': '当前云服务暂不可用，等待自动恢复；视频预处理独立继续',
                        'next_probe_at': max(max(row['retry_at'], row['probe_until']) for row in active),
                        'recovery': 'single_real_request_half_open'}
        return read_json(self.path) if self.path.is_file() else {'active': False}

    def blocks(self, stage):
        if stage == 'stt' and self.config.get('speech_recognition', {}).get('provider') != 'aliyun_qwen':
            return False
        if stage in {'understanding', 'stt'} and self.config.get('runtime', {}).get('provider_circuit_enabled'):
            from .provider_control import Circuit
            binding = self.config.get('mllm', {})
            if stage == 'stt':
                binding = self.config.get('speech_recognition', {}).get('connection') or binding
            rows = Circuit(self.config['storage']['local_runtime_root'], binding, stage).state()
            if any(max(row['retry_at'], row['probe_until']) > time.time() for row in rows):
                return True
        return (stage in {'understanding', 'stt'}
                and self.config.get('mllm', {}).get('provider') == 'aliyun'
                and self.state(stage).get('active', False))

    def record_failure(self, result):
        if self.config.get('runtime', {}).get('provider_circuit_enabled'):
            return False  # Shared transport owns scoped circuits and half-open admission.
        if self.config.get('mllm', {}).get('provider') != 'aliyun':
            return False
        if 'arrearage' not in str(result.get('error', '')).lower():
            return False
        previous = self.state()
        atomic_json(self.path, {'active': True, 'reason': 'Arrearage',
                               'message': '百炼账户不可用；仅隔离云端理解和STT，预处理继续',
                               'opened_at': previous.get('opened_at', time.time()),
                               'next_probe_at': max(previous.get('next_probe_at', 0), time.time() + 900),
                               'automatic_recovery': True})
        return True

    def probe_if_due(self):
        state = read_json(self.path) if self.path.is_file() else {'active': False}
        if not state.get('active'):
            return state
        if not self.blocks('understanding') or self.config.get('device_day', {}).get('preprocessing_only'):
            return state
        if time.time() < state.get('next_probe_at', 0):
            return state
        # Reserve the next attempt before I/O; crashes/restarts cannot make a
        # rapid paid-probe loop. The service supplies one independent worker.
        state = state | {'next_probe_at': time.time() + 900, 'last_probe_at': time.time()}
        atomic_json(self.path, state)
        from .ai_settings import verify_and_activate
        try:
            result = verify_and_activate(self.config['mllm'], '', self.config)
            if result.get('activated'):
                state = state | {'active': False, 'recovered_at': time.time(),
                                 'message': '连接验证通过，自动恢复云端队列'}
            else:
                state = state | {'last_probe_status': 'provider_unavailable'}
        except Exception as exc:
            state = state | {'last_probe_status': type(exc).__name__}
        atomic_json(self.path, state)
        return state
