"""Manage only explicitly configured, owned loopback model process groups."""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request

PORTS = {'vlm': 18110, 'asr': 18111, 'ocr': 18112}


@functools.lru_cache(maxsize=1)
def service_module():
    # The service implementation belongs to this repository, independent of the
    # private runtime root and of ambient Python package search paths.
    path = Path(__file__).resolve().with_name('lab_model_service.py')
    spec = importlib.util.spec_from_file_location('vc_lab_owned_model_service', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_ticks(pid):
    # /proc comm can contain spaces; field 22 follows the closing parenthesis.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


class ModelServices:
    def __init__(self, root, config_path):
        self.source = service_module()
        self.root = self.source.runtime_root(root)
        self.config_path = self.source.local_path(config_path, owned=True)
        self.config = self.source.load_configuration(self.config_path)
        # Validate an existing credential even for status; never provision one.
        self.source.read_token(self.config)

    def settings(self, name):
        if name not in PORTS:
            raise ValueError('Unknown model service')
        return self.source.service_settings(self.config, name)

    def port(self, name):
        value = self.settings(name).get('port', PORTS[name])
        if type(value) is not int or not 1 <= value <= 65535:
            raise ValueError('Explicit service port must be in 1..65535')
        return value

    def interpreter(self, name):
        value = self.settings(name).get('python', str(self.root / f'envs/{name}/bin/python'))
        # Virtual-environment interpreters commonly link to a local base Python.
        # The selected path is explicit; both it and its target must be local.
        path = Path(value)
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError('Existing local model interpreter required')
        self.source.local_path(path.resolve())
        if str(path).startswith(('/mnt/', '/media/', '//')):
            raise ValueError('Model interpreter must be local')
        return path

    def pid_path(self, name):
        self.settings(name)
        return self.root / f'runtime/{name}.pid.json'

    def owned(self, name):
        path = self.pid_path(name)
        if not path.exists():
            return None
        self.source.local_path(path, owned=True)
        if path.stat().st_mode & 0o077 or path.stat().st_size > 65536:
            raise RuntimeError('Model PID record permissions/size are unsafe')
        record = json.loads(path.read_text())
        pid = record['pid']
        if type(pid) is not int or pid <= 1:
            raise RuntimeError('Invalid owned model PID')
        proc = Path(f'/proc/{pid}')
        if not proc.exists():
            return None
        interpreter = self.interpreter(name)
        if (proc.stat().st_uid != os.getuid() or start_ticks(pid) != record['start_ticks']
                or record.get('runtime_root') != str(self.root)
                or record.get('config_path') != str(self.config_path)
                or record['args'][0] != str(interpreter)
                or record.get('executable_realpath') != str(interpreter.resolve())
                or proc.joinpath('exe').resolve() != interpreter.resolve()
                or os.getpgid(pid) != pid):
            raise RuntimeError('Model PID ownership/identity changed; refusing process-group action')
        return record

    def command(self, name, *, probe_port=True):
        settings = self.settings(name)
        port = self.port(name)
        if probe_port:
            # A pre-existing listener, including another team's service, stays intact.
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
        python = str(self.interpreter(name))
        if name == 'vlm' and settings.get('engine', 'vllm') not in {'vllm', 'joyai-hf'}:
            raise ValueError('Unknown explicit VLM engine')
        if name == 'vlm' and settings.get('engine', 'vllm') == 'vllm':
            source, _ = self.source.checkpoint(self.root, settings.get('checkpoint', 'Qwen3.5-9B'), self.config)
            args = [python, '-m', 'vllm.entrypoints.cli.main', 'serve', str(source),
                '--host', '127.0.0.1', '--port', str(port),
                '--served-model-name', settings.get('served_model_name', 'vc-lab-vlm'),
                '--dtype', 'bfloat16', '--gpu-memory-utilization', str(settings.get('gpu_memory_utilization', .46)),
                '--max-model-len', str(settings.get('max_model_len', 32768)), '--max-num-seqs', '1',
                '--max-num-batched-tokens', '2048', '--enforce-eager', '--reasoning-parser', 'qwen3',
                '--mm-processor-cache-gb', '0.25',
                '--limit-mm-per-prompt', '{"image":{"count":18,"width":768,"height":768},"video":0}',
                '--mm-processor-kwargs', '{"max_pixels":589824}']
        else:
            args = [python, '-u', str(Path(__file__).resolve().with_name('lab_model_service.py')),
                    '--runtime-root', str(self.root), '--config', str(self.config_path),
                    '--kind', name, '--port', str(port)]
        environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
            VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', VLLM_API_KEY=self.source.read_token(self.config),
            HF_HOME=str(self.root / 'runtime/cache/huggingface'), XDG_CACHE_HOME=str(self.root / 'runtime/cache'),
            TORCHINDUCTOR_CACHE_DIR=str(self.root / 'runtime/cache/torchinductor'),
            TRITON_CACHE_DIR=str(self.root / 'runtime/cache/triton'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            CUDA_VISIBLE_DEVICES=str(settings.get('cuda_visible_devices', '0')),
            TMPDIR=str(self.root / 'runtime/install-tmp'), NO_PROXY='localhost,127.0.0.1,::1')
        return args, environment

    def record_process(self, name, pid, args):
        self.source.private_directory(self.root, 'runtime')
        record = dict(pid=pid, start_ticks=start_ticks(pid), port=self.port(name), args=args,
                      runtime_root=str(self.root), config_path=str(self.config_path),
                      executable_realpath=str(self.interpreter(name).resolve()))
        target = self.pid_path(name)
        if target.is_symlink():
            raise RuntimeError('PID record must not be a symbolic link')
        with tempfile.NamedTemporaryFile('w', dir=target.parent, prefix='.', delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(json.dumps(record, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)

    def prepare_runtime(self):
        for relative in ('runtime/logs', 'runtime/install-tmp', 'runtime/cache/huggingface',
                         'runtime/cache/torchinductor', 'runtime/cache/triton'):
            self.source.private_directory(self.root, relative)

    def start(self, name):
        if self.owned(name):
            return {'service': name, 'status': 'already_running'}
        args, environment = self.command(name)
        self.prepare_runtime()
        with (self.root / f'runtime/logs/{name}.log').open('ab') as log:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
        self.record_process(name, process.pid, args)
        return {'service': name, 'status': 'starting', 'pid': process.pid, 'port': self.port(name)}

    def foreground(self, name):
        if self.owned(name):
            raise RuntimeError('Owned service is already running')
        args, environment = self.command(name)
        self.prepare_runtime()
        if os.getpgid(0) != os.getpid():
            os.setsid()
        self.record_process(name, os.getpid(), args)
        os.execvpe(args[0], args, environment)

    def stop(self, name):
        record = self.owned(name)
        if not record:
            return {'service': name, 'status': 'not_running'}
        os.killpg(record['pid'], signal.SIGTERM)
        return {'service': name, 'status': 'stop_requested', 'pid': record['pid']}

    def status(self, name):
        record = self.owned(name)
        if not record:
            return {'service': name, 'status': 'not_running'}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            request = urllib.request.Request(f'http://127.0.0.1:{self.port(name)}/health',
                headers={'Authorization': 'Bearer ' + self.source.read_token(self.config)})
            with opener.open(request, timeout=3) as response:
                ready = response.status == 200
        except Exception:
            ready = False
        return {'service': name, 'status': 'ready' if ready else 'starting_or_failed',
                'pid': record['pid'], 'port': self.port(name)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('action', choices=['start', 'stop', 'status', 'run'])
    parser.add_argument('services', nargs='*', choices=list(PORTS))
    args = parser.parse_args(argv)
    if args.action == 'run' and len(args.services) != 1:
        parser.error('run needs exactly one service')
    manager = ModelServices(args.runtime_root, args.config)
    if args.action == 'run':
        manager.foreground(args.services[0])
        return
    operation = getattr(manager, args.action)
    for name in args.services or PORTS:
        print(json.dumps(operation(name)), flush=True)
        if args.action == 'start':
            time.sleep(.1)


if __name__ == '__main__':
    main()
