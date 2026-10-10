"""Operational display uses local heartbeat/process metadata, never model work."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from visioncortex.device_day_overview import (
    SERVICE_STATUS_SCRIPT, processing_service_status, render_service_status,
)


@pytest.fixture
def operational_files(tmp_path):
    config = {'storage': {'local_runtime_root': str(tmp_path / 'runtime')}}
    state = tmp_path / 'runtime' / 'state'
    state.mkdir(parents=True)
    host = state / 'WorkerStatus.json'
    host.write_text(json.dumps({'pid': 99, 'status': 'running', 'at': 100}))
    companion = state / 'DeviceDayCompanionStatus.json'
    data = {'schema_version': 'visioncortex-device-day-companion/1', 'pid': 123,
            'process_start_ticks': 456, 'status': 'running', 'at': 999}
    companion.write_text(json.dumps(data))
    proc = tmp_path / 'proc' / '123'
    proc.mkdir(parents=True)
    (proc / 'stat').write_text('123 (python (companion)) ' + ' '.join(['S'] + ['0'] * 18 + ['456']))
    return config, host, companion, data, tmp_path / 'proc'


def inspect(files):
    config, _, _, _, proc_root = files
    return processing_service_status(config, now=1000, proc_root=proc_root)


def test_fresh_companion_and_stale_host_remain_distinct_and_read_only(operational_files):
    _, host, path, _, _ = operational_files
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (host, path)}
    result = inspect(operational_files)
    assert result['companion']['status'] == 'running'
    assert result['companion']['process_identity_verified'] is True
    assert result['host']['status'] == 'heartbeat_expired'
    page = render_service_status(result)
    assert '自动处理执行器：在线' in page
    assert '原宿主：心跳已过期' in page
    assert '心跳过期不等于处理已停止' in page
    assert '最近核验' in page and '定时更新的状态快照' in page
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original} == original


@pytest.mark.parametrize(('change', 'expected'), [
    ({'at': 980}, 'expired'), ({'at': 1010}, 'unavailable'),
    ({'at': float('nan')}, 'unavailable'), ({'at': float('inf')}, 'unavailable'),
    ({'at': True}, 'unavailable'), ({'at': '999'}, 'unavailable'),
    ({'at': 10 ** 500}, 'unavailable'), ({'pid': True}, 'unavailable'),
    ({'pid': -2}, 'unavailable'), ({'process_start_ticks': None}, 'unavailable'),
    ({'process_start_ticks': 457}, 'unverified'), ({'status': []}, 'unavailable'),
    ({'schema_version': 'wrong'}, 'unavailable'), ({'status': 'draining'}, 'draining'),
    ({'status': 'stopped'}, 'stopped'),
])
def test_online_requires_fresh_timestamp_and_exact_process_incarnation(operational_files, change, expected):
    _, _, path, data, _ = operational_files
    path.write_text(json.dumps(data | change))
    result = inspect(operational_files)
    assert result['companion']['status'] == expected
    if expected != 'draining':
        assert not result['companion']['process_identity_verified']
        assert '自动处理执行器：在线' not in render_service_status(result)
    else:
        assert '正在收尾' in render_service_status(result)


@pytest.mark.parametrize('process', ['absent', 'zombie', 'malformed'])
def test_dead_or_unreadable_process_cannot_appear_online(operational_files, process):
    path = operational_files[-1] / '123' / 'stat'
    if process == 'absent':
        path.unlink()
    elif process == 'zombie':
        path.write_text(path.read_text().replace(' S ', ' Z '))
    else:
        path.write_text('malformed')
    assert inspect(operational_files)['companion']['status'] == 'unverified'


@pytest.mark.parametrize('contents', [None, 'broken', '[]', '{"status": {}}'])
def test_missing_or_invalid_status_does_not_invent_service_outage(operational_files, contents):
    _, host, companion, _, _ = operational_files
    for path in (host, companion):
        if contents is None:
            path.unlink()
        else:
            path.write_text(contents)
    result = inspect(operational_files)
    expected = 'missing' if contents is None else 'unavailable'
    assert result['host']['status'] == expected
    assert result['companion']['status'] == expected
    page = render_service_status(result)
    assert '自动处理执行器：在线' not in page
    assert '自动处理执行器：已停止' not in page


def test_fresh_attention_keeps_verified_online_but_exposes_need_to_check(operational_files):
    _, _, path, data, _ = operational_files
    path.write_text(json.dumps(data | {'error': 'monitor_bridge:OSError'}))
    result = inspect(operational_files)
    assert result['companion']['process_identity_verified']
    assert '在线，有状态待核查' in render_service_status(result)
    assert 'monitor_bridge' not in render_service_status(result)


def test_saved_html_expires_on_initial_load_and_while_left_open(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js unavailable')
    script = tmp_path / 'status.js'
    script.write_text('''const assert = require('node:assert/strict');
let current = 1000000;
Date.now = () => current;
const label = {dataset: {statusExpiry: '1090'}, textContent: '自动处理执行器：在线'};
global.document = {querySelector: () => label};
let timer;
global.setInterval = fn => {timer = fn;};
''' + SERVICE_STATUS_SCRIPT + '''
assert.equal(label.textContent, '自动处理执行器：在线');
current = 1091000;
timer();
assert.match(label.textContent, /状态已过期/);
label.textContent = '自动处理执行器：在线';
''' + SERVICE_STATUS_SCRIPT + '''
assert.match(label.textContent, /状态已过期/);
assert.match(label.textContent, /不代表已停止/);
''')
    subprocess.run([node, str(Path(script))], check=True, capture_output=True, text=True)
