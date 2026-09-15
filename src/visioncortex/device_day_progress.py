"""Local queue observations; never infer experiment absence from pending work."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import asyncio
from threading import Lock
import json
import os
import sqlite3
import time
import math
from statistics import median
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from html import escape

from .device_day_schedule import priority_date

PROGRESS_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='queue-progress')


class ProgressUnavailable(RuntimeError):
    """A failed queue read must never be reported as zero completed work."""


class ProgressPoller:
    """Coalesce concurrent browser polls; cancellation cannot cancel shared work."""

    def __init__(self, settings_factory, *, executor=PROGRESS_WORKER, reader=None):
        self.settings_factory = settings_factory
        self.executor = executor
        self.reader = reader or snapshot
        self.lock = Lock()
        self.future = None
        self.started = 0.0

    async def read(self, timeout=5):
        with self.lock:
            if self.future is None or (self.future.done() and time.monotonic() - self.started >= 1):
                # Settings may touch disk too; keep it off the ASGI event loop.
                self.future = self.executor.submit(lambda: self.reader(self.settings_factory()))
                self.started = time.monotonic()
            future = self.future
        return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout)

STAGES = ('retention', 'vision', 'stt', 'understanding', 'report')


def snapshot(config):
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    now = time.time()
    days, jobs, errors, states, timing_rows = {}, [], [], {}, {}
    failures = {}
    component_samples = {}
    missing_inputs = {}
    def bucket(day):
        return days.setdefault(day, {'recordings': set(), 'cameras': set(), 'stages': {
            s: {'completed': 0, 'queued': 0, 'running': 0, 'failed': 0, 'expired': 0} for s in STAGES}})
    def date_of(us):
        return datetime.fromtimestamp(us/1e6, ZoneInfo('Asia/Shanghai')).date().isoformat()
    path = root / 'observed-inventory.json'
    if path.is_file() or (root / 'observed-inventory.sqlite3').is_file():
        try:
            from .observed_inventory import read_inventory
            inventory = read_inventory(root)
            for row in inventory.get('recordings', []):
                if not row.get('recording_start_us'):
                    continue
                b = bucket(date_of(row['recording_start_us']))
                b['recordings'].add(row['recording_id'])
                b['cameras'].add(row['camera_key'])
        except (OSError, ValueError, KeyError):
            errors.append('采集清单暂不可读')
    for stage in STAGES:
        path = root / f'queue-{stage}.sqlite3'
        if not path.is_file():
            continue
        try:
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=2)) as db:
                rows = db.execute("SELECT recording_id,status,lease_until,queued_at,updated_at, "
                                  "json_extract(payload,'$.camera_key'),json_extract(payload,'$.recording_start_us'),"
                                  "json_extract(payload,'$.recording_end_us'),wall_seconds,completed_at,result FROM recordings")
                for rid, status, lease, _queued, updated, camera, start, end, wall, completed_at, raw_result in rows:
                    if not start:
                        continue
                    day = date_of(start)
                    b = bucket(day)
                    b['recordings'].add(rid)
                    b['cameras'].add(camera)
                    state = 'expired' if status == 'running' and (lease or 0) < now else status
                    counts = b['stages'][stage]
                    counts[state] = counts.get(state, 0)+1
                    states.setdefault(rid, {})[stage] = (state, day)
                    try:
                        result = json.loads(raw_result or '{}')
                        if not isinstance(result, dict):
                            result = {}
                    except (ValueError, TypeError):
                        result = {}
                    if state == 'failed':
                        reason = result.get('error_type') or result.get('status') or 'unknown'
                        group = failures.setdefault(day, {}).setdefault(stage, {})
                        group[reason] = group.get(reason, 0) + 1
                        if stage == 'retention' and reason == 'FileNotFoundError':
                            missing_inputs.setdefault(day, set()).add(rid)
                    if stage == 'vision' and state == 'completed' and result.get('component_timings'):
                        component_samples.setdefault(day, []).append((completed_at or 0, result['component_timings']))
                    if stage == 'vision' and state == 'completed' and wall and wall > 5:
                        timing_rows.setdefault(day, []).append((completed_at or 0, wall))
                    if state == 'running':
                        jobs.append({'stage': stage, 'date': day, 'camera': camera, 'recording_id': rid,
                                     'start_us': start, 'end_us': end, 'last_update_seconds': max(0, now-updated)})
        except (sqlite3.Error, OSError):
            raise ProgressUnavailable(f'{stage} 队列暂不可读；保留上次成功快照') from None
    from .device_day_activity import observations
    active = observations()
    for item in jobs:
        item.update(active.get((item['stage'], item['recording_id']), {'phase': '等待工作线程进入处理或恢复租约'}))
    for day, b in days.items():
        missing = {rid for rid in missing_inputs.get(day, set())
                   if states.get(rid, {}).get('vision', ('missing',))[0] != 'completed'}
        b['missing_input_count'] = len(missing)
        b['processing_total'] = len(b['recordings']) - len(missing)
        b['total'] = len(b.pop('recordings'))
        b['camera_count'] = len(b.pop('cameras'))
        for counts in b['stages'].values():
            counts['not_enqueued'] = max(0, b['total']-sum(counts.values()))
    from .device_day_night_schedule import night_schedule
    from .device_day_contract import DEPENDENCIES
    from .device_day_provider_gate import ProviderGate
    schedule = night_schedule(config)
    try:
        provider = ProviderGate(config).state()
    except (OSError, ValueError):
        provider = {'active': None, 'reason': 'state_unavailable'}
        errors.append('云端状态暂不可读')
    waiting = {day: {s: {} for s in STAGES} for day in days}
    for row in states.values():
        for stage, (status, day) in row.items():
            if status not in {'queued', 'expired'}:
                continue
            parents = [row.get(p, ('missing', day))[0] for p in DEPENDENCIES[stage]]
            reason = ('upstream_failed' if 'failed' in parents else
                      'upstream_pending' if any(p != 'completed' for p in parents) else
                      'provider_blocked' if stage in {'stt', 'understanding'} and provider.get('active')
                      and config.get('mllm', {}).get('provider') == 'aliyun' else
                      'night_window' if stage in {'understanding', 'report'} and not schedule['open'] else
                      'lease_recovery' if status == 'expired' else 'pending_validation')
            counts = waiting[day][stage]
            counts[reason] = counts.get(reason, 0) + 1
    cleanup = (config.get('device_day') or {}).get('capture_video_link_cleanup') or {}
    timings = {}
    for day, values in timing_rows.items():
        samples = sorted(wall for _, wall in sorted(values, reverse=True)[:100])
        timings[day] = {'samples': len(samples), 'median_seconds': median(samples),
                        'p90_seconds': samples[math.ceil(len(samples)*.9)-1],
                        'scope': 'latest_100_completed_jobs_over_5s_including_io_and_postprocessing_not_gpu_inference'}
    from .device_day_latency import snapshot as latency_snapshot, render as render_latency
    latency = latency_snapshot(config, now=now)
    component_timings = {}
    for day, samples in component_samples.items():
        values = {}
        for _, sample in sorted(samples, key=lambda x: x[0], reverse=True)[:100]:
            for name, seconds in sample.items():
                if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0:
                    values.setdefault(name, []).append(seconds)
        component_timings[day] = {name: {'samples': len(v), 'median_seconds': median(v),
                                       'p95_seconds': sorted(v)[math.ceil(len(v)*.95)-1]}
                                  for name, v in values.items()}
    return {'storage_maintenance': os.environ.get('VISIONCORTEX_STORAGE_MAINTENANCE', '0') == '1',
            'capture_link_cleanup': {'enabled': bool(cleanup.get('enabled')),
            'native_links_verified': bool(cleanup.get('native_links_verified')),
            'capture_readers_verified': bool(cleanup.get('capture_readers_verified'))},
            'night_schedule': schedule, 'provider': {k: provider.get(k) for k in
                ('active', 'reason', 'last_probe_at', 'last_probe_status')}, 'waiting': waiting,
            'observed_at': now, 'focus_date': priority_date(root), 'days': days, 'vision_timings': timings,
            'failure_categories': failures, 'vision_component_timings': component_timings,
            'running': jobs, 'errors': errors + latency['errors'], 'latency': latency,
            'latency_html': render_latency(latency) + render_diagnostics(failures, component_timings),
            'scope': 'queue_records_not_current_version_acceptance'}


def render_diagnostics(failures, timings):
    from .device_day_activity import PHASES
    stage_names = dict(zip(STAGES, ('归档', '预处理', '录音识别', '多模态', '日报'), strict=True))
    rows = []
    for day, stages in sorted(failures.items(), reverse=True):
        for stage, reasons in stages.items():
            for reason, count in reasons.items():
                label = '找不到输入文件（需核对路径，不能据此认定已删除）' if reason == 'FileNotFoundError' else reason
                rows.append(f'<li>{escape(day)} · {stage_names[stage]} · {escape(label)}：{count} 个</li>')
    costs = []
    for day, phases in sorted(timings.items(), reverse=True):
        for name, values in phases.items():
            costs.append(f'<tr><td>{escape(day)}</td><td>{escape(PHASES.get(name, name))}</td>'
                         f'<td>{values["samples"]}</td><td>{values["median_seconds"]:.1f} 秒</td>'
                         f'<td>{values["p95_seconds"]:.1f} 秒</td></tr>')
    return ('<details><summary>失败原因与预处理分项耗时</summary><ul>' + ''.join(rows) + '</ul>'
            '<p>失败输入单独核查，其他可用分片继续处理。以下仅统计启用分项记录后的任务，'
            '每日期最近 100 个；包含供帧、推理和后处理，各阶段可能嵌套，不能直接相加。</p>'
            '<table><thead><tr><th>日期</th><th>环节</th><th>样本</th><th>中位数</th><th>P95</th></tr>'
            '</thead><tbody>' + ''.join(costs) + '</tbody></table></details>')
