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
        self.last_good = None
        self.last_error = None
        self.snapshot_path = None
        self.consumer_state = {}

    def _published(self):
        path = self.snapshot_path
        if path is None:
            return None
        try:
            cached = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(cached.get('days'), dict) and cached.get('observed_at'):
                if self.consumer_state.get('consumers', {}).get('observed_at', 0) > cached.get('consumers', {}).get('observed_at', 0):
                    cached = cached | self.consumer_state
                return cached
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return None

    def _remember(self, value):
        if self.last_good is None or value.get('observed_at', 0) >= self.last_good.get('observed_at', 0):
            self.last_good = value

    def _collect(self):
        config = self.settings_factory()
        path = (Path(config['storage']['local_runtime_root']) / 'device-day' / 'ProgressSnapshot.json'
                if config.get('storage', {}).get('local_runtime_root') else None)
        self.snapshot_path = path
        cached = self._published()
        if cached is not None and self.reader is snapshot:
            # A deployment may keep an older independently supervised
            # progress publisher. Refresh consumer ownership locally without
            # restarting analysis or writing a competing progress snapshot.
            from .device_day_consumers import consumer_snapshot
            consumers = consumer_snapshot(config)
            self.consumer_state = {'consumers': consumers, 'consumer_html': render_consumers(consumers)}
            return cached | self.consumer_state
        # Only the publisher writes the shared file. A slow HTTP fallback must
        # never overwrite a newer independently published observation.
        return self.reader(config)

    def _result(self, value, *, stale=False):
        return value | {'snapshot_stale': stale,
                        'snapshot_age_seconds': max(0, time.time() - value.get('observed_at', time.time())),
                        'refresh_error': self.last_error}

    async def read(self, timeout=5):
        published = None
        if self.reader is snapshot and self.snapshot_path is not None:
            try:
                published = await asyncio.wait_for(asyncio.to_thread(self._published), min(timeout, .5))
            except asyncio.TimeoutError:
                pass
        with self.lock:
            # Consume a finished result BEFORE starting the next refresh. A
            # read taking >5s used to be discarded on every subsequent poll.
            if self.future is not None and self.future.done():
                try:
                    self._remember(self.future.result())
                    self.last_error = None
                except Exception as exc:
                    self.last_error = type(exc).__name__
            if published is not None:
                self._remember(published)
                self.last_error = None
            if self.future is None or (self.future.done() and time.monotonic() - self.started >= 1):
                # Settings may touch disk too; keep it off the ASGI event loop.
                self.future = self.executor.submit(self._collect)
                self.started = time.monotonic()
            future = self.future
            if self.last_good is not None:
                return self._result(self.last_good, stale=bool(self.last_error) or time.time()-self.last_good.get('observed_at', 0)>15)
        try:
            value = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout)
            self._remember(value)
            return self._result(self.last_good, stale=time.time()-self.last_good.get('observed_at', 0)>15)
        except (asyncio.TimeoutError, ProgressUnavailable):
            if self.last_good is not None:
                return self._result(self.last_good, stale=True)
            raise

STAGES = ('retention', 'vision', 'stt', 'understanding', 'report')


def queue_lifecycle(records, *, limit=100):
    """Bounded local hints; queue reports never prove artifact publication."""
    selected = sorted(records.values(), key=lambda row: row['capture_us'], reverse=True)[:limit]
    rows = []
    for item in selected:
        stages = item['stage_states']
        visual = stages.get('vision', 'pending')
        archived = stages.get('retention', 'pending')
        label = ('预处理完成，等待归档' if visual == 'completed' and archived != 'completed'
                 else '本地队列记录：原片归档已完成' if archived == 'completed'
                 else '本地队列记录：等待输入或预处理')
        rows.append({key: value for key, value in item.items() if key != 'capture_us'} | {
            'archive_status': 'completed' if archived == 'completed' else 'pending',
            'preprocessing': {stage: stages.get(stage, 'pending') for stage in ('vision', 'stt')},
            'publication_status': item.get('publication_status', 'unverified'),
            'publication_verified': False, 'retention_policy': {'status': 'unknown'},
            'scope': 'local_queue_hints_not_receipt_or_artifact_acceptance',
            'label': label + '；正式发布证据尚未核实'})
    return rows, {'source': 'local_queues', 'available': bool(records),
                  'record_count': len(records), 'returned_records': len(rows), 'limit': limit,
                  'publication_verified': False,
                  'reason': None if records else 'local_queue_lifecycle_unavailable'}


def snapshot(config):
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    now = time.time()
    days, jobs, errors, states, timing_rows = {}, [], [], {}, {}
    queue_details, capture_times = {}, {}
    failures = {}
    component_samples = {}
    missing_inputs = {}
    source_signatures = {}
    lifecycle_records = {}
    from .input_availability import Availability
    availability = Availability(root).states() if (root/'InputAvailability.sqlite3').is_file() else {}
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
                source_signatures[row['recording_id']] = row.get('source_signature')
                capture_times[row['recording_id']] = max(row.get('recording_start_us') or 0,
                                                        row.get('recording_end_us') or 0)
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
                columns = {row[1] for row in db.execute('PRAGMA table_info(recordings)')}
                limit_column = 'retry_limit' if 'retry_limit' in columns else '0 AS retry_limit'
                started_column = 'started_at' if 'started_at' in columns else 'NULL AS started_at'
                rows = db.execute("SELECT recording_id,status,lease_until,queued_at,updated_at, "
                                  "json_extract(payload,'$.camera_key'),json_extract(payload,'$.recording_start_us'),"
                                  "json_extract(payload,'$.recording_end_us'),wall_seconds,completed_at,result,"
                                  "json_extract(payload,'$.source_signature'),attempts,input_status," + limit_column +
                                  ',' + started_column + " FROM recordings")
                for rid, status, lease, queued, updated, camera, start, end, wall, completed_at, raw_result, signature, attempts, input_status, authorized_limit, started_at in rows:
                    if not start:
                        continue
                    day = date_of(start)
                    b = bucket(day)
                    b['recordings'].add(rid)
                    b['cameras'].add(camera)
                    state = 'expired' if status == 'running' and (lease or 0) < now else status
                    source_signatures.setdefault(rid, signature)
                    if status not in {'running', 'completed'} and input_status != 'ready':
                        state = 'input_' + input_status
                    if (status not in {'running', 'completed'} and availability.get(rid, {}).get('state', 'ready') != 'ready'
                            and availability[rid].get('signature') == signature):
                        state = 'input_' + availability[rid]['state']
                    if availability.get(rid, {}).get('state') == 'missing' and availability[rid].get('signature') == signature:
                        missing_inputs.setdefault(day, set()).add(rid)
                    counts = b['stages'][stage]
                    counts[state] = counts.get(state, 0)+1
                    states.setdefault(rid, {})[stage] = (state, day)
                    queue_details[(rid, stage)] = {'attempts': attempts, 'retry_limit': authorized_limit,
                                                   'updated_at': updated}
                    capture_times.setdefault(rid, max(start or 0, end or 0))
                    try:
                        result = json.loads(raw_result or '{}')
                        if not isinstance(result, dict):
                            result = {}
                    except (ValueError, TypeError):
                        result = {}
                    if signature == source_signatures[rid]:
                        lifecycle = lifecycle_records.setdefault(rid, {
                            'recording_id': rid, 'archive': day + '_' + str(camera),
                            'capture_us': max(start or 0, end or 0), 'stage_states': {},
                            'input_ready': {}, 'timings': {}})
                        lifecycle['stage_states'][stage] = state
                        lifecycle['input_ready'][stage] = input_status
                        components = result.get('component_timings')
                        timing = {name: value for name, value in (components if isinstance(components, dict) else {}).items()
                                  if type(value) in (int, float) and math.isfinite(value) and value >= 0}
                        if started_at is not None and queued is not None and started_at >= queued:
                            timing['queue_wait_seconds'] = started_at - queued
                        lifecycle['timings'][stage] = timing
                        publication = result.get('formal_publication')
                        if publication == 'completed':
                            lifecycle['publication_status'] = 'reported_completed'
                        elif publication == 'pending' and lifecycle.get('publication_status') != 'reported_completed':
                            lifecycle['publication_status'] = 'pending'
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
    active = observations(root)
    for item in jobs:
        item.update(active.get((item['stage'], item['recording_id']), {'phase': '等待工作线程进入处理或恢复租约'}))
    for day, b in days.items():
        for rid in b['recordings']:
            available = availability.get(rid, {})
            if available.get('state', 'ready') != 'ready' and available.get('signature') == source_signatures.get(rid):
                if available['state'] == 'missing':
                    missing_inputs.setdefault(day, set()).add(rid)
                for stage, counts in b['stages'].items():
                    if stage not in states.get(rid, {}):
                        state = 'input_' + available['state']
                        counts[state] = counts.get(state, 0)+1
                        states.setdefault(rid, {})[stage] = (state, day)
        missing = {rid for rid in missing_inputs.get(day, set())
                   if states.get(rid, {}).get('vision', ('missing',))[0] != 'completed'}
        b['missing_input_count'] = len(missing)
        b['processing_total'] = len(b['recordings']) - len(missing)
        b['total'] = len(b.pop('recordings'))
        b['camera_count'] = len(b.pop('cameras'))
        for counts in b['stages'].values():
            counts['not_enqueued'] = max(0, b['total']-sum(counts.values()))
    from .device_day_night_schedule import night_schedule, paused_stages
    from .device_day_contract import DEPENDENCIES
    from .device_day_provider_gate import ProviderGate
    schedule = night_schedule(config)
    paused = paused_stages(config)
    try:
        provider = ProviderGate(config).state()
    except (OSError, ValueError):
        provider = {'active': None, 'reason': 'state_unavailable'}
        errors.append('云端状态暂不可读')
    # Observer hosts need no execution package or NAS receipt-tree traversal.
    # Reuse the local rows already read for queue counts, with explicit proof
    # limits. Missing lifecycle evidence cannot suppress the entire snapshot.
    lifecycle, lifecycle_observation = queue_lifecycle(lifecycle_records)
    from .device_day_consumers import consumer_snapshot
    from .device_day_retry import retry_limit
    consumers = consumer_snapshot(config, now=now)
    settings = config.get('device_day') or {}
    cutoff = (now - settings.get('live_priority_seconds', 14400)) * 1_000_000
    waiting = {day: {s: {} for s in STAGES} for day in days}
    for identifier, row in states.items():
        for stage, (status, day) in row.items():
            if status not in {'queued', 'expired', 'waiting_for_prerequisite', 'failed',
                              'input_missing', 'input_unavailable', 'needs_camera_role'}:
                continue
            dependencies = () if settings.get('inplace_preprocessing') and stage in {'vision', 'stt'} else DEPENDENCIES[stage]
            parents = [row.get(p, ('missing', day))[0] for p in dependencies]
            details = queue_details.get((identifier, stage), {})
            scope = 'recent' if capture_times.get(identifier, 0) >= cutoff else 'history'
            coverage = consumers['stages'][stage]['scopes'][scope]['status']
            live_hold = any(owner['owner_state'] == 'verified' and owner.get('backfill_status') == 'waiting_for_live'
                            for owner in consumers['stages'][stage]['owners'])
            storage_hold = any(owner['owner_state'] == 'verified' and (
                owner.get('backfill_status') == 'waiting_for_storage'
                or (owner.get('last_result') or {}).get('status') == 'waiting_for_storage')
                for owner in consumers['stages'][stage]['owners'])
            reason = ('paused_by_user' if stage in paused else
                      status if status in {'input_missing', 'input_unavailable'} else
                      'retry_exhausted' if status == 'failed' and details.get('attempts', 0)
                      >= retry_limit(details, settings.get('failure_retry_limit') or 3) else
                      'failure_cooldown' if status == 'failed' and now - details.get('updated_at', 0) < 60 else
                      'prerequisite_not_verified' if status == 'waiting_for_prerequisite' else
                      'upstream_failed' if 'failed' in parents else
                      'upstream_pending' if any(p != 'completed' for p in parents) else
                      'provider_blocked' if stage in {'stt', 'understanding'} and provider.get('active')
                      and config.get('mllm', {}).get('provider') == 'aliyun' else
                      'night_window' if stage in {'understanding', 'report'} and not schedule['open'] else
                      'waiting_for_storage' if storage_hold else
                      'camera_role_unconfigured' if status == 'needs_camera_role' else
                      'lease_recovery' if status == 'expired' else
                      'disabled' if settings.get('enabled') is False and coverage == 'disabled' else
                      'no_consumer' if coverage == 'no_consumer' else
                      'consumer_evidence_unavailable' if coverage == 'evidence_unavailable' else
                      'history_waiting_for_live' if scope == 'history' and live_hold else
                      'waiting_for_history_dispatch' if scope == 'history' else 'waiting_for_dispatch')
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
    lifecycle_html = '<ul>' + ''.join(f'<li>{escape(row["archive"])} · {escape(row["recording_id"])}：{escape(row["label"])}'
        + ('；外部采集保留期限未知' if row['retention_policy']['status'] == 'unknown' else '') + '</li>'
        for row in lifecycle if row['publication_status'] != 'completed') + '</ul>'
    return {'consumers': consumers, 'consumer_html': render_consumers(consumers),
            'input_lifecycle': lifecycle, 'input_lifecycle_observation': lifecycle_observation,
            'storage_maintenance': os.environ.get('VISIONCORTEX_STORAGE_MAINTENANCE', '0') == '1',
            'process_since_us': config.get('device_day', {}).get('process_since_us'),
            'capture_link_cleanup': {'enabled': bool(cleanup.get('enabled')),
            'native_links_verified': bool(cleanup.get('native_links_verified')),
            'capture_readers_verified': bool(cleanup.get('capture_readers_verified'))},
            'night_schedule': schedule, 'provider': {k: provider.get(k) for k in
                ('active', 'reason', 'last_probe_at', 'last_probe_status')}, 'waiting': waiting,
            'observed_at': now, 'focus_date': priority_date(root), 'days': days, 'vision_timings': timings,
            'failure_categories': failures, 'vision_component_timings': component_timings,
            'running': jobs, 'errors': errors + latency['errors'], 'latency': latency,
            'latency_html': lifecycle_html + render_latency(latency) + render_diagnostics(failures, component_timings),
            'scope': 'queue_records_not_current_version_acceptance'}


def render_consumers(consumers):
    names = dict(retention='原片归档', vision='视觉预处理', stt='语音识别', understanding='多模态理解', report='日报')
    labels = {'covered': '有已核验消费者', 'no_consumer': '缺少已核验消费者', 'no_ready_work': '暂无可运行任务',
              'disabled': '未启用', 'paused_by_user': '已暂停', 'waiting_for_night_window': '等待运行时段',
              'evidence_unavailable': '状态暂不可核实'}
    holds = {'waiting_for_warm_model': '等待已有模型就绪', 'waiting_for_monitor': '等待采集监控恢复',
             'waiting_for_live': '优先处理最新任务', 'waiting_for_idle': '等待空闲',
             'waiting_for_memory': '等待可用内存', 'waiting_for_storage': '等待本地存储达到留空要求',
             'waiting_for_provider': '等待云服务恢复',
             'waiting_for_night_window': '等待运行时段', 'status_unavailable': '调度状态暂不可核实',
             'cold_warm_disabled_capacity': '等待可预留实时容量的模型池',
             'model_preparation_busy': '等待模型准备工位', 'model_preparation_cooldown': '模型准备失败，等待重试',
             'waiting_for_prerequisite_validation': '等待原文件与前序回执通过校验',
             'paused_by_user': '已暂停', 'quantum_elapsed': '已保存断点并让出时间片',
             'resource_busy': '等待空闲资源', 'history_running_elsewhere': '历史任务由其他进程处理'}
    rows = []
    for stage, item in consumers['stages'].items():
        for scope, count in item['scopes'].items():
            waiting = ''
            if scope == 'history':
                reasons = {holds[owner['backfill_status']] for owner in item['owners']
                           if owner['owner_state'] == 'verified' and owner.get('backfill_status') in holds}
                if reasons:
                    waiting = '（' + '；'.join(sorted(reasons)) + '）'
            elif any(owner['owner_state'] == 'verified' and
                     (owner.get('last_result') or {}).get('status') == 'waiting_for_storage'
                     for owner in item['owners']):
                waiting = '（' + holds['waiting_for_storage'] + '）'
            rows.append('<tr><td>' + names[stage] + (' · 最新' if scope == 'recent' else ' · 历史') +
                        '</td><td>' + labels[count['status']] + waiting + '</td><td>' + str(count['locally_ready']) +
                        '</td><td>' + str(count['blocked']) + '</td></tr>')
    return ('<details><summary>自动处理消费者与积压</summary><p>可运行数量依据本地元数据；执行前仍须核验回执和原文件。'
            '未提供新版本身份心跳的旧服务，消费者状态暂不可核实。</p>'
            '<div class="table-wrap"><table><thead><tr><th>处理范围</th><th>消费者状态</th><th>待核验可运行</th>'
            '<th>等待输入或前序</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></details>')


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
