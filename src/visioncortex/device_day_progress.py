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
            repair = retry_observation(config)
            health = processing_health(config)
            self.consumer_state = {'consumers': consumers, 'consumer_html': render_consumers(consumers),
                                   'processing_health': health, 'processing_health_html': render_processing_health(health),
                                   'retry_repair': repair, 'retry_repair_html': render_retry_observation(repair)}
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


def _workload_counts(categories):
    """Input dependencies are not processing backlog; access faults still are."""
    result = dict.fromkeys(('completed', 'running', 'processing_backlog', 'system_attention',
                            'waiting_for_input', 'scheduled_wait', 'outside_processing_scope'), 0)
    attention = {'inventory_unobserved', 'source_version_mismatch', 'input_unavailable',
                 'awaiting_request_reconciliation', 'blocked', 'camera_role_unconfigured',
                 'input_not_processable', 'retry_exhausted'}
    for category, count in categories.items():
        group = ('waiting_for_input' if category in {'input_missing', 'input_waiting', 'upstream_input_waiting'} else
                 'scheduled_wait' if category in {'paused_by_user', 'waiting_for_night_window'} else
                 category if category in {'completed', 'running', 'outside_processing_scope'} else
                 'processing_backlog')
        result[group] += count
        if category in attention:
            result['system_attention'] += count
    return result


def processing_health(config, *, now=None):
    """Observe progress and useful backlog separately from process liveness.

    Only local catalogs are read. Missing observations are not proof that media
    was deleted, and runnable metadata never accepts a receipt or model result.
    """
    current = time.time() if now is None else now
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    settings = config.get('device_day') or {}
    threshold = settings.get('progress_stall_seconds', 1800)
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 60:
        raise ValueError('device_day.progress_stall_seconds must be finite and >= 60')
    queues, errors, inventory, availability = {}, [], None, {}
    availability_path = root / 'InputAvailability.sqlite3'
    if availability_path.is_file():
        try:
            from .sqlite_store import connection
            with connection(availability_path, readonly=True, timeout=.25) as db:
                availability = {row['id']: dict(row) for row in db.execute('SELECT id,signature,state FROM inputs')}
        except (OSError, sqlite3.Error):
            errors.append('input_availability_unavailable')
    inventory_path = root / 'observed-inventory.sqlite3'
    if inventory_path.is_file():
        try:
            from .sqlite_store import connection
            with connection(inventory_path, readonly=True, timeout=.25) as db:
                inventory = {row[0]: row[1] for row in db.execute(
                    "SELECT id,json_extract(payload,'$.source_signature') FROM observations")}
        except (OSError, sqlite3.Error):
            errors.append('current_inventory_unavailable')
    for stage in STAGES:
        path = root / f'queue-{stage}.sqlite3'
        queues[stage] = {}
        if not path.is_file():
            continue
        try:
            from .sqlite_store import connection
            with connection(path, readonly=True, timeout=.25) as db:
                for row in db.execute("SELECT recording_id,status,input_status,lease_until,queued_at,updated_at,"
                    "completed_at,attempts,retry_limit,json_extract(payload,'$.source_signature') AS signature,"
                    "json_extract(payload,'$.configured_role') AS role,"
                    "json_extract(payload,'$.processable') AS processable,"
                    "json_extract(payload,'$.available') AS available,"
                    "json_extract(payload,'$.recording_start_us') AS capture_us,"
                    "json_extract(result,'$.status') AS result_status,json_extract(result,'$.reason') AS reason,"
                    "json_extract(result,'$.retry_at') AS retry_at FROM recordings"):
                    queues[stage][row['recording_id']] = dict(row)
        except (OSError, sqlite3.Error):
            errors.append(f'{stage}_queue_unavailable')
    from .device_day_contract import dependencies_for
    from .device_day_night_schedule import paused_stages, stage_admitted
    from .device_day_retry import retry_limit

    def input_state(rid, row):
        observed = availability.get(rid, {})
        if observed.get('signature') == row['signature'] and observed.get('state', 'ready') != 'ready':
            return observed['state']
        return row['input_status']

    def upstream_input_waiting(rid, row, parents, seen=()):
        for parent in parents:
            if parent in seen:
                continue
            prior = queues[parent].get(rid)
            if not prior or prior['status'] == 'completed' or prior['signature'] != row['signature']:
                continue
            if prior['status'] == 'running' and (prior['lease_until'] or 0) >= current:
                continue
            if input_state(rid, prior) in {'missing', 'waiting'}:
                return True
            dependencies = dependencies_for(settings, {'recording_start_us': prior['capture_us'] or 0})[parent]
            if settings.get('inplace_preprocessing') and parent in {'vision', 'stt'}:
                dependencies = ()
            if upstream_input_waiting(rid, prior, dependencies, (*seen, parent)):
                return True
        return False

    stages, alerts = {}, []
    for stage, rows in queues.items():
        counts, ready_since, completed = {}, [], []
        for rid, row in rows.items():
            status = row['status']
            if row['completed_at'] and status == 'completed':
                completed.append(row['completed_at'])
            record = {'recording_start_us': row['capture_us'] or 0}
            cutoff = settings.get('process_since_us')
            parents = dependencies_for(settings, record)[stage]
            if settings.get('inplace_preprocessing') and stage in {'vision', 'stt'}:
                parents = ()
            category = (
                'completed' if status == 'completed' else
                'outside_processing_scope' if cutoff and (row['capture_us'] or 0) < cutoff else
                'running' if status == 'running' and (row['lease_until'] or 0) >= current else
                'input_' + input_state(rid, row) if input_state(rid, row) != 'ready' else
                'inventory_unobserved' if inventory is not None and rid not in inventory else
                'source_version_mismatch' if inventory is not None and inventory.get(rid) != row['signature'] else
                'awaiting_request_reconciliation' if row['reason'] == 'submitted_unknown' else
                'blocked' if status == 'blocked' else
                'camera_role_unconfigured' if row['role'] not in {'first_person', 'third_person'} else
                'input_not_processable' if not (row['processable'] if row['processable'] is not None else row['available']) else
                'upstream_input_waiting' if upstream_input_waiting(rid, row, parents) else
                'waiting_for_prerequisite' if status == 'waiting_for_prerequisite' or any(
                    queues[parent].get(rid, {}).get('status') != 'completed' for parent in parents) else
                'retry_exhausted' if status == 'failed' and row['attempts'] >= retry_limit(
                    row, settings.get('failure_retry_limit') or 3) else
                'retry_backoff' if (row['retry_at'] or 0) > current else
                'paused_by_user' if stage in paused_stages(config) else
                'waiting_for_night_window' if not stage_admitted(config, stage) else
                'locally_ready')
            counts[category] = counts.get(category, 0) + 1
            if category in {'locally_ready', 'running', 'retry_backoff'}:
                ready_since.append(row['queued_at'])
        last_completed = max(completed, default=None)
        anchor = max(last_completed or 0, min(ready_since)) if ready_since else None
        stalled = anchor is not None and current - anchor >= threshold
        workload = _workload_counts(counts)
        state = ('evidence_unavailable' if f'{stage}_queue_unavailable' in errors else
                 'no_completed_progress' if stalled else 'working' if ready_since else
                 'blocked' if workload['processing_backlog'] else
                 'waiting_for_input' if workload['waiting_for_input'] else
                 'scheduled_wait' if workload['scheduled_wait'] else 'idle')
        stages[stage] = {'status': state, 'categories': counts, 'workload': workload, 'last_completed_at': last_completed,
                         'seconds_without_completion': max(0, current - anchor) if anchor is not None else None}
        if stalled:
            alerts.append({'stage': stage, 'reason': 'no_completed_progress'})
    recovery = {}
    try:
        from .multimodal_usage import UsageLedger
        recovery = UsageLedger(config, readonly=True).recovery_status()
        if recovery.get('unknown_outcomes_require_reconciliation'):
            alerts.append({'stage': 'understanding', 'reason': 'awaiting_request_reconciliation'})
        if recovery.get('admission_available') is False:
            alerts.append({'stage': 'understanding', 'reason': 'local_receipt_capacity'})
    except (OSError, sqlite3.Error, ValueError):
        errors.append('request_recovery_status_unavailable')
    from .input_availability import integrity_status
    integrity = integrity_status(config, now=current)
    if integrity['active_alert_count']:
        alerts.append({'stage': 'archive', 'reason': 'archive_integrity_alerts'})
    if integrity['status'] not in {'not_configured', 'running', 'disabled'}:
        alerts.append({'stage': 'archive', 'reason': 'integrity_service_unavailable'})
    return {'status': 'attention_required' if alerts else 'evidence_unavailable' if errors else 'observing',
            'observed_at': current, 'stall_threshold_seconds': threshold, 'stages': stages,
            'workload_unit': 'stage_tasks', 'workload_complete': not errors, 'input_loss_evidence': 'NOT_PROVEN',
            'request_recovery': recovery, 'input_integrity': integrity,
            'alerts': alerts, 'errors': errors, 'local_metadata_only': True,
            'readiness_boundary': 'receipt_and_media_validation_required'}


def render_processing_health(health):
    labels = {'no_completed_progress': '长时间没有新增完成任务',
              'awaiting_request_reconciliation': '已提交请求结果待核对',
              'local_receipt_capacity': '本地响应恢复额度已满',
              'archive_integrity_alerts': '输入或归档校验有待处理异常',
              'integrity_service_unavailable': '自动校验服务未就绪或心跳过期'}
    rows = ''.join(f'<li>{escape(item["stage"])}：{labels.get(item["reason"], escape(item["reason"]))}</li>'
                   for item in health['alerts'])
    categories = {'locally_ready': '本地元数据就绪', 'running': '租约有效', 'input_missing': '输入待核对',
                  'input_waiting': '等待输入就绪', 'upstream_input_waiting': '前序等待输入',
                  'inventory_unobserved': '当前清单未观察到', 'source_version_mismatch': '来源版本待核对',
                  'awaiting_request_reconciliation': '请求结果待核对', 'waiting_for_prerequisite': '等待前置阶段',
                  'retry_backoff': '重试退避', 'retry_exhausted': '失败预算耗尽', 'blocked': '待处理阻塞',
                  'input_unavailable': '输入不可访问', 'camera_role_unconfigured': '机位待绑定',
                  'input_not_processable': '输入尚不可处理', 'paused_by_user': '已暂停',
                  'waiting_for_night_window': '等待处理时间窗', 'outside_processing_scope': '处理范围外'}
    counts = []
    for stage, state in health['stages'].items():
        workload = state.get('workload') or _workload_counts(state['categories'])
        if not any(key != 'completed' for key in state['categories']):
            continue
        details = '；'.join(f'{categories.get(key, escape(key))} {count}'
                           for key, count in state['categories'].items() if key != 'completed')
        counts.append(f'<li>{escape(stage)}：本系统待处理 {workload["processing_backlog"]}'
                      f'（其中待排查 {workload["system_attention"]}）；运行中 {workload["running"]}；'
                      f'输入等待 {workload["waiting_for_input"]}（不计处理积压）；'
                      f'按计划等待 {workload["scheduled_wait"]}<small>{details}</small></li>')
    errors = '<p>部分本地状态不可读，进度核验不完整。</p>' if health['errors'] else ''
    return ('<details open><summary>处理进度核验</summary><p>服务心跳与处理进度分别核验；'
            '输入等待单列，不计处理积压。访问故障、来源版本和处理失败仍属本系统待排查。'
            '缺失标记不代表已确认原片丢失或责任归属；队列就绪状态仍须通过原文件与回执校验。'
            '以下按阶段任务计数，同一分片可涉及多个阶段。</p>' + errors + '<ul>' + rows + '</ul><ul>'
            + ''.join(counts) + '</ul>' + render_integrity(health.get('input_integrity', {})) + '</details>')


def render_integrity(state):
    if not state or state.get('status') == 'not_configured':
        return ''
    reasons = {'archive_content_conflict': '归档内容与回执哈希冲突',
               'derived_content_conflict': '派生产物与回执哈希冲突',
               'derived_artifact_missing': '派生产物缺失', 'published_index_missing': '已发布索引缺失，等待重放',
               'archive_receipt_missing': '归档回执缺失', 'archive_receipt_invalid': '归档回执结构待核对',
               'archive_receipt_unverified': '归档来源身份待核对', 'archive_source_binding_changed': '归档路径绑定待核对',
               'completed_receipt_missing': '已完成阶段回执缺失', 'completed_receipt_unverified': '已完成回执身份待核对',
               'recorder_input_missing': '采集原件缺失，未重建采集目录',
               'input_revalidation_rejected': '原件内容或来源身份未通过重验',
               'input_seal_unavailable': '缺少可核验的输入封存',
               'archive_missing_source_unverified': '归档缺失且原件尚未核验',
               'integrity_verification_unavailable': '存储或校验暂不可用',
               'recovery_retry_exhausted': '同一修复已重试，需进一步排查',
               'verification_budget_exceeded': '本条校验量超过配置上限',
               'input_changed_during_verification': '校验期间文件发生变化',
               'input_recovery_state_changed': '恢复前队列状态已变化'}
    reasons['storage_identity_unverified'] = '存储挂载或卷身份未通过核验'
    reasons['execution_recipe_changed'] = '执行配方已变化，须走正常准入'
    rows = ''.join('<li>' + escape(item['stage']) + ' · ' + escape(item['id']) + '：'
                   + escape(reasons.get(item['reason'], item['reason'])) + '</li>' for item in state.get('alerts', []))
    count = state.get('active_alert_count', 0)
    service = state.get('service') or {}
    outcomes = {'verified': '最近分片核验通过', 'restored': '归档文件已补回', 'retry_authorized': '已授权一次恢复重试',
                'attention_required': '发现待处理异常', 'idle': '等待下次检查', 'busy': '等待运行任务释放校验锁',
                'disabled': '校验未启用', 'yielded_to_foreground': '为前台任务让出资源',
                'storage_unavailable': '等待存储恢复'}
    last = outcomes.get((service.get('last_result') or {}).get('status'), '尚无检查结果')
    transitions = {'raised': '新增异常', 'changed': '异常变化', 'resolved': '异常解除', 'recovered': '完成自动恢复'}
    event_reasons = reasons | {'restored': '归档文件已补回', 'retry_authorized': '已授权一次恢复重试'}
    events = ''.join('<li>' + escape(transitions.get(event['transition'], event['transition'])) + ' · '
                     + escape(event['key']) + '：' + escape(event_reasons.get(event['reason'], event['reason']))
                     + '</li>' for event in state.get('events', []))
    coverage = state.get('coverage') or {}
    coverage_text = ('<p>完成归档队列记录 ' + str(coverage['completed_queue_records']) + '；当前有效校验 '
                     + str(coverage['verified_current']) + '；尚未核验或已变化 ' + str(coverage['unverified_or_changed'])
                     + '；超过复查时间 ' + str(coverage['overdue_current']) + '。</p>' if coverage else '')
    return ('<section class="integrity-status" aria-live="polite"><h3>自动校验与归档完整性</h3>'
            '<p' + (' role="alert"' if count or not state.get('available') else '') + '>'
            + ('校验服务运行中' if state.get('available') and state.get('status') == 'running' else
               '校验未启用' if state.get('status') == 'disabled' else '校验服务状态待检查')
            + f'；待处理异常 {count} 条。最近检查：' + escape(last) + '。</p><ul>' + rows + '</ul>'
            + coverage_text
            + '<details><summary>最近告警与恢复事件</summary><ul>' + events + '</ul></details>'
            '<p>异常跨重启保留；重复异常不反复通知，变化和恢复会记入事件。校验按分片轮转，'
            '服务在线或单条通过不代表全部历史归档已完成检查。</p></section>')


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
                admission_column = 'processing_cohort' if 'processing_cohort' in columns else 'NULL AS processing_cohort'
                rows = db.execute("SELECT recording_id,status,lease_until,queued_at,updated_at, "
                                  "json_extract(payload,'$.camera_key'),json_extract(payload,'$.recording_start_us'),"
                                  "json_extract(payload,'$.recording_end_us'),wall_seconds,completed_at,result,"
                                  "json_extract(payload,'$.source_signature'),attempts,input_status," + limit_column +
                                  ',' + started_column + ',' + admission_column + " FROM recordings")
                for rid, status, lease, queued, updated, camera, start, end, wall, completed_at, raw_result, signature, attempts, input_status, authorized_limit, started_at, cohort in rows:
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
                    queue_details[(rid, stage)].update(processing_cohort=cohort, reason=result.get('reason'))
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
        recordings = b.pop('recordings')
        b['total'] = len(recordings)
        b['camera_count'] = len(b.pop('cameras'))
        for stage, counts in b['stages'].items():
            counts['not_enqueued'] = max(0, b['total']-sum(counts.values()))
            # Missing originals exclude only unfinished work for THIS stage.
            # A completed historical/no-audio result or an active lease still
            # belongs to its own denominator, regardless of other stages.
            excluded = {rid for rid in missing_inputs.get(day, set()) & recordings
                        if states.get(rid, {}).get(stage, ('missing',))[0] not in {'completed', 'running'}}
            counts['missing_input_count'] = len(excluded)
            counts['processing_total'] = b['total'] - len(excluded)
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
    repair = retry_observation(config, now=now)
    settings = config.get('device_day') or {}
    cutoff = (now - settings.get('live_priority_seconds', 14400)) * 1_000_000
    waiting = {day: {s: {} for s in STAGES} for day in days}
    for identifier, row in states.items():
        for stage, (status, day) in row.items():
            if status not in {'queued', 'expired', 'waiting_for_prerequisite', 'failed',
                              'input_missing', 'input_unavailable', 'needs_camera_role', 'blocked'}:
                continue
            dependencies = () if settings.get('inplace_preprocessing') and stage in {'vision', 'stt'} else DEPENDENCIES[stage]
            parents = [row.get(p, ('missing', day))[0] for p in dependencies]
            details = queue_details.get((identifier, stage), {})
            scope = 'recent' if capture_times.get(identifier, 0) >= cutoff else 'history'
            if details.get('processing_cohort') in {'live', 'current_recovery'}:
                scope = 'recent'
            coverage = consumers['stages'][stage]['scopes'][scope]['status']
            live_hold = any(owner['owner_state'] == 'verified' and owner.get('backfill_status') == 'waiting_for_live'
                            for owner in consumers['stages'][stage]['owners'])
            storage_hold = any(owner['owner_state'] == 'verified' and (
                owner.get('backfill_status') == 'waiting_for_storage'
                or (owner.get('last_result') or {}).get('status') == 'waiting_for_storage')
                for owner in consumers['stages'][stage]['owners'])
            reason = ('paused_by_user' if stage in paused else
                      (details.get('reason') or 'blocked') if status == 'blocked' else
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
    health = processing_health(config, now=now)
    return {'consumers': consumers, 'consumer_html': render_consumers(consumers),
            'processing_health': health, 'processing_health_html': render_processing_health(health),
            'retry_repair': repair, 'retry_repair_html': render_retry_observation(repair),
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


def retry_observation(config, *, now=None):
    """Read only the repair owner's bounded local heartbeat, never its verifier."""
    try:
        from .device_day_retry_worker import retry_service_snapshot
        return retry_service_snapshot(config, now=now)
    except (ImportError, OSError, ValueError, TypeError):
        return {'available': False, 'reason': 'repair_service_observation_unavailable', 'local_metadata_only': True}


def render_retry_observation(repair):
    title = '<details><summary>自动修复重试服务</summary>'
    if not repair.get('available'):
        label = ('尚无本地服务心跳，运行状态未核实' if repair.get('reason') == 'repair_service_observation_absent'
                 else '本地服务快照暂不可核实')
        return title + '<p>' + label + '。</p></details>'
    enabled = ('已启用' if repair.get('configured_enabled') is True else
               '未启用' if repair.get('configured_enabled') is False else '启用配置未核实')
    owner = repair.get('owner_state')
    status = repair.get('status')
    if owner == 'verified':
        label = ('已核验进程正在核验修复条件' if status == 'running' and repair.get('phase') == 'checking'
                 else '已核验进程等待下一轮核验' if status == 'running'
                 else '已核验进程，自动修复未启用' if status == 'disabled'
                 else '进程状态已核实，当前不在核验')
    else:
        label = {'stale': '服务心跳已过期', 'stopped': '服务已停止', 'failed': '服务发生错误'}.get(
            owner, '进程身份尚未核实')
    identity = ' · 进程 ' + str(repair['pid']) if owner == 'verified' and type(repair.get('pid')) is int else ''
    last = repair.get('last_result') or {}
    reasons = {'waiting': '本轮没有可授权候选', 'running_elsewhere': '另一进程正在核验',
               'verification_rejected': '候选未通过当前输入、前序回执或修复核验',
               'stale_or_already_applied': '候选已变化或同一修复已授权', 'granted': '已授予有限重试预算'}
    reason = reasons.get(last.get('status'), '尚无本轮核验结果')
    if last.get('status') == 'verification_rejected' and isinstance(last.get('reason'), str):
        reason += '；阻塞原因：' + retry_rejection_label(last['reason'])
    error = last.get('error_type') or repair.get('error_type')
    if error:
        reason += '（' + escape(str(error)) + '）'
    scanned = last.get('scanned')
    counts = [('outside_scope', '不在自动修复范围'), ('unknown_or_disabled', '原因未识别或未授权'),
              ('cooling', '等待冷却')]
    hints = ([f'本轮检查 {scanned} 条本地候选'] if type(scanned) is int else [])
    hints += [str(last[key]) + ' ' + label for key, label in counts if type(last.get(key)) is int and last[key] > 0]
    kinds = {'provider_binding': '服务账户绑定', 'database_lock': '数据库锁', 'storage_access': '存储访问',
             'vision_nms': '视觉 NMS', 'receipt_projection': '回执映射',
             'resource_coordination': '资源协调', 'storage_capacity': '存储留空'}
    allowed = repair.get('allowed_classes')
    scope = ('；'.join(kinds.get(kind, escape(str(kind))) for kind in allowed) or '未配置自动修复类别'
             if isinstance(allowed, list) else '范围未核实')
    return (title + '<p>' + enabled + ' · ' + label + identity + '。自动核验范围：' + scope + '。</p>'
            '<p>最近服务快照累计：授权 ' + str(repair.get('granted_count', 0)) + ' 次，核验未通过 '
            + str(repair.get('rejected_count', 0)) + ' 次，已完成核验轮次 '
            + str(repair.get('completed_tick_count', 0)) + '。' + reason + '。</p>'
            + ('<p>' + ' · '.join(hints) + '。</p>' if hints else '')
            + '<p>授权仅增加有限重试预算，不代表模型已执行、阶段已完成或成果已发布；原失败记录保留。</p></details>')


def retry_rejection_label(reason):
    names = dict(zip(STAGES, ('归档', '视觉预处理', '录音识别', '多模态理解', '日报'), strict=True))
    for prefix, label in [('parent_queue_not_completed_', '前序队列尚未完成'),
                          ('parent_receipt_or_artifact_unverified_', '前序回执或产物尚未通过核验'),
                          ('parent_queue_revision_not_bound_to_accepted_receipt_', '前序队列版本尚未绑定已接受回执'),
                          ('consumer_owner_not_unique_or_fresh_', '消费者身份不唯一或心跳过期')]:
        stage = reason.removeprefix(prefix)
        if reason.startswith(prefix) and stage in names:
            return names[stage] + '：' + label
    labels = {
        'repair_already_authorized': '同一修复已授予预算',
        'failed_budget_state_changed': '失败记录或重试预算状态已变化',
        'current_source_proof_incomplete_or_stale': '当前原文件核验证明不足或已过期',
        'current_parent_context_proof_incomplete': '当前前序回执或上下文证明不足',
        'specific_repair_proof_incomplete': '此类修复尚无足够验证证据',
        'current_proof_changed_before_grant': '授权前核验证明已变化',
        'current_verification_rejected': '当前核验条件未满足',
        'queue_input_differs_from_current_inventory': '队列输入与当前采集清单不一致',
        'archived_source_snapshot_or_stable_bytes_unverified': '归档原片快照或稳定字节核验未通过',
        'current_input_availability_not_ready': '当前输入尚未就绪',
        'current_queue_recipe_changed_normal_admission_required': '当前执行配方已变化，须走正常任务准入',
        'effective_runtime_recipe_differs_from_reviewed_preflight': '有效运行配方与审核预检不一致',
        'source_view_guard_rejected_current_namespace': '当前存储视图未通过核验',
        'source_view_guard_not_proven': '当前存储视图尚未核实',
        'failed_target_has_unreconciled_lease': '失败记录仍有未协调的运行租约',
        'current_local_capacity_admission_closed': '当前本地容量尚未允许新任务准入',
    }
    return labels.get(reason, escape(reason))


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
                        '</td><td>' + str(count.get('running', 0)) + '</td></tr>')
    return ('<details><summary>自动处理消费者</summary><p>可运行数量依据本地元数据；执行前仍须核验回执和原文件。'
            '输入等待、系统待排查和处理积压见处理进度核验。'
            '未提供新版本身份心跳的旧服务，消费者状态暂不可核实。</p>'
            '<div class="table-wrap"><table><thead><tr><th>处理范围</th><th>消费者状态</th><th>待核验可运行</th>'
            '<th>运行中</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></details>')


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
