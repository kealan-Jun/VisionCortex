"""Incremental, local/NAS-readable overview of published evidence and live queues.

An index is the authority for published clips; queue completion is only an
execution metric. Neither is human ground truth or a current release receipt.
"""
from datetime import datetime
import html
import json
import math
from pathlib import Path
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .device_day_contract import ARTIFACT_DIRECTORIES, atomic_bytes, atomic_json, validate_archive_name
from .device_day_progress import snapshot
from .report_brand import brand_logo_data_url
from .application.device_day_readmodel import DeviceDayReadModels, ReadModelDeferred
from .receipt_projection import ProjectionDeferred, projected_json

STAGE_NAMES = {'retention': '原片归档', 'vision': 'YOLO 预处理', 'stt': '录音识别',
               'understanding': '多模态理解', 'report': '日报更新'}
WAIT_LABELS = {'paused_by_user': '按用户要求暂停',
               'prerequisite_not_verified': '等待前置产物校验或修复',
               'upstream_failed': '上游失败待恢复', 'upstream_pending': '上游未完成',
               'provider_blocked': '云端账户不可用', 'night_window': '等待夜间窗口',
               'lease_recovery': '租约过期待恢复', 'pending_validation': '等待调度（旧版本状态）',
               'waiting_for_dispatch': '等待实时任务调度', 'waiting_for_history_dispatch': '等待历史任务调度',
               'history_waiting_for_live': '历史任务等待实时容量', 'no_consumer': '缺少已核验消费者',
               'consumer_evidence_unavailable': '消费者状态暂不可核实', 'retry_exhausted': '重试预算耗尽，待核验修复',
               'failure_cooldown': '失败后等待冷却', 'input_missing': '输入缺失，待归档原件核验',
               'waiting_for_storage': '等待本地存储达到留空要求',
               'submitted_unknown': '已提交请求结果未知，等待回执核对',
               'anomalous_inactive_request_plan': '重复视觉请求计划异常；保留来源、时间记录与诊断',
               'blocked': '任务阻塞，等待明确处置依据',
               'input_unavailable': '输入暂不可用', 'camera_role_unconfigured': '尚未配置机位角色', 'disabled': '未启用'}


def processing_service_status(config, *, now=None, proc_root=Path('/proc')):
    """Read each owner's status independently; a stale host is not a stopped lane.

    Only the companion's fresh heartbeat bound to a live process incarnation
    proves it online. Never update WorkerStatus or infer completion/throughput.
    """
    now = time.time() if now is None else now
    root = Path(config['storage']['local_runtime_root']) / 'state'

    def read(name):
        try:
            value = json.loads((root / name).read_text())
            return value if isinstance(value, dict) else {'status': 'unavailable'}
        except FileNotFoundError:
            return {'status': 'missing'}
        except (OSError, ValueError):
            return {'status': 'unavailable'}

    def valid_time(value):
        return type(value) in (int, float) and 0 <= value <= now + 5 and math.isfinite(value)

    host = read('WorkerStatus.json')
    host_at = host.get('at')
    host_state = host.get('status')
    if not isinstance(host_state, str):
        host_state = 'unavailable'
    if host_state == 'running':
        host_state = ('unavailable' if not valid_time(host_at) else
                      'heartbeat_expired' if now - host_at > 15 else 'running')
    host = {'status': host_state if host_state in {
        'running', 'heartbeat_expired', 'stopped', 'missing', 'unavailable'} else 'unavailable',
        'heartbeat_at': host_at if valid_time(host_at) else None}

    raw = read('DeviceDayCompanionStatus.json')
    companion = {'status': 'unavailable', 'process_identity_verified': False}
    result = {'observed_at': now, 'snapshot_expires_at': now + 90,
              'companion': companion, 'host': host}
    if not isinstance(raw.get('status'), str):
        return result
    if raw.get('status') in {'missing', 'unavailable'}:
        companion['status'] = raw['status']
        return result
    pid, ticks, at = raw.get('pid'), raw.get('process_start_ticks'), raw.get('at')
    if (raw.get('schema_version') != 'visioncortex-device-day-companion/1'
            or raw.get('status') not in {'running', 'draining', 'stopped'}
            or type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0
            or not valid_time(at)):
        return result
    companion.update(pid=pid, heartbeat_at=at)
    if now - at > 15:
        companion['status'] = 'expired'
        return result
    if raw['status'] == 'stopped':
        companion['status'] = 'stopped'  # A recorded exit, not a claim about the host.
        return result
    try:
        stat = (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        if int(stat[19]) != ticks or stat[0] in {'Z', 'X', 'x'}:
            companion['status'] = 'unverified'
            return result
    except (OSError, ValueError, IndexError):
        companion['status'] = 'unverified'
        return result
    companion.update(status=raw['status'], process_identity_verified=True,
                     needs_attention=bool(raw.get('error') or raw.get('ancillary_errors')))
    return result


def render_service_status(data):
    if not data:
        return ''
    companion = data['companion']
    labels = {'running': '在线', 'draining': '正在收尾', 'expired': '状态已过期',
              'unverified': '暂未确认运行', 'missing': '尚无状态记录',
              'unavailable': '状态暂不可读', 'stopped': '已记录退出'}
    host_labels = {'running': '心跳正常', 'heartbeat_expired': '心跳已过期',
                   'stopped': '已记录退出', 'missing': '尚无心跳记录', 'unavailable': '状态暂不可读'}
    checked = datetime.fromtimestamp(data['observed_at'], ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
    label = labels.get(companion['status'], '状态暂不可读')
    if companion.get('needs_attention'):
        label += '，有状态待核查'
    host_label = host_labels.get(data['host']['status'], '状态暂不可读')
    return (f'<p><strong data-companion-status data-status-expiry="{data["snapshot_expires_at"]}">'
            f'自动处理执行器：{label}</strong><small>最近核验：{checked}（北京时间）；这是定时更新的状态快照。</small></p>'
            f'<details><summary>查看原宿主状态</summary><p>原宿主：{host_label}。'
            '宿主心跳与自动处理执行器分别记录；心跳过期不等于处理已停止。</p></details>')


SERVICE_STATUS_SCRIPT = """(() => {
  const label = document.querySelector('[data-companion-status]');
  if (!label) return;
  const expires = Number(label.dataset.statusExpiry);
  const refresh = () => {
    if (!Number.isFinite(expires) || Date.now() / 1000 > expires) {
      label.textContent = '自动处理执行器：状态已过期，请刷新核验（不代表已停止）';
    }
  };
  refresh();
  setInterval(refresh, 1000);
})();"""


def union_seconds(intervals):
    end, total = 0, 0
    for start, stop in sorted(intervals):
        if stop > start:
            total += max(0, stop - max(start, end))
            end = max(end, stop)
    return total / 1e6


def summarize(name, index, config, *, report_exists=False, understanding_exists=False, root=None):
    """Count unique published records/materials at their actual grain."""
    validate_archive_name(name)
    if index.get('archive') != name:
        raise ValueError('Archive/index identity mismatch')
    camera, day = name[11:], name[:10]
    records = {r['recording_id']: r for r in index.get('recordings', [])}
    segments = {s['segment_id']: s for s in index.get('segments', [])}
    if any(s['recording_id'] not in records for s in segments.values()):
        raise ValueError('Segment has no source recording')
    active = [s for s in segments.values() if s.get('activity') == 'active']
    sources = {s['retained']['path']: s for r in records.values() for s in r.get('sources', [])}
    understood = {s['segment_id'] for s in index.get('understandings', [])
                  if s.get('status') == 'completed' and s.get('windows')} & segments.keys()
    keys = {f['path'] for s in active for f in s.get('key_frames', [])}
    covered = {s['recording_id'] for s in segments.values()}
    role = config.get('collection_ingest', {}).get('camera_role_map', {}).get(camera, 'unknown')
    from .device_day_narrative import retention_summary
    from .device_day_retention_projection import interval_views
    from .device_day_content_paths import clip_name, layout_contract
    # Reuse the interval owner without copying model windows into a summary.
    views = [(child, interval) for child, _, interval in interval_views(
        {'segments': list(segments.values()), 'understandings': []}) if interval is not None]
    expected_paths = {child['segment_id']: ('ProcessedClips/Clips/' + clip_name(child) + '/'
        + layout_contract()['activities'][child['activity']] + '.mp4') for child, _ in views}
    guarded = []
    for candidate in segments.values():
        rows = []
        for interval in candidate.get('retention_intervals', []):
            receipt = interval.get('receipt') or {}
            valid = ((interval.get('video') or {}).get('path') == expected_paths.get(interval.get('segment_id'))
                and interval.get('processing_status') == 'completed'
                and isinstance(receipt.get('path'), str) and receipt['path'].startswith('ProcessedClips/')
                and receipt['path'].endswith('.json'))
            # Do not even hash an original that was mislabeled as an output.
            rows.append(interval if valid else interval | {'video': None})
        guarded.append(candidate | {'retention_intervals': rows})
    retained = retention_summary(index | {'segments': guarded}, root)
    proofs = {(row['candidate_segment_id'], row['start_ms'], row['end_ms']):
              row.get('actual_video_verification') for row in retained['partitions']}
    intervals = []
    for child, interval in views:
        proof = proofs.get((child['candidate_segment_id'], child['start_ms'], child['end_ms'])) or {}
        expected = expected_paths[child['segment_id']]
        available = (root is not None and interval['video_value'] == 'selected'
            and interval.get('processing_status') == 'completed' and proof.get('status') == 'PROVEN'
            and (child.get('video') or {}).get('path') == expected)
        reason = ('actual_files_not_checked' if root is None else
                  'processing_not_completed' if interval.get('processing_status') != 'completed' else
                  'selected_path_not_frozen_clip' if (child.get('video') or {}).get('path') != expected else
                  proof.get('reason', 'exact_export_not_verified'))
        intervals.append({key: child.get(key) for key in ('segment_id', 'candidate_segment_id',
            'recording_id', 'start_us', 'end_us', 'start_ms', 'end_ms', 'source_ref', 'json_path', 'key_frames')} | {
            'activity': child['activity'], 'activity_category': interval.get('activity_category', 'unknown'),
            'video_value': interval['video_value'], 'processing_status': interval.get('processing_status', 'unknown'),
            'reason': interval.get('reason'), 'video': child.get('video') if available else None,
            'understanding_path': interval.get('understanding_path'),
            'actual_video_verification': proof.get('status', 'NOT_PROVEN') if available else 'NOT_PROVEN',
            'verification_reason': reason,
            'actual_duration_ms': proof.get('duration_ms') if available else None})
    selected = [row for row in intervals if row['video']]
    return {'archive': name, 'date': day, 'camera': camera, 'role': role, 'updated_at': index.get('updated_at'),
            'index_path': name + '/ProcessedClips/Index.json', 'indexed_recordings': len(records),
            'preprocessed_recordings': len(covered), 'active_count': len(active),
            'inactive_count': sum(s.get('activity') == 'inactive' for s in segments.values()),
            'keyframes': len(keys), 'understood_segments': len(understood), 'segment_count': len(segments),
            'video_files': sum(s['kind'] == 'video' for s in sources.values()),
            'audio_files': sum(s['kind'] == 'audio_audio' for s in sources.values()),
            'csv_files': sum(p.lower().endswith('.csv') for p in sources),
            'source_bytes': sum(s['retained'].get('size_bytes', 0) for s in sources.values()),
            'stt_transcribed_recordings': sum(r.get('transcription', {}).get('status') == 'completed'
                                             and r.get('transcription', {}).get('outcome') == 'transcribed'
                                             for r in records.values()),
            'report_exists': report_exists, 'understanding_exists': understanding_exists,
            'selected_count': retained['selected_interval_count'], 'available_selected_count': len(selected),
            'pending_value_count': retained['pending_interval_count'],
            'confirmed_no_value_count': retained['confirmed_no_value_interval_count'],
            'selected_video_seconds': sum(row['actual_duration_ms'] for row in selected) / 1000,
            'current_files_checked': root is not None, 'intervals': intervals,
            'candidates': [{key: s.get(key) for key in ('segment_id', 'recording_id', 'start_us', 'end_us',
                'start_ms', 'end_ms', 'source_ref', 'json_path', 'video', 'historical_scope')} for s in segments.values()],
            'active': [{k: s.get(k) for k in ('segment_id', 'recording_id', 'start_us', 'end_us',
                'start_ms', 'end_ms', 'video', 'source_ref', 'json_path', 'key_frames')} for s in active]}


def assemble(progress, archives, errors):
    dates = sorted(set(progress['days']) | {a['date'] for a in archives}, reverse=True)
    days = []
    for date in dates:
        rows = [a for a in archives if a['date'] == date]
        queued = progress['days'].get(date)
        sums = {k: sum(a[k] for a in rows) for k in ('indexed_recordings', 'preprocessed_recordings',
                'active_count', 'inactive_count', 'keyframes', 'video_files', 'audio_files', 'csv_files',
                'source_bytes', 'stt_transcribed_recordings', 'understood_segments', 'segment_count')}
        sums.update({key: sum(a.get(key, 0) for a in rows) for key in ('selected_count',
            'available_selected_count', 'pending_value_count', 'confirmed_no_value_count', 'selected_video_seconds')})
        days.append({'date': date, **sums, 'archives': rows, 'queue': queued,
                     'discovered_recordings': queued['total'] if queued else None,
                     'camera_count': queued['camera_count'] if queued else len(rows),
                     'activity_seconds': union_seconds((s['start_us'], s['end_us']) for a in rows
                         if not a.get('historical_snapshot') for s in a.get('intervals', []) if s.get('video')),
                     'waiting': progress.get('waiting', {}).get(date, {}),
                     'read_errors': [e for e in errors if e['archive'].startswith(date)]})
    return {'updated_at': progress['observed_at'], 'days': days, 'progress': progress, 'errors': errors,
            'scope': 'published_indexes_and_queue_snapshots_not_human_ground_truth',
            'evidence_status': 'PARTIAL_EVIDENCE', 'refresh_seconds': 30}


class ArchiveOverview:
    def __init__(self):
        self.cache = {}
        self.repair_after = {}
        # The compatibility CLI/combined publisher shares the bounded public
        # reader. Its summaries have their own cache; do not retain a second
        # whole projected device/day here or hydrate dense CV diagnostics.
        self._published_views = DeviceDayReadModels(lambda value: value, cache_limit=0)

    @staticmethod
    def _version(root, path, references=()):
        values = []
        for source in (path, root/'Comment/TimeIndex.json',
                root/'LaboratoryDailyReport/LaboratoryDailyReport.json',
                root/'MultimodalUnderstanding/Index.json', *(root/ref for ref in references)):
            try:
                info = source.stat()
                values.append((str(source), source.is_symlink(), info.st_dev, info.st_ino,
                               info.st_size, info.st_mtime_ns, info.st_ctime_ns))
            except OSError as exc:
                values.append((str(source), type(exc).__name__))
        return tuple(values)

    def publish(self, runner):
        progress = snapshot(runner.config)
        # Local observations run on this background worker, never the GPU or
        # capture discovery lanes. An unavailable sampler cannot block output.
        from .device_day_acceptance import observe, render as render_acceptance
        import sqlite3
        try:
            progress['acceptance_observations'] = observe(runner.config, progress)
            progress['latency_html'] = progress.get('latency_html', '') + render_acceptance(progress['acceptance_observations'])
        except (OSError, sqlite3.Error, ValueError, KeyError):
            progress.setdefault('errors', []).append('持续运行观察记录暂不可用')
        archives, errors, seen = [], [], set()
        from .device_day_schedule import processing_cutoff, in_processing_scope
        if processing_cutoff(runner.config.get('device_day', {})):
            from .observed_inventory import read_inventory
            from .device_day_contract import archive_name
            names = {archive_name(r['camera_key'], r['recording_start_us'])
                     for r in read_inventory(runner.runtime_root)['recordings']
                     if in_processing_scope(runner.config['device_day'], r)}
            previous_path = runner.runtime_root/'ArchiveOverview.json'
            previous = {}
            if previous_path.is_file():
                try:
                    if previous_path.stat().st_size > 32 * 1024**2:
                        raise ReadModelDeferred('Previous overview exceeds its 32 MiB source limit')
                    previous = projected_json(previous_path, max_resident_bytes=16 * 1024**2)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append({'archive': previous_path.name, 'reason': type(exc).__name__,
                                   'read_status': 'deferred', 'detail': str(exc)[:240]})
            archives = [a | {'historical_snapshot': True, 'current_files_checked': False,
                        'available_selected_count': 0, 'selected_video_seconds': 0,
                        'intervals': [row | {'video': None, 'actual_video_verification': 'NOT_PROVEN',
                            'verification_reason': 'historical_snapshot_not_current_file_verification'}
                            for row in a.get('intervals', [])]} for d in previous.get('days', [])
                        for a in d['archives'] if a['archive'] not in names]
            roots = [runner.archive_root/name for name in sorted(names, reverse=True)]
        else:
            roots = sorted(runner.archive_root.iterdir(), reverse=True)
        # Current exact exports use the owning cached artifact verifier. Never
        # read originals/frames or invoke a model to build this derived view.
        for root in roots:
            try:
                validate_archive_name(root.name)
            except ValueError:
                continue
            if root.is_symlink():
                continue
            seen.add(root.name)
            try:
                path = root / 'ProcessedClips/Index.json'
                path.stat()
                previous = self.cache.get(root.name)
                references = previous[2] if previous else ()
                version = self._version(root, path, references)
                if not previous or previous[0] != version:
                    before_load = self._version(root, path)
                    index = self._published_views.index(root).value
                    references = tuple(sorted({ref['path'] for segment in index.get('segments', [])
                        for row in segment.get('retention_intervals', []) for key in ('video', 'receipt')
                        if isinstance((ref := row.get(key)), dict) and isinstance(ref.get('path'), str)
                        and not Path(ref['path']).is_absolute() and '..' not in Path(ref['path']).parts}))
                    before = self._version(root, path, references)
                    if before[:len(before_load)] != before_load:
                        raise ValueError('Published index changed during overview read')
                    data = summarize(root.name, index, runner.config, root=root,
                        report_exists=(root / 'LaboratoryDailyReport/LaboratoryDailyReport.json').is_file(),
                        understanding_exists=(root / 'MultimodalUnderstanding/Index.json').is_file())
                    if self._version(root, path, references) != before:
                        raise ValueError('Published artifacts changed during overview verification')
                    self.cache[root.name] = (before, data, references)
                archives.append(self.cache[root.name][1])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # Never silently substitute the last good value for an unreadable index.
                self.cache.pop(root.name, None)
                errors.append({'archive': root.name, 'reason': type(exc).__name__,
                    'read_status': 'deferred' if isinstance(exc, (ReadModelDeferred, ProjectionDeferred)) else 'unavailable',
                    'detail': str(exc)[:240]})
        self.cache = {k: v for k, v in self.cache.items() if k in seen}
        data = assemble(progress, archives, errors)
        data['processing_services'] = processing_service_status(runner.config)
        data['archive_root'] = str(runner.archive_root.resolve())
        body = render(data).encode()
        atomic_json(runner.runtime_root / 'ArchiveOverview.json', data)
        atomic_bytes(runner.runtime_root / 'ArchiveOverview.html', body)
        atomic_bytes(runner.archive_root / 'Readme.html', body)
        # Rebuild one absent derived index from existing canonical receipts on
        # this independent worker. Never rerun a model or fabricate a completion.
        from .device_day_recovery import repair_missing_index
        from .device_day_schedule import processing_cutoff
        repair = None
        for error in ([] if processing_cutoff(runner.config.get('device_day', {})) else errors):
            if error['reason'] == 'FileNotFoundError' and self.repair_after.get(error['archive'], 0) < progress['observed_at']:
                self.repair_after[error['archive']] = progress['observed_at'] + 300
                repair = repair_missing_index(runner, error['archive'])
                break
        return {'status': 'partial' if errors else 'completed', 'updated_at': data['updated_at'], 'archives': len(archives),
                'active_segments': sum(a['active_count'] for a in archives),
                'selected_videos': sum(a.get('available_selected_count', 0) for a in archives),
                'errors': errors, 'index_repair': repair}


def render(data):
    def esc(value):
        return html.escape(str(value), quote=True)
    def stamp(us):
        return datetime.fromtimestamp(us / 1e6, ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
    def link(archive, path, label, fragment='', *, watch=False):
        # Restrict every link to the five existing archive directories.
        if (not path or '\\' in path or ':' in path or Path(path).is_absolute() or '..' in Path(path).parts
                or Path(path).parts[0] not in ARTIFACT_DIRECTORIES):
            return '<span>素材引用不可用</span>'
        href = quote(archive + '/' + path, safe='/') + fragment
        return (f'<a data-artifact {"data-watch class=play-button" if watch else ""} data-archive="{esc(archive)}" data-path="{esc(path)}" '
                f'data-fragment="{esc(fragment)}" href="{esc(href)}" target="_blank" rel="noopener">{esc(label)}</a>')
    progress = data['progress']
    rows, details = [], []
    for day in data['days']:
        date, q = day['date'], day['queue']
        discovered = day['discovered_recordings']
        total = q.get('processing_total', discovered) if q else discovered
        missing = q.get('missing_input_count', 0) if q else 0
        processed = day['preprocessed_recordings']
        stages = q['stages'] if q else {}
        vision = stages.get('vision', {})
        pending = max(0, total-processed) if total is not None else None
        rows.append(f'''<tr data-day="{date}"><th><a href="#Day{date}">{date}</a><small>{day['camera_count']} 台设备</small></th>
<td>{discovered if discovered is not None else '未知'}<small>已发现分片；{missing} 个输入缺失，单独记录</small></td>
<td><b>{processed}</b> / {total if total is not None else '?'}<progress value="{processed}" max="{max(1,total or processed)}"></progress><small>已排除缺失输入；{pending if pending is not None else '未知'} 尚无发布结果</small></td>
<td><a class="activity" href="#Day{date}">{day['available_selected_count']} 个实际视频</a><small>选中计划 {day['selected_count']} 个；实际文件 {day['selected_video_seconds']:.1f} 秒</small></td>
<td>{day['keyframes']} 张</td><td>{day['audio_files']} / {day['stt_transcribed_recordings']}<small>原录音 / 已转写分片</small></td>
<td>{day['understood_segments']} / {day['segment_count']}<small>已理解 / 已发布区间</small></td>
<td>{vision.get('running',0)} 运行 · {vision.get('failed',0)} 失败<small>{'；'.join(f'{n} {WAIT_LABELS[k]}' for k,n in day['waiting'].get('vision',{}).items()) or '无排队项'}</small>{'<small class="error">索引读取不全</small>' if day['read_errors'] else ''}</td></tr>''')
        stage_rows = ''.join(f'<tr><th>{label}</th><td>{stages.get(stage,{}).get("completed",0)}</td>'
            f'<td>{stages.get(stage,{}).get("running",0)}</td><td>{stages.get(stage,{}).get("failed",0)}</td>'
            f'<td>{stages.get(stage,{}).get("not_enqueued",0)}</td><td>'
            + ('；'.join(f'{n} {WAIT_LABELS[k]}' for k,n in day['waiting'].get(stage,{}).items()) or '无') + '</td></tr>'
            for stage,label in STAGE_NAMES.items())
        device_rows, clip_rows = [], []
        for a in day['archives']:
            name = a['archive']
            refs = [link(name, 'ProcessedClips/Index.json', '总索引')]
            refs.append(link(name, 'Comment/TimeIndex.json', '时间索引 JSON'))
            if a['report_exists']:
                refs.append(link(name,'LaboratoryDailyReport/LaboratoryDailyReport.json','日报 JSON（已有部分）'))
            if a['understanding_exists']:
                refs.append(link(name,'MultimodalUnderstanding/Index.json','理解 JSON 索引'))
            role = {'first_person':'第一人称','third_person':'第三人称'}.get(a['role'],'视角待配置')
            if a.get('historical_snapshot'):
                refs.append('<small>历史快照：当前文件未重新核验，保留原追溯引用。</small>')
            elif not a.get('current_files_checked'):
                refs.append('<small>当前文件尚未核验。</small>')
            candidate_links = []
            for candidate in a.get('candidates', a.get('active', [])):
                source = candidate.get('source_ref') or {}
                fragment = f'#t={(candidate.get("start_ms") or 0)/1000:.3f},{(candidate.get("end_ms") or 0)/1000:.3f}'
                candidate_links.append('<li>' + link(name, candidate.get('json_path'), '预处理候选 JSON') + ' · '
                    + link(name, source.get('path'), '原片对应时间', fragment)
                    + (' · ' + link(name, candidate['video'].get('path'), '历史候选视频（未按当前精确范围重新认定）')
                       if candidate.get('video') else '') + '</li>')
            trace = ('<details><summary>原片、预处理候选及历史视频追溯</summary><ul>'
                     + ''.join(candidate_links) + '</ul></details>')
            device_rows.append(f'<tr><th>{esc(a["camera"])}<small>{role}</small></th><td>{a["video_files"]}</td>'
                f'<td>{a["active_count"]} / {a["inactive_count"]}</td><td>{a["csv_files"]}</td><td>{a["audio_files"]}</td>'
                f'<td>{" · ".join(refs)}<small>索引更新：{esc(a["updated_at"])}</small>{trace}</td></tr>')
            for s in a.get('intervals', []):
                ref = s.get('video') or {}
                frames = ''.join(f'<li>{link(name,f["path"],"动作关键帧 " + str(i+1))}</li>' for i,f in enumerate(s.get('key_frames') or []))
                nas_path = ''
                if ref.get('path') and '素材引用不可用' not in link(name, ref['path'], ''):
                    nas_path = str(Path(data.get('archive_root', 'VisionCortexExperimentArchive'))/name/ref['path'])
                location = (f'<details class="nas-location"><summary>在 NAS 中的位置</summary>'
                            f'<input readonly aria-label="NAS 视频路径" value="{esc(nas_path)}">'
                            '<button data-copy-path>复制路径</button></details>') if nas_path else ''
                category = {'experiment': '实验', 'non_experiment': '非实验', 'mixed': '混合', 'unknown': '类别未知'}.get(s.get('activity_category'), '类别未知')
                value = {'selected': '选中', 'pending': '待判定', 'confirmed_no_value': '确认无价值'}.get(s.get('video_value'), '价值未知')
                state = {'completed': '处理完成', 'queued': '排队', 'failed': '失败', 'partial': '部分完成',
                    'waiting_for_value_review': '等待价值判断', 'waiting_for_evidence': '等待证据核验',
                    'waiting_for_input': '等待输入', 'submitted_unknown': '调用结果未知'}.get(s.get('processing_status'), '状态未知')
                source = s.get('source_ref') or {}
                fragment = f'#t={(s.get("start_ms") or 0)/1000:.3f},{(s.get("end_ms") or 0)/1000:.3f}'
                video_link = (link(name, ref.get('path'), '打开选中视频') if ref else
                    '<span>无当前已验证视频</span>')
                clip_rows.append((s['start_us'], f'<tr><td>{stamp(s["start_us"])}—{stamp(s["end_us"])}</td>'
                    f'<td>{esc(a["camera"])}<small>{role}</small></td>'
                    f'<td>{category} · {value} · {state}<small>{"实际文件已核验" if ref else "文件可读性未确认；不能据此判无价值"}</small>'
                    f'<small>{esc(s.get("reason") or "未提供判断依据")}</small></td><td>{video_link}{location}'
                    f'<small>{link(name,s.get("json_path"),"区间 JSON")} · {link(name,s.get("understanding_path"),"Understanding.json")}</small>'
                    f'<small>{link(name,source.get("path"),"原片时间映射（追溯）",fragment)}</small></td>'
                    f'<td><details><summary>{len(s.get("key_frames") or [])} 张动作关键帧</summary><ul>{frames}</ul></details></td></tr>'))
        clips = ''.join(row for _,row in sorted(clip_rows))
        timing = progress.get('vision_timings', {}).get(date)
        timing_text = (f'最近 {timing["samples"]} 个有耗时记录的 YOLO 分片任务：中位数 {timing["median_seconds"]:.1f} 秒，'
                       f'P90 {timing["p90_seconds"]:.1f} 秒。包含读取、供帧与后处理，含历史版本，不是纯 GPU 推理耗时。'
                       if timing else '尚无可统计的 YOLO 分片任务耗时。')
        details.append(f'''<section id="Day{date}" data-day="{date}" data-activity-count="{day['available_selected_count']}"><h2>{date} · 视频筛选区间</h2>
<p>实际已验证选中视频 {day['available_selected_count']} 个，选中计划 {day['selected_count']} 个；待判定 {day['pending_value_count']} 个，确认无价值 {day['confirmed_no_value_count']} 个。原片与资料 {day['source_bytes']/1024**3:.2f} GiB。</p>
<p>预处理候选分类：活动 {day['active_count']} 个、非活动 {day['inactive_count']} 个；这些数量不代表选中视频或完整动作。已验证视频按采集时间跨视角去重 {day['activity_seconds']:.1f} 秒。</p>
<div class="table-wrap"><table><thead><tr><th>采集时间</th><th>设备与视角</th><th>活动／价值／处理状态</th><th>视频与 JSON</th><th>关键帧入口</th></tr></thead><tbody>{clips or '<tr><td colspan="5">尚无精确留存区间。原候选及历史引用保留在下方追溯入口；不能据此判定没有有价值活动。</td></tr>'}</tbody></table></div>
<details><summary>查看各设备的原片、采集资料与报告</summary><div class="table-wrap"><table><thead><tr><th>设备</th><th>原视频</th><th>活动 / 无活动区间</th><th>CSV</th><th>原录音</th><th>查阅入口</th></tr></thead><tbody>{''.join(device_rows) or '<tr><td colspan="6">尚无可读取索引</td></tr>'}</tbody></table></div></details>
<details><summary>查看五个阶段的处理状态与 YOLO 耗时</summary><div class="table-wrap"><table><thead><tr><th>阶段</th><th>任务完成</th><th>运行</th><th>失败</th><th>尚未入队</th><th>入队后的等待原因</th></tr></thead><tbody>{stage_rows}</tbody></table></div><p>{timing_text}</p><p>任务完成数来自执行队列，可能含历史版本及无录音的正常结束；日报任务逐分片更新同一份报告，不代表同等数量的日报。</p></details></section>''')
    running = ''.join(f'<li>{j["date"]} · {esc(j["camera"])} · {STAGE_NAMES[j["stage"]]} · {stamp(j["start_us"])}—{stamp(j["end_us"])} · {esc(j.get("phase","处理中"))}</li>' for j in progress['running'])
    updated = datetime.fromtimestamp(data['updated_at'],ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    schedule = progress['night_schedule']
    night = f'多模态和日报在 {schedule["start"]}–次日 {schedule["end"]} 运行。' if schedule['enabled'] else '夜间时段限制未启用。'
    if schedule.get('paused_stages'):
        night = '按用户要求暂停：' + '、'.join(STAGE_NAMES[s] for s in schedule['paused_stages']) + '。其他阶段继续按就绪顺序处理。'
    cloud = '<p class="error">云端账户当前不可用（Arrearage），录音识别和多模态等待服务恢复；原片归档与 YOLO 独立运行。</p>' if progress.get('provider',{}).get('active') else ''
    errors = ''.join(f'<li>{esc(e["archive"])}：{esc("总索引待恢复" if e["reason"] == "FileNotFoundError" else "兼容总览读取已延后；请打开当前设备日视图或原始 JSON" if e.get("read_status") == "deferred" else "总索引读取异常")}</li>' for e in data['errors'])
    errors += ''.join(f'<li>{esc(e)}</li>' for e in progress['errors'])
    activity_dates = ''.join(f'<a class="date-chip" href="#Day{d["date"]}">{d["date"]}<b>{d["available_selected_count"]} 个实际视频</b></a>'
                            for d in data['days'] if d['available_selected_count'])
    ordered_details = [row for day, row in zip(data['days'], details, strict=True) if day['available_selected_count']]
    ordered_details += [row for day, row in zip(data['days'], details, strict=True) if not day['available_selected_count']]
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisionCortex 实验数据总览</title>
<style>body{{margin:0;background:#eaf2f3;color:#112d32;font:15px/1.6 system-ui}}main{{max-width:1480px;margin:auto;padding:28px}}header,section{{background:#fbfdfd;padding:24px;margin:0 0 20px;border:1px solid #cbdadc;border-radius:14px}}header{{border-top:6px solid #1e4a52}}header img{{width:48px;float:left;margin-right:14px}}h1{{font-size:27px;margin:0}}h2{{font-size:21px}}h3{{font-size:17px}}a{{color:#216d77;text-underline-offset:3px}}small{{display:block;color:#526d73;font-size:12px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;text-align:left}}th,td{{padding:12px;border-bottom:1px solid #dbe6e8;vertical-align:top}}th{{font-weight:600}}td{{min-width:70px}}progress{{display:block;width:130px;height:9px;accent-color:#208a68}}.activity{{font-size:19px;font-weight:700}}details{{margin:15px 0}}td details{{margin:0}}summary{{cursor:pointer;color:#216d77}}.error{{color:#a34435}}select,button{{font:inherit;padding:8px 12px;border:1px solid #cbdadc;border-radius:6px;background:white}}nav{{display:flex;gap:16px;align-items:center;flex-wrap:wrap}}[hidden]{{display:none!important}}:focus-visible{{outline:3px solid #cf8337}}@media(max-width:700px){{main{{padding:10px}}header,section{{padding:16px}}h1{{font-size:22px}}}}@media print{{nav{{display:none}}section{{break-inside:avoid}}}}</style></head><body><main>
<header><img src="{brand_logo_data_url()}" alt="VisionCortex"><h1>实验数据总览</h1><p>数据更新：<time data-updated="{data['updated_at']}">{updated}</time>（北京时间） · 后台每 30 秒汇总，页面每 60 秒刷新</p>
<nav><label>采集日期 <select id="DateFilter"><option value="">全部日期</option>{''.join(f'<option>{d["date"]}</option>' for d in data['days'])}</select></label><button id="Refresh">刷新</button><a href="#Operations">后台处理进度</a><a data-app href="/#/day-timeline">返回应用时间线</a></nav></header>
<section id="Experiments"><h2>选中视频，从这里看</h2><nav class="date-chips">{activity_dates or '当前暂无已验证发布的精确选中视频。'}</nav><p>实验和非实验的有价值视频都列在对应日期；只有实际文件和精确导出回执已核验的选中区间提供视频入口。</p><small>原始录像、预处理候选与历史整段视频分别追溯，不计作当前选中视频。JSON 直接提供理解、留存原因和原片映射；文件完整性核验不等于完整物理动作或观看质量已经确认。</small></section>
{''.join(ordered_details)}
<div id="Operations"></div>
<section><h2>当前运行情况</h2>{render_service_status(data.get('processing_services'))}<p>原片、YOLO 按新分片持续处理。{night}无新增采集的日期不会被当作“无实验活动”。</p>{cloud}
<ul>{running or '<li>当前没有有效运行任务，具体等待原因见各日期的阶段状态。</li>'}</ul>
{progress.get('processing_health_html', '')}{progress.get('consumer_html', '')}{progress.get('latency_html', '')}
<p>采集端原视频软链接替换：{'已启用' if progress['capture_link_cleanup']['enabled'] else '未启用，原始数据保持原路径'}。</p>{'<details class="error"><summary>查看索引与队列读取异常（正在后台核对恢复）</summary><ul>'+errors+'</ul></details>' if errors else ''}</section>
<section><h2>每天处理到哪里了</h2><p>发布结果按原始分片去重；选中视频文件、筛选计划、预处理候选和阶段执行记录分别计数。</p>
<div class="table-wrap"><table><thead><tr><th>采集日期</th><th>采集量</th><th>已有预处理结果</th><th>选中视频</th><th>候选动作关键帧</th><th>录音 / 转写</th><th>多模态理解</th><th>YOLO 当前状态</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="8">尚无可读取的采集数据</td></tr>'}</tbody></table></div></section>
<p>选中入口来自精确留存区间及其实际视频、导出回执；缺失或尚未核验的文件不显示成已完成视频。无活动区间的场景采样图片不计入动作关键帧。统计来源：各设备的 ProcessedClips/Index.json、Comment/TimeIndex.json 的同一留存区间投影及本地阶段队列。</p>
<footer>原始目录 MetaData／MetaVideo 归采集端所有；结果保持 ProcessedClips、MultimodalUnderstanding、LaboratoryDailyReport、Comment。ExperimentActivity／NoExperimentActivity 是活动分类，不是价值决定或完整动作证明。理解直接阅读 Understanding.json，日报直接阅读 LaboratoryDailyReport.json。</footer></main>
<dialog id="Playback"><form method="dialog"><button aria-label="关闭播放器">关闭</button></form><h2 id="PlaybackTitle">实验视频</h2><div id="PlaybackBody"></div></dialog>
<style>header{{padding:16px 24px}}header p{{margin:6px 0 12px}}h2{{margin-top:0}}.date-chip{{border:1px solid #b9d4d7;padding:6px 12px;border-radius:8px;text-decoration:none;background:#edf7f5}}.date-chip b{{display:block;font-size:12px}}.play-button{{display:inline-block;padding:8px 14px;background:#1e4a52;color:white;border-radius:6px;text-decoration:none;white-space:nowrap;margin-bottom:6px}}button{{cursor:pointer}}.nas-location input{{font:12px monospace;max-width:100%;width:420px;box-sizing:border-box;padding:8px}}td small,td input{{overflow-wrap:anywhere}}dialog{{border:1px solid #bfd3d6;border-radius:12px;width:min(1320px,94vw);max-height:90vh;box-sizing:border-box;color:#112d32}}dialog::backdrop{{background:#102e35b3}}dialog form{{float:right}}dialog video{{display:block;width:100%;max-height:65vh;background:#132328}}.view-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,310px),1fr));gap:16px}}.view-grid article{{min-width:0;border:1px solid #d5e3e6;padding:12px;border-radius:8px}}.view-grid h3{{overflow-wrap:anywhere}}.view-grid video{{aspect-ratio:1.6;max-height:45vh}}.playback-controls{{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:14px 0}}.playback-controls input{{flex:1;min-width:150px}}section[id]{{scroll-margin-top:12px}}</style>
<script>{Path(__file__).with_name('web').joinpath('archive-playback.js').read_text()}</script>
<script>{SERVICE_STATUS_SCRIPT}</script>
<script>{Path(__file__).with_name('web').joinpath('archive-overview.js').read_text()}</script></body></html>'''
