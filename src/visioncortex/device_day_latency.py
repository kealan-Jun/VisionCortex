"""Durable discovery and queue timings, separate from media execution identity.

NAS mtimes may be preserved by an uploader. They are estimates, never receiver
upload acknowledgements. Existing jobs receive no reconstructed timestamps.
"""
from datetime import datetime
import math
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo


def _root(config):
    return Path(config['storage']['local_runtime_root']) / 'device-day'


def observe(config, records, discovery=None, *, now=None):
    now = time.time() if now is None else now
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / 'latency.sqlite3', timeout=2) as db:
        db.executescript('''PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS observations (
            recording_id TEXT PRIMARY KEY, camera TEXT NOT NULL, date TEXT NOT NULL,
            source_signature TEXT NOT NULL, first_observed_at REAL NOT NULL,
            revision_observed_at REAL NOT NULL, ready_at REAL, source_mtime_at REAL,
            cohort TEXT NOT NULL, start_us INTEGER NOT NULL, end_us INTEGER, published_at REAL);
        ''')
        today = datetime.fromtimestamp(now, ZoneInfo('Asia/Shanghai')).date().isoformat()
        for r in records:
            if not r.get('recording_start_us') or not r.get('source_signature'):
                continue
            day = datetime.fromtimestamp(r['recording_start_us']/1e6, ZoneInfo('Asia/Shanghai')).date().isoformat()
            info = (discovery or {}).get(r['recording_id'], {})
            at = info.get('observed_at', now)
            cohort = 'historical_backfill' if day < today else info.get('cohort', 'startup_inventory')
            # The history discovery lane also visits today's directories on
            # startup. Lane selection does not make today's data historical.
            if day >= today and cohort == 'historical_backfill':
                cohort = 'startup_inventory'
            ready = at if r.get('processable') else None
            try:
                mtime = datetime.fromisoformat(r.get('updated_at', '')).timestamp()
            except (TypeError, ValueError):
                mtime = None
            db.execute('''INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL)
              ON CONFLICT(recording_id) DO UPDATE SET
              revision_observed_at=CASE WHEN source_signature!=excluded.source_signature
                THEN excluded.revision_observed_at ELSE revision_observed_at END,
              ready_at=CASE WHEN source_signature!=excluded.source_signature THEN excluded.ready_at
                ELSE COALESCE(ready_at,excluded.ready_at) END,
              published_at=CASE WHEN source_signature!=excluded.source_signature THEN NULL ELSE published_at END,
              source_mtime_at=excluded.source_mtime_at, source_signature=excluded.source_signature,
              end_us=excluded.end_us, cohort=CASE
                WHEN cohort='historical_backfill' AND excluded.cohort='startup_inventory'
                THEN excluded.cohort ELSE cohort END
            ''', (r['recording_id'], r['camera_key'], day, r['source_signature'], at, at,
                  ready, mtime, cohort, r['recording_start_us'], r.get('recording_end_us')))


def published(config, records, *, now=None):
    """Called only after the corresponding ProcessedClips/Index.json is atomic."""
    path = _root(config) / 'latency.sqlite3'
    if not path.is_file():
        return
    at = time.time() if now is None else now
    try:
        with sqlite3.connect(path, timeout=2) as db:
            db.executemany('''UPDATE observations SET published_at=? WHERE recording_id=?
              AND source_signature=? AND ready_at<=? AND published_at IS NULL''',
              [(at, r['recording_id'], r.get('source_signature'), at) for r in records])
    except (OSError, sqlite3.Error):
        import logging
        logging.getLogger(__name__).warning('Index publication latency persistence unavailable')


def _elapsed(start, end):
    # A prior completion or changed input is not a zero-latency completion.
    return round(end-start, 3) if start is not None and end is not None and end >= start else None


def snapshot(config, *, now=None):
    now = time.time() if now is None else now
    root = _root(config)
    result = {'scope': 'observed_metadata_to_preprocessing_not_camera_recording_latency',
              'upload_completion_time_available': False, 'errors': [], 'recent': [], 'cohorts': {},
              'poll_seconds': (config.get('collection_ingest') or {}).get('poll_seconds'),
              'settle_seconds': (config.get('collection_ingest') or {}).get('settle_seconds')}
    path = root / 'latency.sqlite3'
    if not path.is_file():
        return result
    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=2) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute('SELECT * FROM observations')]
        queues = {}
        for stage in ('retention', 'vision'):
            path = root / f'queue-{stage}.sqlite3'
            queues[stage] = {}
            if not path.is_file():
                continue
            with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=2) as db:
                db.row_factory = sqlite3.Row
                columns = {r[1] for r in db.execute('PRAGMA table_info(recordings)')}
                started = 'started_at' if 'started_at' in columns else 'NULL AS started_at'
                for r in db.execute('SELECT recording_id,status,queued_at,completed_at,attempts,wall_seconds,'
                                    "json_extract(payload,'$.source_signature') AS source_signature,"+started+' FROM recordings'):
                    queues[stage][r['recording_id']] = dict(r)
        uploads = {}
        upload_path = root / 'UploadReceipts.sqlite3'
        if upload_path.is_file():
            from .sqlite_store import connection
            with connection(upload_path, readonly=True) as db:
                uploads = {(r['recording_id'], r['signature']): r['completed'] for r in db.execute('SELECT * FROM receipts')}
        for r in rows:
            uploaded = uploads.get((r['recording_id'], r['source_signature']))
            stages = {s: q.get(r['recording_id'], {}) for s, q in queues.items()}
            # Link only to the exact source revision observed, never a prior run.
            stages = {s: v if v.get('source_signature') == r['source_signature'] else {} for s, v in stages.items()}
            a, v = stages['retention'], stages['vision']
            ready, start, done = r['ready_at'], v.get('started_at'), r['published_at']
            valid_start = start if _elapsed(ready, start) is not None else None
            valid_done = done if valid_start is not None and _elapsed(start, done) is not None else None
            r.update({'upload_completed_at': uploaded, 'discovery_delay_seconds': _elapsed(uploaded, r['revision_observed_at']),
                'upload_to_preprocessing_start_seconds': _elapsed(uploaded, valid_start),
                'upload_to_preprocessing_completed_seconds': _elapsed(uploaded, valid_done),
                'upload_clock_source': 'receiver_reported' if uploaded is not None else None,
                'mtime_to_first_observation_estimate_seconds': _elapsed(r['source_mtime_at'], r['revision_observed_at']),
                'observed_to_ready_seconds': _elapsed(r['first_observed_at'], ready),
                'ready_to_archive_queue_seconds': _elapsed(ready, a.get('queued_at')),
                'archive_queue_seconds': _elapsed(a.get('queued_at'), a.get('started_at')),
                'archive_run_seconds': a.get('wall_seconds'),
                'archive_to_vision_queue_seconds': _elapsed(a.get('completed_at'), v.get('queued_at')),
                'vision_queue_seconds': _elapsed(v.get('queued_at'), start),
                'vision_run_seconds': v.get('wall_seconds') if _elapsed(valid_start, v.get('completed_at')) is not None else None,
                'ready_to_start_seconds': _elapsed(ready, valid_start),
                'ready_to_completed_seconds': _elapsed(ready, valid_done),
                'waiting_to_start_seconds': _elapsed(ready, now) if valid_start is None and ready and v.get('status') != 'completed' else None,
                'vision_started_at': valid_start, 'preprocessing_completed_at': valid_done,
                'vision_status': v.get('status', 'not_enqueued'), 'attempts': v.get('attempts', 0)})
        rows.sort(key=lambda r: r['revision_observed_at'], reverse=True)
        result['recent'] = rows[:50]
        result['upload_completion_time_available'] = any(r['upload_completed_at'] is not None for r in rows)
        metrics = ('discovery_delay_seconds', 'upload_to_preprocessing_start_seconds', 'upload_to_preprocessing_completed_seconds', 'ready_to_start_seconds', 'ready_to_completed_seconds', 'vision_queue_seconds', 'vision_run_seconds')
        for cohort in ('live_observation', 'startup_inventory', 'historical_backfill'):
            samples = [r for r in rows if r['cohort'] == cohort and r['revision_observed_at'] >= now-86400]
            result['cohorts'][cohort] = {'observations_24h': len(samples), 'metrics': {}}
            for metric in metrics:
                values = sorted(r[metric] for r in samples if r[metric] is not None)
                result['cohorts'][cohort]['metrics'][metric] = {
                    'samples': len(values), 'p50_seconds': values[math.ceil(len(values)*.5)-1] if values else None,
                    'p95_seconds': values[math.ceil(len(values)*.95)-1] if values else None,
                    'max_seconds': max(values) if values else None}
    except (sqlite3.Error, OSError, ValueError):
        result['errors'].append('分片时延记录暂不可读')
    return result


def render(data):
    """Shared HTML for the application and self-contained NAS overview."""
    from html import escape
    def seconds(n):
        return f'{n:.1f} 秒' if n is not None else '尚无测量'
    def clock(n):
        return datetime.fromtimestamp(n, ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S') if n is not None else '尚未确认'
    cohort_names = {'live_observation': '持续监控发现', 'startup_inventory': '启动时已有数据',
                    'historical_backfill': '历史补跑'}
    rows = []
    for name, group in data.get('cohorts', {}).items():
        m = group['metrics']
        start, end = m['ready_to_start_seconds'], m['ready_to_completed_seconds']
        rows.append(f'<tr><td>{cohort_names[name]}</td><td>{group["observations_24h"]}</td>'
                    f'<td>{seconds(start["p50_seconds"])} / {seconds(start["p95_seconds"])}'
                    f'<small>{start["samples"]} 个已启动样本</small></td>'
                    f'<td>{seconds(end["p50_seconds"])} / {seconds(end["p95_seconds"])}'
                    f'<small>{end["samples"]} 个已完成样本</small></td></tr>')
    def state(r):
        if r['vision_started_at'] is not None:
            return seconds(r['ready_to_start_seconds'])
        if r['vision_status'] == 'completed':
            return '已有结果，缺少本次时延样本'
        return '仍未启动：'+seconds(r['waiting_to_start_seconds'])
    recent = ''.join(f'<tr><td>{r["date"]} {datetime.fromtimestamp(r["start_us"]/1e6, ZoneInfo("Asia/Shanghai")).strftime("%H:%M:%S")}<br>{escape(r["camera"])}</td>'
                     f'<td>上传 {clock(r.get("upload_completed_at"))}<br>发现 {clock(r["first_observed_at"])}<br>可处理 {clock(r["ready_at"])}<br>上传至发现 {seconds(r.get("discovery_delay_seconds"))}<br>上传至启动 {seconds(r.get("upload_to_preprocessing_start_seconds"))}</td>'
                     f'<td>等待 {seconds(r["archive_queue_seconds"])}<br>执行 {seconds(r["archive_run_seconds"])}</td>'
                     f'<td>{state(r)}</td>'
                     f'<td>{seconds(r["vision_queue_seconds"])}</td><td>{seconds(r["vision_run_seconds"])}</td>'
                     f'<td>{seconds(r["ready_to_completed_seconds"])}</td></tr>' for r in data.get('recent', [])[:20])
    return ('<h3>新分片多久开始处理</h3><p>从确认文件可处理开始计时，包含归档和调度等待；'
            '结束时间为预处理任务成功落盘，不等待夜间多模态。以下统计最近 24 小时发现的分片。</p>'
            '<p>采集端尚未提供可信的 NAS 上传完成回执，写入完成到发现的时延暂无法精确测量；'
            '文件修改时间只作估计，不能当作上传完成时间。启动盘点和历史补跑单独统计。</p>'
            '<div class="table-wrap"><table><thead><tr><th>来源</th><th>发现分片</th>'
            '<th>可处理 → 启动 P50 / P95</th><th>可处理 → 落盘 P50 / P95</th></tr></thead>'
            f'<tbody>{"".join(rows) or "<tr><td colspan=4>尚无时延样本，后续自动累计。</td></tr>"}</tbody></table></div>'
            '<details><summary>最近分片的归档与预处理时延</summary><div class="table-wrap"><table><thead>'
            '<tr><th>采集时段 / 相机</th><th>监控时刻</th><th>原片归档</th><th>可处理至启动</th><th>YOLO 队列等待</th><th>预处理耗时</th>'
            f'<th>可处理至落盘</th></tr></thead><tbody>{recent}</tbody></table></div></details>')


def upload_completed(config, receipt):
    """Authenticated receiver's explicit completion time, never reconstructed."""
    from .observed_inventory import read_inventory
    from .sqlite_store import connection
    root = _root(config)
    record = next((r for r in read_inventory(root).get('recordings', [])
                   if r['recording_id'] == receipt['recording_id']), None)
    if record is None or record.get('source_signature') != receipt['source_signature']:
        raise ValueError('尚未发现此分片版本，请在发现后重试；不能关联到其他版本')
    at = receipt['completed_at']
    if not isinstance(at, (int,float)) or not math.isfinite(at) or not 0 < at <= time.time()+5:
        raise ValueError('上传完成时间无效或接收端时钟超前')
    root.mkdir(parents=True, exist_ok=True)
    with connection(root / 'UploadReceipts.sqlite3') as db:
        db.execute('''CREATE TABLE IF NOT EXISTS receipts(recording_id TEXT,signature TEXT,
                      completed REAL,received REAL,PRIMARY KEY(recording_id,signature))''')
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT completed FROM receipts WHERE recording_id=? AND signature=?',
                         (receipt['recording_id'],receipt['source_signature'])).fetchone()
        if row and row[0] != at:
            raise ValueError('该分片已有不同的上传回执；不能覆盖原始时间')
        db.execute('INSERT OR IGNORE INTO receipts VALUES(?,?,?,?)',
                   (receipt['recording_id'],receipt['source_signature'],at,time.time()))
    return {'status':'recorded', 'clock_source':'receiver_reported', **receipt}
