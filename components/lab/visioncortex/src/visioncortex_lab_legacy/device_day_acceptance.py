"""Persist bounded production observations, never infer quality from queue health."""
import json
import sqlite3
from pathlib import Path


def observe(config, progress):
    from .device_day_latency import live_observations
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    root.mkdir(parents=True, exist_ok=True)
    now = progress['observed_at']
    counts = {'completed': 0, 'queued': 0, 'running': 0, 'expired': 0, 'failed': 0}
    for day in progress['days'].values():
        for key, value in day['stages']['vision'].items():
            counts[key] = counts.get(key, 0) + value
    with sqlite3.connect(root / 'production-observations.sqlite3', timeout=2) as db:
        db.executescript('''PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS samples(at REAL PRIMARY KEY, counts TEXT, errors INTEGER);
          CREATE TABLE IF NOT EXISTS live(recording_id TEXT, signature TEXT, observed REAL,
            started REAL, completed REAL, PRIMARY KEY(recording_id,signature));''')
        first = db.execute('SELECT MIN(at) FROM samples').fetchone()[0]
        first = now if first is None else first
        errors = list(progress.get('errors', []))
        try:
            live_rows = live_observations(config, max(first, now - 14*86400))
        except (OSError, sqlite3.Error, ValueError):
            live_rows = []
            errors.append('实时分片持久时延记录暂不可读；保留既有观察，不推断零新增')
        db.execute('INSERT OR REPLACE INTO samples VALUES(?,?,?)',
                   (now, json.dumps(counts), len(errors)))
        for r in live_rows:
            db.execute('''INSERT INTO live VALUES(?,?,?,?,?) ON CONFLICT(recording_id,signature)
              DO UPDATE SET started=CASE WHEN live.completed IS NOT NULL THEN live.started
                ELSE COALESCE(excluded.started,live.started) END,
                completed=COALESCE(live.completed,excluded.completed)''',
                       (r['recording_id'], r['source_signature'], r['revision_observed_at'],
                        r.get('vision_started_at'), r.get('preprocessing_completed_at')))
        db.execute('DELETE FROM samples WHERE at<?', (now - 14*86400,))
        db.execute('DELETE FROM live WHERE observed<?', (now - 14*86400,))
        rows = db.execute('SELECT at,errors FROM samples ORDER BY at').fetchall()
        live, completed = db.execute('SELECT COUNT(*),COUNT(completed) FROM live').fetchone()
    elapsed = rows[-1][0] - rows[0][0]
    max_gap = max((b[0]-a[0] for a, b in zip(rows, rows[1:], strict=False)), default=0)
    return {'observed_hours': elapsed/3600, 'samples': len(rows), 'max_sample_gap_seconds': max_gap,
            'samples_with_read_errors': sum(bool(r[1]) for r in rows), 'live_inputs': live,
            'live_published': completed, 'vision': counts, 'evidence_status': 'PARTIAL_EVIDENCE',
            'errors': errors,
            'live_input_scope': 'durable_current_source_revisions_and_previously_recorded_revisions',
            'scope': 'observation_only_not_continuous_load_or_quality_acceptance',
            'missing_gates': ['8–12小时持续采集负载验证', '真实实验识别与跨视角质量验收',
                              '夜间云端调用与次晨日报验证']}


def render(data):
    return ('<h3>持续运行观察</h3>'
            f'<p>已记录 {data["observed_hours"]:.1f} 小时 / {data["samples"]} 次观察；'
            f'期间新发现实时分片 {data["live_inputs"]} 个，已发布 {data["live_published"]} 个。'
            f'最大采样间隔 {data["max_sample_gap_seconds"]:.0f} 秒。</p>'
            '<p>实时分片按持久源版本计数，不受最近列表长度限制；未被采样即已被后续版本替换的历史版本无法追溯补计。</p>'
            '<p>运行时间不等于持续负载验收。没有新增采集的时段不能证明生产吞吐能力；'
            '识别质量与夜间日报仍需分别验证。</p>')
