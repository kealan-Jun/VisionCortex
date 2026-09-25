"""Local and durable work phases; observations never confer artifact acceptance."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
import time
import json
from pathlib import Path
import uuid
import sqlite3
import logging

_CURRENT = ContextVar('device_day_activity', default=None)
_LOCK = Lock()
_ACTIVE = {}


def current_stage():
    value = _CURRENT.get()
    return value[0] if value else None


PHASES = {
 'io_queue_seconds': '等待共享 I/O 名额',
 'input_validation_seconds': '验证封口输入身份与稳定性',
 'input_hash_seconds': '输入内容哈希',
 'decode_inference_seconds': '解码与模型处理',
 'archive_copy_seconds': '原始素材归档复制',
 'archive_hash_seconds': '归档内容与持久化校验',
 'formal_publication_seconds': '正式设备日索引发布',
 'retention_work': '原片及采集资料归档（含完整性校验）',
 'stt_work': '录音归档与转写', 'understanding_work': '多模态理解', 'report_work': '整合并发布日报',
 'claimed': '检查分片与输出目录', 'prerequisites': '校验原片及上游产物',
 'source_validation_and_setup_seconds': '校验源文件、时间戳与模型',
 'coarse_scan_seconds': '粗扫：解码供帧与YOLO推理',
 'coarse_planning_seconds': '筛选精扫范围',
 'fine_scan_seconds': '精扫：解码供帧与YOLO推理',
 'fine_index_and_coverage_seconds': '建立精扫索引并检查覆盖',
 'fine_activity_audit_seconds': '实验活动与连续性复核',
 'materialization_seconds': '切分视频与提取素材',
 'audit_publication_seconds': '发布预处理结果',
}


@contextmanager
def job(stage, recording_id, *, root=None):
    key = (stage, recording_id)
    token = _CURRENT.set(key)
    with _LOCK:
        row = _ACTIVE[key] = {'phase': stage+'_work' if stage+'_work' in PHASES else 'claimed',
                        'phase_started': time.time(), 'phase_seconds': {}, 'frame_counts': {},
                        'root': root, 'token': uuid.uuid4().hex, 'persisted_at': 0}
    _persist(key, row, force=True)
    try:
        yield row
    finally:
        _persist(key, row, done=True)
        with _LOCK:
            _ACTIVE.pop(key, None)
        _CURRENT.reset(token)


@contextmanager
def phase(name):
    key = _CURRENT.get()
    if key is None or name not in PHASES:
        yield
        return
    start = time.monotonic()
    with _LOCK:
        old = (_ACTIVE[key]['phase'], _ACTIVE[key]['phase_started'])
        _ACTIVE[key].update(phase=name, phase_started=time.time())
        row = dict(_ACTIVE[key])
    _persist(key, row, force=True)
    try:
        yield
    finally:
        with _LOCK:
            row = _ACTIVE.get(key)
            if row is not None:
                row['phase_seconds'][name] = row['phase_seconds'].get(name, 0) + time.monotonic()-start
                row.update(phase=old[0], phase_started=old[1])
        if row is not None:
            _persist(key, row, force=True)


def observations(root=None):
    with _LOCK:
        current = {key: _public(row) for key,row in _ACTIVE.items()}
    path = Path(root) / 'WorkPhases.sqlite3' if root else None
    if path and path.is_file():
        from .sqlite_store import connection
        try:
            with connection(path, readonly=True, timeout=.2) as db:
                for row in db.execute('SELECT * FROM phases'):
                    current.setdefault((row['stage'], row['id']), _public(json.loads(row['payload'])))
        except (OSError, sqlite3.Error, ValueError):
            pass
    return current


def counted(phase_name, count):
    row = None
    key = _CURRENT.get()
    if key is not None:
        with _LOCK:
            row = _ACTIVE.get(key)
            if row is not None:
                row['frame_counts'][phase_name] = row['frame_counts'].get(phase_name, 0)+count
    if row is not None:
        _persist(key, row)


def _public(row):
    return {'phase': PHASES[row['phase']], 'phase_elapsed_seconds': max(0,time.time()-row['phase_started']),
            'phase_seconds': dict(row['phase_seconds']), 'frame_counts': dict(row['frame_counts']),
            'phase_observed_at': row.get('persisted_at', time.time())}


def _persist(key, row, *, force=False, done=False):
    try:
        _write_phase(key, row, force=force, done=done)
    except (OSError, sqlite3.Error):
        logging.getLogger(__name__).debug("Phase telemetry unavailable", exc_info=True)


def _write_phase(key, row, *, force=False, done=False):
    if not row.get('root'):
        return
    if not (force or done) and time.time() - row['persisted_at'] < 1:
        return
    from .sqlite_store import connection
    path = Path(row['root']) / 'WorkPhases.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    with connection(path, timeout=.2) as db:
        db.execute('CREATE TABLE IF NOT EXISTS phases(stage TEXT,id TEXT,token TEXT,payload TEXT,PRIMARY KEY(stage,id))')
        if done:
            db.execute('DELETE FROM phases WHERE stage=? AND id=? AND token=?', (*key, row['token']))
        else:
            row['persisted_at'] = time.time()
            payload = {k:v for k,v in row.items() if k != 'root'}
            db.execute('INSERT OR REPLACE INTO phases VALUES(?,?,?,?)', (*key,row['token'],json.dumps(payload)))
