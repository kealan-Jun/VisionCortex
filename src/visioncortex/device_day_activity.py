"""Process-local live work phases; observations never confer artifact acceptance."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
import time

_CURRENT = ContextVar('device_day_activity', default=None)
_LOCK = Lock()
_ACTIVE = {}
PHASES = {
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
def job(stage, recording_id):
    key = (stage, recording_id)
    token = _CURRENT.set(key)
    with _LOCK:
        row = _ACTIVE[key] = {'phase': stage+'_work' if stage+'_work' in PHASES else 'claimed',
                        'phase_started': time.time(), 'phase_seconds': {}, 'frame_counts': {}}
    try:
        yield row
    finally:
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
    try:
        yield
    finally:
        with _LOCK:
            row = _ACTIVE.get(key)
            if row is not None:
                row['phase_seconds'][name] = row['phase_seconds'].get(name, 0) + time.monotonic()-start
                row.update(phase=old[0], phase_started=old[1])


def observations():
    with _LOCK:
        return {key: {'phase': PHASES[row['phase']], 'phase_elapsed_seconds': max(0,time.time()-row['phase_started']),
                      'phase_seconds': dict(row['phase_seconds']), 'frame_counts':dict(row['frame_counts'])} for key,row in _ACTIVE.items()}


def counted(phase_name, count):
    key = _CURRENT.get()
    if key is not None:
        with _LOCK:
            row = _ACTIVE.get(key)
            if row is not None:
                row['frame_counts'][phase_name] = row['frame_counts'].get(phase_name, 0)+count
