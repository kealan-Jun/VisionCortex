"""Bounded HTTP views of published device/day files; never rewrite evidence."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
import json
from pathlib import Path
import stat
import sys
import threading

from ..device_day_contract import safe_child


MAX_SOURCE_BYTES = 128 * 1024**2
MAX_PROJECTED_BYTES = 16 * 1024**2
MAX_RESPONSE_BYTES = 8 * 1024**2
MAX_CACHE_BYTES = 32 * 1024**2
MAX_CACHE_ENTRIES = 512
MAX_READERS = 2
CANONICAL_OMIT = {('recordings', None, 'processing', name)
                  for name in ('batches', 'scan_reports', 'audit_artifacts')} | {
                      ('segments', None, 'activity_audit')}
TIME_OMIT = {('streams', 'images'), ('readable_content',)}


class ReadModelDeferred(ValueError):
    """Keep published files intact and report an explicit unavailable view."""


def generation(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ReadModelDeferred('Published JSON must be a regular file')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def resident_size(value, limit):
    total, pending = 0, [iter((value,))]
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        total += 2 * sys.getsizeof(item)
        if total > limit:
            return total
        if isinstance(item, dict):
            pending.append(chain(item.keys(), item.values()))
        elif isinstance(item, (tuple, list)):
            pending.append(iter(item))
    return total


@dataclass(frozen=True)
class PublishedView:
    generation: tuple
    value: dict
    body: bytes
    cost: int


class DeviceDayReadModels:
    def __init__(self, recording_status, *, source_limit=MAX_SOURCE_BYTES,
                 resident_limit=MAX_PROJECTED_BYTES, response_limit=MAX_RESPONSE_BYTES,
                 cache_limit=MAX_CACHE_BYTES, cache_entries=MAX_CACHE_ENTRIES,
                 readers=MAX_READERS):
        self.recording_status = recording_status
        self.source_limit, self.resident_limit = source_limit, resident_limit
        self.response_limit, self.cache_limit = response_limit, cache_limit
        self.cache_entries, self.readers = cache_entries, readers
        self.lock = threading.Lock()
        self.cache, self.flights = OrderedDict(), {}
        self.cache_bytes = 0

    def _read(self, path, kind, *, transform=None):
        path = Path(path)
        before, key = generation(path), (str(path), kind)
        if before[2] > self.source_limit:
            raise ReadModelDeferred('Published JSON exceeds the bounded HTTP source limit; open its original file directly')
        with self.lock:
            cached = self.cache.get(key)
            if cached and cached.generation == before:
                self.cache.move_to_end(key)
                return cached
            if cached:
                self.cache_bytes -= self.cache.pop(key).cost
            event = self.flights.get(key)
            if event is None:
                if len(self.flights) >= self.readers:
                    raise ReadModelDeferred('Published view readers are busy; retry shortly')
                event = self.flights[key] = threading.Event()
                owner = True
            else:
                owner = False
        if not owner:
            if not event.wait(0.15):
                raise ReadModelDeferred('This published view is being read; retry shortly')
            # A failed owner never makes an earlier successful generation current.
            return self._read(path, kind, transform=transform)
        try:
            from ..receipt_projection import projected_json
            @contextmanager
            def opener(mode):
                with path.open(mode) as handle:
                    limit = self.source_limit
                    class Reader:
                        read_bytes = 0
                        def read(self, size=-1):
                            size = limit + 1 if size < 0 else min(size, limit + 1 - self.read_bytes)
                            block = handle.read(size)
                            self.read_bytes += len(block)
                            if self.read_bytes > limit:
                                raise ReadModelDeferred('Published JSON grew beyond the HTTP source limit')
                            return block
                    yield Reader()
            value = projected_json(path, omit=CANONICAL_OMIT if kind == 'index' else TIME_OMIT,
                                   max_resident_bytes=self.resident_limit, opener=opener)
            if kind == 'index':
                for field in ('recordings', 'segments', 'understandings'):
                    if not isinstance(value.setdefault(field, []), list):
                        raise ReadModelDeferred('Invalid canonical view arrays')
                value = self.recording_status(value)
            elif (not isinstance(value.get('recordings'), list)
                  or not isinstance(value.get('streams'), dict)
                  or any(not isinstance(value['streams'].get(k, []), list)
                         for k in ('video', 'audio', 'multimodal'))):
                raise ReadModelDeferred('Invalid published time index')
            if any(not isinstance(row, dict) or not isinstance(row.get('recording_id'), str)
                   for row in value.get('recordings', [])):
                raise ReadModelDeferred('Invalid published recording identity')
            if value.get('archive', path.parent.parent.name) != path.parent.parent.name:
                raise ReadModelDeferred('Published JSON belongs to another device/day')
            if transform is not None:
                value = transform(value)
            value = value | {'read_projection': {
                'source': 'ProcessedClips/Index.json' if kind == 'index' else 'Comment/TimeIndex.json',
                'dense_evidence_omitted': True, 'canonical_file_unchanged': True}}
            body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
            if len(body) > self.response_limit:
                raise ReadModelDeferred('Published HTTP response exceeds its encoded limit')
            if generation(path) != before:
                raise ReadModelDeferred('Published JSON changed during reading; retry shortly')
            cost = resident_size(value, self.resident_limit) + len(body)
            if cost - len(body) > self.resident_limit:
                raise ReadModelDeferred('Published view exceeds its resident limit')
            model = PublishedView(before, value, body, cost)
            with self.lock:
                if cost <= self.cache_limit:
                    while self.cache and (len(self.cache) >= self.cache_entries
                                          or self.cache_bytes + cost > self.cache_limit):
                        self.cache_bytes -= self.cache.popitem(last=False)[1].cost
                    self.cache[key] = model
                    self.cache_bytes += cost
            return model
        except (KeyError, TypeError, AttributeError) as exc:
            raise ReadModelDeferred('Invalid published JSON references') from exc
        finally:
            with self.lock:
                self.flights.pop(key).set()

    def index(self, root):
        return self._read(safe_child(root, 'ProcessedClips/Index.json'), 'index')

    def time_index(self, root):
        return self._read(safe_child(root, 'Comment/TimeIndex.json'), 'time')

    def timeline_index(self, root):
        """Adapt published file times without hydrating dense CV diagnostics."""
        try:
            data = self.time_index(root).value
        except FileNotFoundError:
            return self.index(root).value
        if data.get('schema_version') != 'visioncortex-file-time-index/2':
            return self.index(root).value
        try:
            return self._timeline_index(data)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ReadModelDeferred('Invalid published timeline references') from exc

    @staticmethod
    def _timeline_index(data):
        if data.get('schema_version') != 'visioncortex-file-time-index/2':
            raise ReadModelDeferred('Timeline lookup requires a version 2 time index')
        streams = data['streams']
        originals = {r['recording_id']: r for r in streams.get('video', []) if r['kind'] == 'original_video'}
        audio = {r['recording_id']: r for r in streams.get('audio', [])}
        records = []
        for row in data['recordings']:
            rid = row['recording_id']
            video, sound = originals.get(rid, {}), audio.get(rid, {})
            transcript = sound.get('transcription') or {}
            records.append({**row, 'camera_key': data['archive'][11:],
                'sources': row.get('sources') or ([{'kind': 'video', 'retained': video['file']}]
                                                 if video.get('file') else []),
                'processing': {'clock_mapping': video.get('clock_mapping') or {}},
                'audio': {'start_us': sound.get('start_us'), 'end_us': sound.get('end_us'),
                          'artifacts': [{'kind': 'audio_audio', **sound['file']}] if sound.get('file') else []},
                'transcription': {'status': transcript.get('status', row.get('transcription_status')),
                    'outcome': transcript.get('outcome', row.get('transcription_outcome')),
                    'transcript_file': transcript.get('text_file') or row.get('transcript_file'),
                    'comments': [s | {'transcript_path': s.get('transcript_file') or transcript.get('text_file')}
                                 for s in transcript.get('sentences', [])]}})
        videos = [r for r in streams.get('video', []) if r['kind'] == 'video_segment']
        parents = {r['segment_id'] for r in videos if r.get('record_scope') != 'retention_interval'}
        segments = [{'segment_id': r['segment_id'], 'recording_id': r['recording_id'],
                     'start_us': r['start_us'], 'end_us': r['end_us'],
                     'start_ms': r.get('source_start_ms'), 'end_ms': r.get('source_end_ms'),
                     'activity': r['activity'], 'source_ref': r.get('source_video'),
                     'video': r.get('file'), 'json_path': r.get('metadata_file'),
                     'key_frames': [], 'scene_frames': []} for r in videos
                    if r.get('record_scope') != 'retention_interval' or r.get('candidate_segment_id') not in parents]
        meanings = [{'segment_id': r['segment_id'], 'status': r.get('status'),
                     'content_file': r['content_file'], 'windows': []}
                    for r in streams.get('multimodal', []) if r.get('content_file')]
        return {'archive': data['archive'], 'recordings': records, 'segments': segments,
                'understandings': meanings, 'read_model_source': 'Comment/TimeIndex.json',
                'evidence_status': 'PARTIAL_EVIDENCE', 'dense_evidence_omitted': True}

    def summary(self, root):
        try:
            return self._read(safe_child(root, 'Comment/TimeIndex.json'), 'summary',
                              transform=lambda data: self._summary(root, data, time_index=True)).value
        except FileNotFoundError:
            return self._summary(root, self.index(root).value, time_index=False)

    def _summary(self, root, data, *, time_index):
        if not time_index:
            candidates = data['segments']
            intervals = [row for parent in candidates for row in parent.get('retention_intervals', [])]
            records = data['recordings']
            transcript_count = sum(len((r.get('transcription') or {}).get('comments') or []) for r in records)
            partitioned = bool(intervals) or not candidates
            covered = {row['segment_id'] for row in candidates if row.get('retention_intervals')}
            updated = data.get('updated_at')
        else:
            video = data['streams'].get('video', [])
            candidates = [row for row in video if row.get('kind') == 'video_segment'
                          and row.get('record_scope') != 'retention_interval']
            intervals = [row for row in video if row.get('kind') == 'video_segment'
                         and row.get('record_scope') == 'retention_interval']
            known = {row.get('segment_id') for row in data['streams'].get('multimodal', [])}
            per_recording = {}
            for row in candidates:
                per_recording.setdefault(row['recording_id'], set()).add(row['segment_id'])
            records = [row | {'preprocessing_ready': bool(per_recording.get(row['recording_id'])),
                'current_results_ready': bool(per_recording.get(row['recording_id']))
                and per_recording[row['recording_id']].issubset(known)} for row in data['recordings']]
            transcript_count = sum(len((row.get('transcription') or {}).get('sentences') or [])
                                   for row in data['streams'].get('audio', []))
            partitioned = bool(intervals) or not candidates
            covered = {row.get('candidate_segment_id') for row in intervals}
            updated = data.get('source_index_updated_at')
        fields = ('recording_id', 'start_us', 'end_us', 'capture_complete', 'stages',
                  'current_results_ready', 'preprocessing_ready')
        return {'archive': root.name, 'segment_count': len(candidates),
            'preprocessing_candidate_count': len(candidates), 'transcript_count': transcript_count,
            'unpartitioned_candidate_count': sum(row['segment_id'] not in covered for row in candidates),
            'updated_at': updated, 'recordings': [{k: row.get(k) for k in fields} for row in records],
            **{label: sum(row.get('video_value') == value for row in intervals) if partitioned else None
               for label, value in (('selected_interval_count', 'selected'),
                                    ('pending_interval_count', 'pending'),
                                    ('confirmed_no_value_interval_count', 'confirmed_no_value'))},
            'read_status': 'available',
            'read_model_source': 'Comment/TimeIndex.json' if time_index else 'ProcessedClips/Index.json'}

    @staticmethod
    def report(root):
        relative = 'LaboratoryDailyReport/LaboratoryDailyReport.json'
        path = safe_child(root, relative)
        return {'report_exists': path.is_file(), 'report_path': relative}
