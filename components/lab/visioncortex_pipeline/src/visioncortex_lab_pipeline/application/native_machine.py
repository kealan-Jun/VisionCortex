"""Machine requests reference the existing device/day queues and publications.

This adapter never starts a model, discovers a NAS tree, retries a failed native
job, or writes into a native archive. Native stage leases remain the sole owners.
"""
import json
from pathlib import Path
import time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..device_day_contract import archive_name, atomic_json, digest, read_json, safe_child
from ..device_day_verification import verify_artifact_cached
from ..device_day_inputs import hash_content
from ..local_storage import require_local_path
from ..schemas import ViewInput
from ..sqlite_store import connection


class NativeDeviceDayConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    runtime_root: Path
    archive_root: Path
    # Existing read-only machine input alias -> actual native camera identity.
    input_roots: dict[str, str] = Field(min_length=1)

    @model_validator(mode='after')
    def absolute_roots(self):
        if not all(p.is_absolute() for p in [self.runtime_root, self.archive_root, *map(Path, self.input_roots)]):
            raise ValueError('Native integration requires absolute storage roots')
        return self


class NativePathManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    experiment_id: str = Field(min_length=1, pattern=r'^[A-Za-z0-9_.-]+$')
    views: list[ViewInput] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_views(self):
        if len({v.view_id for v in self.views}) != len(self.views):
            raise ValueError('view_id must be unique')
        return self


def _database(config, stage):
    root = require_local_path(config.runtime_root)
    return root/'device-day'/f'queue-{stage}.sqlite3'


def _rows(config, stage, *, camera=None, recording_id=None):
    clauses, args = [], []
    if camera is not None:
        clauses.append("json_extract(payload,'$.camera_key')=?")
        args.append(camera)
    if recording_id is not None:
        clauses.append('recording_id=?')
        args.append(recording_id)
    with connection(_database(config, stage), readonly=True) as db:
        return [dict(row) for row in db.execute('SELECT * FROM recordings WHERE '+' AND '.join(clauses), args)]


def bind_inputs(manifest, config):
    """Match exact camera/files to the native discovery owner's current rows."""
    requests = []
    for view in manifest.views:
        for source in ([view] if view.video else view.segments):
            video = Path(source.video)
            roots = [(Path(root), camera) for root, camera in config.input_roots.items()
                     if video.is_relative_to(Path(root))]
            if len(roots) != 1 or source.timestamps_csv is None:
                raise ValueError('Input needs one explicit native camera mapping and timestamp CSV')
            root, camera = roots[0]
            relative = video.relative_to(root).as_posix()
            if Path(source.timestamps_csv).parent != video.parent:
                raise ValueError('Timestamp CSV must belong to the requested camera slice')
            matches = []
            for row in _rows(config, 'vision', camera=camera):
                record = json.loads(row['payload'])
                if record.get('relative_path') == camera+'/'+relative:
                    matches.append((row, record))
            if len(matches) != 1:
                raise LookupError('Native discovery has not registered this exact camera slice')
            row, record = matches[0]
            if record.get('configured_role') != view.role.value:
                raise ValueError('Declared camera role differs from native configuration')
            if (Path(record['frames_path']).name != Path(source.timestamps_csv).name
                    or video.stat().st_size != record['size_bytes']):
                raise ValueError('Machine input differs from the native recording version')
            if source.audio and Path(source.audio).name not in {Path(f['path']).name for f in record.get('audio', {}).get('files', [])}:
                raise ValueError('Audio is not associated with this native camera slice')
            requests.append({'recording_id': row['recording_id'], 'camera': camera, 'view_id': view.view_id,
                'role': view.role.value, 'source_signature': record['source_signature'],
                'source_video': str(video), 'source_clock': str(source.timestamps_csv),
                'video_size_bytes': video.stat().st_size, 'video_mtime_ns': video.stat().st_mtime_ns,
                'clock_size_bytes': Path(source.timestamps_csv).stat().st_size,
                'clock_mtime_ns': Path(source.timestamps_csv).stat().st_mtime_ns,
                'requires_stt': bool(source.audio),
                'start_us': record['recording_start_us'], 'end_us': record['recording_end_us'],
                'archive': archive_name(camera, record['recording_start_us'])})
    return requests


def enqueue(native, manifest, config, run_id, archive):
    requests = bind_inputs(manifest, config)
    native.queue.save_run(run_id, {'state': 'queued', 'progress': 0, 'experiment_id': archive,
                                  'workflow': 'device_day', 'message': '等待原生自动处理队列的结果'})
    native.queue.enqueue(run_id, 'native_device_day', {'native': config.model_dump(mode='json'),
        'requests': requests, 'archive': archive})


def inspect(config, requests):
    """A queue completion alone is insufficient: require readable published units."""
    results, states = [], []
    for request in requests:
        stage_rows = {}
        required = {'vision', 'understanding', 'report'} | ({'stt'} if request.get('requires_stt') else set())
        for stage in ('retention', 'vision', 'stt', 'understanding', 'report'):
            rows = _rows(config, stage, recording_id=request['recording_id'])
            if not rows:
                states.append({'stage': stage, 'state': 'queued', 'recording_id': request['recording_id']})
                continue
            row = rows[0]
            record = json.loads(row['payload'])
            if record.get('source_signature') != request['source_signature']:
                raise ValueError('Native recording input version changed')
            if row['status'] == 'failed' and stage in required:
                # Keep native diagnostics local; no paths/provider text is exposed.
                raise RuntimeError('NATIVE_'+stage.upper()+'_FAILED')
            stage_rows[stage] = row
            states.append({'stage': stage, 'state': row['status'], 'recording_id': request['recording_id'],
                           'revision': row['revision']})
        if not required <= stage_rows.keys() or any(stage_rows[s]['status'] != 'completed' for s in required):
            continue
        root = safe_child(config.archive_root, request['archive'])
        index = read_json(root/'ProcessedClips/Index.json')
        recordings = [r for r in index.get('recordings', []) if r['recording_id'] == request['recording_id']]
        segments = [s for s in index.get('segments', []) if s['recording_id'] == request['recording_id']]
        ids = {s['segment_id'] for s in segments}
        meanings = [m for m in index.get('understandings', []) if m['segment_id'] in ids]
        if not recordings or not segments or ids != {m['segment_id'] for m in meanings}:
            continue  # Native publication is still catching up with its stage queues.
        refs = []
        for segment in segments:
            for field, input_field in [('source_ref', 'source_video'), ('clock_ref', 'source_clock')]:
                reference = segment[field]
                if (not verify_artifact_cached(root, reference)
                        or hash_content(Path(request[input_field]), expected=reference['sha256']) != reference['sha256']):
                    raise ValueError('Native output refers to a different submitted source version')
            refs.extend([*segment.get('key_frames', []), *segment.get('scene_frames', [])])
            if segment.get('video'):
                refs.append(segment['video'])
            for key in ('json_path',):
                if segment.get(key) and not safe_child(root, segment[key]).is_file():
                    raise ValueError('Native segment metadata is missing')
        for meaning in meanings:
            if meaning.get('status') == 'partial':
                raise ValueError('Native understanding is still partial')
            for window in meaning.get('windows', []):
                refs.extend(window.get('source_frames', []))
                for field in ('input', 'model_receipt'):
                    if window.get(field) and not safe_child(root, window[field]).is_file():
                        raise ValueError('Native model provenance is missing')
        if not refs or not all(verify_artifact_cached(root, ref) for ref in refs):
            raise ValueError('Native published evidence is missing or changed')
        report = root/'LaboratoryDailyReport/LaboratoryDailyReport.json'
        if not report.is_file() or not report.with_suffix('.html').is_file():
            continue
        # Match the original read-only input and current source version. Native
        # processing/receipt owners supply the hashes and model execution keys.
        for prefix in ('video', 'clock'):
            path = Path(request['source_'+('video' if prefix == 'video' else 'clock')])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (request[prefix+'_size_bytes'], request[prefix+'_mtime_ns']):
                raise ValueError('Submitted source changed while waiting for native results')
        results.append({'request': request, 'recordings': recordings, 'segments': segments,
                        'understandings': meanings,
                        'native_revisions': {s: r['revision'] for s, r in stage_rows.items()},
                        'report': 'LaboratoryDailyReport/LaboratoryDailyReport.html',
                        'report_scope': 'native_device_day; includes other same-day slices',
                        'index_path': 'ProcessedClips/Index.json'})
    return {'ready': len(results) == len(requests), 'states': states, 'device_days': results,
            'workflow': 'device_day', 'evidence_classification': 'PARTIAL_EVIDENCE',
            'physical_action_confirmed': False, 'additional_model_invocations': 0,
            'time_basis': 'native_capture_clock; physical cross-camera alignment is separately evidenced'}


def execute(job, settings, stop):
    from . import runtime_host as host
    config = NativeDeviceDayConfig.model_validate(job.payload['native'])
    while not stop.is_set():
        state = inspect(config, job.payload['requests'])
        if state['ready']:
            path = Path(settings['storage']['local_runtime_root'])/'state/NativeDeviceDayLinks'/f'{job.run_id}.json'
            atomic_json(path, {'schema_version': 'visioncortex-native-machine-link/1', 'run_id': job.run_id,
                'native': job.payload['native'], 'requests': job.payload['requests'], 'result_digest': digest(state),
                'created_at': time.time()})
            host._update(job.run_id, state='completed', progress=1, workflow='device_day',
                         evidence_classification='PARTIAL_EVIDENCE', native_result_receipt=str(path),
                         archive_url='/api/archives/'+job.payload['archive'],
                         message='原生自动处理结果已发布，来源与证据可读取')
            return
        running = any(s['state'] == 'running' for s in state['states'])
        host._update(job.run_id, state='running' if running else 'queued',
                     progress=sum(s['state'] == 'completed' for s in state['states'])/max(1, len(state['states'])),
                     workflow='device_day', native_stages=state['states'],
                     message='由原生自动处理队列执行；复用已有转写及分析结果')
        stop.wait(2)
    from ..runtime_control import ExecutionCancelled
    raise ExecutionCancelled('Native result wait interrupted by worker shutdown')


def result(native, config, run_id):
    job = native.queue.get_job(run_id)
    if not job or job['kind'] != 'native_device_day':
        return None
    payload = job['payload']
    if payload['native'] != config.model_dump(mode='json'):
        raise ValueError('Native result binding configuration changed')
    data = inspect(config, payload['requests'])
    if not data['ready']:
        raise ValueError('Native results are not fully published')
    run = native.queue.load_run(run_id) or {}
    receipt = read_json(Path(run.get('native_result_receipt') or ''))
    if receipt.get('run_id') != run_id or receipt.get('result_digest') != digest(data):
        raise ValueError('The referenced native result version was replaced')
    return data


def evidence(data, uid):
    for day in data['device_days']:
        request = day['request']
        frames = [f for s in day['segments'] for f in [*s.get('key_frames', []), *s.get('scene_frames', [])]]
        frames.extend(f for m in day['understandings'] for w in m.get('windows', []) for f in w.get('source_frames', []))
        for frame in frames:
            if frame['frame_id'] == uid:
                return {'workflow': 'device_day', 'evidence_uid': uid, 'view_id': request['view_id'],
                    'native_archive': request['archive'], 'frame_kind': frame['frame_kind'],
                    'global_timestamp_us': frame.get('capture_us'), 'artifact': frame,
                    'source': {'source_segment_path': request['source_video'],
                        'source_timestamps_csv': request['source_clock'],
                        'virtual_local_ms': frame['local_ms'], 'source_segment_virtual_start_ms': 0,
                        'source_frame_index': frame.get('source_frame_index'),
                        'source_time_basis': frame.get('source_time_basis')},
                    'physical_action_confirmed': False, 'evidence_classification': 'PARTIAL_EVIDENCE'}
    return None


def file_path(data, config, recording_id, relative):
    for day in data['device_days']:
        if day['request']['recording_id'] != recording_id:
            continue
        refs = []
        for segment in day['segments']:
            refs.extend([*segment.get('key_frames', []), *segment.get('scene_frames', []),
                         segment['source_ref'], segment['clock_ref']])
            if segment.get('video'):
                refs.append(segment['video'])
        refs.extend(f for m in day['understandings'] for w in m.get('windows', []) for f in w.get('source_frames', []))
        ref = next((r for r in refs if r['path'] == relative), None)
        root = safe_child(config.archive_root, day['request']['archive'])
        if ref and verify_artifact_cached(root, ref):
            return safe_child(root, relative)
    raise ValueError('File is outside the submitted recording evidence')
