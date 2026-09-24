"""Independent capture preprocessing and archive publication, with legacy routing.

Only local receipt inspection occurs at scheduling time. Model outputs retain
canonical MetaVideo references from their first execution; physical read paths
are never part of a new stage's content identity.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
import time

from .device_day_contract import VERSION, digest, read_json, safe_child, validate_segment
from .device_day_activity import phase
from .device_day_inputs import (
    InputChanged, binding, copy_sealed, durable_json, identity, input_path, input_context,
    processing_input, seal_input, validate_after, verify_content,
)
from .device_day_io import slot


PRIMARY = ('vision', 'stt', 'retention')


def marker(runner, record):
    from .device_day_contract import archive_name
    name = archive_name(record['camera_key'], record['recording_start_us'])
    return safe_child(runner.backend_root, f'device-day-receipts/{name}/{record["recording_id"]}/input-state.json')


def active(runner, record, *, initialize=False):
    path = marker(runner, record)
    if path.is_file():
        return True
    if not runner.settings.get('inplace_preprocessing', False):
        return False
    # Preserve any historical execution, including incomplete model checkpoints.
    # A queued record with no execution is migrated in place without clearing it.
    if has_legacy_execution(path):
        return False
    if not initialize:
        return True
    from .device_day import exclusive
    try:
        with ExitStack() as locks:
            for stage in ('all', *PRIMARY):
                locks.enter_context(exclusive(runner.runtime_root / 'locks' / f'{record["recording_id"]}.{stage}.lock'))
            if path.is_file():
                return True
            if has_legacy_execution(path):
                return False
            old = path.with_name('retention.json')
            if old.is_file():
                durable_json(path.parent / 'history' / f'legacy-retention-{digest(read_json(old))}.json', read_json(old))
            durable_json(path, {'schema_version': 'visioncortex-inplace-input/1', 'recording_id': record['recording_id'],
                                'recording': record, 'created_at': first_queued_at(runner, record),
                                'retention_policy': retention_policy(runner, record)})
            return True
    except BlockingIOError:
        # Another new stage may be committing the short local mode marker.
        # Reuse that decision instead of deferring its independent input.
        for _ in range(10):
            if path.is_file():
                return True
            time.sleep(.01)
        return False


def retention_policy(runner, record):
    policy = runner.settings.get('capture_retention') or {}
    policy = policy | policy.get('cameras', {}).get(record['camera_key'], {})
    media = [('video', record, record['recording_end_us'])]
    if record.get('audio', {}).get('status') == 'provided':
        media.append(('audio', record['audio'], record['audio'].get('end_us')))
    sources = []
    for kind, metadata, ended in media:
        rule = policy | policy.get('by_kind', {}).get(kind, {})
        deadline = metadata.get('source_expires_at')
        if deadline is None and rule.get('seconds') is not None and ended is not None:
            deadline = ended / 1e6 + rule['seconds']
        sources.append({'kind': kind, 'deadline': deadline,
                        'status': 'known' if deadline is not None else 'unknown',
                        'cleanup_risk': bool(metadata.get('source_cleanup_risk', rule.get('cleanup_risk', False))),
                        'safety_margin_seconds': rule.get('safety_margin_seconds', 3600)})
    deadlines = [s['deadline'] for s in sources if s['deadline'] is not None]
    return {'deadline': min(deadlines) if deadlines else None,
            'status': 'known' if all(s['status'] == 'known' for s in sources) else 'unknown',
            'cleanup_risk': any(s['cleanup_risk'] for s in sources), 'sources': sources,
            'safety_margin_seconds': max(s['safety_margin_seconds'] for s in sources),
            'basis': 'explicit_operator_policy' if deadlines or any(s['cleanup_risk'] for s in sources)
                     else 'external_collector_policy_not_supplied'}


def archive_priority(runner, record, created):
    policy = retention_policy(runner, record)
    urgent = policy['cleanup_risk'] or (policy['deadline'] is not None and
                                       time.time() >= policy['deadline'] - policy['safety_margin_seconds'])
    aged = time.time() - created >= runner.settings.get('archive_max_wait_seconds', 900)
    return urgent, aged


def receipt(runner, record, stage):
    path = runner._receipt(runner.layout(record), record, stage)
    return read_json(path) if path.is_file() else {}


def enqueue(runner, record, stage):
    """Return admission without reading media or changing historical completions."""
    if not record.get('processable', record.get('available')) or record.get('configured_role') not in {'first_person', 'third_person'}:
        return False
    if not active(runner, record, initialize=True):
        return False
    if stage not in PRIMARY:
        # Legacy downstream executor is used only after both inputs are archived
        # and their independent model receipts are formally published.
        if receipt(runner, record, 'publication').get('status') != 'completed':
            return False
        return None
    state = read_json(marker(runner, record))
    policy = retention_policy(runner, record)
    if state.get('retention_policy') != policy:
        update_state(runner, record, retention_policy=policy)
    urgent, aged = archive_priority(runner, record, state['created_at'])
    visual = receipt(runner, record, 'vision')
    admitted = not (stage == 'retention' and not (urgent or aged or visual.get('status') in {'completed', 'failed'}))
    if stage == 'stt' and record.get('audio', {}).get('status') in {'pending_publication', 'association_mismatch'}:
        admitted = False
    record = record | {'archive_date': runner.layout(record).name[:10],
                       'archive_deadline': policy['deadline'], 'source_retention_status': policy['status'],
                       'archive_urgent': urgent, 'archive_aged': aged}
    # Scheduling only: model/configuration provenance remains in the executable
    # stage key. Preserve an old queue row until its original model receipt is
    # validated by process(); never bulk erase queues or completed caches.
    inputs = ([record['source_signature'], record.get('audio', {}).get('source_signature')] if stage == 'retention' else
              record.get('audio', {}).get('source_signature') if stage == 'stt' else record['source_signature'])
    revision = digest(['inplace-queue/1', stage, record['recording_id'], inputs, runner._key(stage, record, {})])
    runner.queues[stage].enqueue(record, revision)
    archived = receipt(runner, record, 'retention')
    if (archived.get('status') == 'completed' and archived.get('input_binding_version') == 1
            and archived.get('recording', {}).get('source_signature') == record.get('source_signature')):
        # Local receipt only admits a verification attempt. process() still
        # verifies archived bytes before models or publication may consume them.
        with runner.queues[stage].connect() as db:
            db.execute("UPDATE recordings SET input_status='ready' WHERE recording_id=? AND revision=? "
                       "AND status!='running'", (record['recording_id'], revision))
    runner.queues[stage].resume_prerequisite(record['recording_id'], revision)
    return admitted


def verified_archive(runner, record):
    saved = receipt(runner, record, 'retention')
    if saved.get('status') != 'completed':
        return None
    if saved.get('recording', {}).get('source_signature') != record.get('source_signature'):
        raise InputChanged('Discovery source version differs from archived input')
    if saved.get('input_binding_version') != 1:
        raise InputChanged('Unrecognized archive binding')
    layout = runner.layout(record)
    with slot(runner.config, copy=True), phase('archive_hash_seconds'):
        if not all(verify_content(safe_child(layout.root, s['retained']['path']), s['retained']) for s in saved['sources']):
            raise InputChanged('Archived originals failed verification')
    return saved


def _archive(runner, record, layout):
    state = read_json(marker(runner, record))
    urgent, _ = archive_priority(runner, record, state['created_at'])
    seals, sources = [], {}
    # Copy each independently sealed media set before touching the next one.
    # A disappearing audio sidecar must not prevent saving an expiring video.
    for stage in ('vision', 'stt'):
        if stage == 'stt' and record.get('audio', {}).get('status') in {'pending_publication', 'association_mismatch'}:
            continue
        with slot(runner.config, copy=True, urgent=urgent), phase('input_validation_seconds'):
            seal = seal_input(runner, record, stage)
            seals.append(seal)
            update_state(runner, record, input_ready={stage: 'ready'})
        with slot(runner.config, copy=True, urgent=urgent, whole_copy=True):
            for source in seal['sources']:
                key = source['retained']['path']
                if key in sources and sources[key]['sha256'] != source['sha256']:
                    raise InputChanged('Video/audio seals refer to different source versions')
                if key not in sources:
                    copy_sealed(runner.config, source, safe_child(layout.root, key), urgent=urgent)
                sources[key] = source
    audio = dict(seals[-1]['audio']) if len(seals) > 1 else dict(record.get('audio', {}))
    audio.pop('files', None)
    audio['artifacts'] = [s['retained'] | {'kind': s['kind']} for s in sources.values() if s['kind'].startswith('audio_')]
    return {'input_binding_version': 1, 'recording': record, 'capture_root': runner.config['collection_ingest']['source_root'],
            'sources': list(sources.values()), 'artifacts': [s['retained'] for s in sources.values()], 'audio': audio,
            'input_bindings': {s['stage'].removeprefix('input-'): binding(s) for s in seals},
            'capture_deletion': 'disabled_by_user'}


def execute(runner, layout, record, *, stage, retry):
    """Called under the existing stage OS lock and queue lease."""
    from .device_day import now, visual_input
    from .stage_execution import StageExecutor
    from .publication_journal import PublicationJournal
    if stage == 'all':
        # Explicit sequential CLI execution prefers preprocessing; service
        # workers remain independent and retention can preempt for risk/age.
        order = ('retention', 'vision', 'stt') if archive_priority(runner, record, read_json(marker(runner, record))['created_at'])[0] else PRIMARY
        results = [runner.process(record, stage=s, retry=retry) for s in order]
        failed = next((r for r in results if r['status'] != 'completed'), None)
        if failed:
            return failed
        for downstream in ('understanding', 'report'):
            result = runner.process(record, stage=downstream, retry=retry)
            if result.get('status') != 'completed':
                return result
        return result
    if stage not in PRIMARY:
        if receipt(runner, record, 'publication').get('status') != 'completed':
            return {'status': 'waiting_for_publication', 'recording_id': record['recording_id']}
        return runner._process(layout, record, stage=stage, retry=retry)
    journal = PublicationJournal(runner.runtime_root)
    journal.begin(record)
    path = runner._receipt(layout, record, stage)
    started = time.perf_counter()
    key = None
    try:
        archived = verified_archive(runner, record)
        if stage == 'retention':
            audio_signature = record.get('audio', {}).get('source_signature')
            if archived and archived['recording'].get('audio', {}).get('source_signature') == audio_signature:
                result = archived
            else:
                result = _archive(runner, record, layout)
            key = runner._key('retention', record, record)
        else:
            with slot(runner.config), phase('input_validation_seconds'):
                seal = seal_input(runner, record, stage)
                update_state(runner, record, input_ready={stage: 'ready'})
                if archived and archived.get('input_bindings', {}).get(stage) != binding(seal):
                    archived = None  # Late independent audio has not been copied yet.
                value = processing_input(layout, seal, archived)
                before = {ref: (identity(p), str(p.resolve())) for ref, p in layout.capture_sources.items()}
            inputs = visual_input(value) if stage == 'vision' else value
            key = runner._key(stage, record, inputs)
            previous = runner._load(path, key, layout)
            if previous and previous.get('input_binding') == binding(seal):
                result = previous
            else:
                old = read_json(path) if path.is_file() else {}
                if old.get('status') == 'failed' and old.get('key') == key and not retry:
                    return old
                if old:
                    durable_json(path.parent / 'history' / f'{stage}-{digest(old)}.json', old)
                durable_json(path, {'status': 'running', 'stage': stage, 'key': key,
                                    'recording_id': record['recording_id'], 'input_binding': binding(seal)})
                with slot(runner.config), phase('decode_inference_seconds'), input_context(seal):
                    validate_after(layout.capture_sources, before)
                    function = runner._backend().vision if stage == 'vision' else runner._backend().transcribe
                    result = StageExecutor(runner.config).run(stage, function, layout, value, key)
                    validate_after(layout.capture_sources, before)
                if stage == 'vision':
                    if not result.get('segments'):
                        raise ValueError('A completed recording requires activity or inactivity records')
                    for segment in result['segments']:
                        validate_segment(segment)
                    result |= {'vision_input_digest': digest(visual_input(value))}
                result |= {'input_binding': binding(seal), 'input_binding_version': 1}
        if result.get('status') != 'completed' or result.get('key') != key:
            import os
            from .device_day_inputs import sync_directory
            references = [(layout.root, r) for r in result.get('artifacts', [])]
            for reference in result.get('audit_artifacts', []):
                if reference.get('storage_root') != 'local_cache_root':
                    raise ValueError('Unknown audit artifact storage root')
                references.append((runner.backend_root, reference))
            for root, reference in references:
                output = safe_child(root, reference['path'])
                with output.open('rb') as handle:
                    os.fsync(handle.fileno())
                sync_directory(output.parent)
            result |= {'schema_version': VERSION, 'status': 'completed', 'stage': stage, 'key': key,
                       'recording_id': record['recording_id'], 'completed_at': now(),
                       'wall_seconds': time.perf_counter() - started}
            durable_json(path, result)
        update_state(runner, record, **({'archive': 'completed'} if stage == 'retention' else
                                       {'preprocessing': {stage: 'completed'}}))
        # Queue completion describes this stage only. Publication failures retain
        # the journal and never revoke a successful model or archive receipt.
        try:
            published = publish(runner, record)
        except (OSError, ValueError):
            published = False
        return {'schema_version': VERSION, 'recording_id': record['recording_id'], 'archive': layout.name,
                'stage': stage, 'status': 'completed', 'formal_publication': 'completed' if published else 'pending',
                'message': '预处理完成，等待归档' if stage == 'vision' and not archived else '阶段结果已持久保存',
                'capture_deletion': 'disabled_by_user'}
    except Exception as exc:
        failure = {'schema_version': VERSION, 'input_binding_version': 1, 'stage': stage, 'key': key, 'status': 'failed',
                   'recording_id': record['recording_id'], 'error_type': type(exc).__name__, 'message': str(exc)[:1000]}
        # Preserve the last completed checkpoint even when a source check fails.
        if path.is_file() and read_json(path).get('status') == 'completed':
            durable_json(path.parent / 'history' / f'{stage}-{digest(read_json(path))}.json', read_json(path))
        durable_json(path.with_name(stage + '-attempt.json'), failure)
        if not path.is_file() or read_json(path).get('status') != 'completed':
            durable_json(path, failure)
        durable_json(path.with_name('publication.json'), failure | {'stage': 'publication', 'status': 'blocked'})
        update_state(runner, record, publication='blocked', last_failure=failure)
        from .runtime_control import ExecutionCancelled
        if isinstance(exc, ExecutionCancelled):
            return failure | {'status': 'cancelled'}
        if stage == 'stt':
            from .device_day_provider_gate import ProviderGate
            from .provider_control import ProviderUnavailable, classify
            ProviderGate(runner.config).record_failure({'error': str(exc)})
            if isinstance(exc, ProviderUnavailable) or classify(exc):
                return failure | {'status': 'waiting_for_provider'}
        return failure


def _publish(runner, record):
    """No inference: bind completed outputs, then atomically publish the index."""
    from .device_day import exclusive, visual_input
    layout = runner.layout(record)
    with exclusive(runner.runtime_root / 'locks' / f'{record["recording_id"]}.publication.lock'):
        archived = verified_archive(runner, record)
        if not archived:
            return False
        visual = receipt(runner, record, 'vision')
        if visual.get('status') != 'completed':
            return False
        if (visual.get('input_binding') != archived['input_bindings'].get('vision') or
                visual.get('vision_input_digest') != digest(visual_input(archived))):
            raise InputChanged('Preprocessing and archive bind different input versions')
        if not runner._load(runner._receipt(layout, record, 'vision'), runner._key('vision', record, visual_input(archived)), layout):
            raise InputChanged('Preprocessing artifacts or execution identity failed verification')
        for stage in ('vision', 'stt'):
            seal_path = input_path(runner, record, stage)
            if not seal_path.is_file():
                continue
            seal = read_json(seal_path)
            # Existing captures must still be the sealed versions. A missing
            # capture is recoverable only after independently verified archival.
            for source in seal['sources']:
                p = Path(source['original_path'])
                if p.exists() and (identity(p) != source['identity'] or
                                   str(p.resolve()) != source.get('resolved_path', str(p.resolve()))):
                    raise InputChanged('Capture was replaced after preprocessing')
        if visual.get('retention_digest') != digest(archived):
            visual = visual | {'retention_digest': digest(archived)}
            durable_json(runner._receipt(layout, record, 'vision'), visual)
        stt = receipt(runner, record, 'stt')
        stt_ready = (stt.get('status') == 'completed' and stt.get('input_binding') == archived['input_bindings'].get('stt'))
        if stt_ready:
            if not runner._load(runner._receipt(layout, record, 'stt'), runner._key('stt', record, archived), layout):
                raise InputChanged('Transcription artifacts or execution identity failed verification')
        publication = {'status': 'publishing', 'recording_id': record['recording_id'],
                       'retention_digest': digest(archived), 'vision_digest': digest(visual),
                       'stt_digest': digest(stt) if stt_ready else None, 'at': time.time()}
        path = runner._receipt(layout, record, 'publication')
        durable_json(path, publication)
        with phase('formal_publication_seconds'):
            if not runner.refresh_index(layout, recording_id=record['recording_id']):
                return False
        from .device_day_inputs import sync_directory
        sync_directory(layout.index.parent)
        sync_directory(layout.comments)
        durable_json(path, publication | {'status': 'completed', 'completed_at': time.time()})
        update_state(runner, record, publication='completed')
        from .capture_link_cleanup import submit_after_preprocessing
        submit_after_preprocessing(runner, layout, record)
        return True


def publication_filter(runner, layout, stages):
    """The index may reference only a verified, bound publication generation."""
    retained = stages.get('retention') or {}
    if retained.get('input_binding_version') != 1:
        return stages
    record = retained['recording']
    state = receipt(runner, record, 'publication')
    result = deepcopy(stages)
    if (state.get('status') not in {'publishing', 'completed'} or state.get('retention_digest') != digest(retained)
            or state.get('vision_digest') != digest(stages.get('vision'))):
        result.pop('vision', None)
        result.pop('stt', None)
    elif state.get('stt_digest') != digest(stages.get('stt')):
        result.pop('stt', None)
    return result


def update_state(runner, record, **changes):
    """Small local lifecycle projection, independent of large model receipts."""
    path = marker(runner, record)
    with state_lock(runner.runtime_root / 'locks' / f'{record["recording_id"]}.state.lock'):
        state = read_json(path)
        for key, value in changes.items():
            state[key] = (state.get(key, {}) | value) if isinstance(value, dict) else value
        state['updated_at'] = time.time()
        durable_json(path, state)


def record_timings(runner, record, stage, measured):
    if not marker(runner, record).is_file():
        return
    timing = dict(measured)
    if stage in runner.queues:
        with runner.queues[stage].connect() as db:
            row = db.execute('SELECT queued_at,started_at FROM recordings WHERE recording_id=?', (record['recording_id'],)).fetchone()
            if row and row['started_at'] is not None:
                timing['queue_wait_seconds'] = max(0, row['started_at'] - row['queued_at'])
    update_state(runner, record, timings={stage: timing})


def progress(config):
    """Local metadata only; no archive/capture stats or model receipts."""
    if not config.get('storage', {}).get('local_cache_root'):
        return []
    root = Path(config['storage']['local_cache_root']) / 'device-day-receipts'
    rows = []
    for path in root.glob('*/*/input-state.json'):
        state = read_json(path)
        record = state['recording']
        visual = state.get('preprocessing', {}).get('vision', 'pending')
        archive = state.get('archive', 'pending')
        publication = state.get('publication', 'pending')
        label = ('发布受阻，保留可恢复结果' if publication == 'blocked' else
                 '预处理完成，等待归档' if visual == 'completed' and archive != 'completed' else
                 '正式索引已发布' if publication == 'completed' else
                 '归档已完成，等待预处理或发布' if archive == 'completed' else '等待输入或预处理')
        rows.append({'recording_id': record['recording_id'], 'archive': path.parent.parent.name,
                     'input_ready': state.get('input_ready', {}), 'preprocessing': state.get('preprocessing', {}),
                     'archive_status': archive, 'publication_status': publication, 'label': label,
                     'retention_policy': state['retention_policy'], 'timings': state.get('timings', {})})
    return rows


@contextmanager
def state_lock(path):
    from .device_day import exclusive
    deadline = time.monotonic() + 10
    while True:
        lock = exclusive(path)
        try:
            lock.__enter__()
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)
    try:
        yield
    finally:
        import sys
        lock.__exit__(*sys.exc_info())


def validate_settings(settings):
    import math
    if not isinstance(settings.get('inplace_preprocessing', False), bool):
        raise ValueError('inplace_preprocessing must be boolean')
    for name, default in (('nas_io_slots', 3), ('archive_copy_workers', 1)):
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name, default in (('archive_max_wait_seconds', 900), ('nas_io_timeout_seconds', 3600),
                          ('archive_copy_bytes_per_second', 8 * 1024 * 1024)):
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    policy = settings.get('capture_retention') or {}
    policies = [policy, *policy.get('cameras', {}).values()]
    policies += [rule for item in policies for rule in item.get('by_kind', {}).values()]
    for item in policies:
        if not isinstance(item.get('cleanup_risk', False), bool):
            raise ValueError('capture_retention.cleanup_risk must be boolean')
        for name in ('seconds', 'safety_margin_seconds'):
            value = item.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0 or name == 'seconds' and value == 0):
                raise ValueError(f'capture_retention.{name} must be a finite valid interval or null')


def has_legacy_execution(path):
    if any(path.with_name(stage + '.json').exists() for stage in ('vision', 'stt')):
        return True
    retained = path.with_name('retention.json')
    return retained.is_file() and read_json(retained).get('status') == 'completed'


def publish(runner, record):
    try:
        return _publish(runner, record)
    except BlockingIOError:
        return False
    except (OSError, ValueError) as exc:
        failure = {'stage': 'publication', 'status': 'blocked', 'recording_id': record['recording_id'],
                   'error_type': type(exc).__name__, 'message': str(exc)[:1000], 'at': time.time()}
        durable_json(runner._receipt(runner.layout(record), record, 'publication'), failure)
        update_state(runner, record, publication='blocked', last_failure=failure)
        raise


def execution_identity():
    """Cover the new execution boundary without hashing queue/UI scheduling."""
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    names = {'execute', '_archive', 'verified_archive', '_publish', 'publish', 'publication_filter'}
    return digest([ast.dump(node, include_attributes=False) for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in names])


def first_queued_at(runner, record):
    observed = time.time()
    for stage in PRIMARY:
        with runner.queues[stage].connect() as db:
            row = db.execute('SELECT queued_at FROM recordings WHERE recording_id=?', (record['recording_id'],)).fetchone()
            if row:
                observed = min(observed, row['queued_at'])
    return observed
