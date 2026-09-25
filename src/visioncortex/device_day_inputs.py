"""Sealed capture inputs and content-bound relocation to the frozen archive."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from threading import Lock
import os
from pathlib import Path
import time
import uuid

from .device_day_contract import VERSION, atomic_json, digest, read_json, safe_child
from .device_day_activity import phase
from .device_day_io import pace_copy, slot


class InputChanged(ValueError):
    pass


_CONTENT = OrderedDict()
_CONTENT_LOCK = Lock()
_HASH_LOCKS = [Lock() for _ in range(32)]


def remember_content(path, checksum):
    # Only call after this process hashed stable bytes itself. Never deserialize
    # a persisted stat tuple as a trusted content verification.
    key = (str(path), tuple(identity(path)), checksum)
    with _CONTENT_LOCK:
        _CONTENT[key] = True
        _CONTENT.move_to_end(key)
        while len(_CONTENT) > 2048:
            _CONTENT.popitem(last=False)


def hash_content(path):
    # Single-flight hashing shares video/embedded-audio validation, including
    # when both independent stages seal the same source concurrently.
    with _HASH_LOCKS[hash(str(path)) % len(_HASH_LOCKS)]:
        before = identity(path)
        prefix = (str(path), tuple(before))
        with _CONTENT_LOCK:
            known = next((key[2] for key in reversed(_CONTENT) if key[:2] == prefix), None)
        if known is not None:
            return known
        checksum = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                checksum.update(block)
        if before != identity(path):
            raise InputChanged('Input changed during content verification')
        result = checksum.hexdigest()
        remember_content(path, result)
        return result


def verify_content(path, reference):
    return identity(path)[2] == reference['size_bytes'] and hash_content(path) == reference['sha256']


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise InputChanged('Sealed input disappeared or is no longer a regular file')
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def durable_json(path, value):
    atomic_json(path, value)
    sync_directory(path.parent)


def sync_directory(path):
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def source_path(layout, source):
    """Only runner-created layouts may resolve a sealed capture input."""
    return getattr(layout, 'capture_sources', {}).get(source['retained']['path']) or safe_child(layout.root, source['retained']['path'])


def source_relative(layout, path):
    for reference, physical in getattr(layout, 'capture_sources', {}).items():
        if Path(path) == physical:
            return reference
    return layout.relative(path)


def validate_snapshot(seal, *, paths=None, hashes=False):
    for source in seal['sources']:
        path = (paths or {}).get(source['retained']['path'], Path(source['original_path']))
        expected = source['identity']
        if identity(path) != expected or str(path.resolve()) != source.get('resolved_path', str(path.resolve())):
            raise InputChanged(f'Sealed input changed: {source["kind"]}')
        if hashes and not verify_content(path, source):
            raise InputChanged(f'Sealed input content changed: {source["kind"]}')


def input_path(runner, record, stage):
    return runner._receipt(runner.layout(record), record, 'input-' + stage)


def seal_input(runner, record, stage):
    """Video and audio seals are independent; supplied hashes never confer trust."""
    from .device_day import exclusive
    from .nas_recordings import _inspect
    layout = runner.layout(record)
    path = input_path(runner, record, stage)
    with exclusive(runner.runtime_root / 'locks' / f'{record["recording_id"]}.input-{stage}.lock'):
        if path.is_file():
            saved = read_json(path)
            expected = (record.get('audio', {}).get('source_signature') if stage == 'stt' else record['source_signature'])
            if saved['source_signature'] == expected:
                if any(saved['recording'].get(k) != record.get(k) for k in (
                        'recording_id', 'camera_key', 'configured_role', 'recording_start_us', 'recording_end_us')):
                    raise InputChanged('Sealed device/date/role identity changed')
                return saved
            # Video version replacement is never silently accepted. Audio may
            # arrive later; preserve its previous independent no-audio receipt.
            if stage != 'stt' or saved.get('audio', {}).get('status') == 'provided':
                raise InputChanged('Discovery now describes a different sealed input')
            durable_json(path.parent / 'history' / f'input-{stage}-{digest(saved)}.json', saved)
        root = Path(runner.config['collection_ingest']['source_root'])
        original = Path(record['video_path'])
        if original.is_symlink() or not original.resolve().is_relative_to(root.resolve()):
            raise InputChanged('Capture input escapes configured source root')
        fresh = _inspect(root, original, time.time(), runner.config['collection_ingest'].get('settle_seconds', 120),
                         bool(runner.config['collection_ingest'].get('discover_plain_video_csv', False)))
        if (not fresh.get('processable', fresh.get('available')) or fresh['source_signature'] != record['source_signature']
                or fresh['camera_key'] != record['camera_key']
                or fresh['recording_start_us'] != record['recording_start_us']
                or fresh['recording_end_us'] != record['recording_end_us']):
            raise InputChanged('Capture completion, identity or source signature changed')
        audio = record.get('audio', {})
        if stage == 'stt':
            if fresh.get('audio', {}).get('source_signature') != audio.get('source_signature'):
                raise InputChanged('Audio changed before sealing')
            if audio.get('status') in {'pending_publication', 'association_mismatch'}:
                raise InputChanged('Audio is not independently ready')
        files = [('video', original), ('clock', Path(record['frames_path']))] if stage == 'vision' else []
        if stage == 'vision':
            prefix = original.name[:-len('rgb.mp4')] if original.name.endswith('rgb.mp4') else ''
            files.extend(('sidecar', p) for suffix in ('meta.json', 'recording_ready.json', 'calibration.json')
                         if (p := original.with_name(prefix + suffix)).is_file())
        else:
            files = [('audio_' + s['kind'], Path(s['path'])) for s in audio.get('files', [])]
            if not any(kind == 'audio_audio' for kind, _ in files):
                files.append(('video', original))  # Embedded audio remains supported.
        sources = []
        for kind, source in files:
            if source.is_symlink() or not source.resolve().is_relative_to(original.parent.resolve()):
                raise InputChanged('Capture sidecar escapes recording directory')
            before = identity(source)
            other_path = input_path(runner, record, 'stt' if stage == 'vision' else 'vision')
            other = read_json(other_path) if other_path.is_file() else {}
            known = next((s for s in other.get('sources', []) if s['original_path'] == str(source)
                          and s['identity'] == before), None)
            with phase('input_hash_seconds'):
                if known:
                    if not verify_content(source, known):
                        raise InputChanged('Shared source content no longer matches its seal')
                    checksum = known['sha256']
                else:
                    checksum = hash_content(source)
            if before != identity(source):
                raise InputChanged('Capture changed while sealing content')
            remember_content(source, checksum)
            reference = {'path': layout.relative(layout.source_path(record, kind, source)),
                         'size_bytes': before[2], 'sha256': checksum}
            sources.append({'kind': kind, 'original_path': str(source), 'resolved_path': str(source.resolve()), 'identity': before,
                            'size_bytes': before[2], 'mtime_ns': before[3], 'sha256': checksum, 'retained': reference})
        seal = {'schema_version': VERSION, 'status': 'ready', 'stage': 'input-' + stage,
                'recording': record, 'recording_id': record['recording_id'], 'sources': sources,
                'source_signature': audio.get('source_signature') if stage == 'stt' else record['source_signature'],
                'audio': {k: v for k, v in audio.items() if k != 'files'}, 'sealed_at': time.time()}
        validate_snapshot(seal)
        durable_json(path, seal)
        return seal


def binding(seal):
    record = seal['recording']
    return digest({'recording': {k: record.get(k) for k in ('recording_id', 'camera_key', 'configured_role',
                   'recording_start_us', 'recording_end_us', 'recording_session_id')},
                   'sources': [{'kind': s['kind'], 'sha256': s['sha256'], 'size_bytes': s['size_bytes']}
                               for s in seal['sources']]})


def processing_input(layout, seal, archived=None):
    sources = seal['sources']
    paths = {}
    if archived:
        archived_refs = {digest(s['retained']) for s in archived['sources']}
        if not all(digest(s['retained']) in archived_refs for s in sources):
            raise InputChanged('Archive does not bind this input version')
        for source in sources:
            if not verify_content(safe_child(layout.root, source['retained']['path']), source['retained']):
                raise InputChanged('Archived input failed content verification')
            paths[source['retained']['path']] = safe_child(layout.root, source['retained']['path'])
    else:
        validate_snapshot(seal, hashes=True)
        paths = {s['retained']['path']: Path(s['original_path']) for s in sources}
    layout.capture_sources = paths
    audio = dict(seal['audio'])
    audio['artifacts'] = [s['retained'] | {'kind': s['kind']} for s in sources if s['kind'].startswith('audio_')]
    return {'input_binding_version': 1, 'recording': seal['recording'], 'sources': sources, 'audio': audio,
            'artifacts': [], 'input_binding': binding(seal)}


def _verify_owned_copy(path, reference):
    """Re-read our exclusive temporary copy after an SMB metadata refresh.

    Never relax source/previous-artifact verification. Every attempt reads all
    bytes and requires the sealed size/hash and the same device/inode; only
    timestamps may settle, within three passes, before publication is allowed.
    """
    owned = identity(path)
    if owned[2] != reference['size_bytes']:
        raise InputChanged('Durable archive copy size differs from sealed input')
    for _ in range(3):
        before = identity(path)
        if before[:3] != owned[:3]:
            raise InputChanged('Owned archive copy was replaced or resized')
        checksum = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                checksum.update(block)
        after = identity(path)
        if after[:3] != owned[:3] or checksum.hexdigest() != reference['sha256']:
            raise InputChanged('Durable archive copy failed verification')
        if before == after:
            return True
    raise InputChanged('Owned archive copy metadata did not stabilize')


def copy_sealed(config, source, target, *, urgent=False):
    """One source copy/hash pass followed by bounded full-copy verification."""
    original = Path(source['original_path'])
    reference = source['retained']
    if target.exists():
        with slot(config, copy=True, urgent=urgent), phase('archive_hash_seconds'):
            if not verify_content(target, reference):
                raise InputChanged('Immutable MetaVideo destination collision')
        return
    if (identity(original) != source['identity']
            or str(original.resolve()) != source.get('resolved_path', str(original.resolve()))):
        raise InputChanged('Capture input changed before archival copy')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.partial')
    checksum = hashlib.sha256()
    try:
        with original.open('rb') as incoming, temporary.open('xb') as outgoing:
            while True:
                with slot(config, copy=True, urgent=urgent), phase('archive_copy_seconds'):
                    block = incoming.read(8 * 1024 * 1024)
                    if block:
                        checksum.update(block)
                        outgoing.write(block)
                if not block:
                    break
                pace_copy(config, len(block))
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if checksum.hexdigest() != reference['sha256'] or identity(original) != source['identity']:
            raise InputChanged('Capture input changed during archival copy')
        with slot(config, copy=True, urgent=urgent), phase('archive_hash_seconds'):
            if not _verify_owned_copy(temporary, reference):
                raise InputChanged('Durable archive copy failed verification')
        # Retention stage lock owns this destination. Never replace existing media.
        if target.exists():
            raise InputChanged('Archive destination appeared during copy')
        os.replace(temporary, target)
        sync_directory(target.parent)
        remember_content(target, reference['sha256'])
    finally:
        temporary.unlink(missing_ok=True)


def validate_after(paths, before):
    for reference, path in paths.items():
        if (identity(path), str(path.resolve())) != before[reference]:
            raise InputChanged('Input changed or disappeared during preprocessing')


def canonical_stt(value):
    audio_sources = [s['retained'] for s in value['sources'] if s['kind'].startswith('audio_')]
    if not any(s['kind'] == 'audio_audio' for s in value['sources']):
        audio_sources += [s['retained'] for s in value['sources'] if s['kind'] == 'video']
    return {'recording': {k: value['recording'].get(k) for k in ('recording_id', 'camera_key',
             'recording_start_us', 'recording_end_us')}, 'audio': value.get('audio'),
            'sources': audio_sources}


_ACTIVE_INPUT = ContextVar('device_day_sealed_input', default=None)


@contextmanager
def input_context(seal):
    token = _ACTIVE_INPUT.set(seal)
    try:
        yield
    finally:
        _ACTIVE_INPUT.reset(token)


def plan_sources(view, info):
    """Path-independent recall-plan identity only for runner-verified inputs."""
    selected, media = view.model_dump(mode='json'), info.model_dump(mode='json')
    seal = _ACTIVE_INPUT.get()
    if seal:
        sources = {source['kind']: source for source in seal['sources']}
        selected['video'] = media['path'] = 'sha256:' + sources['video']['sha256']
        selected['timestamps_csv'] = 'sha256:' + sources['clock']['sha256']
    return selected, media


def relocate_evidence(view, evidence):
    """Keep the original ledger, rebind its native-frame read with byte proof."""
    seal = _ACTIVE_INPUT.get()
    frame = evidence.source_frame
    if not seal or frame is None or frame.source_path == view.video:
        return evidence, None
    source = next(s for s in seal['sources'] if s['kind'] == 'video')
    if (str(frame.source_path) != source.get('resolved_path', source['original_path'])
            or frame.source_size_bytes != source['size_bytes'] or frame.source_mtime_ns != source['mtime_ns']
            or not verify_content(view.video, source)):
        raise InputChanged('Native frame ledger does not bind the relocated source')
    before = frame.model_dump(mode='json')
    relocated = frame.model_copy(update={'source_path': view.video.resolve(),
                                        'source_mtime_ns': view.video.stat().st_mtime_ns})
    return evidence.model_copy(update={'source_frame': relocated}), {
        'input_binding': binding(seal), 'original_source_frame': before,
        'verified_meta_video': source['retained'], 'ledger_rewritten': False}


def plan_config(config):
    """New scheduling options must not invalidate an old recall checkpoint."""
    from copy import deepcopy
    value = deepcopy(config)
    for name in ('inplace_preprocessing', 'nas_io_slots', 'archive_copy_workers', 'archive_copy_bytes_per_second',
                 'archive_max_wait_seconds', 'nas_io_timeout_seconds', 'capture_retention'):
        value.get('device_day', {}).pop(name, None)
    speech = value.get('speech_recognition') or {}
    if speech.get('adapter_sha256'):
        from .device_day_runtime_identity import compatible_runtime_hash
        speech['adapter_sha256'] = compatible_runtime_hash(Path(__file__).with_name('speech_qwen.py'), speech['adapter_sha256'])
    return value
