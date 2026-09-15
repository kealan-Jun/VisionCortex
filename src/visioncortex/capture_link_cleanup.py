"""Verified atomic source-video replacement. Never unlink before link creation.

Requires native symlink support visible to all capture readers. The current
production CIFS mount failed its capability probe, so production is disabled.
"""
import os
from pathlib import Path
import uuid

from .device_day_contract import DIRECTORIES, atomic_json, digest, file_hash, safe_child, verify_artifact


def replace_capture_video_with_link(archive, retention, vision, *, enabled=False, backend_root=None):
    if not enabled:
        return {'status':'disabled_until_native_link_verified','replaced':False}
    archive = Path(archive)
    if retention.get('status')!='completed' or vision.get('status')!='completed':
        raise ValueError('Retention and preprocessing must be complete')
    if vision.get('retention_digest') != digest(retention):
        raise ValueError('Preprocessing does not bind this retained recording')
    for receipt in [retention,vision]:
        if not receipt.get('artifacts') or not all(verify_artifact(archive,r) for r in receipt['artifacts']):
            raise ValueError('Missing or changed archived evidence')
    for ref in vision.get('audit_artifacts',[]):
        if backend_root is None or ref.get('storage_root')!='local_cache_root' or not verify_artifact(Path(backend_root),ref):
            raise ValueError('Missing preprocessing audit')
    record = retention['recording']
    main = next(s for s in retention['sources'] if s['kind']=='video')
    source = Path(record['video_path'])
    root = Path(retention['capture_root']).resolve()
    target = safe_child(archive,main['retained']['path'])
    if str(source)!=main['original_path'] or not source.parent.resolve().is_relative_to(root):
        raise ValueError('Source outside sealed capture root')
    if source.parent.resolve().is_relative_to(archive.resolve()) or source == target:
        raise ValueError('Archive is not a cleanup source')
    if not target.resolve().is_relative_to((archive/DIRECTORIES[0]).resolve()):
        raise ValueError('Target must be the retained MetaVideo')
    if file_hash(target)!=main['sha256']:
        raise ValueError('Retained video mismatch')
    relative = os.path.relpath(target,source.parent)
    if source.is_symlink():
        if source.resolve()!=target.resolve():
            raise ValueError('Existing link points to a different target')
        return {'status':'already_linked','replaced':False,'target':relative}
    before = source.stat()
    identity = (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
    if before.st_size!=main['size_bytes'] or before.st_mtime_ns!=main['mtime_ns'] or file_hash(source)!=main['sha256']:
        raise ValueError('Capture source changed')
    temporary = source.with_name('.'+source.name+'.'+uuid.uuid4().hex+'.link')
    try:
        # Failure here leaves the original file untouched. Never use mfsymlinks
        # or a client-only absolute path to impersonate a native NAS link.
        os.symlink(relative,temporary)
        if not temporary.is_symlink() or temporary.resolve()!=target.resolve() or file_hash(temporary)!=main['sha256']:
            raise ValueError('Link is not readable as the verified retained video')
        after = source.lstat()
        if identity != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError('Capture source changed before replacement')
        os.replace(temporary,source)
    finally:
        temporary.unlink(missing_ok=True)
    result = {'status':'linked','replaced':True,'original_path':str(source),'target':relative,
              'sha256':main['sha256'],'retention_digest':digest(retention)}
    atomic_json(archive/DIRECTORIES[0]/'Metadata'/'Cleanup'/f"{record['recording_id']}.json",result)
    return result


def submit_after_preprocessing(runner, layout, recording):
    """Per-recording hook; never waits for semantic/report stages.

    Enabling requires both native NAS support and recorder-reader compatibility.
    Queue intent lives locally so a restart does not lose an unfinished cleanup.
    """
    settings = runner.settings.get('capture_video_link_cleanup') or {}
    if not settings.get('enabled'):
        return
    if not (settings.get('native_links_verified') and settings.get('capture_readers_verified')):
        raise ValueError('Native NAS link and capture reader gates are not verified')
    request = runner.runtime_root/'capture-link-requests'/f"{recording['recording_id']}.json"
    atomic_json(request,{'status':'queued','recording':recording})
    resume_pending(runner)


def resume_pending(runner):
    settings = runner.settings.get('capture_video_link_cleanup') or {}
    if not all(settings.get(k) for k in ['enabled','native_links_verified','capture_readers_verified']):
        return
    from concurrent.futures import ThreadPoolExecutor
    from .device_day_contract import read_json
    with runner._backend_lock:
        if not hasattr(runner,'_capture_cleanup_executor'):
            runner._capture_cleanup_executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='capture-link-cleanup')
            runner._capture_cleanup_jobs = {}
        for request in (runner.runtime_root/'capture-link-requests').glob('*.json'):
            if request in runner._capture_cleanup_jobs and not runner._capture_cleanup_jobs[request].done():
                continue
            payload=read_json(request)
            if payload.get('status') not in {'queued','waiting_for_index'}:
                continue
            def perform(request=request,payload=payload):
                recording=payload['recording']
                layout=runner.layout(recording)
                try:
                    retention=read_json(runner._receipt(layout,recording,'retention'))
                    vision=read_json(runner._receipt(layout,recording,'vision'))
                    expected=runner._key('retention',recording,recording)
                    if not runner._matches_key(retention.get('key'),expected):
                        raise ValueError('Retention recipe changed before cleanup')
                    from .device_day import visual_input
                    if vision.get('key')!=runner._key('vision',recording,visual_input(retention)):
                        raise ValueError('Preprocessing recipe changed before cleanup')
                    index=read_json(layout.index)
                    ids={x['segment_id'] for x in index.get('segments',[])}
                    wanted={x['segment_id'] for x in vision.get('segments',[])}
                    if not wanted or not wanted.issubset(ids):
                        atomic_json(request,payload|{'status':'waiting_for_index'})
                        return
                    result=replace_capture_video_with_link(layout.root,retention,vision,enabled=True,backend_root=runner.backend_root)
                    atomic_json(request,payload|{'status':'completed','result':result})
                except Exception as exc:
                    atomic_json(request,payload|{'status':'blocked','error_type':type(exc).__name__,'message':str(exc)[:500]})
            runner._capture_cleanup_jobs[request]=runner._capture_cleanup_executor.submit(perform)
