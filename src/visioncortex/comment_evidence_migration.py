"""Copy, verify and recoverably retire only generated Comment evidence.

The caller owns the device/day index lock and all recording stage locks. No
capture files, completed model receipts or public transcripts are rewritten.
"""
import os
from pathlib import Path
import stat

from .comment_evidence import PREFIXES, backend_path
from .device_day_contract import atomic_json, file_hash, read_json


def _identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Comment evidence must be a regular file')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def inventory(source, *, max_files, max_bytes):
    files, size = [], 0
    for directory, children, names in os.walk(source, followlinks=False):
        for name in children:
            if (Path(directory)/name).is_symlink():
                raise ValueError('Comment evidence directory cannot be a symlink')
        for name in sorted(names):
            path = Path(directory)/name
            identity = _identity(path)
            size += identity[2]
            files.append((path.relative_to(source).as_posix(), identity))
            if len(files) > max_files or size > max_bytes:
                raise ValueError('Comment evidence migration exceeds its declared bound')
    return sorted(files)


def migrate(layout, *, max_files=200000, max_bytes=32*1024**3, checkpoint=lambda: None):
    from .device_day import copy_verified
    from .device_day_content_paths import _publish_aliases, aliases
    from .device_day_contract import safe_child
    results = []
    for prefix in PREFIXES:
        relative = prefix.rstrip('/')
        source = safe_child(layout.root, relative)
        target = backend_path(layout.root, relative, layout.backend_root)
        journal = target.parent/'Migration.json'
        retired = target.parent/'RetiredEvidence'
        if not source.exists():
            if journal.is_file():
                saved = read_json(journal)
                if saved.get('status') == 'verified' and retired.is_dir():
                    references = saved['files']
                    old = inventory(retired, max_files=max_files, max_bytes=max_bytes)
                    if [name for name, _ in old] != [ref['path'] for ref in references]:
                        raise ValueError('Interrupted Comment evidence retirement differs')
                    for reference in references:
                        checkpoint()
                        for base in (target, retired):
                            path = safe_child(base, reference['path'])
                            if (_identity(path)[2] != reference['size_bytes']
                                    or file_hash(path) != reference['sha256']):
                                raise ValueError('Interrupted Comment evidence retirement is unverified')
                    saved['status'] = 'completed'
                    atomic_json(journal, saved)
                if saved.get('status') != 'completed':
                    raise ValueError('Interrupted Comment evidence retirement needs verification')
                results.append(saved)
            continue
        if source.is_symlink() or not source.is_dir():
            raise ValueError('Comment evidence source is not an owned directory')
        if retired.exists():
            raise ValueError('Returning legacy evidence must not overwrite its retirement copy')
        items = inventory(source, max_files=max_files, max_bytes=max_bytes)
        references = []
        for name, identity in items:
            checkpoint()
            original, destination = safe_child(source, name), safe_child(target, name)
            if _identity(original) != identity:
                raise ValueError('Comment evidence changed before copy')
            reference = copy_verified(original, destination)
            if _identity(original) != identity or file_hash(destination) != reference['sha256']:
                raise ValueError('Comment evidence changed during copy')
            references.append(reference | {'path': name})
        if inventory(source, max_files=max_files, max_bytes=max_bytes) != items:
            raise ValueError('Comment evidence membership changed during migration')
        for (name, identity), reference in zip(items, references, strict=True):
            checkpoint()
            original, destination = safe_child(source, name), safe_child(target, name)
            if (_identity(original) != identity or file_hash(original) != reference['sha256']
                    or file_hash(destination) != reference['sha256']):
                raise ValueError('Comment evidence failed final verification')
        # Preserve old Comment/Stt redirects in the existing publication reader,
        # outside Comment. All referenced model receipts retain their bytes.
        _publish_aliases(layout, 'Comment', {
            k: v for k, v in aliases(layout.root).items() if k.startswith('Comment/')})
        result = {'schema_version': 'visioncortex-comment-evidence-migration/1',
            'archive': layout.name, 'source': relative,
            'backend': target.relative_to(layout.backend_root).as_posix(),
            'files': references, 'status': 'verified', 'model_calls': 0,
            'raw_media_changed': False, 'completed_receipts_changed': False}
        atomic_json(journal, result)
        # Keep a complete verified rollback copy; never unlink an evidence tree.
        os.rename(source, retired)
        result['status'] = 'completed'
        atomic_json(journal, result)
        results.append(result)
    return results
