"""Private Comment storage, with confined read compatibility for old receipts.

Logical archive references and their hashes remain unchanged. The configured
backend is an explicit capability; an archive document cannot choose a root.
"""
from pathlib import Path

PREFIXES = ('Comment/.Evidence/', 'Comment/Audio/.Evidence/', 'Comment/OCR/.Evidence/')
NAMESPACE = 'device-day-comment-evidence'


def is_private(relative):
    return relative + '/' in PREFIXES or relative.startswith(PREFIXES)


def backend_path(root, relative, backend_root):
    from .device_day_contract import safe_child, validate_archive_name
    if backend_root is None or not is_private(relative):
        raise ValueError('Comment evidence requires its configured backend root')
    day = validate_archive_name(Path(root).name)
    return safe_child(Path(backend_root), f'{NAMESPACE}/{day}/{relative}')


def resolve(root, relative, backend_root=None):
    """Read a logical reference; never use a damaged backend's legacy copy."""
    from .device_day_contract import safe_child
    from .device_day_content_paths import aliases, legacy_reference, relocated
    if legacy_reference(relative):
        relative = relocated(relative, aliases(root))
    original = safe_child(root, relative)
    if backend_root is not None and is_private(relative):
        target = backend_path(root, relative, backend_root)
        # During a rolling copy, old receipts may still use unmoved files.
        # Once committed, missing backend bytes must fail closed.
        prefix = next(p for p in PREFIXES if (relative + '/').startswith(p))
        owner = backend_path(root, prefix.rstrip('/'), backend_root)
        if target.exists() or (owner.parent/'Migration.json').is_file():
            return target
    return original


def relative_path(root, path, backend_root=None):
    from .device_day_contract import safe_child
    try:
        relative = path.relative_to(root).as_posix()
        safe_child(root, relative)
        return relative
    except ValueError:
        if backend_root is None:
            raise
        base = Path(backend_root) / NAMESPACE / root.name
        relative = path.relative_to(base).as_posix()
        if backend_path(root, relative, backend_root) != path:
            raise ValueError('File is not owned Comment evidence') from None
        return relative


def artifact(root, path, backend_root=None):
    from .device_day_contract import file_hash
    relative = relative_path(root, path, backend_root)
    checked = resolve(root, relative, backend_root)
    if checked != path:
        raise ValueError('Comment evidence reference has another physical owner')
    return {'path': relative, 'size_bytes': checked.stat().st_size, 'sha256': file_hash(checked)}
