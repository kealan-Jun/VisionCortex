"""Shared archive-relative file resolution and truthful storage error mapping."""
import stat
from .device_day_contract import safe_child


def resolve(root, relative, *, directories=None, suffixes=None, historical=False):
    if historical:
        # Offline archives historically allow links that resolve within their
        # archive. Preserve that reader without relaxing recorder v1 paths.
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Artifact escapes archive')
    else:
        path = safe_child(root, relative)
    if directories and path.relative_to(root).parts[0] not in directories:
        raise ValueError('Unsupported archive directory')
    if suffixes and path.suffix.lower() not in suffixes:
        raise ValueError('Unsupported artifact type')
    if not stat.S_ISREG(path.stat().st_mode):
        raise FileNotFoundError('Artifact is not a regular file')
    return path


def storage_status(error):
    if isinstance(error, (FileNotFoundError, NotADirectoryError)):
        return 404, '文件不存在'
    if isinstance(error, PermissionError):
        return 403, '没有读取文件的权限'
    if isinstance(error, ValueError):
        return 404, '无效的归档文件引用'
    return 503, '存储暂不可读，已保留文件记录，请稍后重试'
