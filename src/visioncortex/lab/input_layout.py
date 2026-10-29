"""Read existing capture paths in place; never reorganize producer files."""
from __future__ import annotations

from pathlib import PurePosixPath

from .archive_layout import day_directory
from .components import enable_components
from .domain import Conflict


def photo_location(relative):
    """Bind path aliases to FR's unchanged filename, date and batch semantics."""
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError('Invalid photo path')
    parts = relative.split('/')
    if any(part in {'', '.', '..'} for part in parts):
        raise ValueError('Photo paths must be normalized and relative')
    if len(parts) == 5 and parts[0] == 'voice_photos':
        parts = parts[1:]
        if parts[1] == 'MetaData':
            raise ValueError('Invalid legacy voice-photo path')
    if len(parts) == 5 and parts[1:3] == ['MetaData', 'Photos']:
        parts = [parts[0], 'MetaData', *parts[3:]]
    if len(parts) != 4:
        raise ValueError('Unsupported photo input layout')
    if parts[1] == 'MetaData':
        day, separator, camera = parts[0].partition('_')
        if not separator:
            raise ValueError('Missing capture device identity')
    else:
        camera, day = parts[:2]
    if day_directory(camera, day+'T00:00:00+08:00') != day+'_'+camera:
        raise ValueError('Invalid capture device or date')
    enable_components()
    from field_recognition.capture.photo_sources import photo_location as original
    from field_recognition.capture.photo_sources import PhotoLocation
    # FR historically restricts camera spelling to 100 characters without dots.
    # Validate the actual key with VC, and reuse FR for filename/time checks.
    checked = original('/'.join(['path-validation-camera', day, parts[2], parts[3]]))
    return PhotoLocation(camera, checked.day, checked.moment, checked.filename)


def upload_markers(root, path):
    """Include ancestor upload state so completion immediately wakes discovery."""
    path.relative_to(root)
    current = path.parent
    markers = []
    while True:
        marker = current/'.uploading'
        if marker.exists() or marker.is_symlink():
            markers.append(PurePosixPath(marker.relative_to(root)).as_posix())
        if current == root:
            break
        current = current.parent
    return markers


def require_finished_upload(root, path):
    if upload_markers(root, path):
        raise Conflict('Capture directory is still uploading; wait for completion')
