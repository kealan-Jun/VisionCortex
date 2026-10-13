"""Explicit input-directory aliases; preserve the original recorder identity."""
from pathlib import Path, PurePosixPath
from glob import escape
import hashlib
import re


def validate(settings):
    additional = settings.get('additional_camera_directories') or []
    if not isinstance(additional, list) or any(
            not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
            or name in {'.', '..'} for name in additional):
        raise ValueError('Invalid additional capture directories')
    for rule in settings.get('directory_camera_bindings') or []:
        if set(rule) != {'directory_glob', 'camera_key'}:
            raise ValueError('Camera directory binding requires directory_glob and camera_key')
        parts = PurePosixPath(rule['directory_glob']).parts
        if (not parts or parts[0] not in additional or '..' in parts
                or '\\' in rule['directory_glob']):
            raise ValueError('Camera binding must stay inside an additional capture directory')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', rule['camera_key']):
            raise ValueError('Invalid bound camera key')


def bind_camera(settings, record):
    relative = PurePosixPath(record.get('relative_path') or '')
    if not relative.parts or relative.parts[0] not in (settings.get('additional_camera_directories') or []):
        return record
    matches = [rule for rule in settings.get('directory_camera_bindings') or []
               if relative.parent.match(rule['directory_glob'])]
    if len(matches) > 1:
        raise ValueError('Capture directory has ambiguous camera bindings')
    original = record.get('recorder_camera_key', record.get('camera_key'))
    if not matches:
        # Reused recorder IDs must never inherit the other laboratory's role.
        suffix = hashlib.sha256('/'.join(relative.parts[:2]).encode()).hexdigest()[:12]
        return record | {'camera_key': 'unmapped-'+suffix,
                         'source_group': relative.parts[0],
                         'recorder_camera_key': original, 'configured_role': None,
                         'camera_binding_status': 'needs_directory_binding'}
    return record | {'camera_key': matches[0]['camera_key'], 'recorder_camera_key': original,
                     'source_group': camera_group(settings, matches[0]['camera_key']),
                     'camera_identity_source': 'configured_directory_binding',
                     'camera_binding_status': 'configured'}


def camera_group(settings, camera):
    return (settings.get('camera_group_map') or {}).get(camera, '')


METADATA_LAYOUT = 'visioncortex-recorder-metadata/1'


def metadata_location(video):
    """Identify a recorder-owned device/day without inferring its view role."""
    from .device_day_contract import safe_child, validate_archive_name
    video = Path(video)
    for parent in video.parents:
        if parent.name != 'MetaData':
            continue
        try:
            name = validate_archive_name(parent.parent.name)
        except ValueError:
            continue
        safe_child(parent.parent, video.relative_to(parent.parent).as_posix())
        return {'archive': name, 'camera_key': name[11:], 'root': parent}
    return None


def metadata_directories(config):
    """Join existing discovery with recorder-owned MetaData only; create nothing."""
    from .device_day_contract import safe_child, validate_archive_name
    settings = config['collection_ingest']
    if not settings.get('discover_recorder_metadata', True):
        return []
    if not settings.get('source_root') or not config.get('storage', {}).get('archive_root'):
        return []
    source = Path(settings['source_root'])
    archive = Path(config['storage']['archive_root'])
    if not archive.is_relative_to(source) or not archive.is_dir() or archive.is_symlink():
        return []
    if archive != source:
        safe_child(source, archive.relative_to(source).as_posix())
    cameras = settings.get('camera_directories') or []
    result = set()
    dates = settings.get('metadata_capture_dates') or ([settings['capture_date']] if settings.get('capture_date') else [])
    day_patterns = [escape(date) for date in dates] if dates else ['????-??-??']
    camera_patterns = [escape(name) for name in cameras] if cameras else ['*']
    patterns = [f'{day}_{camera}' for day in day_patterns for camera in camera_patterns]
    for pattern in patterns:
        for day in archive.glob(pattern):
            try:
                validate_archive_name(day.name)
                date = day.name[:10]
                if ((settings.get('capture_date') and date != settings['capture_date'])
                        or (settings.get('capture_since_date') and date < settings['capture_since_date'])
                        or (settings.get('metadata_capture_dates') and date not in settings['metadata_capture_dates'])):
                    continue
                raw = safe_child(archive, day.name + '/MetaData')
                if raw.is_dir():
                    result.add(raw)
            except ValueError:
                continue
    return sorted(result, reverse=True)
