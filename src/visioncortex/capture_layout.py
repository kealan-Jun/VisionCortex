"""Explicit input-directory aliases; preserve the original recorder identity."""
from pathlib import PurePosixPath
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
