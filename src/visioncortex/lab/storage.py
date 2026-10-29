"""Private, explicit NAS identities; bounded local workflows need no NAS setup.

VC_LAB_STORAGE_MANIFEST names a local untracked JSON file. It contains a
vc-lab-storage/1 document with bindings: mount_path, share, project_root,
marker_path and marker_sha256. Optional marker_fields are checked as a subset.
No site mount, server, project marker or customer destination is bundled here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


_REMOTE_FS = {'cifs','smb3','nfs','nfs4','fuse.sshfs'}


def _mount_info(path):
    result = subprocess.run(['findmnt','-J','-T',str(path),'-o','TARGET,SOURCE,FSTYPE'],
                            capture_output=True,text=True,check=True,timeout=10)
    return [m for m in json.loads(result.stdout).get('filesystems',[]) if m.get('fstype')!='autofs']


def _manifest(required=False):
    name = os.environ.get('VC_LAB_STORAGE_MANIFEST')
    if not name:
        if required:
            raise OSError('Explicit private VC_LAB_STORAGE_MANIFEST is required before NAS access')
        return []
    path = Path(name)
    if (not path.is_absolute() or str(path).startswith(('/mnt/','/media/'))
            or path.is_symlink() or path.resolve()!=path or not path.is_file() or path.stat().st_size>65536):
        raise OSError('Storage manifest must be a bounded regular file on local storage')
    document = json.loads(path.read_text())
    bindings = document.get('bindings')
    if document.get('schema_version')!='vc-lab-storage/1' or not isinstance(bindings,list) or not 1<=len(bindings)<=8:
        raise OSError('Explicit versioned private storage bindings required')
    seen = set()
    for b in bindings:
        if not isinstance(b,dict):
            raise OSError('Invalid private storage binding')
        for key in ('mount_path','project_root','marker_path'):
            p = Path(b.get(key,''))
            if not p.is_absolute() or '..' in p.parts:
                raise OSError('Absolute lexical storage paths required')
        mount,root,marker = (Path(b[k]) for k in ('mount_path','project_root','marker_path'))
        if (not root.is_relative_to(mount) or not marker.is_relative_to(mount)
                or not isinstance(b.get('share'),str) or not b['share'].startswith('//')
                or not re.fullmatch('[0-9a-f]{64}',b.get('marker_sha256',''))
                or str(mount) in seen or b.get('marker_fields') is not None and not isinstance(b['marker_fields'],dict)):
            raise OSError('Invalid explicit share/root/marker identity')
        seen.add(str(mount))
    return bindings


def is_nas(path):
    path = Path(path).absolute()
    if str(path).startswith(('/mnt/','/media/','//')):
        return True
    if any(path.is_relative_to(Path(b['mount_path'])) for b in _manifest()):
        return True
    try:
        return any(m.get('fstype') in _REMOTE_FS for m in _mount_info(path))
    except (OSError,subprocess.SubprocessError):
        # Local operation does not require a Linux mount-inspection utility.
        return False


def verify_mount(mount, source):
    if _mount_info(mount) != [{'target':str(mount),'source':source,'fstype':'cifs'}]:
        raise OSError('NAS CIFS mount identity mismatch; no local fallback allowed')


def verify_required_roots(marker_sha256=None):
    bindings = _manifest(required=True)
    # Verify every kernel mount before touching a marker under any mount.
    for b in bindings:
        verify_mount(Path(b['mount_path']),b['share'])
    if marker_sha256 is not None and marker_sha256 not in {b['marker_sha256'] for b in bindings}:
        raise OSError('Supplied marker pin differs from private storage manifest')
    verified = []
    for b in bindings:
        marker = Path(b['marker_path'])
        if marker.is_symlink() or marker.resolve()!=marker or not marker.is_file() or marker.stat().st_size>65536:
            raise OSError('Regular bounded storage marker required')
        raw = marker.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=b['marker_sha256']:
            raise OSError('Private storage marker SHA mismatch')
        fields = json.loads(raw)
        if not isinstance(fields,dict) or not fields or any(fields.get(k)!=v for k,v in b.get('marker_fields',{}).items()):
            raise OSError('Private storage marker identity mismatch')
        verified.append(dict(b,verified_marker_fields=fields))
    return verified


def verify_root(path: Path, marker_sha256: str | None=None, *, project_destination=True):
    lexical = Path(path).absolute()
    if is_nas(lexical):
        bindings = verify_required_roots(marker_sha256)
        matches = [b for b in bindings if lexical.is_relative_to(Path(b['mount_path']))]
        if not matches or project_destination and not any(lexical.is_relative_to(Path(b['project_root'])) for b in matches):
            raise OSError('NAS path outside explicitly configured project root')
    if lexical.resolve()!=lexical:
        raise OSError('Storage root must not traverse symbolic links')
    return lexical


def _check(mount,source,marker,fields):
    if not fields:
        raise OSError('Empty NAS volume marker')
    key = 'id' if 'id' in fields else next(iter(fields))
    return dict(mount_path=str(mount),filesystem='cifs',source=source,marker_path=str(marker),
                marker_id_field=key,marker_id=fields[key],marker_fields=fields)


def storage_binding(root,marker_sha256=None):
    root = verify_root(root,marker_sha256)
    remote = is_nas(root)
    config = dict(runtime=dict(local_only=not remote))
    if remote:
        checks = [_check(b['mount_path'],b['share'],b['marker_path'],b['verified_marker_fields'])
                  for b in verify_required_roots(marker_sha256)]
        config['device_day'] = dict(publication_storage_checks=checks,recovery=dict(storage_checks=checks))
    return config


def publication_config(settings):
    root = settings.incoming
    config = storage_binding(root,settings.nas_marker_sha256)
    config['storage'] = dict(archive_root=str(root),local_runtime_root=str(settings.runtime_root),local_cache_root=str(settings.cache))
    return config
