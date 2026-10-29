"""One device/day archive shared by native video, photographs and live readouts."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from uuid import uuid4

from .components import ROOT, pipeline_module
from .domain import Conflict, instant
from .media import hash_file, safe_path
from .storage import is_nas, publication_config, verify_root, _mount_info, _REMOTE_FS

CONTRACT_PATH = '.System/ArchiveContract.json'


def contract_bytes():
    return (ROOT / 'contracts/lab/archive-layout-v2.json').read_bytes()


def day_directory(camera_key, moment):
    # VC's original validator and Asia/Shanghai day boundary are authoritative.
    at = instant(moment)
    delta = at - datetime(1970, 1, 1, tzinfo=timezone.utc)
    micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    return pipeline_module('device_day_contract').archive_name(camera_key, micros)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Conflict('Duplicate archive compatibility JSON key')
        result[key] = value
    return result


def _json_contract(raw):
    def invalid_constant(value):
        raise Conflict('Nonfinite archive compatibility JSON value')
    document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    if not isinstance(document, dict):
        raise Conflict('Archive compatibility JSON must be an object')
    return document


def _compatible_contract(root, path, actual, size, expected):
    name = os.environ.get('VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST')
    if not name:
        return False
    manifest = Path(name)
    try:
        # This private configuration stays local, even when the archive is NAS.
        if (not manifest.is_absolute() or str(manifest) != name or '..' in manifest.parts or
                is_nas(manifest) or manifest.resolve(strict=True) != manifest or not manifest.is_file()):
            raise Conflict('Archive compatibility manifest must be a canonical local non-NAS file')
        mounts = _mount_info(manifest)
        if (len(mounts) != 1 or not mounts[0].get('fstype') or
                mounts[0]['fstype'] in _REMOTE_FS or str(mounts[0].get('source', '')).startswith('//')):
            raise Conflict('Archive compatibility manifest local filesystem is unverified')
        digest, length = hash_file(manifest, 64 * 1024)
        with manifest.open('rb') as stream:
            raw = stream.read(64 * 1024 + 1)
        if len(raw) != length or hashlib.sha256(raw).hexdigest() != digest:
            raise Conflict('Archive compatibility manifest changed while reading')
        document = _json_contract(raw)
        if set(document) != {'schema_version', 'bindings'} or document['schema_version'] != 'vc-lab-archive-compatibility/1':
            raise Conflict('Unsupported archive compatibility manifest')
        bindings = document['bindings']
        if not isinstance(bindings, list) or not 1 <= len(bindings) <= 32:
            raise Conflict('Bounded explicit archive compatibility bindings required')
        selected, seen = [], set()
        installed = hashlib.sha256(expected).hexdigest()
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {'archive_root', 'source_sha256', 'installed_contract_sha256', 'metadata_aliases'}:
                raise Conflict('Incomplete archive compatibility binding')
            archive = binding['archive_root']
            if (type(archive) is not str or not Path(archive).is_absolute() or
                    str(Path(archive)) != archive or '..' in Path(archive).parts):
                raise Conflict('Archive compatibility binding requires an exact absolute root')
            for key in ('source_sha256', 'installed_contract_sha256'):
                if type(binding[key]) is not str or not re.fullmatch('[0-9a-f]{64}', binding[key]):
                    raise Conflict('Archive compatibility binding requires exact SHA256 pins')
            aliases = binding['metadata_aliases']
            if (not isinstance(aliases, dict) or not 1 <= len(aliases) <= 3 or
                    set(aliases) - {'project', 'schema_version', 'supersedes'} or
                    any(type(value) is not str or not value or len(value.encode()) > 4096 for value in aliases.values())):
                raise Conflict('Only explicit project/schema_version/supersedes string aliases are allowed')
            identity = (archive, binding['source_sha256'], binding['installed_contract_sha256'])
            if identity in seen:
                raise Conflict('Ambiguous duplicate archive compatibility bindings')
            seen.add(identity)
            if identity == (str(root), actual, installed):
                selected.append(binding)
        if len(selected) != 1:
            return False
        with path.open('rb') as stream:
            raw = stream.read(128 * 1024 + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != actual:
            raise Conflict('Archive source contract changed while reading')
        source, target = _json_contract(raw), _json_contract(expected)
        for key, old_value in selected[0]['metadata_aliases'].items():
            if source.get(key) != old_value or key not in target:
                raise Conflict('Archive metadata alias differs from the pinned source contract')
            source[key] = target[key]
        # JSON preserves types: True must not compare equal to the integer 1.
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        return canonical(source) == canonical(target)
    except Conflict:
        raise
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        raise Conflict('Archive compatibility manifest or pinned source is unavailable/invalid') from exc


def verify_archive_contract(root):
    root = verify_root(Path(root))
    expected = contract_bytes()
    try:
        path = safe_path(root, CONTRACT_PATH)
    except FileNotFoundError:
        if not os.environ.get('VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST'):
            raise
        from .existing_archive_binding import verify_existing_native_root
        return verify_existing_native_root(root, expected)
    actual, size = hash_file(path, 128 * 1024)
    if size == len(expected) and actual == hashlib.sha256(expected).hexdigest():
        return path
    if not _compatible_contract(root, path, actual, size, expected):
        raise Conflict('NAS archive format differs from the installed unified contract')
    return path


def _immutable_write(path, raw, guard):
    if path.exists():
        if path.stat().st_size != len(raw) or hash_file(path, 128 * 1024)[0] != hashlib.sha256(raw).hexdigest():
            raise Conflict('Existing archive contract or instructions differ; an explicit migration is required')
    elif not guard.guarded_atomic(path, [raw]):
        if is_nas(path):
            raise OSError('NAS archive initialization requires a pinned write guard; no local fallback')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name('.'+path.name+'.'+str(uuid4())+'.partial')
        try:
            with temporary.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    digest, size = hash_file(path, 128 * 1024)
    if digest != hashlib.sha256(raw).hexdigest() or size != len(raw):
        raise Conflict('Unified archive initialization failed read-back verification')
    return dict(path=str(path), sha256=digest, size_bytes=size)


def archive_guide(contract):
    return ('实验室统一自动留存目录\n'
             '优先复用VC目录；Comment/Audio保存录音产出，Comment/OCR保存FR读数产出。\n'
             '日期按Asia/Shanghai和原始拍摄时间；时间未知时明确标记，不当作已校时。\n'
             'MetaData由采集端写入，处理端保留原名、原字节，已有MetaVideo不搬迁。\n'
             '数据库、队列、环境和当前模型在SSD；NAS保存原始证据、处理产物及不可变回执。\n'
             '下面是格式说明，不代表已经收到数据或完成识别。\n\n'
             'YYYY-MM-DD_<camera_key>/\n'+''.join('  '+name+'/\n'+''.join('    '+child+'/\n' for child in item.get('children', {})) for name,item in contract['directories'].items())+
             '\n.System/ArchiveContract.json 为版本化格式合同。\n'
             '软件设置VC_LAB_SOURCE_ROOT为本目录，VC_LAB_UNIFIED_ARCHIVE_REQUIRED=1。\n'
             '创建本合同不启用采集连接，不生成实验结果、设备档案或材料库存。\n')


def prepare_archive(settings):
    """Prepare the format only; do not register sources or start processing."""
    from .rules import Rules
    Rules().identity()
    if settings.source_root is None:
        raise ValueError('Unified archive preparation requires an explicit source/output root')
    writable = replace(settings, unified_archive_required=False)
    root = writable.incoming
    contract = json.loads(contract_bytes())
    guide = archive_guide(contract)
    guard = pipeline_module('device_day_write_guard')
    with guard.storage_guard(publication_config(writable)):
        from .existing_archive_binding import _read
        active = guard.CURRENT.get()
        active.mkdir(root)
        if (_read(root, CONTRACT_PATH, 128 * 1024, active, optional=True) is None
                and _read(root, 'DataRetentionGuide.docx', 2 * 1024**2, active, optional=True) is not None):
            raise Conflict('Existing native guide requires explicit read-only root binding; empty bootstrap is forbidden')
        refs = [_immutable_write(safe_path(root, CONTRACT_PATH, must_exist=False), contract_bytes(), guard),
                _immutable_write(safe_path(root, 'README.txt', must_exist=False), guide.encode(), guard)]
    verify_archive_contract(root)
    return dict(root=str(root), contract=contract['schema_version'], files=refs, customer_sources_started=0, data_created=False)


def upgrade_archive(settings, expected_previous_sha256):
    """Upgrade only a prepared archive with no device/day data; retain a replayable journal."""
    import fcntl
    from .rules import Rules
    Rules().identity()
    if settings.source_root is None:
        raise ValueError('Archive upgrade requires an explicit root')
    writable = replace(settings, unified_archive_required=False)
    root = writable.incoming
    previous = (ROOT/'contracts/lab/archive-layout.json').read_bytes()
    old_hash = hashlib.sha256(previous).hexdigest()
    current = contract_bytes()
    new_hash = hashlib.sha256(current).hexdigest()
    if expected_previous_sha256 != old_hash:
        raise Conflict('Upgrade must name the exact previously frozen archive contract')
    guard = pipeline_module('device_day_write_guard')
    lock = writable.runtime_root/('archive-contract-'+hashlib.sha256(str(root).encode()).hexdigest()+'.lock')
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle, guard.storage_guard(publication_config(writable)):
        fcntl.flock(handle,fcntl.LOCK_EX)
        if any(p.is_dir() and not p.name.startswith('.') for p in root.iterdir()):
            raise Conflict('Archive already contains data; automatic historical relocation is not allowed')
        contract_path = safe_path(root,CONTRACT_PATH)
        actual,_ = hash_file(contract_path,128*1024)
        if actual not in {old_hash,new_hash}:
            raise Conflict('Existing archive contract differs from the authorized predecessor')
        history = '.System/ArchiveContracts/'+old_hash
        refs = [_immutable_write(safe_path(root,history+'/ArchiveContract.json',must_exist=False),previous,guard)]
        guide_path = safe_path(root,'README.txt')
        new_guide = archive_guide(json.loads(current)).encode()
        plan_path = safe_path(root,'.System/ArchiveContractUpdates/'+old_hash+'-'+new_hash+'.json',must_exist=False)
        if plan_path.exists():
            if plan_path.stat().st_size>128*1024:
                raise Conflict('Archive update journal exceeds its bound')
            plan = json.loads(plan_path.read_text())
            if plan.get('from_sha256')!=old_hash or plan.get('to_sha256')!=new_hash:
                raise Conflict('Archive update journal has a different source or target')
            old_guide_path = safe_path(root,history+'/README.txt')
            if hash_file(old_guide_path,128*1024)[0]!=plan['old_guide_sha256']:
                raise Conflict('Previous archive instructions failed history verification')
        else:
            if actual!=old_hash:
                raise Conflict('Changed archive contract has no recoverable update journal')
            hash_file(guide_path,128*1024)
            old_guide = guide_path.read_bytes()
            refs.append(_immutable_write(safe_path(root,history+'/README.txt',must_exist=False),old_guide,guard))
            plan = dict(schema_version='vc-lab-archive-contract-update/1',from_sha256=old_hash,to_sha256=new_hash,old_guide_sha256=hashlib.sha256(old_guide).hexdigest(),new_guide_sha256=hashlib.sha256(new_guide).hexdigest(),historical_data_moved=False,customer_connections_started=0)
            raw = (json.dumps(plan,sort_keys=True,indent=2)+'\n').encode()
            refs.append(_immutable_write(plan_path,raw,guard))
        if plan['new_guide_sha256']!=hashlib.sha256(new_guide).hexdigest():
            raise Conflict('Archive update journal targets different instructions')
        for path,raw,allowed in ((contract_path,current,{old_hash,new_hash}),(guide_path,new_guide,{plan['old_guide_sha256'],plan['new_guide_sha256']})):
            if hash_file(path,128*1024)[0] not in allowed:
                raise Conflict('Archive metadata changed outside the recorded update')
            # Both writes have a durable journal. A retry after interruption
            # accepts either the previous or the exact new bytes and completes.
            if not guard.guarded_atomic(path,[raw]):
                if is_nas(path):
                    raise OSError('NAS contract update requires its pinned write guard')
                temporary = path.with_name('.'+path.name+'.'+str(uuid4())+'.partial')
                try:
                    with temporary.open('xb') as stream:
                        stream.write(raw);stream.flush();os.fsync(stream.fileno())
                    os.replace(temporary,path)
                    descriptor = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                finally:
                    temporary.unlink(missing_ok=True)
            if hash_file(path,128*1024)[0]!=hashlib.sha256(raw).hexdigest():
                raise Conflict('Updated archive metadata failed read-back verification')
    verify_archive_contract(root)
    return dict(root=str(root),contract=json.loads(current)['schema_version'],previous_contract_preserved=True,historical_data_moved=False,customer_sources_started=0)
