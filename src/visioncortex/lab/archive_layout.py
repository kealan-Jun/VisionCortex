"""One device/day archive shared by native video, photographs and live readouts."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from uuid import uuid4

from .components import ROOT, pipeline_module
from .domain import Conflict, instant
from .media import hash_file, safe_path
from .storage import is_nas, publication_config

CONTRACT_PATH = '.System/ArchiveContract.json'


def contract_bytes():
    return (ROOT / 'contracts/lab/archive-layout-v2.json').read_bytes()


def day_directory(camera_key, moment):
    # VC's original validator and Asia/Shanghai day boundary are authoritative.
    at = instant(moment)
    delta = at - datetime(1970, 1, 1, tzinfo=timezone.utc)
    micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    return pipeline_module('device_day_contract').archive_name(camera_key, micros)


def verify_archive_contract(root):
    path = safe_path(root, CONTRACT_PATH)
    expected = contract_bytes()
    if path.stat().st_size != len(expected) or hash_file(path, 128 * 1024)[0] != hashlib.sha256(expected).hexdigest():
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
