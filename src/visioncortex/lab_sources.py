"""Isolated laboratory archives and queues sharing one worker's model pools."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import os
from pathlib import Path
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LabSource(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(pattern=r'^[A-Z][A-Za-z0-9]*$')
    source_root: str
    archive_root: str
    cache_root: str
    runtime_root: str
    camera_directory_glob: str
    camera_role_map: dict[str, Literal['first_person', 'third_person']]


def laboratory_configs(config):
    """Apply isolation after deployment environment overrides; no storage access."""
    labs = [LabSource.model_validate(row) for row in
            config.get('device_day', {}).get('additional_labs', [])]
    if not labs:
        return {}
    if not config.get('device_day', {}).get('enabled') or not config.get('collection_ingest', {}).get('enabled'):
        raise ValueError('Additional laboratories require enabled NAS device-day processing')
    storage = config['storage']
    occupied = [Path(os.path.abspath(storage[key])) for key in ('archive_root', 'local_cache_root')]
    runtimes = {Path(os.path.abspath(storage['local_runtime_root']))}
    sources = {Path(os.path.abspath(config['collection_ingest']['source_root']))}
    result = {}
    for lab in labs:
        if lab.name in result:
            raise ValueError('Laboratory names must be unique')
        for key in ('source_root', 'archive_root', 'cache_root', 'runtime_root'):
            value = getattr(lab, key)
            if not Path(value).is_absolute() or '..' in Path(value).parts:
                raise ValueError('Laboratory storage roots must be absolute without traversal')
        source, archive, cache, runtime = [Path(os.path.abspath(getattr(lab, key)))
            for key in ('source_root', 'archive_root', 'cache_root', 'runtime_root')]
        if source in sources or runtime in runtimes:
            raise ValueError('Laboratories must have separate input and runtime roots')
        if str(runtime).startswith(('//', '/mnt/', '/media/', '/run/user/')):
            raise ValueError('Laboratory runtime must be on local storage')
        for output in (archive, cache):
            if any(output.is_relative_to(other) or other.is_relative_to(output) for other in occupied):
                raise ValueError('Laboratory archive/cache roots must not overlap')
            if source.is_relative_to(output):
                raise ValueError('Laboratory output cannot contain its capture root')
            occupied.append(output)
        sources.add(source)
        runtimes.add(runtime)
        child = deepcopy(config)
        child['device_day']['additional_labs'] = []
        # Historical compatibility receipts belong to the original lab only.
        for key in ('completed_vision_receipts', 'completed_stage_receipts'):
            child['device_day'].pop(key, None)
        child['collection_ingest'].update(
            source_root=str(source), camera_directories=[],
            camera_directory_glob=lab.camera_directory_glob,
            camera_role_map=dict(lab.camera_role_map), snapshot_path=None,
            lab_name=lab.name)
        staging = archive / '.VisionCortex-Run-Staging'
        child['storage'].update(
            archive_root=str(archive), local_cache_root=str(cache),
            local_runtime_root=str(runtime), local_input_root=str(runtime / 'Input-Manifests'),
            local_staging_root=str(staging), index_csv=str(archive / 'ExperimentRecordIndex.csv'),
            device_registry_path=None, source_path_migration_receipts=[], yolo_annotation_queue_path=None)
        child['project']['output_root'] = str(staging)
        child.setdefault('runtime', {})['resource_root'] = (
            config.get('runtime', {}).get('resource_root') or storage['local_runtime_root'])
        result[lab.name] = child
    return result


def select_laboratory(config, name):
    try:
        return laboratory_configs(config)[name]
    except KeyError as exc:
        raise ValueError(f'Unknown laboratory: {name}') from exc


@contextmanager
def additional_laboratories(settings_factory, primary, monitor_loop):
    """Own each lab once, stop all admissions before draining shared inference."""
    from .device_day_contract import atomic_json
    from .device_day_service import DeviceDayService
    from .runtime_process import worker_owner

    configs = laboratory_configs(settings_factory())
    if configs and not settings_factory().get('runtime', {}).get('resource_limits', {}).get('vision'):
        raise ValueError('Additional laboratories require shared vision resource limits')
    active = []
    with ExitStack() as owners:
        try:
            for name, config in configs.items():
                owners.enter_context(worker_owner(config))
                def factory(name=name):
                    return select_laboratory(settings_factory(), name)
                service = DeviceDayService(factory, primary.gpu_lock)
                path = Path(config['storage']['local_runtime_root']) / 'state' / 'nas-recording-monitor.json'
                def publish(settings, payload, path=path):
                    atomic_json(path, payload)
                thread = threading.Thread(target=monitor_loop, args=(config,),
                    kwargs={'service': service, 'stop': service.stop_event, 'publish': publish},
                    name='nas-monitor-'+name, daemon=True)
                active.append((service, thread))
                service.start()
                thread.start()
            yield
        finally:
            for service, _ in active:
                service.stop_event.set()
                service.wakeup.set()
            for service, thread in active:
                if thread.ident is not None:
                    thread.join(timeout=5)
                service.stop()
