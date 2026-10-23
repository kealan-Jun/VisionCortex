"""Bounded legacy import port, executed against the pinned production package.

Only an explicitly configured legacy recording may create missing retained
copies. Recorder-native inputs and every model stage are rejected here.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


def retain(config, record):
    from visioncortex.capture_layout import bind_camera
    from visioncortex.device_day import DeviceDayRunner
    from visioncortex.device_day_recovery_worker import verify_storage

    options = config.get('device_day', {}).get('legacy_ingest') or {}
    root = Path(config['collection_ingest']['source_root'])
    original = Path(record['video_path'])
    relative = original.relative_to(root)
    configured = bind_camera(config['collection_ingest'], record)
    if (options.get('enabled') is not True or record.get('input_layout')
            or not 4 <= len(relative.parts) <= 5 or '..' in relative.parts
            or relative.parts[0] not in options.get('directories', [])
            or relative.parts[0] not in config['collection_ingest'].get('additional_camera_directories', [])
            or relative.as_posix() != record.get('relative_path') or original.name != 'rgb.mp4' or original.is_symlink()
            or not original.resolve().is_relative_to((root / relative.parts[0]).resolve())
            or configured.get('camera_key') != record.get('camera_key')
            or configured.get('camera_binding_status') != 'configured'
            or config['collection_ingest']['camera_role_map'].get(record['camera_key']) not in {'first_person', 'third_person'}):
        raise ValueError('Legacy archive copy requires its exact configured source binding')
    verify_storage(config)
    scoped = deepcopy(config)
    scoped['device_day']['raw_originals_read_only'] = False
    scoped['device_day']['delete_capture_sources'] = False
    scoped['device_day'].setdefault('capture_video_link_cleanup', {})['enabled'] = False

    class RetentionOnly(DeviceDayRunner):
        def _backend(self):
            raise RuntimeError('Legacy intake cannot execute models')

        def refresh_index(self, layout, *, recording_id=None):
            # The unchanged primary publisher owns canonical indices.
            return False

    runner = RetentionOnly(scoped, backend=object())
    result = runner.process(record, stage='retention')
    if result.get('status') == 'completed':
        # The user requested this exact old source now. Admit its ready CV
        # work through the native source-bound recovery cohort, so a busy
        # live-only history selector cannot leave it undiscovered again.
        # The existing worker still owns claims, priorities, input verification,
        # retry limits and actual model execution. No paid visual stage is admitted.
        from visioncortex.device_day_retention_worker import RetentionWorker
        result['vision_admitted'] = RetentionWorker(stage='vision', admission_cohort='current_recovery').admit(runner, record)
    return result


def main():
    from visioncortex.config import load_config

    path = Path(sys.argv[1])
    if hashlib.sha256(path.read_bytes()).hexdigest() != sys.argv[2]:
        raise ValueError('Legacy intake configuration changed')
    data = sys.stdin.buffer.read(1024**2 + 1)
    if len(data) > 1024**2:
        raise ValueError('Legacy intake request exceeds its bound')
    print(json.dumps(retain(load_config(path), json.loads(data))))


if __name__ == '__main__':
    main()
