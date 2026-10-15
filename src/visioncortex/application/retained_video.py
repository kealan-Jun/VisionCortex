"""Materialize a new derived generation, preserving originals and old outputs."""
from pathlib import Path
import uuid

from ..device_day_contract import artifact, atomic_json, digest, read_json
from ..device_day_inputs import hash_content, identity
from ..media.retained_intervals import disposition, publish_disposition


def materialize(layout, source, segment, meaning, recording, context=None, *, segments=None, extractor=None):
    """The caller owns source resolution and verified storage admission.

    Each full bounded analysis interval is kept, including uncertain boundaries.
    This function never deletes any input or replaces a historical generation.
    """
    from ..video_io import extract_clip, probe_video
    from ..device_day_content import time_folder
    extractor = extractor or extract_clip
    source = Path(source)
    before = identity(source)
    if hash_content(source) != segment['source_ref']['sha256']:
        raise ValueError('Derived video source differs from the sealed original')
    decision = disposition(segment, meaning, recording, context, segments=segments)
    folder = layout.processed / time_folder(segment['start_us'], segment['end_us']) / digest(decision)[:24]
    manifest = folder / 'Retention.json'
    if manifest.is_file():
        saved = read_json(manifest)
        if all(saved.get(k) == v for k, v in decision.items()) and saved.get('video'):
            from ..device_day_verification import verify_artifact_cached
            if verify_artifact_cached(layout.root, saved['video']):
                return saved
    if decision['decision'] == 'omit_static_derivative':
        return publish_disposition(folder, decision)
    from ..device_day_write_guard import guarded_mkdir, verify_before_submission
    guarded_mkdir(folder)
    target = folder / 'Activity.mp4'
    temporary = folder / ('.' + uuid.uuid4().hex + '.mp4')
    try:
        verify_before_submission()
        extractor(source, temporary, segment['start_ms'], segment['end_ms'] - segment['start_ms'], 'libx264')
        info = probe_video(temporary)
        expected = segment['end_ms'] - segment['start_ms']
        if abs(info.duration_ms - expected) > 1000 or identity(source) != before:
            raise ValueError('Derived video duration or original source changed')
        verify_before_submission()
        temporary.replace(target)
        value = publish_disposition(folder, decision, video_present=True) | {'video': artifact(layout.root, target),
            'source_mapping': {'source_start_ms': segment['start_ms'], 'source_end_ms': segment['end_ms'],
                               'derived_start_ms': 0, 'derived_end_ms': info.duration_ms}}
        atomic_json(manifest, value)
        return value
    finally:
        temporary.unlink(missing_ok=True)
