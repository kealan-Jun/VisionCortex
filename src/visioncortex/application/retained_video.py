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
    from ..device_day_content_paths import clip_name, layout_contract, private_folder
    from ..device_day_contract import safe_child
    from ..device_day_verification import verify_artifact_cached
    extractor = extractor or extract_clip
    source = Path(source)
    before = identity(source)
    if hash_content(source) != segment['source_ref']['sha256']:
        raise ValueError('Derived video source differs from the sealed original')
    decision = disposition(segment, meaning, recording, context, segments=segments)
    folder = private_folder(layout, 'ProcessedClips', 'Retention', segment['segment_id'], digest(decision)[:24])
    manifest = folder / 'Retention.json'
    if manifest.is_file():
        saved = read_json(manifest)
        if all(saved.get(k) == v for k, v in decision.items()) and saved.get('video'):
            if verify_artifact_cached(layout.root, saved['video']):
                return saved
    if decision['decision'] == 'omit_static_derivative':
        return publish_disposition(folder, decision)
    from ..device_day_write_guard import guarded_mkdir, verify_before_submission
    guarded_mkdir(folder)
    activity = layout_contract()['activities'][segment['activity']]
    target = safe_child(layout.root, 'ProcessedClips/Clips/' + clip_name(segment) + '/' + activity + '.mp4')
    original_video = segment.get('video')
    if original_video and verify_artifact_cached(layout.root, original_video):
        existing = safe_child(layout.root, original_video['path'])
        if existing != target:
            from ..device_day import copy_verified
            copy_verified(existing, target)
        value = publish_disposition(folder, decision, video_present=True) | {'video': artifact(layout.root, target)}
        atomic_json(manifest, value)
        return value
    if target.exists():
        raise ValueError('Unbound clip output exists; refusing to overwrite historical media')
    guarded_mkdir(target.parent)
    temporary = target.parent / ('.' + uuid.uuid4().hex + '.mp4')
    try:
        verify_before_submission()
        extractor(source, temporary, segment['start_ms'], segment['end_ms'] - segment['start_ms'], 'libx264')
        info = probe_video(temporary)
        expected = segment['end_ms'] - segment['start_ms']
        if abs(info.duration_ms - expected) > 1000 or identity(source) != before:
            raise ValueError('Derived video duration or original source changed')
        if original_video and original_video['path'] == target.relative_to(layout.root).as_posix():
            recovered = artifact(layout.root, temporary)
            if any(recovered[field] != original_video[field] for field in ('size_bytes', 'sha256')):
                raise ValueError('Recovered video differs from the immutable original clip receipt')
        verify_before_submission()
        temporary.replace(target)
        value = publish_disposition(folder, decision, video_present=True) | {'video': artifact(layout.root, target),
            'source_mapping': {'source_start_ms': segment['start_ms'], 'source_end_ms': segment['end_ms'],
                               'derived_start_ms': 0, 'derived_end_ms': info.duration_ms}}
        atomic_json(manifest, value)
        return value
    finally:
        temporary.unlink(missing_ok=True)
