"""Video-only operation descriptions, distinct from static scene observations."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .device_day_call_templates import CLIP_UNDERSTANDING
from .device_day_models import FrameObservation, SceneStep, validate_understanding
from .media_time import capture_us


class ExperimentStep(SceneStep):
    action: str = Field(min_length=1, max_length=200)
    objects: list[str] = Field(min_length=1)
    before_state: str = Field(min_length=1, max_length=1000)
    after_state: str = Field(min_length=1, max_length=1000)
    visible_evidence: str = Field(min_length=1, max_length=1500)
    outcome: Literal['visible_change', 'incomplete_or_uncertain']


class StepUnderstanding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    summary: str = Field(min_length=1, max_length=5000)
    activity_observed: Literal['active', 'inactive', 'uncertain']
    experiment_steps: list[ExperimentStep]
    frame_observations: list[FrameObservation]
    uncertainties: list[str]


class WorkActivity(ExperimentStep):
    category: Literal['computer', 'phone', 'organizing', 'cleaning', 'walking', 'waiting', 'other']


class AllDayUnderstanding(StepUnderstanding):
    activity_kind: Literal['experiment', 'non_experiment', 'static', 'uncertain']
    observability: Literal['clear', 'partial', 'obscured']
    work_activities: list[WorkActivity]


PROMPT = CLIP_UNDERSTANDING.prompt


def enabled(settings, recording):
    cutoff = settings.get('experiment_steps_since_us')
    return bool(cutoff and recording['recording_start_us'] >= cutoff)


def validate(payload, images, start_ms, end_ms, origin_us, clock, *, require_all_day=False):
    model = AllDayUnderstanding if require_all_day or any(k in payload for k in AllDayUnderstanding.model_fields.keys() - StepUnderstanding.model_fields.keys()) else StepUnderstanding
    fields = {k: payload[k] for k in model.model_fields if k in payload}
    parsed = model.model_validate(fields).model_dump(mode='json')
    steps = parsed.pop('experiment_steps')
    work = parsed.pop('work_activities', [])
    if parsed['activity_observed'] == 'inactive' and steps:
        raise ValueError('Inactive scene cannot contain experiment steps')
    normalization = None
    if parsed.get('activity_kind') == 'static' and (steps or work or parsed['observability'] != 'clear'):
        # Keep the paid raw response intact. Conflicting labels cannot authorize
        # omission, but valid observations need not be discarded and billed again.
        normalization = {'model_activity_kind': 'static', 'effective_activity_kind': 'uncertain',
                         'reason': '静止标签与动作或可见性冲突，保守按不确定保留',
                         'raw_response_unchanged': True}
        parsed['activity_kind'] = 'uncertain'
        parsed['uncertainties'].append(normalization['reason'])
    if any(s['comment_ids'] for s in [*steps, *work]):
        raise ValueError('Video-only steps cannot cite speech')
    base = dict(parsed, steps=[{k: s[k] for k in SceneStep.model_fields} for s in [*steps, *work]])
    validated = validate_understanding(base, images, [], start_ms, end_ms, origin_us, clock)
    for step, normalized in zip([*steps, *work], validated['steps'], strict=True):
        step.update(normalized)
        step.update(start_us=capture_us(clock, step['start_ms']), end_us=capture_us(clock, step['end_ms']))
    if model is AllDayUnderstanding:
        validated.update(activity_kind=parsed['activity_kind'], observability=parsed['observability'],
                         work_activities=work, understanding_schema='visioncortex-all-day-understanding/1')
    if normalization:
        validated['conservative_normalization'] = normalization
    return {**validated, 'experiment_steps': steps, 'step_schema': 'visioncortex-experiment-step/1',
            'content_source': 'video_only', 'human_reviewed': False, 'accuracy': 'NOT_PROVEN'}


def reviewed_window(root, segment, clock):
    """Load an explicit, bounded user-requested reanalysis with intact evidence."""
    from .device_day_content import time_folder
    from .device_day_contract import read_json, safe_child
    from .device_day_verification import verify_artifact_cached
    path = root/'MultimodalUnderstanding'/time_folder(segment['start_us'], segment['end_us'])/'Analysis/StepReview.json'
    if not path.is_file():
        return None
    manifest = read_json(path)
    if (manifest.get('segment_id') != segment['segment_id'] or manifest.get('source_video') != segment['source_ref']
            or not all(verify_artifact_cached(root, manifest[k]) for k in ('input', 'model_receipt'))):
        raise ValueError('Step review identity or execution evidence changed')
    request = read_json(safe_child(root, manifest['input']['path']))
    result = read_json(safe_child(root, manifest['model_receipt']['path']))['model_result']
    metadata = request['metadata']
    if (metadata['segment_id'] != segment['segment_id'] or metadata['source_ref'] != segment['source_ref']
            or metadata.get('comments') or metadata.get('protocol') or result.get('status') != 'completed'
            or metadata['start_ms'] != segment['start_ms'] or metadata['end_ms'] != segment['end_ms']
            or not all(verify_artifact_cached(root, f) for f in metadata['frames'])):
        raise ValueError('Step review does not match its video interval and images')
    parsed = validate(result, metadata['frames'], segment['start_ms'], segment['end_ms'], clock['origin_us'], clock)
    return {**parsed, 'start_ms': segment['start_ms'], 'end_ms': segment['end_ms'],
            'input': manifest['input']['path'], 'model_receipt': manifest['model_receipt']['path'],
            'source_frames': metadata['frames'], 'coverage': metadata['coverage'],
            'usage': result.get('usage'), 'response_cache_reused': False, 'explicit_step_review': True}


def readable_meaning(meaning):
    """Do not relabel legacy captions as structured experimental steps."""
    from copy import deepcopy
    value = deepcopy(meaning)
    windows = (value or {}).get('windows', [])
    for window in windows:
        if 'experiment_steps' not in window:
            window['scene_and_behavior_observations'] = window.pop('steps', [])
    structured = bool(windows) and all('experiment_steps' in w for w in windows)
    inactive = bool(windows) and all(w.get('activity_observed') == 'inactive' for w in windows)
    steps = [s for w in windows for s in w.get('experiment_steps', [])] if structured else ([] if inactive else None)
    status = ('structured_model_output' if structured else 'no_operation_observed_in_supplied_frames' if inactive
              else 'legacy_observations_only' if windows else 'pending')
    return value, steps, status
