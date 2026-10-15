"""Video-only operation descriptions, distinct from static scene observations."""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
    objects: list[str]
    category: Literal['computer', 'phone', 'organizing', 'cleaning', 'walking', 'waiting', 'other']

    @model_validator(mode='after')
    def require_target_for_object_work(self):
        if not self.objects and self.category not in {'walking', 'waiting'}:
            raise ValueError('Object-directed work requires a visible object')
        return self


class InvalidStepUnderstanding(ValueError):
    """A received model response failed its structure or evidence contract."""


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
    fields = {k: deepcopy(payload[k]) for k in model.model_fields if k in payload}
    adjustments = []
    work_candidates = fields.get('work_activities', [])
    for number, step in enumerate(work_candidates if isinstance(work_candidates, list) else []):
        if not isinstance(step, dict):
            continue
        missing = [key for key in ('basis', 'outcome') if key not in step]
        if missing:
            if 'basis' in missing:
                step['basis'] = 'uncertain'
            step['outcome'] = 'incomplete_or_uncertain'
            adjustments.append({'collection': 'work_activities', 'index': number, 'missing_fields': missing,
                'reason': '非实验活动缺少判断依据或结果状态，保守记为未确认，不补造已完成结论',
                'raw_response_unchanged': True})
    try:
        parsed = model.model_validate(fields).model_dump(mode='json')
    except ValidationError as exc:
        raise InvalidStepUnderstanding(str(exc)) from exc
    steps = parsed.pop('experiment_steps')
    work = parsed.pop('work_activities', [])
    if parsed['activity_observed'] == 'inactive' and steps:
        raise InvalidStepUnderstanding('Inactive scene cannot contain experiment steps')
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
        raise InvalidStepUnderstanding('Video-only steps cannot cite speech')
    frames = {frame['frame_id']: frame for frame in images}
    for collection, rows in [('experiment_steps', steps), ('work_activities', work)]:
        for number, step in enumerate(rows):
            # Unknown references and wholly unsupported intervals remain errors.
            # Do not stretch action times to accommodate unrelated context frames.
            if not set(step['frame_ids']) <= frames.keys():
                continue
            inside = [identifier for identifier in step['frame_ids']
                      if step['start_ms']-1 <= frames[identifier]['local_ms'] <= step['end_ms']+1]
            outside = [identifier for identifier in step['frame_ids'] if identifier not in inside]
            if inside and outside:
                step.update(frame_ids=inside, basis='uncertain', outcome='incomplete_or_uncertain')
                adjustments.append({'collection': collection, 'index': number,
                    'discarded_out_of_interval_frame_ids': outside, 'accepted_frame_ids': inside,
                    'action_times_unchanged': True, 'raw_response_unchanged': True,
                    'reason': '剔除动作时间外的上下文引用；剩余画面不足以确认完整动作及结果，保守记为未确认'})
    parsed['uncertainties'].extend(row['reason'] for row in adjustments)
    base = dict(parsed, steps=[{k: s[k] for k in SceneStep.model_fields} for s in [*steps, *work]])
    try:
        validated = validate_understanding(base, images, [], start_ms, end_ms, origin_us, clock)
    except (ValueError, KeyError, TypeError) as exc:
        raise InvalidStepUnderstanding(str(exc)) from exc
    for step, normalized in zip([*steps, *work], validated['steps'], strict=True):
        step.update(normalized)
        step.update(start_us=capture_us(clock, step['start_ms']), end_us=capture_us(clock, step['end_ms']))
    if model is AllDayUnderstanding:
        validated.update(activity_kind=parsed['activity_kind'], observability=parsed['observability'],
                         work_activities=work, understanding_schema='visioncortex-all-day-understanding/1')
    if normalization:
        validated['conservative_normalization'] = normalization
    if adjustments:
        validated['step_normalizations'] = adjustments
    return {**validated, 'experiment_steps': steps, 'step_schema': 'visioncortex-experiment-step/1',
            'content_source': 'video_only', 'human_reviewed': False, 'accuracy': 'NOT_PROVEN'}


def repair_window(config, analyzer, root, metadata, image_paths, request_path, result_path, origin_us, clock):
    """One fixed-template correction of a known received invalid response.

    The original prompt and paid key stay unchanged. Unknown submissions and
    provider/IO errors cannot enter this path. Every correction has its own exact
    identity, input, response and accounting; retries reuse that same unit.
    """
    from . import scene_requests
    from .device_day_call_templates import CLIP_VALIDATION_REPAIR
    from .device_day_contract import artifact, atomic_json, digest, read_json, verify_artifact

    original_input, original_result = read_json(request_path), read_json(result_path)
    raw = original_result.get('model_result') or {}
    if (original_input.get('prompt') != PROMPT or original_input.get('metadata') != metadata
            or original_input.get('input_key') != digest([metadata, PROMPT])
            or original_result.get('input_key') != original_input['input_key']
            or raw.get('status') != 'completed'
            or not all(verify_artifact(root, frame) for frame in metadata['frames'])):
        raise ValueError('Correction requires intact original video input and a received response')
    def validator(value):
        return validate(value, metadata['frames'], metadata['start_ms'], metadata['end_ms'],
                        origin_us, clock, require_all_day=True)
    try:
        validator(raw)
    except InvalidStepUnderstanding as exc:
        problem = str(exc)[:1000]
    else:
        raise ValueError('Valid observations do not require a paid correction')
    parent_key = scene_requests.request_key(config, PROMPT, metadata)
    context = {'schema_version': 'visioncortex-step-validation-repair/1',
        'parent_request_key': parent_key, 'parent_response_digest': digest(raw), 'validation_error': problem,
        'previous_observations': {key: raw[key] for key in AllDayUnderstanding.model_fields if key in raw}}
    correction_metadata = metadata | {'validation_repair': context}
    folder = request_path.parent/'ValidationRepair'/digest([parent_key, digest(raw), CLIP_VALIDATION_REPAIR.receipt()])
    source = {'original_input': artifact(root, request_path), 'original_result': artifact(root, result_path),
              'validation_error': problem, 'maximum_corrections': 1, 'originals_unchanged': True}
    pointer = folder/'RepairSource.json'
    if pointer.is_file() and read_json(pointer) != source:
        atomic_json(folder/'History'/('Source'+digest(read_json(pointer))+'.json'), read_json(pointer))
    atomic_json(pointer, source)
    repaired_raw, parsed, reused = scene_requests.window(config, analyzer, CLIP_VALIDATION_REPAIR.prompt,
        correction_metadata, image_paths, folder/'Input.json', folder/'Result.json', validator)
    parsed['validation_repair'] = {'parent_request_key': parent_key, 'parent_response_digest': digest(raw),
        'source': artifact(root, pointer), 'maximum_corrections': 1, 'originals_unchanged': True,
        'original_additional_usage': scene_requests.additional_usage(raw, original_result.get('response_cache_reused', False))}
    return repaired_raw, parsed, reused, folder/'Input.json', folder/'Result.json'


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
