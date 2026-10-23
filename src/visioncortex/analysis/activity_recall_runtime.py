"""Bounded local-model review of unselected source intervals, within CV only."""
from contextlib import closing
from copy import deepcopy
import math
from pathlib import Path

from .activity_recall import RECIPE, observations, selected_runs


def options(config):
    value = config.get('device_day', {}).get('local_activity_recall') or {}
    if not value.get('enabled', False):
        return None
    result = {'sample_fps': value.get('sample_fps', 1.),
              'image_size': value.get('image_size', 1280),
              'maximum_frames': value.get('maximum_frames', 4096),
              'actor_confidence': value.get('actor_confidence', .25),
              'laboratory_context_gap_norm': value.get('laboratory_context_gap_norm', .12),
              'motion_threshold_norm': value.get('motion_threshold_norm', .01),
              'minimum_observations': value.get('minimum_observations', 3)}
    for name, lower, upper in [('sample_fps', .5, 4), ('image_size', 640, 1920),
            ('maximum_frames', 3, 16384), ('actor_confidence', .15, .9),
            ('laboratory_context_gap_norm', .01, .2), ('motion_threshold_norm', .002, .1),
            ('minimum_observations', 3, 20)]:
        number = result[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError('Invalid bounded local activity recall option: '+name)
    for name in ('image_size', 'maximum_frames', 'minimum_observations'):
        if type(result[name]) is not int:
            raise ValueError('Local activity recall requires integer '+name)
    return result


def _negative(active, start, end):
    previous = start
    for left, right in sorted(active):
        left, right = max(start, left), min(end, right)
        if left > previous:
            yield previous, left
        previous = max(previous, right)
    if previous < end:
        yield previous, end


def review(view, info, active, start, end, config, *, source_ref, cache_directory,
           infer=None, frames=None, actor_fallback=None, source_origin_ms=None):
    """All promoted ranges bind real decoded pixels and pinned local models.

    This creates video-context candidates only. It never changes formal action
    acceptance, creates action keyframes, calls a paid provider or discards an
    interval as no-value. Saved successful samples survive bounded retries.
    """
    from ..device_day_contract import atomic_json, digest, read_json
    from ..runtime_control import check_yield
    from ..stage_dependencies import stage_source_manifest
    from fractions import Fraction
    setting = options(config)
    if setting is None:
        return [], {'status': 'disabled', 'recipe': RECIPE}
    model = config.get('models', {}).get('open_vocabulary_key_frame') or {}
    if not model.get('enabled') or not source_ref.get('sha256') or not source_ref.get('size_bytes'):
        raise ValueError('Local activity recall requires pinned models and a sealed source')
    if infer is None:
        from ..coarse_recall import _yolo_world_detections
        infer = _yolo_world_detections
    if actor_fallback is None:
        from ..archive import _grounding_dino_key_frame_detections
        actor_fallback = _grounding_dino_key_frame_detections
    if frames is None:
        from ..video_io import iter_view_sampled_frames
        frames = iter_view_sampled_frames
    if source_origin_ms is None:
        import json
        from ..source_frames import _run
        metadata = json.loads(_run(['ffprobe', '-v', 'error', '-show_entries', 'format=start_time',
            '-of', 'json', str(view.video)], capture_output=True, check=True, timeout=20).stdout)
        source_origin_ms = float(metadata.get('format', {}).get('start_time', 0))*1000
    negatives = list(_negative(active, start, end))
    rates = [min(4., max(setting['sample_fps'], setting['minimum_observations']*1000/(right-left)))
             for left, right in negatives]
    expected = sum(math.ceil((right-left)*rate/1000-1e-8) for (left, right), rate in zip(negatives, rates, strict=True))
    if expected > setting['maximum_frames']:
        raise ValueError('Local activity recall exceeds the explicit frame budget')
    intervals, reports = [], []
    settings = deepcopy(model)
    fallback = settings.get('grounding_dino_fallback') or {}
    fallback.setdefault('prompt_map', {}).update({'gloved hand': 'gloved_hand', 'bare hand': 'hand'})
    settings['grounding_dino_fallback'] = fallback
    consumer_sources = stage_source_manifest('vision')
    for number, (left, right) in enumerate(negatives):
        rate = rates[number]
        period = 1000/rate
        identity = {'recipe': RECIPE, 'source': source_ref, 'view': view.view_id,
            'role': view.role.value, 'start_ms': left, 'end_ms': right,
            'options': setting, 'effective_sample_fps': rate, 'models': settings,
            'consumer_sources': consumer_sources}
        path = Path(cache_directory)/(digest(identity)+'.json')
        saved = read_json(path) if path.is_file() else {}
        if saved and (saved.get('identity') != identity or saved.get('checksum') != digest(saved.get('samples'))):
            raise ValueError('Local activity recall cache changed')
        rows = saved.get('samples', [])
        if saved.get('status') != 'completed':
            previous = {r['requested_ms']: r for r in rows if r.get('status') == 'completed'}
            rows = []
            receipt = {}
            with closing(frames(view, info, left, right, rate, setting['image_size'],
                                None, False, 2, receipt=receipt)) as decoded:
                for item in decoded:
                    check_yield(checkpoint=False)
                    _, requested, image = item
                    old = previous.get(requested)
                    if old is not None:
                        rows.append(old)
                        continue
                    native = getattr(item, 'source_frame', None)
                    row = {'requested_ms': requested, 'local_ms': requested, 'actors': [], 'objects': [],
                           'source_verified': False, 'status': 'source_unverified'}
                    if native is not None and native.status == 'resolved' and not native.terminal_frame_hold:
                        row['source_frame'] = native.model_dump(mode='json')
                        # Match the decoder's recorded BGR image, never a seek
                        # target or a repeated terminal frame.
                        from hashlib import sha256
                        if sha256(image.tobytes()).hexdigest() != native.decoded_pixels_sha256:
                            raise ValueError('Local review source pixels do not match native identity')
                        row['source_verified'] = True
                        row['local_ms'] = float(Fraction(native.time_base)*native.source_pts*1000)-source_origin_ms
                        if not start <= row['local_ms'] < end:
                            raise ValueError('Local review native PTS escaped its available source window')
                        boxes, invocation = infer(image, settings)
                        actors, objects = observations(boxes, confidence=setting['actor_confidence'],
                            maximum_gap=setting['laboratory_context_gap_norm'])
                        calls = {'yolo_world': invocation}
                        if not actors and fallback.get('enabled'):
                            extra, invocation = actor_fallback(image, {'hand', 'gloved_hand'}, settings)
                            boxes = boxes + extra
                            calls['grounding_dino_actor_recovery'] = invocation
                            actors, objects = observations(boxes, confidence=setting['actor_confidence'],
                                maximum_gap=setting['laboratory_context_gap_norm'])
                        row.update(actors=actors, objects=objects, model_receipts=calls,
                                   status='completed', physical_action_confirmed=False)
                    rows.append(row)
                    # Persist small sample results incrementally; source video
                    # and decoded images are never copied into this cache.
                    atomic_json(path, {'identity': identity, 'samples': rows,
                                       'checksum': digest(rows), 'status': 'partial'})
            required = math.floor((right-left)/period+1e-8)
            observed = [r['local_ms'] for r in rows]
            covered = (bool(observed) and observed[0] <= left+period
                and observed[-1] >= right-period*1.5
                and all(0 < b-a <= period*1.1 for a, b in zip(observed, observed[1:], strict=False)))
            if len(rows) < required or not covered or any(r['status'] != 'completed' for r in rows):
                raise ValueError('Local activity recall has incomplete source/model coverage')
            atomic_json(path, {'identity': identity, 'samples': rows,
                               'checksum': digest(rows), 'status': 'completed'})
        selected, decisions = selected_runs(rows, period_ms=period,
            motion_threshold=setting['motion_threshold_norm'], minimum_observations=setting['minimum_observations'])
        intervals.extend(selected)
        reports.append({'window_index': number, 'start_ms': left, 'end_ms': right,
            'samples': rows, 'decisions': decisions, 'selected_intervals': selected,
            'cache_identity': digest(identity), 'sample_coverage': 'complete'})
    return intervals, {'status': 'completed', 'recipe': RECIPE, 'sample_fps': setting['sample_fps'],
        'sample_count': sum(len(r['samples']) for r in reports), 'windows': reports,
        'scope': 'local_lab_work_video_recall_not_action_confirmation',
        'paid_requests': 0, 'physical_action_confirmed': False, 'formal_events_mutated': False}
