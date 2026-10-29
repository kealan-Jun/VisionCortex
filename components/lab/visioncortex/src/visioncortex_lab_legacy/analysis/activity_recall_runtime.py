"""Bounded local-model review of unselected source intervals, within CV only."""
from contextlib import closing
from copy import deepcopy
import math
from pathlib import Path

from .activity_recall import RECIPE, observations, selected_runs


def recover_actor_crop(image, previous, objects, settings, infer, *, confidence, maximum_gap, actor_fallback=None):
    """Re-detect a missed hand on this frame; never propagate an old label."""
    from hashlib import sha256
    from .activity_recall import ACTORS
    height, width = image.shape[:2]
    recovered, receipts = [], []
    if len(objects) < 2:
        return recovered, receipts
    for actor in previous[:2]:
        a, b, c, d = actor['xyxy_norm']
        cx, cy = (a+c)/2, (b+d)/2
        # Include the existing tracker's .10 source-coordinate association
        # radius plus hand extent, rather than clipping a hand that moved
        # between adjacent observations.
        half_w, half_h = max(.16, c-a), max(.16, d-b)
        x1, x2 = max(0, int((cx-half_w)*width)), min(width, math.ceil((cx+half_w)*width))
        y1, y2 = max(0, int((cy-half_h)*height)), min(height, math.ceil((cy+half_h)*height))
        if x2-x1 < 32 or y2-y1 < 32:
            continue
        crop = image[y1:y2, x1:x2]
        crop_settings = dict(settings, grounded_actor_maximum_box_area_norm=.85)
        boxes, receipt = infer(crop, crop_settings)
        if receipt.get('status') != 'executed':
            raise ValueError('Local recall actor crop did not complete')
        def mapped_actors(boxes):
            mapped = []
            for box in boxes:
                if box['class_name'] not in ACTORS:
                    continue
                q, r, s, t = box['xyxy_norm']
                mapped.append(box | {'xyxy_norm': [(x1+q*(x2-x1))/width, (y1+r*(y2-y1))/height,
                    (x1+s*(x2-x1))/width, (y1+t*(y2-y1))/height], 'local_recall_origin': 'current_frame_actor_crop'})
            return observations([*objects, *mapped], confidence=confidence, maximum_gap=maximum_gap)[0]
        actors = mapped_actors(boxes)
        fallback_receipt = None
        if not actors and actor_fallback is not None:
            boxes, fallback_receipt = actor_fallback(crop, ACTORS, crop_settings)
            if fallback_receipt.get('status') != 'executed':
                raise ValueError('Local recall grounded actor crop did not complete')
            actors = mapped_actors(boxes)
        recovered.extend(actors)
        receipts.append({'invocation': receipt, 'source_crop_xyxy_pixels': [x1, y1, x2, y2],
                         'grounding_dino_invocation': fallback_receipt,
                         'crop_pixels_sha256': sha256(crop.tobytes()).hexdigest(),
                         'accepted_actor_count': len(actors), 'prior_label_propagated': False})
        if actors:
            break
    return recovered, receipts


def options(config):
    value = config.get('device_day', {}).get('local_activity_recall') or {}
    if not isinstance(value, dict) or type(value.get('enabled', False)) is not bool:
        raise ValueError('local_activity_recall.enabled must be a boolean')
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
    if (not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                for t in (start, end)) or not 0 <= start < end
            or info is not None and end > info.duration_ms+1e-6):
        raise ValueError('Local activity recall requires a real available source window')
    model = config.get('models', {}).get('open_vocabulary_key_frame') or {}
    import re
    if (not model.get('enabled') or not re.fullmatch('[a-f0-9]{64}', str(source_ref.get('sha256', '')))
            or type(source_ref.get('size_bytes')) is not int or source_ref['size_bytes'] <= 0):
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
            def checkpoint(status):
                atomic_json(path, {'identity': identity, 'samples': rows,
                                   'checksum': digest(rows), 'status': status})
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
                        if (native.source_path.resolve() != Path(view.video).resolve()
                                or native.source_size_bytes != source_ref['size_bytes']):
                            raise ValueError('Local review native frame belongs to a different sealed source')
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
                        if invocation.get('status') != 'executed':
                            raise ValueError('Local recall model invocation did not complete')
                        actors, objects = observations(boxes, confidence=setting['actor_confidence'],
                            maximum_gap=setting['laboratory_context_gap_norm'])
                        calls = {'yolo_world': invocation}
                        if not actors and fallback.get('enabled'):
                            extra, invocation = actor_fallback(image, {'hand', 'gloved_hand'}, settings)
                            if invocation.get('status') != 'executed':
                                raise ValueError('Local recall actor recovery did not complete')
                            boxes = boxes + extra
                            calls['grounding_dino_actor_recovery'] = invocation
                            actors, objects = observations(boxes, confidence=setting['actor_confidence'],
                                maximum_gap=setting['laboratory_context_gap_norm'])
                        if (not actors and rows and rows[-1].get('actors')
                                and 0 < row['local_ms']-rows[-1]['local_ms'] <= period*1.1):
                            actors, crop_receipts = recover_actor_crop(image, rows[-1]['actors'], objects,
                                settings, infer, confidence=setting['actor_confidence'],
                                maximum_gap=setting['laboratory_context_gap_norm'],
                                actor_fallback=actor_fallback if fallback.get('enabled') else None)
                            calls['yolo_world_current_frame_actor_crops'] = crop_receipts
                        row.update(actors=actors, objects=objects, model_receipts=calls,
                                   status='completed', physical_action_confirmed=False)
                    rows.append(row)
                    # Bound NAS checkpoint writes and lost work to 16 local
                    # samples. Never copy source video or decoded images.
                    if len(rows) % 16 == 0:
                        checkpoint('partial')
            checkpoint('partial')
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
        # A completely observed working run between two existing fine-audited
        # process anchors closes only the sub-sample edge context. Unknown
        # samples or a missing adjacent process never authorize a bridge.
        if (len(selected) == 1 and decisions[0]['sample_count'] == len(rows)
                and any(abs(b-left) < 1e-6 for _, b in active)
                and any(abs(a-right) < 1e-6 for a, _ in active)):
            observed = selected[0]
            selected = [(left, right)]
            decisions[0].update(start_ms=left, end_ms=right,
                observed_start_ms=observed[0], observed_end_ms=observed[1],
                edge_context={'basis': 'adjacent_fine_process_continuity',
                              'ranges': [[left, observed[0]], [observed[1], right]],
                              'maximum_edge_ms': period*1.5})
        intervals.extend(selected)
        reports.append({'window_index': number, 'start_ms': left, 'end_ms': right,
            'samples': rows, 'decisions': decisions, 'selected_intervals': selected,
            'cache_identity': digest(identity), 'sample_coverage': 'complete'})
    return intervals, {'status': 'completed', 'recipe': RECIPE, 'sample_fps': setting['sample_fps'],
        'sample_count': sum(len(r['samples']) for r in reports), 'windows': reports,
        'scope': 'local_lab_work_video_recall_not_action_confirmation',
        'paid_requests': 0, 'physical_action_confirmed': False, 'formal_events_mutated': False}
