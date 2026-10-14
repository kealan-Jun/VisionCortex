"""Bounded native-PTS sampling, paid-work checkpoints and explicit coverage."""
import math
from fractions import Fraction

from .device_day_contract import artifact, atomic_bytes, atomic_json, digest, media_interval_name, media_time_name, safe_child
from .device_day_coverage import apply_plan_reviews, frame_coverage, limits, manifest, request_plan, shadow_episodes
from .media_time import capture_us


def enabled(settings, recording):
    cutoff = settings.get('understanding_coverage_since_us')
    return bool(cutoff and recording['recording_start_us'] >= cutoff)


class UnderstandingPlanPaused(ValueError):
    """An anomalous plan remains retained for an explicit, auditable disposition."""
    reason = 'anomalous_inactive_request_plan'

    def __init__(self, plan_ref, partial_receipt=None):
        super().__init__('Anomalous inactive request plan held before paid submission; original task and coverage retained')
        self.plan_ref = plan_ref
        self.partial_receipt = partial_receipt


class UnderstandingCheckpointYield(Exception):
    """A saved paid window yields its queue turn, never discarding its receipt."""
    reason = 'understanding_request_quantum'

    def __init__(self, plan_ref, completed_new_requests, partial_receipt=None):
        super().__init__('Completed request quantum saved; remaining windows are resumable')
        self.plan_ref, self.completed_new_requests = plan_ref, completed_new_requests
        self.partial_receipt = partial_receipt


def frame_windows(source, start, end, active, settings, *, required_times=(), details=None, pixels=True):
    """Select on native decoded-frame PTS; split capacity without dropping a tail.

    Decoding starts at the source origin so its frame ordinal joins the native
    ffprobe ledger. Missing or ambiguous identities fail closed. Baseline and
    event sampling retain their independent cadence; action/boundary samples
    and a bounded overlap are additional inputs, never a replacement baseline.
    """
    import cv2
    from .source_frames import SourceFrameTrace
    bound = limits(settings)
    maximum, window_ms = bound['maximum_frames'], bound['maximum_window_ms']
    spacing = 1000 / bound['active_fps'] if active else bound['inactive_interval_ms']
    trace = SourceFrameTrace(source, 0, end, bound['active_fps'])
    if trace.failure or not trace.time_base or not trace.positions:
        raise ValueError('Understanding native source PTS unavailable')
    native = {}
    for rows in trace.positions.values():
        for pts, index in rows:
            if index is None or index in native:
                raise ValueError('Understanding native frame identity is ambiguous')
            native[index] = (float(pts * Fraction(trace.time_base) * 1000) - trace.origin_ms, pts)
    within = [(index, row[0]) for index, row in native.items() if start <= row[0] < end]
    if not within:
        raise ValueError('Understanding interval has no native frames')
    mandatory = set()
    required_times = sorted(set(required_times))
    for stamp in required_times:
        if not math.isfinite(stamp) or not start <= stamp < end:
            raise ValueError('Required action frame is outside the source interval')
        mandatory.add(min(within, key=lambda row: abs(row[1] - stamp))[0])
    details = details if details is not None else {}
    capture = (cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        if pixels else None)
    if pixels and not capture.isOpened():
        capture.release()
        raise ValueError('Understanding source could not be opened')
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if pixels else len(native)
    previous_time, ordinal, left, next_sample, next_event = -1.0, 0, start, start, start
    pending, last = [], None
    initial = trace.stat
    def unchanged():
        current = source.stat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(current, field) != getattr(initial, field) for field in fields):
            raise ValueError('Understanding source changed during native frame selection')
    def overlap(rows, stamp):
        count = bound['overlap_frames']
        return [row for row in rows[-count:] if stamp - row[1] < window_ms] if count else []
    try:
        while True:
            ok, frame = capture.read() if pixels else (ordinal in native, None)
            if not ok:
                if expected > 0 and ordinal < expected:
                    raise ValueError('Video decoder ended before the declared frame count')
                break
            if ordinal not in native:
                raise ValueError('Decoded understanding frame has no native PTS identity')
            stamp, pts = native[ordinal]
            ordinal += 1
            if not math.isfinite(stamp) or stamp <= previous_time:
                raise ValueError('Video decoder did not provide increasing frame timestamps')
            previous_time = stamp
            if stamp < start:
                continue
            if stamp >= end:
                break
            if pending and stamp - left >= window_ms:
                unchanged()
                yield left, stamp, pending
                pending = overlap(pending, stamp)
                left = pending[0][1] if pending else stamp
            width = min(768, frame.shape[1]) if pixels else None
            reasons = []
            if stamp >= next_sample:
                reasons.append('experiment_baseline' if active else 'scene_interval_sample')
                next_sample += (math.floor((stamp - next_sample) / spacing) + 1) * spacing
            if active and any(abs(stamp - t) <= bound['event_radius_ms'] for t in required_times) and stamp >= next_event:
                reasons.append('action_event_neighborhood')
                next_event = stamp + 1000 / bound['event_fps'] - 1e-6
            if ordinal - 1 in mandatory:
                reasons.append('required_action_frame')
            if last is None:
                reasons.append('segment_start_boundary')
            item = (ordinal - 1, stamp, cv2.resize(frame, (width, round(frame.shape[0]*width/frame.shape[1]))) if pixels else None)
            last = item
            details[item[0]] = {'source_pts': pts, 'source_time_base': trace.time_base,
                                'selection_reasons': reasons}
            if reasons:
                if len(pending) >= maximum:
                    unchanged()
                    yield left, stamp, pending
                    pending = overlap(pending, stamp)
                    left = pending[0][1] if pending else stamp
                pending.append(item)
        if last is not None and (not pending or pending[-1][0] != last[0]):
            details[last[0]]['selection_reasons'].append('segment_end_boundary')
            if len(pending) >= maximum:
                unchanged()
                yield left, last[1], pending
                pending = overlap(pending, last[1])
                left = pending[0][1] if pending else last[1]
            pending.append(last)
        if pending:
            unchanged()
            yield left, end, pending
        elif last is None:
            raise ValueError('Understanding interval has no decoded frames')
    finally:
        if capture is not None:
            capture.release()


def understand(backend, layout, recording, vision, context, key):
    import cv2
    from .device_day_models import SCENE_PROMPT, validate_understanding
    from .mllm import ArkAnalyzer
    from . import scene_requests
    from . import device_day_steps
    from .device_day_activity import phase
    from .device_day_window_inputs import selections, saved_metadata, decode_selected
    structured_steps = device_day_steps.enabled(backend.settings, recording)
    prompt = SCENE_PROMPT + '\n本任务只理解视频画面，不提供或引用录音转写。coverage给出实际输入覆盖；每帧文字简短，重复场景可简述。'
    if structured_steps:
        prompt = device_day_steps.PROMPT
    with phase('visual_request_planning_seconds'):
        plan = apply_plan_reviews(request_plan(vision['segments'], context, backend.settings),
                                  backend.settings, recording, vision['key'])
        audit = manifest(recording, vision, plan, shadow_episodes(vision['segments'], context))
        audit_path = layout.understanding/'RequestPlans'/recording['recording_id']/key/'SamplingCoverage.json'
        atomic_json(audit_path, audit)
    audit_ref = layout.relative(audit_path)
    held_segments = {hold['segment_id'] for hold in plan['holds']}
    runnable = [segment for segment in vision['segments'] if segment['segment_id'] not in held_segments]
    if not runnable and held_segments:
        raise UnderstandingPlanPaused(audit_ref)
    # A long unrelated inactive tail cannot prevent the same recording's
    # experiment evidence from advancing. Holds remain per source segment;
    # the recording remains incomplete until every hold has a disposition.
    runnable.sort(key=lambda segment: segment['activity'] != 'active')
    if held_segments:
        audit['status'] = 'partial_with_segment_holds'
        atomic_json(audit_path, audit)
    analyzer = ArkAnalyzer(backend.config)
    clock = vision.get('clock_mapping') or {'origin_us': recording['recording_start_us']}
    artifacts, meanings = [], []
    new_requests = 0
    quantum = limits(backend.settings)['requests_per_turn']
    def partial_receipt(current_segment=None, current_results=None):
        completed = list(meanings)
        if current_results:
            completed.append({'segment_id': current_segment['segment_id'], 'activity': current_segment['activity'],
                'mode': 'event_preserving_native_pts_sampling' if current_segment['activity'] == 'active' else 'interval_sampling',
                'status': 'partial', 'content_source': 'video_only', 'evidence_status': 'PARTIAL_EVIDENCE',
                'physical_action_confirmed': False, 'windows': list(current_results)})
        return {'vision_key': vision['key'], 'context_digest': None, 'content_source': 'video_only',
            'sampling_coverage_manifest': audit_ref, 'understandings': completed,
            'held_segment_ids': sorted(held_segments),
            'artifacts': [*artifacts, artifact(layout.root, audit_path)]}
    try:
        for segment in runnable:
            active = segment['activity'] == 'active'
            source = safe_child(layout.root, segment['source_ref']['path'])
            from .device_day_content_paths import analysis_folder
            folder = analysis_folder(layout, segment, key)
            results = []
            previously_selected = set()
            selected_windows = selections(source, segment, backend.settings, recording, vision['key'],
                folder/'SelectedWindows.json')
            artifacts.append(artifact(layout.root, folder/'SelectedWindows.json'))
            frames_iter = iter(selected_windows)
            try:
                for window_index, selected in enumerate(frames_iter):
                    review = next(row.get('review') for row in plan['segments'] if row['segment_id'] == segment['segment_id'])
                    if review and window_index >= review['maximum_requests']:
                        audit.update(status='bounded_review_limit_exhausted')
                        atomic_json(audit_path, audit)
                        raise UnderstandingPlanPaused(audit_ref, partial_receipt(segment, results))
                    left, right = selected['start_ms'], selected['end_ms']
                    directory = folder / media_interval_name(left, right)
                    request_path, result_path = directory/'Input.json', directory/'Result.json'
                    metadata = saved_metadata(layout, request_path, prompt, segment, selected, clock,
                        previously_selected, recording['recording_id'])
                    if quantum and new_requests >= quantum:
                        audit.update(status='scheduled_checkpoint', actual_request_coverage='validated_selected_frames_only')
                        atomic_json(audit_path, audit)
                        raise UnderstandingCheckpointYield(audit_ref, new_requests, partial_receipt(segment, results))
                    if metadata is None:
                        images = []
                        frame_details = {f['source_frame_index']: f for f in selected['frames']}
                        for ordinal, stamp, frame, pixel_hash in decode_selected(source, selected):
                            path = directory / 'SceneFrames' / f'{media_time_name(stamp)}.jpg'
                            ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            if not ok:
                                raise ValueError('Understanding frame encoding failed')
                            atomic_bytes(path, data.tobytes())
                            ref = artifact(layout.root, path)
                            selection = dict(frame_details[ordinal])
                            selection['selection_reasons'] = [*selection['selection_reasons'],
                                *(['window_overlap_context'] if ordinal in previously_selected else [])]
                            previously_selected.add(ordinal)
                            images.append({**ref, 'frame_id': 'frame-' + digest([recording['recording_id'], ref['sha256'], stamp])[:24],
                                'frame_kind': 'scene_sample', 'local_ms': stamp, 'capture_us': capture_us(clock, stamp),
                                'source_frame_index': ordinal, 'source_path': layout.relative(source),
                                'source_time_basis': 'exact_native_decoder_pts',
                                **selection, 'decoded_pixels_sha256': pixel_hash,
                                'pixel_time_binding': 'exact_decoder_pts_before_scale', 'native_pts_to_pixels_verified': True})
                        coverage = frame_coverage(images, left, right, active, backend.settings)
                        metadata = {'schema_version': 'visioncortex-device-day/1', 'segment_id': segment['segment_id'],
                            'activity': segment['activity'], 'start_ms': left, 'end_ms': right, 'frames': images,
                            'comments': [], 'protocol': None, 'source_ref': segment['source_ref'],
                            'content_source': 'video_only', 'coverage': coverage, 'physical_action_confirmed': False}
                    else:
                        images = metadata['frames']
                        coverage = metadata['coverage']
                        previously_selected.update(f['source_frame_index'] for f in images)
                    request = {'request_key': scene_requests.request_key(backend.config, prompt, metadata),
                        'segment_id': segment['segment_id'], 'source_ref': segment['source_ref'],
                        'selected_frames': [{k: f[k] for k in ('frame_id', 'local_ms', 'source_pts', 'source_time_base', 'selection_reasons')}
                                            for f in images], 'coverage': coverage,
                        'status': 'planned', 'input': layout.relative(request_path),
                        'model_receipt': layout.relative(result_path), 'result_identity': None}
                    audit['requests'].append(request)
                    atomic_json(audit_path, audit)
                    def validate(raw):
                        return (device_day_steps.validate(raw, images, left, right, recording['recording_start_us'], clock)
                                if structured_steps else validate_understanding(raw, images, [], left, right, recording['recording_start_us'], clock))
                    request['status'] = 'submission_or_cache_lookup_started'
                    atomic_json(audit_path, audit)
                    try:
                        with phase('visual_request_work_seconds'):
                            raw, parsed, reused = scene_requests.window(backend.config, analyzer, prompt, metadata,
                                [(f['frame_id'], safe_child(layout.root, f['path'])) for f in images], request_path, result_path, validate)
                    except Exception as exc:
                        from .multimodal_usage import SubmittedUnknown
                        request['status'] = ('submitted_unknown' if isinstance(exc, SubmittedUnknown) else
                                             'no_validated_result; inspect_durable_transport_receipt')
                        audit['status'] = 'partial'
                        atomic_json(audit_path, audit)
                        raise
                    request.update(status='validated_return', result_identity=digest(raw), response_cache_reused=reused)
                    audit['actual_request_coverage'] = 'validated_selected_frames_partial'
                    new_requests += int(not reused)
                    atomic_json(audit_path, audit)
                    for step in parsed['steps']:
                        step.update(start_us=capture_us(clock, step['start_ms']), end_us=capture_us(clock, step['end_ms']))
                    results.append({'start_ms': left, 'end_ms': right, **parsed, 'coverage': coverage,
                        'content_source': 'video_only', 'source_frames': images, 'response_cache_reused': reused,
                        'model_receipt': layout.relative(result_path), 'input': layout.relative(request_path), 'usage': raw.get('usage'),
                        'additional_usage': scene_requests.additional_usage(raw, reused)})
                    artifacts.extend(artifact(layout.root, p) for p in (request_path, result_path))
                    artifacts.extend({k: f[k] for k in ('path', 'size_bytes', 'sha256')} for f in images)
            finally:
                if hasattr(frames_iter, 'close'):
                    frames_iter.close()
            meanings.append({'segment_id': segment['segment_id'], 'activity': segment['activity'],
                'mode': 'event_preserving_native_pts_sampling' if active else 'interval_sampling', 'status': 'completed',
                'content_source': 'video_only', 'evidence_status': 'PARTIAL_EVIDENCE',
                'physical_action_confirmed': False, 'windows': results})
    finally:
        analyzer.close()
    if held_segments:
        audit.update(status='waiting_for_plan_review', actual_request_coverage='validated_selected_frames_partial')
        atomic_json(audit_path, audit)
        raise UnderstandingPlanPaused(audit_ref, partial_receipt=partial_receipt())
    audit.update(status='completed_selected_frame_delivery', actual_request_coverage='validated_selected_frames_only')
    atomic_json(audit_path, audit)
    artifacts.append(artifact(layout.root, audit_path))
    return {'vision_key': vision['key'], 'context_digest': None, 'content_source': 'video_only',
            'sampling_coverage_manifest': audit_ref, 'understandings': meanings, 'artifacts': artifacts}
