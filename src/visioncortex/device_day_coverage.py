"""Conservative request planning and inspectable coverage, never human truth."""
import math
import hashlib
import json
from pathlib import Path

from .device_day_contract import digest

POLICY = 'visioncortex-visual-request-plan/1'
MANIFEST = 'visioncortex-sampling-coverage/1'


def apply_plan_reviews(plan, settings, recording, vision_key):
    """An exact bounded decision resolves a hold without asserting stability."""
    for hold in plan['holds']:
        hold.update(required_action='review_continuous_source_or_authorize_bounded_processing',
                    review_target_seconds=900, automatic_paid_retry=False)
    reference = settings.get('understanding_plan_reviews')
    if reference is None:
        return plan
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or not isinstance(reference['path'], str) or not Path(reference['path']).is_absolute()):
        raise ValueError('Plan reviews require an absolute local path and sha256')
    from .local_storage import require_local_path
    path = require_local_path(Path(reference['path']))
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Plan review manifest exceeds 1 MiB')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('Plan review manifest checksum mismatch')
    manifest = json.loads(data)
    if manifest.get('schema_version') != 'visioncortex-plan-reviews/1':
        raise ValueError('Invalid plan review manifest schema')
    reviewed = set()
    plan['review_manifest'] = reference
    for review in manifest.get('entries', []):
        identity = (review.get('recording_id'), review.get('segment_id'))
        if identity in reviewed:
            raise ValueError('Duplicate plan review decision')
        reviewed.add(identity)
        if review.get('recording_id') != recording['recording_id']:
            continue
        if review.get('source_signature') != recording.get('source_signature') or review.get('vision_key') != vision_key:
            raise ValueError('Plan review source or visual identity changed')
        row = next((r for r in plan['segments'] if r['segment_id'] == review.get('segment_id')), None)
        if row is None or review.get('plan_identity') != digest({k: row[k] for k in (
                'segment_id', 'source_ref', 'start_ms', 'end_ms', 'estimated_requests_upper_bound')} | {'policy': plan['policy_version']}):
            raise ValueError('Plan review does not bind the current exact interval and plan')
        if (review.get('action') != 'execute_bounded' or review.get('paid_authorized') is not True
                or type(review.get('maximum_requests')) is not int
                or not row['estimated_requests_upper_bound'] <= review['maximum_requests'] <= 10000
                or not all(isinstance(review.get(k), str) and review[k] for k in ('reviewed_by', 'evidence'))):
            raise ValueError('Plan review requires explicit bounded authorization and review evidence')
        row.update(status='reviewed_bounded_execution', review=review)
        plan['holds'] = [h for h in plan['holds'] if h['segment_id'] != row['segment_id']]
    plan['status'] = 'held_for_plan_review' if plan['holds'] else 'planned'
    return plan


def limits(settings):
    value = {
        'maximum_frames': int(settings.get('understanding_frames_per_request', 48)),
        'maximum_window_ms': float(settings.get('understanding_window_seconds', 30)) * 1000,
        'active_fps': float(settings.get('understanding_active_fps', 4)),
        'event_fps': float(settings.get('understanding_event_fps', 10)),
        'event_radius_ms': float(settings.get('understanding_event_radius_ms', 1000)),
        'inactive_interval_ms': float(settings.get('inactive_sample_seconds', 1)) * 1000,
        'overlap_frames': int(settings.get('understanding_overlap_frames', 2)),
        'inactive_window_limit': int(settings.get('understanding_inactive_window_limit', 12)),
        'requests_per_turn': int(settings.get('understanding_requests_per_turn', 1)),
    }
    if (not all(math.isfinite(v) for v in value.values())
            or not 2 <= value['maximum_frames'] <= 96
            or not 0 < value['maximum_window_ms'] <= 60000
            or not 0 < value['active_fps'] <= value['event_fps'] <= 60
            or not 0 <= value['event_radius_ms'] <= 10000
            or not 0 < value['inactive_interval_ms'] <= 10000
            or not 0 <= value['overlap_frames'] < value['maximum_frames'] - 1
            or value['inactive_window_limit'] < 1 or value['requests_per_turn'] < 0):
        raise ValueError('Invalid bounded understanding request plan settings')
    return value


def value_assessment(segment, segments, context):
    """Keep activity semantics and visibility independent; protect known signals."""
    protection = []
    if segment['activity'] == 'active':
        protection.append('experiment_candidate')
    if segment.get('key_frames'):
        protection.append('selected_action_frames')
    semantic = segment.get('activity_semantics', 'undecided')
    if semantic in {'experiment_waiting', 'ordinary_valuable_activity'}:
        protection.append(semantic)
    if segment.get('experiment_context'):
        protection.append('experiment_context')
    # An inactive gap between two experiment candidates is not an established
    # idle episode. Preserve it as possible experimental waiting.
    if (any(s['activity'] == 'active' and s['end_ms'] <= segment['start_ms'] for s in segments)
            and any(s['activity'] == 'active' and s['start_ms'] >= segment['end_ms'] for s in segments)):
        protection.append('between_experiment_candidates')
        semantic = 'experiment_waiting'
    for field in ('comments', 'events', 'ocr_events', 'bindings', 'audio_events'):
        for event in (context or {}).get(field, []) or []:
            left, right = event.get('start_us'), event.get('end_us')
            if (left is not None and right is not None
                    and left < segment.get('end_us', 0)
                    and (right > segment.get('start_us', 0)
                         or right == left and segment.get('start_us', 0) <= left)):
                protection.append(field)
                break
    visibility = segment.get('visibility', 'unknown')
    observation = ('occluded_or_dark' if visibility == 'dark_or_obscured' else
                   'partially_observable' if visibility == 'sampled_images_available' else
                   visibility if visibility in {'observable', 'occluded', 'blurred', 'source_gap', 'time_uncertain'}
                   else 'unknown')
    if observation in {'occluded_or_dark', 'occluded', 'blurred', 'source_gap', 'time_uncertain'}:
        protection.append('observability_requires_review')
    return {'activity_semantics': semantic if segment['activity'] != 'active' else 'experiment_candidate',
            'observability': observation, 'protection_signals': sorted(set(protection)),
            'audio_observability': segment.get('audio_status', 'unknown'),
            'human_quality_validation': 'NOT_PROVEN'}


def shadow_episodes(segments, context):
    """Reuse sparse scene evidence only as a shadow candidate, never a filter."""
    episodes = []
    for segment in segments:
        assessment = value_assessment(segment, segments, context)
        scenes = sorted(segment.get('scene_frames', []), key=lambda f: f['local_ms'])
        hashes = [frame.get('sha256') for frame in scenes]
        signature = hashes[0] if len(hashes) >= 2 and hashes[0] and len(set(hashes)) == 1 else None
        candidate = bool(signature and segment['activity'] != 'active' and not assessment['protection_signals'])
        intervals = [{'source_ref': segment['source_ref'], 'recording_id': segment.get('recording_id'),
                      'segment_id': segment['segment_id'], 'start_ms': segment['start_ms'], 'end_ms': segment['end_ms']}]
        row = {'episode_id': 'shadow-' + digest(intervals)[:24], 'policy_version': POLICY,
               'camera_key': segment.get('camera_key'), 'capture_session_id': segment.get('capture_session_id'),
               'start_us': segment.get('start_us'), 'end_us': segment.get('end_us'),
               'source_intervals': intervals, 'scene_signature': signature,
               'assessment': assessment, 'status': 'shadow_candidate' if candidate else 'undecided',
               'check_method': 'existing_scene_artifact_exact_hash_comparison',
               'checked_frame_pts_ms': [f['local_ms'] for f in scenes],
               'maximum_check_gap_ms': max((b['local_ms'] - a['local_ms']
                   for a, b in zip(scenes, scenes[1:], strict=False)), default=None),
               'full_interval_checked': False, 'confirmed_stable': False,
               'paid_filter_applied': False, 'exit_reason': 'end_of_known_source_interval',
               'quality': 'NOT_PROVEN', 'missing_gate': 'continuous_original_video_annotation_and_short_action_recall'}
        previous = episodes[-1] if episodes else None
        if (previous and candidate and previous['status'] == 'shadow_candidate'
                and row['camera_key'] and row['capture_session_id']
                and previous['camera_key'] == row['camera_key']
                and previous['capture_session_id'] == row['capture_session_id']
                and previous['end_us'] is not None and previous['end_us'] == row['start_us']
                and previous['scene_signature'] == signature):
            previous['source_intervals'].extend(intervals)
            previous['end_us'] = row['end_us']
            previous['checked_frame_pts_ms'] = None  # Per-source coordinates are not a joined PTS axis.
            previous['maximum_check_gap_ms'] = None
            previous['episode_id'] = 'shadow-' + digest(previous['source_intervals'])[:24]
        else:
            episodes.append(row)
    return episodes


def request_plan(segments, context, settings):
    bound = limits(settings)
    rows, holds = [], []
    capacity = bound['maximum_frames'] - bound['overlap_frames']
    for segment in segments:
        assessment = value_assessment(segment, segments, context)
        active = segment['activity'] == 'active'
        duration = segment['end_ms'] - segment['start_ms']
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Understanding segment has invalid media interval')
        mandatory = len(segment.get('key_frames', []))
        # Event density is a conservative upper estimate. Exact selected native
        # PTS and actual request counts are recorded during bounded decoding.
        event_ms = min(duration, mandatory * 2 * bound['event_radius_ms'])
        frames = (math.ceil((duration - event_ms) * bound['active_fps'] / 1000
                            + event_ms * bound['event_fps'] / 1000) if active else
                  math.ceil(duration / bound['inactive_interval_ms'])) + mandatory + 2
        overlap_ms = bound['overlap_frames'] * (1000 / bound['active_fps'] if active else bound['inactive_interval_ms'])
        useful_window = max(1, bound['maximum_window_ms'] - overlap_ms)
        windows = max(math.ceil(duration / useful_window), math.ceil(frames / capacity))
        paused = (not active and not assessment['protection_signals']
                  and windows > bound['inactive_window_limit'])
        row = {'segment_id': segment['segment_id'], 'source_ref': segment['source_ref'],
               'start_ms': segment['start_ms'], 'end_ms': segment['end_ms'], 'assessment': assessment,
               'estimated_selected_frames_upper_bound': frames, 'estimated_requests_upper_bound': windows,
               'mandatory_action_frames': mandatory, 'status': 'held_for_plan_review' if paused else 'planned',
               'estimated_new_calls': None, 'estimated_tokens': None, 'estimated_cost': None,
               'cache_reuse': 'resolved_per_exact_request_identity_before_submission'}
        rows.append(row)
        if paused:
            holds.append({'segment_id': segment['segment_id'], 'reason': 'anomalous_inactive_request_plan',
                          'estimated_requests_upper_bound': windows, 'limit': bound['inactive_window_limit'],
                          'data_preserved': True, 'quality': 'NOT_PROVEN'})
    return {'policy_version': POLICY, 'limits': bound, 'segments': rows, 'holds': holds,
            'estimated_requests_upper_bound': sum(r['estimated_requests_upper_bound'] for r in rows),
            'status': 'held_for_plan_review' if holds else 'planned',
            'human_semantic_recall': 'NOT_PROVEN', 'paid_requests_authorized_by_this_plan': False}


def manifest(recording, vision, plan, episodes):
    return {'schema_version': MANIFEST, 'policy_version': POLICY,
            'recording_id': recording['recording_id'], 'camera_key': recording.get('camera_key'),
            'vision_key': vision['key'], 'source_intervals': [
                {k: s[k] for k in ('segment_id', 'source_ref', 'start_ms', 'end_ms')} for s in vision['segments']],
            'capture_coverage': vision.get('media_coverage'),
            'local_scan_coverage': [{'segment_id': s['segment_id'], 'coverage': s.get('sampling_coverage')}
                                    for s in vision['segments']],
            'request_plan': plan, 'shadow_episodes': episodes, 'requests': [],
            'status': plan['status'], 'actual_request_coverage': 'no_validated_results_recorded',
            'human_semantic_recall': 'NOT_PROVEN',
            'gaps': ['native_frame_selection_does_not_prove_model_attention_or_semantic_recall'],
            'necessary_region_visibility': 'NOT_PROVEN'}


def frame_coverage(images, left, right, active, settings):
    bound = limits(settings)
    times = [f['local_ms'] for f in images]
    return {'mode': 'event_preserving_native_pts_sampling' if active else 'interval_sampling',
            'start_ms': left, 'end_ms': right, 'input_frames': len(images),
            'first_frame_ms': times[0], 'last_frame_ms': times[-1],
            'first_frame_index': images[0]['source_frame_index'],
            'last_frame_index': images[-1]['source_frame_index'],
            'actual_selected_pts_ms': times,
            'maximum_internal_gap_ms': max((b - a for a, b in zip(times, times[1:], strict=False)), default=0),
            'leading_gap_ms': times[0] - left, 'trailing_gap_ms': right - times[-1],
            'baseline_sample_fps': bound['active_fps'] if active else 1000 / bound['inactive_interval_ms'],
            'event_sample_fps': bound['event_fps'] if active else None,
            'overlap_frames_reserved': bound['overlap_frames'], 'all_decoded_frames_submitted': False,
            'native_pts_verified': all(f.get('source_pts') is not None for f in images),
            'native_pts_to_pixels_verified': all(f.get('native_pts_to_pixels_verified') is True for f in images),
            'pixel_time_binding': ('exact_decoder_pts_before_scale' if all(f.get('native_pts_to_pixels_verified') is True for f in images)
                                   else 'sequential_decoder_ordinal_join; independent_packet_concordance_unverified'),
            'model_attention_to_every_frame_verified': False, 'necessary_region_visibility': 'NOT_PROVEN'}
