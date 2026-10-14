"""Project already validated device indexes without reopening media or models."""

from datetime import date, datetime, timedelta, timezone

from ..device_day_contract import VERSION, digest
from ..media_time import capture_us


def physical_time(value):
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError('Device event requires a physical microsecond clock')
    return (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=value)).isoformat()


def device_index_events(index, archive_id, *, tenant=None):
    if (index.get('schema_version') != VERSION or index.get('archive') != archive_id
            or len(archive_id) < 12 or archive_id[10] != '_'):
        raise ValueError('Device index schema or archive identity mismatch')
    date.fromisoformat(archive_id[:10])
    camera = archive_id[11:]
    if not camera:
        raise ValueError('Device index has no capture device identity')
    # Receipt changes are semantically relevant; a publication clock is not.
    revision = digest({key: index.get(key) for key in ('schema_version', 'archive', 'recordings',
                       'segments', 'understandings', 'cross_view_links', 'aligned_experiments')})
    recordings = {row['recording_id']: row for row in index.get('recordings', [])}
    meanings = {row['segment_id']: row for row in index.get('understandings', [])}
    events = []

    def build(ident, kind, start, end, recording, payload, media=None):
        clock = (recording.get('processing') or {}).get('clock_mapping') or {}
        refs = [{'source': 'visioncortex', 'archive': archive_id, 'index_revision': revision,
                 'recording_id': recording['recording_id'], 'event_id': ident}]
        return {'schema_version': 'device-context-event/1', 'source': 'visioncortex',
                'source_instance': 'visioncortex-device-index', 'event_id': f'visioncortex:{archive_id}:{ident}',
                'revision': revision, 'tenant': tenant, 'type': kind, 'occurred_start': physical_time(start),
                'occurred_end': physical_time(end), 'received_at': None, 'processed_at': None,
                'capture_device_id': camera, 'instrument_id': None, 'wearer_id': None,
                'operator_id': None, 'actor_track_id': None, 'capture_id': recording['recording_id'],
                'capture_session_id': None, 'clock': {'basis': clock.get('basis'),
                    'mapping_version': clock.get('version'), 'uncertainty_us': clock.get('uncertainty_us')},
                'media': media or {'sources': recording.get('sources', [])},
                'associations': {role: {'id': None, 'state': 'unknown', 'evidence_refs': [], 'version': 1}
                                 for role in ('wearer', 'operator', 'instrument', 'experiment', 'sample', 'step')},
                'evidence_refs': refs, 'payload': payload | {'physical_action_verified': False,
                        'receipt_validation_basis': 'caller_verified_device_index'}, 'withdrawn': False}

    for recording in recordings.values():
        start, end = recording.get('start_us'), recording.get('end_us')
        if start is not None and end is not None and end < start:
            raise ValueError('Device recording interval is invalid')
        events.append(build(recording['recording_id'] + ':coverage', 'recording_coverage', start, end, recording,
                            {'capture_complete': recording.get('capture_complete'),
                             'capture_issues': recording.get('capture_issues', []),
                             'stages': recording.get('stages', {}), 'processing': recording.get('processing', {})}))
        transcription, audio = recording.get('transcription') or {}, recording.get('audio') or {}
        comments = transcription.get('comments') or []
        # No input and not-yet-processed are distinct source states. Neither is an empty successful transcript.
        events.append(build(recording['recording_id'] + ':audio', 'speech_availability', start, end, recording,
                            {'audio_status': audio.get('status', 'not_provided'),
                             'transcription_status': transcription.get('status'),
                             'outcome': transcription.get('outcome'), 'model_invocation': transcription.get('model_invocation'),
                             'transcript_file': transcription.get('transcript_file'),
                             'audio_artifacts': audio.get('artifacts', []), 'actual_comment_count': len(comments)}))
        if comments and transcription.get('status') != 'completed':
            raise ValueError('Unvalidated speech comments cannot enter personnel views')
        for comment in comments:
            left, right = comment.get('start_us'), comment.get('end_us')
            if type(left) is not int or type(right) is not int or right < left or not comment.get('transcript_path'):
                raise ValueError('Speech event has no verified clock or transcript reference')
            ident = recording['recording_id'] + ':speech:' + digest([left, right, comment['transcript_path']])[:24]
            events.append(build(ident, 'speech_observation', left, right, recording,
                                {'text': comment.get('text'), 'human_reviewed': False,
                                 'transcript_path': comment['transcript_path'], 'audio_artifacts': audio.get('artifacts', [])}))
    for segment in index.get('segments', []):
        recording = recordings.get(segment.get('recording_id'))
        if recording is None:
            raise ValueError('Video segment has no verified recording')
        start, end = segment.get('start_us'), segment.get('end_us')
        if type(start) is not int or type(end) is not int or end < start or not segment.get('source_ref'):
            raise ValueError('Video segment lacks its physical interval or source reference')
        frames = [*segment.get('key_frames', []), *segment.get('scene_frames', [])]
        media = {'source_ref': segment['source_ref'], 'start_us': start, 'end_us': end,
                 'start_ms': segment.get('start_ms'), 'end_ms': segment.get('end_ms'), 'frames': frames}
        events.append(build(segment['segment_id'] + ':candidate', 'video_observation', start, end, recording,
                            {'segment_id': segment['segment_id'], 'activity': segment.get('activity'),
                             'observation_kind': 'local_scan_candidate', 'review_status': 'unreviewed',
                             'understanding_status': meanings.get(segment['segment_id'], {}).get('status', 'not_processed')}, media))
        for window in meanings.get(segment['segment_id'], {}).get('windows', []):
            if not window.get('model_receipt') or not window.get('input'):
                raise ValueError('Understanding window has no actual model/input receipt reference')
            clock = (recording.get('processing') or {}).get('clock_mapping')
            if not clock:
                # Keep actual selected-frame clocks if an old index lacks its mapping.
                captured = [frame.get('capture_us') for frame in window.get('source_frames', [])]
                captured = [stamp for stamp in captured if type(stamp) is int]
                left, right = (min(captured), max(captured)) if captured else (None, None)
            else:
                left, right = capture_us(clock, window['start_ms']), capture_us(clock, window['end_ms'])
            ident = segment['segment_id'] + ':window:' + digest([window.get('start_ms'), window.get('end_ms'), window['input']])[:24]
            events.append(build(ident, 'video_observation', left, right, recording,
                                {'segment_id': segment['segment_id'], 'observation_kind': 'model_window',
                                 'window': window, 'review_status': 'unreviewed'}, media | {'source_frames': window.get('source_frames', [])}))
    return events, revision
