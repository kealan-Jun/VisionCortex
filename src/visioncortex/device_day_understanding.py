"""Bounded, complete frame delivery for new recorder-native understanding."""
import math

from .device_day_contract import artifact, atomic_bytes, atomic_json, digest, media_interval_name, media_time_name, read_json, safe_child
from .media_time import capture_us


def enabled(settings, recording):
    cutoff = settings.get('understanding_coverage_since_us')
    return bool(cutoff and recording['recording_start_us'] >= cutoff)


def frame_windows(source, start, end, active, settings):
    """Decode consecutively; never seek past or omit frames in an active span.

    OpenCV media timestamps are explicit decoder evidence, not verified native
    PTS. A decode discontinuity fails the stage rather than reporting coverage.
    """
    import cv2
    maximum = int(settings.get('understanding_frames_per_request', 48))
    seconds = float(settings.get('understanding_window_seconds', 30))
    spacing = float(settings.get('inactive_sample_seconds', 1)) * 1000
    if not 2 <= maximum <= 96 or not 0 < seconds <= 60 or not 0 < spacing <= 10000:
        raise ValueError('Invalid bounded understanding coverage settings')
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
    if not capture.isOpened():
        capture.release()
        raise ValueError('Understanding source could not be opened')
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    previous_time, ordinal, left, next_sample = -1.0, 0, start, start
    pending, last = [], None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if expected > 0 and ordinal < expected:
                    raise ValueError('Video decoder ended before the declared frame count')
                break
            stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            ordinal += 1
            if not math.isfinite(stamp) or stamp <= previous_time:
                raise ValueError('Video decoder did not provide increasing frame timestamps')
            previous_time = stamp
            if stamp < start:
                continue
            if stamp >= end:
                break
            if pending and (len(pending) >= maximum or stamp - left >= seconds * 1000):
                yield left, stamp, pending
                left, pending = stamp, []
            width = min(768, frame.shape[1])
            selected = active or stamp >= next_sample
            # Keep at most one unselected tail image so inactive sampling also
            # includes the end of each segment, including a sub-second tail.
            item = (ordinal - 1, stamp, cv2.resize(frame, (width, round(frame.shape[0]*width/frame.shape[1]))))
            last = item
            if selected:
                pending.append(item)
                next_sample = stamp + spacing
        if not active and last is not None and (not pending or pending[-1][0] != last[0]):
            pending.append(last)
        if pending:
            yield left, end, pending
        elif last is None:
            raise ValueError('Understanding interval has no decoded frames')
    finally:
        capture.release()


def understand(backend, layout, recording, vision, context, key):
    import cv2
    from .device_day_models import SCENE_PROMPT, validate_understanding
    from .mllm import ArkAnalyzer
    from . import device_day_semantic_cache as cache
    prompt = SCENE_PROMPT + '\n本任务只理解视频画面，不提供或引用录音转写。coverage给出实际输入覆盖；每帧文字简短，重复场景可简述。'
    analyzer = ArkAnalyzer(backend.config)
    clock = vision.get('clock_mapping') or {'origin_us': recording['recording_start_us']}
    artifacts, meanings = [], []
    try:
        for segment in vision['segments']:
            active = segment['activity'] == 'active'
            source = safe_child(layout.root, segment['source_ref']['path'])
            from .device_day_content_paths import analysis_folder
            folder = analysis_folder(layout, segment, key)
            results = []
            for left, right, frames in frame_windows(source, segment['start_ms'], segment['end_ms'], active, backend.settings):
                directory = folder / media_interval_name(left, right)
                images = []
                for ordinal, stamp, frame in frames:
                    path = directory / 'SceneFrames' / f'{media_time_name(stamp)}.jpg'
                    ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise ValueError('Understanding frame encoding failed')
                    atomic_bytes(path, data.tobytes())
                    ref = artifact(layout.root, path)
                    images.append({**ref, 'frame_id': 'frame-' + digest([ref['sha256'], stamp])[:24],
                        'frame_kind': 'scene_sample', 'local_ms': stamp, 'capture_us': capture_us(clock, stamp),
                        'source_frame_index': ordinal, 'source_path': layout.relative(source),
                        'source_time_basis': 'sequential_decoder_media_timestamp_native_pts_unverified'})
                gaps = [b['local_ms']-a['local_ms'] for a, b in zip(images, images[1:], strict=False)]
                coverage = {'mode': 'all_decoded_frames' if active else 'interval_sampling',
                    'start_ms': left, 'end_ms': right, 'input_frames': len(images),
                    'first_frame_ms': images[0]['local_ms'], 'last_frame_ms': images[-1]['local_ms'],
                    'first_frame_index': images[0]['source_frame_index'], 'last_frame_index': images[-1]['source_frame_index'],
                    'maximum_internal_gap_ms': max(gaps, default=0),
                    'sample_interval_ms': None if active else float(backend.settings.get('inactive_sample_seconds', 1))*1000,
                    'all_decoded_frames_submitted': active, 'native_pts_verified': False,
                    'model_attention_to_every_frame_verified': False}
                metadata = {'schema_version': 'visioncortex-device-day/1', 'segment_id': segment['segment_id'],
                    'activity': segment['activity'], 'start_ms': left, 'end_ms': right, 'frames': images,
                    'comments': [], 'protocol': None, 'source_ref': segment['source_ref'],
                    'content_source': 'video_only', 'coverage': coverage, 'physical_action_confirmed': False}
                input_key = digest([metadata, prompt])
                request_path, result_path = directory/'Input.json', directory/'Result.json'
                response_identity = cache.identity(backend.config, prompt, metadata)
                raw = None
                if request_path.is_file() and result_path.is_file():
                    previous, result = read_json(request_path), read_json(result_path)
                    if previous.get('input_key') == input_key == result.get('input_key'):
                        raw = result.get('model_result')
                atomic_json(request_path, {'input_key': input_key, 'prompt': prompt, 'metadata': metadata})
                raw = raw or cache.load(backend.config, response_identity)
                reused = raw is not None
                if raw is None:
                    raw = analyzer._call(prompt, metadata,
                        [(f['frame_id'], safe_child(layout.root, f['path'])) for f in images], max_images=len(images))
                atomic_json(result_path, {'input_key': input_key, 'model_result': raw, 'response_cache_reused': reused})
                if raw.get('status') != 'completed':
                    from .device_day_provider_gate import ProviderGate
                    ProviderGate(backend.config).record_failure(raw)
                    raise ValueError('Multimodal window failed; preceding windows remain published')
                parsed = validate_understanding(raw, images, [], left, right, recording['recording_start_us'], clock)
                if not reused:
                    cache.save(backend.config, response_identity, raw)
                for step in parsed['steps']:
                    step.update(start_us=capture_us(clock, step['start_ms']), end_us=capture_us(clock, step['end_ms']))
                results.append({'start_ms': left, 'end_ms': right, **parsed, 'coverage': coverage,
                    'content_source': 'video_only', 'source_frames': images, 'response_cache_reused': reused,
                    'model_receipt': layout.relative(result_path), 'input': layout.relative(request_path), 'usage': raw.get('usage')})
                artifacts.extend(artifact(layout.root, p) for p in (request_path, result_path))
                artifacts.extend({k: f[k] for k in ('path', 'size_bytes', 'sha256')} for f in images)
            meanings.append({'segment_id': segment['segment_id'], 'activity': segment['activity'],
                'mode': 'all_decoded_frames' if active else 'interval_sampling', 'status': 'completed',
                'content_source': 'video_only', 'evidence_status': 'PARTIAL_EVIDENCE',
                'physical_action_confirmed': False, 'windows': results})
    finally:
        analyzer.close()
    return {'vision_key': vision['key'], 'context_digest': digest(context), 'understandings': meanings, 'artifacts': artifacts}
