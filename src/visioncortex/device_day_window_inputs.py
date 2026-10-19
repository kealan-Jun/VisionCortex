"""Durable native selections and bounded pixel decoding for resumable windows."""
from fractions import Fraction
import hashlib
import json
import re
import subprocess

from .device_day_contract import atomic_json, digest, read_json, verify_artifact
from .device_day_coverage import limits


class DecodedFrames(list):
    """Selected pixels plus the measured decoder work, including seek preroll."""
    def __init__(self, rows, decoded_frames):
        super().__init__(rows)
        self.decoded_frames = decoded_frames


def source_stat(source):
    value = source.stat()
    return [getattr(value, field) for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')]


def selections(source, segment, settings, recording, vision_key, path):
    """Persist selection metadata, never a new source seal or stat-only trust."""
    from .device_day_understanding import frame_windows
    policy = limits(settings)
    for field in ('requests_per_turn', 'inactive_window_limit'):
        policy.pop(field)
    identity = {'recipe': 'native-window-inputs/2', 'source_ref': segment['source_ref'],
                'recording_id': recording['recording_id'], 'source_signature': recording.get('source_signature'),
                'vision_key': vision_key, 'segment_id': segment['segment_id'], 'activity': segment['activity'],
                'start_ms': segment['start_ms'], 'end_ms': segment['end_ms'],
                'required_times': [f['local_ms'] for f in segment.get('key_frames', [])], 'sampling': policy}
    initial = source_stat(source)
    if path.is_file():
        saved = read_json(path)
        if saved.get('identity') == identity and saved.get('source_stat') == initial:
            if saved.get('windows_digest') != digest(saved['windows']):
                raise ValueError('Saved native window selection was modified')
            return saved['windows']
    details = {}
    windows = []
    for left, right, frames in frame_windows(source, segment['start_ms'], segment['end_ms'],
            segment['activity'] == 'active', settings, required_times=identity['required_times'],
            details=details, pixels=False):
        windows.append({'start_ms': left, 'end_ms': right, 'frames': [
            {'source_frame_index': ordinal, 'local_ms': stamp, **details[ordinal]} for ordinal, stamp, _ in frames]})
    if source_stat(source) != initial:
        raise ValueError('Source changed while planning native windows')
    atomic_json(path, {'identity': identity, 'source_stat': initial, 'windows': windows,
                       'windows_digest': digest(windows)})
    return windows


def saved_metadata(layout, path, prompt, segment, selected, clock, previously_selected, recording_id):
    """Reuse only exact planned images whose bytes and native identities verify."""
    from .media_time import capture_us
    if not path.is_file():
        # Exact input reuse across sibling stage revisions, never a day-wide
        # search or an execution-key alias. Original images stay in place and
        # every native selection and image hash must still match below.
        from .device_day_content import time_folder
        from .device_day_contract import safe_child
        bases = {path.parent.parent.parent,
                 safe_child(layout.root, 'MultimodalUnderstanding/'
                            + time_folder(segment['start_us'], segment['end_us'])
                            + '/Analysis/' + segment['segment_id']),
                 safe_child(layout.root, 'MultimodalUnderstanding/ClipUnderstanding/' + segment['segment_id'])}
        candidates = sorted({p for base in bases for p in base.glob(f'*/{path.parent.name}/Input.json')})
        if len(candidates) > 64:
            raise ValueError('Too many historical versions for exact input reuse')
        for sibling in candidates:
            if sibling == path or not sibling.is_file():
                continue
            previous = saved_metadata(layout, sibling, prompt, segment, selected, clock,
                                      previously_selected, recording_id)
            if previous is not None:
                return previous
        return None
    saved = read_json(path)
    metadata = saved.get('metadata')
    if (saved.get('prompt') != prompt or not isinstance(metadata, dict)
            or saved.get('input_key') != digest([metadata, prompt])
            or any(metadata.get(k) != v for k, v in {
                'segment_id': segment['segment_id'], 'source_ref': segment['source_ref'],
                'activity': segment['activity'], 'start_ms': selected['start_ms'], 'end_ms': selected['end_ms'],
                'content_source': 'video_only', 'comments': [], 'protocol': None}.items())):
        return None
    images = metadata.get('frames', [])
    if len(images) != len(selected['frames']):
        return None
    for image, frame in zip(images, selected['frames'], strict=True):
        reasons = [*frame['selection_reasons'], *(['window_overlap_context']
                   if frame['source_frame_index'] in previously_selected else [])]
        if (any(image.get(k) != frame[k] for k in ('source_frame_index', 'local_ms', 'source_pts', 'source_time_base'))
                or image.get('selection_reasons') != reasons
                or image.get('capture_us') != capture_us(clock, frame['local_ms'])
                or image.get('frame_id') != 'frame-' + digest([recording_id, image.get('sha256'), frame['local_ms']])[:24]
                or not verify_artifact(layout.root, image)):
            return None
    return metadata


def decode_selected(source, selected):
    """Seek to an unfinished window; bind output pixels to exact decoder PTS."""
    import numpy as np
    from .video_io import _ffmpeg_passthrough_arguments
    rows = selected['frames']
    bases = {Fraction(row['source_time_base']) for row in rows}
    if len(bases) != 1:
        raise ValueError('Ambiguous selected native time base')
    initial = source_stat(source)
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,time_base:format=start_time', '-of', 'json', str(source)],
        check=True, capture_output=True, timeout=20).stdout)
    stream = info['streams'][0]
    if Fraction(stream['time_base']) not in bases:
        raise ValueError('Native source time base changed')
    width = min(768, stream['width'])
    height = round(stream['height'] * width / stream['width'])
    if not 0 < height <= 4096:
        raise ValueError('Unbounded source frame dimensions')
    start = max(0, rows[0]['local_ms']/1000 - 1)
    duration = rows[-1]['local_ms']/1000 - start + 1
    selectors = '+'.join(f'eq(pts\\,{row["source_pts"]})' for row in rows)
    output = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'info', '-nostdin', '-copyts',
        '-threads', '2', '-ss', f'{start:.9f}', '-t', f'{duration:.9f}', '-i', str(source),
        '-map', '0:v:0', '-vf', f'showinfo@decode_work,select={selectors},showinfo@window_identity,scale={width}:{height}:flags=bilinear',
        '-an', '-sn', *_ffmpeg_passthrough_arguments(), '-frames:v', str(len(rows)),
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'], check=True, capture_output=True, timeout=90)
    lines = [line for line in output.stderr.decode(errors='replace').splitlines() if 'showinfo@window_identity' in line]
    pts = [int(match[1]) for line in lines if (match := re.search(r'\bn:\s*\d+\s+pts:\s*(-?\d+)', line))]
    time_bases = [Fraction(match[1]) for line in lines if (match := re.search(r'config in time_base:\s*(\d+/\d+)', line))]
    if pts != [row['source_pts'] for row in rows] or time_bases != list(bases):
        raise ValueError('Decoded window pixels do not match selected native PTS')
    if len(output.stdout) != width * height * 3 * len(rows) or source_stat(source) != initial:
        raise ValueError('Decoded window size or source identity changed')
    pixels = np.frombuffer(output.stdout, dtype=np.uint8).reshape((len(rows), height, width, 3))
    work = sum('showinfo@decode_work' in line and re.search(r'\bn:\s*\d+\s+pts:', line) is not None
               for line in output.stderr.decode(errors='replace').splitlines())
    return DecodedFrames([(row['source_frame_index'], row['local_ms'], frame,
             hashlib.sha256(frame.tobytes()).hexdigest()) for row, frame in zip(rows, pixels, strict=True)], work)
