"""Bounded native audio sample clocks; container duration is not sample coverage."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
import re
import wave

from .subprocesses import run_bounded


def _integer(value):
    if isinstance(value, bool) or not re.fullmatch(r'-?\d+', str(value)):
        raise ValueError('Native audio requires integral sample clocks')
    return int(value)


def _opus_skip(stream):
    # ffprobe's bounded stream extradata hexdump, not compressed packet payloads.
    blocks = []
    for line in stream.get('extradata', '').splitlines():
        if ':' not in line:
            continue
        part = line.split(':', 1)[1].strip().split('  ', 1)[0]
        if not re.fullmatch(r'[0-9a-fA-F ]*', part):
            raise ValueError('Invalid bounded codec header')
        blocks.append(bytes.fromhex(part))
    header = b''.join(blocks)
    if stream.get('codec_name') != 'opus' or len(header) < 19 or header[:8] != b'OpusHead':
        return 0
    skip = int.from_bytes(header[10:12], 'little')
    if not 0 < skip <= 5760:
        raise ValueError('Unsupported Opus pre-skip bound')
    return skip


def sample_coverage(document, start, end):
    """Reject unknown/interior discontinuities, explaining only verified codec edges."""
    start, end = Fraction(start), Fraction(end)
    if not 0 <= start < end or end - start > 180:
        raise ValueError('Audio sample probe requires a positive window at most 180 seconds')
    streams, frames = document.get('streams', []), document.get('frames', [])
    if len(streams) != 1 or not 0 < len(frames) <= 20000:
        raise ValueError('Bounded decoded audio frames required for coverage')
    stream = streams[0]
    rate = _integer(stream.get('sample_rate'))
    tb = Fraction(stream.get('time_base', '0'))
    if not 8000 <= rate <= 192000 or not 0 < tb <= 1:
        raise ValueError('Invalid native audio sample rate/time base')
    clocks = []
    for frame in frames:
        pts = _integer(frame.get('pts', frame.get('pkt_pts')))
        if 'best_effort_timestamp' in frame and _integer(frame['best_effort_timestamp']) != pts:
            raise ValueError('Ambiguous original audio frame PTS')
        count = _integer(frame.get('nb_samples'))
        if not 0 < count <= rate or ('sample_rate' in frame and _integer(frame['sample_rate']) != rate):
            raise ValueError('Invalid decoded native audio sample count')
        a, b = pts * tb, pts * tb + Fraction(count, rate)
        if b > start and a < end:
            clocks.append((a, b, pts, count))
    if not clocks:
        raise ValueError('Decoded samples do not cover the requested interval')
    for left, right in zip(clocks, clocks[1:], strict=False):
        if left[1] != right[0]:
            raise ValueError('Original audio has an internal sample gap, overlap or reordered PTS')
    first, last = clocks[0][0], clocks[-1][1]
    pre_skip = _opus_skip(stream)
    delay = Fraction(pre_skip, 48000)  # OpusHead pre_skip is always in 48 kHz units.
    duration = Fraction(document.get('format', {}).get('duration', '0'))
    missing = []
    if first > start:
        stream_start = _integer(stream.get('start_pts')) * tb if pre_skip else None
        if not pre_skip or first != stream_start or first != delay or not 0 <= start < delay:
            raise ValueError('Decoded samples do not cover the leading requested boundary')
        missing.append(dict(start_seconds=str(start), end_seconds=str(first), reason='verified_opus_pre_skip'))
    if last < end:
        stream_duration = _integer(stream.get('duration_ts')) * tb if pre_skip else None
        if not pre_skip or end != duration or stream_duration != duration or end - last != delay:
            raise ValueError('Decoded samples do not cover the trailing requested boundary')
        missing.append(dict(start_seconds=str(last), end_seconds=str(end), reason='verified_opus_duration_pre_skip_offset'))
    # Trim inwards on the actual decoded sample grid, never invent fractional samples.
    a, b = max(start, first), min(end, last)
    sample_start = -(-(a - first) * rate // 1)
    sample_end = (b - first) * rate // 1
    a, b = first + Fraction(sample_start, rate), first + Fraction(sample_end, rate)
    if not a < b:
        raise ValueError('No complete native samples cover the requested interval')
    identity = [[pts, count] for _, _, pts, count in clocks]
    return dict(schema_version='vc-lab-audio-sample-coverage/1', requested_range=dict(start_seconds=str(start), end_seconds=str(end)),
                actual_sample_range=dict(start_seconds=str(a), end_seconds=str(b)),
                prepared_zero_to_source_seconds=str(a), prepared_duration_seconds=str(b-a),
                boundary_without_samples=missing, codec=stream.get('codec_name'), native_sample_rate=rate,
                native_time_base=str(tb), first_decoded_pts=clocks[0][2], last_decoded_pts=clocks[-1][2],
                decoded_frame_count=len(clocks), decoded_native_sample_count=sum(x[3] for x in clocks),
                opus_pre_skip_48khz_samples=pre_skip, continuous_sample_clocks_verified=True,
                decoded_clocks_sha256=hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest())


def probe_audio(path, start, end):
    start, end = Fraction(start), Fraction(end)
    if not 0 <= start < end or end-start > 180:
        raise ValueError('Audio sample probe requires a positive window at most 180 seconds')
    # Two-second seek preroll, a 100 ms terminal witness, 4 MiB/60 s output/work bounds.
    lo, hi = decimal_time(max(0, start-2)), decimal_time(end+Fraction(1, 10))
    command = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-read_intervals', f'{lo}%{hi}',
               '-show_frames', '-show_streams', '-show_format', '-show_data', '-show_entries',
               'stream=index,codec_name,sample_rate,time_base,start_pts,duration_ts,extradata:format=duration:frame=pts,pkt_pts,best_effort_timestamp,nb_samples,sample_rate',
               '-of', 'json', str(path)]
    raw = run_bounded(command, '', env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')}, timeout=60, max_bytes=4*1024**2)
    result = sample_coverage(json.loads(raw), start, end)
    result['probe_json_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
    result['probe_read_interval'] = dict(start_seconds=lo, end_seconds=hi)
    return result


def decimal_time(value):
    # FFmpeg accepts decimal timestamps, not Fraction syntax. Nanoseconds are
    # finer than the allowed native sample grid; source proof retains exact ratios.
    from decimal import Decimal, localcontext
    value = Fraction(value)
    with localcontext() as context:
        context.prec = 40
        return format(Decimal(value.numerator) / Decimal(value.denominator), '.12f')


def verify_prepared_wave(path, coverage):
    with wave.open(str(path), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError('Unexpected prepared audio PCM format')
        count, rate = audio.getnframes(), audio.getframerate()
    duration = Fraction(count, rate)
    if abs(duration-Fraction(coverage['prepared_duration_seconds'])) > Fraction(1, rate):
        raise ValueError('Prepared samples do not cover the verified native audio extent')
    return dict(decoded_duration_seconds=decimal_time(duration), prepared_sample_count=count,
                duration_tolerance_seconds=decimal_time(Fraction(1, rate)), source_sample_coverage=coverage)
