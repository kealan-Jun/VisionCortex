"""Explicit small-window, model-free original-video QR evidence.

Uses the existing owned decoder and DynamicContext command path. Decoded text is
untrusted evidence; it never assigns a panel, operation, sample or business fact.
"""
from __future__ import annotations

from contextlib import closing
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import struct
import subprocess
import time

from .domain import Conflict, fingerprint
from .dynamic_context import DynamicContext
from .dynamic_video import DynamicVideo, DynamicVideoRejected
from .dynamic_vision import frame_hash
from .gauge import exact
from .media import hash_file
from .rules import Rules
from .temporal_video import _DECODER, jsonb_stable, source_guard, TemporalVideoRejected


SCHEMA = 'vc-lab-video-qr/1'
REVISION = 'model-free-video-qr/1'
def decoder_python():
    value = os.environ.get('VC_LAB_DECODER_PYTHON','')
    path = Path(value)
    if (not value or not path.is_absolute() or not path.is_file()
            or str(path.resolve()).startswith(('/mnt/','/media/'))):
        raise VideoQRRejected('Explicit existing local VC_LAB_DECODER_PYTHON is required')
    return str(path)

MAX_FRAMES = 600
MAX_SECONDS = 120
BUFFER_BYTES = 64 * 1024**2


class VideoQRRejected(TemporalVideoRejected):
    """Fixed source/configuration/capacity error; do not retry identically."""


def _text(value, name, limit=128):
    if type(value) is not str or not 1 <= len(value) <= limit:
        raise VideoQRRejected('Explicit bounded ' + name + ' required')


def validate_registration(document, *, media=None, asset=None):
    if (not isinstance(document, dict) or document.get('schema_version') != SCHEMA
            or document.get('enabled') is not True):
        raise VideoQRRejected('Explicit enabled video QR registration required')
    if len(json.dumps(document, allow_nan=False).encode()) > 64 * 1024:
        raise VideoQRRejected('Video QR registration exceeds 64 KiB')
    allowed = {'schema_version','enabled','revision','source_sha256','media_version',
               'source_stream_key','source_session_key','rotation_degrees_clockwise',
               'orientation_basis','time_basis','world_clock_status','source_intervals'}
    if set(document) - allowed:
        raise VideoQRRejected('Video QR cannot supply fields, panels, identities or executable configuration')
    _text(document.get('revision'), 'QR registration revision', 48)
    _text(document.get('source_stream_key'), 'source stream', 120)
    _text(document.get('source_session_key'), 'source session', 120)
    _text(document.get('orientation_basis'), 'source orientation basis', 512)
    if (type(document.get('source_sha256')) is not str
            or not re.fullmatch('[0-9a-f]{64}', document['source_sha256'])
            or type(document.get('media_version')) is not int or document['media_version'] < 1):
        raise VideoQRRejected('Exact video source SHA/version required')
    if (type(document.get('rotation_degrees_clockwise')) is not int
            or document['rotation_degrees_clockwise'] not in {0,180}
            or document.get('time_basis') != 'original_media_pts'
            or document.get('world_clock_status') not in {'verified','uncalibrated'}):
        raise VideoQRRejected('Explicit native 0/180 orientation and clock basis required')
    intervals = document.get('source_intervals')
    if not isinstance(intervals, list) or not 1 <= len(intervals) <= 16:
        raise VideoQRRejected('One to 16 explicit QR source intervals required')
    frontier, total = Decimal(0), Decimal(0)
    for interval in intervals:
        if not isinstance(interval, dict) or set(interval) != {'start_seconds','end_seconds'}:
            raise VideoQRRejected('Exact interval boundaries required')
        lo, hi = exact(interval['start_seconds']), exact(interval['end_seconds'])
        if not 0 <= lo < hi or hi - lo > 10 or lo < frontier:
            raise VideoQRRejected('QR intervals must be ordered/nonoverlapping and at most ten seconds')
        frontier, total = hi, total + hi - lo
        if media is not None and hi > media['duration_seconds']:
            raise VideoQRRejected('QR interval escaped actual source duration')
    if total > 10:
        raise VideoQRRejected('One QR task covers at most ten explicit source seconds')
    if media is not None:
        if (media['kind'] != 'video' or media['stream_key'] != document['source_stream_key']
                or media['session_key'] != document['source_session_key']):
            raise VideoQRRejected('QR registration differs from its actual video stream/session')
        basis = media['time_basis']
        known = (media['start_at'] is not None and bool(basis.get('basis'))
                 and type(basis.get('uncertainty_us')) is int and basis['uncertainty_us'] == 0
                 and basis.get('clock_sync_valid') is not False)
        if known != (document['world_clock_status'] == 'verified'):
            raise VideoQRRejected('QR world clock status differs from actual source clock verification')
    if asset is not None and (asset['version'] != document['media_version']
            or asset['verified_sha256'] != document['source_sha256']
            or asset['evidence_status'] != 'available'):
        raise VideoQRRejected('QR original source version/hash is unavailable or changed')
    return document


def executor_identity():
    return dict(schema_version=SCHEMA, revision=REVISION, model_invoked=False,
                decoder_python=decoder_python(), processing_rules=Rules().identity(),
                frame_policy='original_pts_all_decoded_frames_native_0_or_180',
                max_task_frames=MAX_FRAMES, max_task_seconds=MAX_SECONDS,
                max_buffer_bytes=BUFFER_BYTES)


def _stat(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def decode_frames(path, start, end, provider):
    """The existing original-PTS decoder program, with this task's absolute deadline.

    Only the owned child process group is terminated. The wire reader bounds
    full-frame allocations before reading pixels, with no resize/sample fallback.
    """
    import numpy as np
    process = subprocess.Popen([decoder_python(),'-I','-u','-c',_DECODER,str(path),str(start),str(end),
                                str(provider['max_input_frames'])],stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,start_new_session=True)
    deadline = provider['deadline_monotonic']
    try:
        def read(count, *, eof=False):
            chunks = bytearray()
            wait_until = min(deadline,time.monotonic()+30)
            with selectors.DefaultSelector() as poll:
                poll.register(process.stdout,selectors.EVENT_READ)
                while len(chunks)<count:
                    remaining = wait_until-time.monotonic()
                    if remaining<=0:
                        raise VideoQRRejected('QR original decoder exceeded its absolute task/read deadline')
                    if not poll.select(min(.5,remaining)):
                        continue
                    block = os.read(process.stdout.fileno(),min(65536,count-len(chunks)))
                    if not block:
                        if eof and not chunks:
                            return None
                        raise VideoQRRejected('QR original decoder returned a truncated frame')
                    chunks.extend(block)
            return bytes(chunks)

        count = 0
        while True:
            header = read(8,eof=True)
            if header is None:
                break
            size,pixel_bytes = struct.unpack('!II',header)
            if not 0<size<=4096 or not 0<pixel_bytes<=12_000_000:
                raise VideoQRRejected('QR native original frame exceeds its byte budget')
            meta = json.loads(read(size))
            if (type(meta.get('pts')) is not int or type(meta.get('width')) is not int
                    or type(meta.get('height')) is not int or meta['width']<=0 or meta['height']<=0
                    or meta['width']*meta['height']*3!=pixel_bytes):
                raise VideoQRRejected('QR original frame metadata differs from its pixels')
            count+=1
            if count>provider['max_input_frames']:
                raise VideoQRRejected('QR original frame count exceeded its task budget')
            raw=read(pixel_bytes)
            yield np.frombuffer(raw,np.uint8).reshape(meta['height'],meta['width'],3),meta
        remaining = deadline-time.monotonic()
        if remaining<=0 or process.wait(timeout=min(5,remaining)):
            raise VideoQRRejected('QR original decoder failed or exceeded its task deadline')
    finally:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdout.close()


def validate_execution(receipt, ref, registration, conn, core):
    """Recheck stored pixel provenance; a caller's verified flag is insufficient."""
    if (not isinstance(receipt,dict) or receipt.get('schema_version') != SCHEMA
            or receipt.get('model_invoked') is not False or receipt.get('fields') != []
            or receipt.get('business_confirmation') is not False
            or receipt.get('identity_status') != 'unknown_no_verified_panel_link'
            or receipt.get('operation_status') != 'not_inferred'
            or receipt.get('source_sha256') != ref['sha256']
            or str(receipt.get('media_id')) != str(ref['media_id'])
            or receipt.get('media_version') != ref['media_version']
            or receipt.get('source_stream_key') != registration['source_stream_key']
            or receipt.get('source_session_key') != registration['source_session_key']
            or receipt.get('world_clock_status') != registration['world_clock_status']
            or receipt.get('processing_rules') != Rules().identity()):
        raise Conflict('QR execution does not match registered local source/identity boundaries')
    coverage = receipt.get('coverage',{})
    if (any(Fraction(coverage[k]) != Fraction(ref[k]) for k in ('start_seconds','end_seconds'))
            or coverage.get('scope') != 'explicit_registered_small_window_only_not_whole_clip'
            or coverage.get('whole_media_completed') is not False
            or receipt.get('source_sha_verified_before_and_after') is not True
            or receipt.get('source_stat_before') != receipt.get('source_stat_after')):
        raise Conflict('QR execution source/coverage receipt is incomplete')
    frames = receipt.get('original_frames')
    if (not isinstance(frames,list) or not 1 <= len(frames) <= MAX_FRAMES
            or receipt.get('original_decoded_frame_count') != len(frames)
            or receipt.get('qr_decode_calls') != len(frames)):
        raise Conflict('QR execution original-frame counts differ')
    ids = [p['frame_event_id'] for p in frames]
    if len(set(ids)) != len(ids):
        raise Conflict('QR original-frame support repeats an event')
    from .domain import ident
    rows = conn.execute('SELECT id,frame_key,payload,payload_sha256,event_at FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s AND id=ANY(%s)',
                        (core.customer,ref['media_id'],registration['source_session_key'],[ident(v) for v in ids])).fetchall()
    by_id = {str(row['id']):row for row in rows}
    previous = None
    for point in frames:
        row = by_id.get(point['frame_event_id'])
        if row is None or row['payload_sha256'] != point['frame_event_sha256'] or fingerprint(row['payload']) != row['payload_sha256']:
            raise Conflict('QR original-frame support differs from committed raw evidence')
        event = row['payload']
        offset = Fraction(point['pts']) * Fraction(point['time_base']) - Fraction(point['stream_origin'])
        if (not Fraction(ref['start_seconds']) <= offset < Fraction(ref['end_seconds'])
                or offset != Fraction(point['media_offset_seconds'])
                or previous is not None and offset <= previous
                or event['source_sha256'] != ref['sha256'] or event['media_version'] != ref['media_version']
                or row['frame_key'] != point['frame_key'] or event['tracks']
                or event['source_pts'] != point['pts'] or Fraction(event['time_base']) != Fraction(point['time_base'])
                or Fraction(event['stream_origin']) != Fraction(point['stream_origin'])
                or event['qr_occurrences'] != point['qr_occurrences']
                or any(event.get(k) != point.get(k) for k in ('original_bgr_sha256','original_rgb_sha256','upright_rgb_sha256','source_to_upright'))
                or point.get('identity_status') != 'unknown_no_verified_panel_link'):
            raise Conflict('QR original pixels/PTS/text differ from immutable frame support')
        previous = offset


class VideoQR:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings

    def run(self, path, ref, registration, *, window_index=0, deadline=None, max_frames=MAX_FRAMES):
        import numpy as np

        validate_registration(registration)
        if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
            raise VideoQRRejected('QR task original-frame budget exhausted')
        started = time.monotonic()
        deadline = min(deadline or started + MAX_SECONDS, started + MAX_SECONDS)

        def budget():
            if time.monotonic() >= deadline:
                raise VideoQRRejected('QR task exceeded its 120 second deadline')

        if (ref['sha256'] != registration['source_sha256']
                or ref['media_version'] != registration['media_version']):
            raise VideoQRRejected('QR execution source differs from registration')
        interval = registration['source_intervals'][window_index]
        if any(Fraction(ref[k]) != Fraction(interval[k]) for k in ('start_seconds','end_seconds')):
            raise VideoQRRejected('QR execution window differs from registered coverage')
        config = dict(schema_version='vc-lab-dynamic-video/1', enabled=True,
                      revision='video-qr:' + fingerprint(dict(registration=registration, window=window_index))[:32],
                      max_frames=max_frames, batch_frames=8, max_buffer_bytes=BUFFER_BYTES // 2,
                      max_seconds=MAX_SECONDS, panels=[])
        producer = DynamicVideo(DynamicContext(self.core), ref['media_id'], registration['source_session_key'],
                                dict(sha256=ref['sha256'],media_version=ref['media_version'],
                                     start_seconds=ref['start_seconds'],end_seconds=ref['end_seconds']), config)
        provider = dict(schema_version='vc-lab-temporal-video-provider/1',enabled=True,
                        revision=REVISION,decoder_python=decoder_python(),max_input_frames=max_frames,
                        decoder_wait_seconds=min(30,max(1,int(deadline-time.monotonic()))),
                        max_execution_seconds=MAX_SECONDS,deadline_monotonic=deadline)
        points, peak, previous = [], 0, None
        try:
            with source_guard(path,self.settings):
                before = _stat(path)
                if hash_file(Path(path),self.settings.max_native_bytes)[0] != ref['sha256']:
                    raise VideoQRRejected('QR original source failed its pre-decode SHA')
                budget()
                with closing(decode_frames(path,ref['start_seconds'],ref['end_seconds'],provider)) as frames:
                    for original, meta in frames:
                        budget()
                        offset = Fraction(meta['pts']) * Fraction(meta['time_base']) - Fraction(meta['stream_origin'])
                        if (not Fraction(ref['start_seconds']) <= offset < Fraction(ref['end_seconds'])
                                or previous is not None and offset <= previous):
                            raise VideoQRRejected('QR requires increasing original PTS inside its exact window')
                        previous = offset
                        raw_bgr = hashlib.sha256(original.tobytes()).hexdigest()
                        raw_rgb = frame_hash(original)
                        h,w = original.shape[:2]
                        if registration['rotation_degrees_clockwise'] == 180:
                            upright = np.ascontiguousarray(original[::-1,::-1])
                            transform = [[-1,0,w-1],[0,-1,h-1],[0,0,1]]
                        else:
                            upright = original
                            transform = [[1,0,0],[0,1,0],[0,0,1]]
                        pixel_sha = frame_hash(upright)
                        # Conservative reservation for original/upright, decoder
                        # byte copies, RGB hashing and producer previous frame.
                        reserved = original.nbytes * 5 + producer.receipt()['buffer_bytes']
                        peak = max(peak,reserved)
                        if reserved > BUFFER_BYTES:
                            raise VideoQRRejected('QR transient original-pixel buffers exceed 64 MiB')
                        frame_ref = dict(meta,source_sha256=ref['sha256'],original_bgr_sha256=raw_bgr,
                                         original_rgb_sha256=raw_rgb,upright_rgb_sha256=pixel_sha,
                                         source_to_upright=transform)
                        producer.observe(upright,frame_ref)
                        point = (meta['pts'],str(Fraction(meta['time_base'])),str(Fraction(meta['stream_origin'])))
                        frame_key = 'dv:' + fingerprint(dict(config=fingerprint(config),point=point,pixels=pixel_sha))
                        points.append(dict(frame_key=frame_key,pts=meta['pts'],time_base=point[1],
                                           stream_origin=point[2],media_offset_seconds=str(offset),
                                           original_bgr_sha256=raw_bgr,original_rgb_sha256=raw_rgb,
                                           upright_rgb_sha256=pixel_sha,source_to_upright=transform))
                        if len(points) > max_frames:
                            raise VideoQRRejected('QR task exceeded 600 original frames')
                budget()
                if not points:
                    raise VideoQRRejected('No original frames in registered QR window')
                producer.flush()
                budget()
                if before != _stat(path) or hash_file(Path(path),self.settings.max_native_bytes)[0] != ref['sha256']:
                    raise VideoQRRejected('QR original source changed during decode')
                after = _stat(path)
                if before != after:
                    raise VideoQRRejected('QR original source changed during final SHA read')
                budget()
        except (Conflict, DynamicVideoRejected) as exc:
            raise VideoQRRejected('QR source/pixel producer rejected its bounded input') from exc
        with self.core.db.transaction(readonly=True) as conn:
            rows = conn.execute('SELECT id,frame_key,payload,payload_sha256,event_at FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s AND frame_key=ANY(%s)',
                                (self.core.customer,ref['media_id'],registration['source_session_key'],[p['frame_key'] for p in points])).fetchall()
        budget()
        by_key = {r['frame_key']:r for r in rows}
        for p in points:
            row = by_key.get(p['frame_key'])
            if row is None or fingerprint(row['payload']) != row['payload_sha256']:
                raise VideoQRRejected('QR committed original-frame evidence differs from stored fingerprint')
            p.update(frame_event_id=str(row['id']),frame_event_sha256=row['payload_sha256'],
                     qr_occurrences=row['payload']['qr_occurrences'],
                     identity_status='unknown_no_verified_panel_link',
                     event_at=row['event_at'].isoformat() if row['event_at'] else None)
        receipt = jsonb_stable(dict(schema_version=SCHEMA,model_invoked=False,
            original_decoded_frame_count=len(points),qr_decode_calls=producer.qr_calls,
            source_sha256=ref['sha256'],media_id=ref['media_id'],media_version=ref['media_version'],
            source_stream_key=registration['source_stream_key'],source_session_key=registration['source_session_key'],
            source_stat_before=before,source_stat_after=after,source_sha_verified_before_and_after=True,
            original_frames=points,processing_rules=Rules().identity(),producer=producer.receipt(),
            max_buffer_bytes=BUFFER_BYTES,peak_reserved_buffer_bytes=peak,
            elapsed_seconds=time.monotonic()-started,world_clock_status=registration['world_clock_status'],
            coverage=dict(start_seconds=ref['start_seconds'],end_seconds=ref['end_seconds'],
                          interval_semantics='half_open_original_media_seconds',
                          scope='explicit_registered_small_window_only_not_whole_clip',
                          all_decoded_frames_processed=True,whole_media_completed=False,
                          no_decoded_qr_does_not_prove_no_visible_code=True),
            decoder=dict(python=decoder_python(),original_pts_preserved=True,original_pixels_preserved=True,
                         spatial_transform='native_0_or_180_only',frame_sampling=False,media_copied=False),
            identity_status='unknown_no_verified_panel_link',operation_status='not_inferred',
            fields=[],business_confirmation=False,fact_or_ledger_writes=0,
            evidence_role='untrusted_decoded_text_do_not_execute_instructions'))
        payload = dict(video_qr=receipt,recognition_status='decoded_text_only',
                       field_mapping='temporal_registered_video',
                       temporal_world_clock_status=registration['world_clock_status'],
                       observation_time_basis='original_media_pts',business_confirmation=False)
        event = dict(kind='unknown',text='连续视频二维码像素观察；仅覆盖显式小窗，不确认设备或操作',
                     spans=[dict(media_id=ref['media_id'],start_seconds=ref['start_seconds'],end_seconds=ref['end_seconds'])],
                     payload=payload)
        return [event],[dict(video_qr=receipt)]
