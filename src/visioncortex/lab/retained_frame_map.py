"""RESEARCH ONLY, not imported by default runtime. Opt-in discrete pixel/frame provenance for retained clips, separate from v1.

An export duration is not a content boundary. Every mapped frame keeps original
PTS/Fraction and an actual RGB identity; between-frame/time-tail coverage stays
unknown. Supplied JSON can never claim an app-owned local pixel verification.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time

from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, fingerprint, ident
from .temporal_video import decode_frames, validate_provider, jsonb_stable

SCHEMA = 'vc-lab-retained-frame-map/2'
_AUTHORITY = object()


@dataclass(frozen=True)
class FrameMapLimits:
    max_frames: int = 3600
    max_span_seconds: int = 120
    max_receipt_bytes: int = 4*1024**2
    max_document_bytes: int = 2*1024**2
    max_source_bytes: int = 6*1024**3
    max_clip_bytes: int = 128*1024**2


@dataclass(frozen=True)
class LocalPixelProof:
    document_sha256: str
    source_sha256: str
    clip_sha256: str
    decoded_clip_frames: int
    receipt: dict
    _authority: object


def _fraction(value):
    if isinstance(value, (bool, float)):
        raise ValueError('Exact Fraction/integer/string required')
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError('Invalid exact media time') from exc
    return result


def _sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Pinned lowercase SHA256 required')
    return value


def _read_pin(path, expected, limit):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > limit:
        raise ValueError('Bounded regular receipt file required')
    before = path.stat()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        raw = stream.read(limit+1)
    after = path.stat()
    if len(raw) > limit or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise Conflict('Pixel receipt changed during reading')
    if hashlib.sha256(raw).hexdigest() != _sha(expected):
        raise Conflict('Pixel readback receipt SHA mismatch')
    return json.loads(raw)


def _point(row):
    if type(row.get('pts')) is not int:
        raise ValueError('Original integer PTS required')
    tb = _fraction(row['time_base'])
    origin = _fraction(row.get('stream_origin_seconds', '0'))
    offset = _fraction(row['media_offset_seconds'])
    if tb <= 0 or offset < 0 or offset != row['pts']*tb-origin:
        raise ValueError('PTS/time-base/origin differs from original media offset')
    return dict(pts=row['pts'], time_base=str(tb), stream_origin_seconds=str(origin),
                media_offset_seconds=str(offset), RGB_sha256=_sha(row['RGB_sha256']),
                width=row['width'], height=row['height'])


def load_pixel_readback(receipt_path, receipt_sha256, frame_readback_path,
                        frame_readback_sha256, *, sample_id=None, limits=None,
                        playback_probe_path=None, playback_probe_sha256=None):
    """Load pinned audit artifacts as supplied evidence; never mint local proof.

    Supports the actual sample manifest/readback pair produced by our bounded
    audit and its standalone genuine-source receipt. A negative/no-frame-match
    receipt is rejected rather than promoted by its successful file hashes.
    """
    limits = limits or FrameMapLimits()
    receipt = _read_pin(receipt_path, receipt_sha256, limits.max_receipt_bytes)
    rows = receipt.get('samples')
    if rows is not None:
        found = [r for r in rows if r.get('sample_id') == sample_id]
        if len(found) != 1:
            raise ValueError('Exact unique sample id required')
        receipt = found[0]
    if not (receipt.get('exact_complete_clip_RGB_sequence_matches_original') is True or
            receipt.get('exact_complete_clip_RGB_sequence_matches_genuine_original') is True):
        raise ValueError('Receipt does not establish a complete matching frame sequence')
    readback = _read_pin(frame_readback_path, frame_readback_sha256, limits.max_receipt_bytes)
    clip, source = readback.get('clip'), readback.get('original')
    if not isinstance(clip, list) or not isinstance(source, list) or not 0 < len(clip) == len(source) <= limits.max_frames:
        raise ValueError('All decoded clip frames and equal source correspondences required')
    pairs = []
    for a, b in zip(clip, source):
        a, b = _point(a), _point(b)
        if a['RGB_sha256'] != b['RGB_sha256'] or (a['width'], a['height']) != (b['width'], b['height']):
            raise ValueError('Source/clip pixel identity is not established')
        pairs.append(dict(clip=a, source=b))
    mapping = receipt['source_mapping']
    lo, hi = (Fraction(str(mapping[k]))/1000 for k in ('source_start_ms', 'source_end_ms'))
    probe = receipt.get('clip_probe') or receipt.get('existing_clip_probe')
    # Samples-r1 preserves an earlier child-fd ffprobe failure. A separately
    # supplied playback extent is preferable to inventing it from frame count.
    metadata = (probe or {}).get('metadata')
    probe_pin = None
    if playback_probe_path is not None:
        probes = _read_pin(playback_probe_path, playback_probe_sha256, limits.max_receipt_bytes)
        clip_ref = receipt.get('video') or receipt['published_interval']['video']
        found = [p for p in probes if p.get('relative') == clip_ref['path'] and
                 p.get('day') == receipt.get('day') and p.get('status') == 'success']
        if len(found) != 1:raise ValueError('Unique pinned actual clip probe required')
        metadata = found[0]['metadata'];probe_pin = playback_probe_sha256
    playback = None
    if metadata and metadata.get('format', {}).get('duration') is not None:
        form = metadata['format']; start = Fraction(form.get('start_time', '0'))
        playback = dict(start_seconds=str(start), end_seconds=str(start+Fraction(form['duration'])),
                        duration_seconds=str(Fraction(form['duration'])), basis='actual_container_probe')
    document = dict(schema_version=SCHEMA, source_ref=receipt['source_ref'],
        clip_ref=receipt.get('video') or receipt['published_interval']['video'],
        receipt_pins=dict(receipt_sha256=_sha(receipt_sha256), frame_readback_sha256=_sha(frame_readback_sha256), sample_id=sample_id,
                          playback_probe_sha256=probe_pin),
        nominal_source_interval=dict(start_seconds=str(lo), end_seconds=str(hi), basis='saved_selection_not_verified_event_boundary'),
        content_boundaries=dict(status='unknown', ranges=[]), event_boundaries=dict(status='unknown', ranges=[]),
        context_padding=dict(status='unknown', before_seconds=None, after_seconds=None),
        export=dict(policy=receipt.get('export_policy'), declared_derived_start_ms=mapping.get('derived_start_ms'),
                    declared_derived_end_ms=mapping.get('derived_end_ms'), historical_duration_tolerance_ms=mapping.get('duration_tolerance_ms')),
        playback=playback, frames=pairs, decoded_frame_count=len(pairs),
        world_clock=dict(status='unknown', epoch_origin=None),
        verification='supplied_not_independently_verified', continuous_time_coverage_proven=False,
        original_media_modified=False, business_fact_confirmed=False)
    return validate_frame_map(document, limits=limits)


def validate_frame_map(document, *, limits=None):
    limits = limits or FrameMapLimits()
    doc = json.loads(json.dumps(document, allow_nan=False))
    # Legacy export measurements remain metadata, with stable JSONB numeric
    # meaning. Exact active frame/PTS fields retain their strict types below.
    doc['export'] = jsonb_stable(doc.get('export', {}))
    if doc.get('schema_version') != SCHEMA or doc.get('world_clock') != dict(status='unknown', epoch_origin=None):
        raise ValueError('Versioned source-relative map with unknown world clock required')
    if any(doc.get(k) is True for k in ('continuous_time_coverage_proven', 'business_fact_confirmed', 'original_media_modified')):
        raise ValueError('Discrete pixel mapping cannot confirm continuous coverage, business facts or changed originals')
    doc.update(continuous_time_coverage_proven=False, business_fact_confirmed=False, original_media_modified=False)
    pins = doc.get('receipt_pins')
    if not isinstance(pins, dict):raise ValueError('Pinned actual readback receipts required')
    for name in ('receipt_sha256', 'frame_readback_sha256'):_sha(pins.get(name))
    if pins.get('playback_probe_sha256') is not None:_sha(pins['playback_probe_sha256'])
    for name in ('source_ref', 'clip_ref'):
        _sha(doc[name]['sha256'])
        if type(doc[name].get('size_bytes')) is not int or doc[name]['size_bytes'] <= 0:
            raise ValueError('Pinned positive media byte size required')
    if doc['source_ref']['size_bytes'] > limits.max_source_bytes or doc['clip_ref']['size_bytes'] > limits.max_clip_bytes:
        raise ValueError('Frame-map media byte budget exceeded')
    frames = doc.get('frames')
    if not isinstance(frames, list) or not 0 < len(frames) == doc.get('decoded_frame_count') <= limits.max_frames:
        raise ValueError('Bounded complete decoded clip-frame mapping required')
    old_clip = old_source = old_source_rgb = None
    for pair in frames:
        a, b = _point(pair['clip']), _point(pair['source'])
        if a['RGB_sha256'] != b['RGB_sha256'] or (a['width'], a['height']) != (b['width'], b['height']):
            raise ValueError('Mapped RGB identity or native dimensions differ')
        for point in (a, b):
            if type(point['width']) is not int or type(point['height']) is not int or not 0 < point['width']*point['height'] <= 4_194_304:
                raise ValueError('Bounded native frame dimensions required')
        ca, cb = _fraction(a['media_offset_seconds']), _fraction(b['media_offset_seconds'])
        if old_clip is not None and (ca <= old_clip or cb < old_source):
            raise ValueError('Clip PTS must strictly increase; source PTS must not reverse')
        if old_source == cb and old_source_rgb != b['RGB_sha256']:
            raise ValueError('Repeated source PTS cannot have different RGB pixels')
        old_clip, old_source, old_source_rgb = ca, cb, b['RGB_sha256']
    for kind in ('clip', 'source'):
        extent = _fraction(frames[-1][kind]['media_offset_seconds'])-_fraction(frames[0][kind]['media_offset_seconds'])
        if extent > limits.max_span_seconds:
            raise ValueError('Frame-map selected span exceeds bound')
    lo, hi = (_fraction(doc['nominal_source_interval'][k]) for k in ('start_seconds', 'end_seconds'))
    if not 0 <= lo < hi or hi-lo > limits.max_span_seconds:
        raise ValueError('Bounded nominal source interval required')
    for key in ('content_boundaries', 'event_boundaries'):
        proposal = doc.get(key)
        if not isinstance(proposal, dict) or proposal.get('status') not in {'unknown', 'candidate'} or not isinstance(proposal.get('ranges'), list):
            raise ValueError('Independent unknown/candidate content and event boundaries required')
        ranges = proposal['ranges']
        if len(ranges) > 64 or (proposal['status'] == 'unknown' and ranges) or (proposal['status'] == 'candidate' and not ranges):
            raise ValueError('Bounded explicit boundary candidates required')
        for boundary in ranges:
            a, b = (_fraction(boundary[k]) for k in ('start_seconds', 'end_seconds'))
            ids = boundary.get('evidence_frame_indices')
            if not lo <= a < b <= hi or not isinstance(ids, list) or not 1 <= len(ids) <= 128 or any(type(i) is not int or not 0 <= i < len(frames) for i in ids):
                raise ValueError('Boundary candidates require source-relative bounds and mapped frame evidence')
            if any(not a <= _fraction(frames[i]['source']['media_offset_seconds']) <= b for i in ids):
                raise ValueError('Boundary evidence frames must lie within the declared candidate range')
            if not all(isinstance(boundary.get(k), str) and 0 < len(boundary[k]) <= 1024 for k in ('basis', 'evidence_ref')):
                raise ValueError('Attributable boundary basis and evidence reference required')
            if boundary.get('physical_completion_confirmed') is not False or type(boundary.get('start_censored')) is not bool or type(boundary.get('end_censored')) is not bool:
                raise ValueError('Candidate boundaries must preserve censored edges and never confirm physical completion')
    padding = doc.get('context_padding')
    if padding != dict(status='unknown', before_seconds=None, after_seconds=None):
        if (not isinstance(padding, dict) or padding.get('status') != 'declared_processing_context' or
                not isinstance(padding.get('basis'), str) or not padding['basis'] or
                any(not 0 <= _fraction(padding.get(k)) <= limits.max_span_seconds for k in ('before_seconds', 'after_seconds'))):
            raise ValueError('Padding needs explicit processing-context provenance, separate from event boundaries')
    playback = doc.get('playback')
    gaps = []
    if playback is not None:
        start, end, duration = (_fraction(playback[k]) for k in ('start_seconds', 'end_seconds', 'duration_seconds'))
        first = _fraction(frames[0]['clip']['pts'])*_fraction(frames[0]['clip']['time_base'])
        last = _fraction(frames[-1]['clip']['pts'])*_fraction(frames[-1]['clip']['time_base'])
        if not start <= first <= last < end or end-start != duration or duration > limits.max_span_seconds+1:
            raise ValueError('Container playback does not enclose mapped frames')
        if start < first:gaps.append(dict(start_seconds=str(start), end_seconds=str(first), reason='before_first_decoded_frame_time', status='unknown'))
        gaps.append(dict(start_seconds=str(last), end_seconds=str(end), reason='after_last_decoded_frame_timestamp_no_duration_proof', status='unknown'))
    doc['unmapped_playback_extents'] = gaps
    doc['mapping_semantics'] = 'discrete decoded frame identities only; no interpolation or continuous-time/business/action-completion proof'
    if len(json.dumps(doc, allow_nan=False).encode()) > limits.max_document_bytes:
        raise ValueError('Frame-map document exceeds bounded checkpoint')
    return doc


@contextmanager
def _held_media(paths, settings):
    """Pinned directory/file descriptors; no local fallback for NAS sources."""
    from .storage import is_nas, storage_binding, verify_root
    from .components import pipeline_module
    with ExitStack() as stack:
        guard = None
        if any(is_nas(p) for p in paths):
            for p in paths:
                verify_root(Path(p),settings.nas_marker_sha256,project_destination=False)
            remote = next(Path(p) for p in paths if is_nas(p))
            config = storage_binding(remote,settings.nas_marker_sha256)
            config['storage'] = dict(archive_root=str(settings.incoming),local_runtime_root=str(settings.runtime_root))
            guard = stack.enter_context(pipeline_module('device_day_write_guard').storage_guard(config))
        held = []
        for path in map(Path, paths):
            if path.is_symlink() or path.absolute().resolve() != path.absolute():raise ValueError('Media path must not traverse symlinks')
            if guard and is_nas(path):
                guard.validate();mount, absolute = guard.mount_for(path)
                parent = guard.directory(mount, absolute.relative_to(mount.path).parts[:-1]);stack.callback(os.close, parent)
                fd = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            else:
                if str(path.absolute()).startswith(('/mnt/', '/media/')):raise ValueError('Unverified source mount')
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):raise ValueError('Original regular media required')
            held.append((fd, before))
        yield held
        for fd, before in held:
            after = os.fstat(fd)
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise Conflict('Media changed across local pixel verification')
        if guard:guard.validate()


def _hash_fd(fd, size, maximum):
    if size > maximum:raise ValueError('Local verification byte budget exceeded')
    h = hashlib.sha256();offset = 0
    while offset < size:
        raw = os.pread(fd, min(1024**2, size-offset), offset)
        if not raw:raise Conflict('Media truncated during verification')
        h.update(raw);offset += len(raw)
    return h.hexdigest()


def verify_local_pixels(document, source_path, clip_path, decoder_provider, settings, *, limits=None):
    """Actual app-owned local byte/PTS/RGB validation, never a supplied result.

    No callback/model is accepted. A fixed owned decoder is invoked through
    the existing bounded subprocess API. Only this function mints the in-process
    authority accepted by registration; JSON flags and receipt claims do not.
    """
    import numpy as np
    limits = limits or FrameMapLimits();doc = validate_frame_map(document, limits=limits)
    validate_provider(decoder_provider)
    started = time.monotonic();counts = {};providers = dict(decoder_provider, max_input_frames=600)
    with _held_media([source_path, clip_path], settings) as held:
        for index, (kind, reference, maximum) in enumerate([('source', doc['source_ref'], limits.max_source_bytes), ('clip', doc['clip_ref'], limits.max_clip_bytes)]):
            fd, st = held[index]
            if st.st_size != reference['size_bytes'] or _hash_fd(fd, st.st_size, maximum) != reference['sha256']:
                raise Conflict('Local source/clip bytes differ from pinned readback')
            expected = {(p[kind]['pts'], p[kind]['time_base']): p[kind] for p in doc['frames']}
            points = list(expected.values());lo = min(_fraction(p['media_offset_seconds']) for p in points)
            hi = max(_fraction(p['media_offset_seconds'])+_fraction(p['time_base']) for p in points)
            if kind == 'clip':
                if doc['playback'] is None:raise ValueError('Actual playback bounds required before local proof')
                probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=start_time,duration',
                    '-of', 'json', f'/proc/{os.getpid()}/fd/{fd}'], capture_output=True, text=True, timeout=30)
                if probe.returncode or len(probe.stdout.encode()) > 65536:raise Conflict('Actual clip playback probe failed')
                actual = json.loads(probe.stdout).get('format', {})
                if (actual.get('duration') is None or
                        _fraction(actual.get('start_time', '0')) != _fraction(doc['playback']['start_seconds']) or
                        _fraction(actual['duration']) != _fraction(doc['playback']['duration_seconds'])):
                    raise Conflict('Actual clip playback differs from the declared container extent')
                origin = _fraction(points[0]['stream_origin_seconds'])
                lo = max(Fraction(0), _fraction(doc['playback']['start_seconds'])-origin)
                hi = _fraction(doc['playback']['end_seconds'])-origin
            seen = set();decoded = 0;cursor = lo
            while cursor < hi:
                end = min(hi, cursor+10)
                # A child opens the parent's held file descriptor namespace.
                path = Path(f'/proc/{os.getpid()}/fd/{fd}')
                for bgr, meta in decode_frames(path, cursor, end, providers):
                    decoded += 1
                    if decoded > limits.max_frames+600:raise ValueError('Decoded local verification frame bound exceeded')
                    key = (meta['pts'], str(_fraction(meta['time_base'])))
                    point = expected.get(key)
                    if point is None:
                        if kind == 'clip':raise Conflict('Unmapped decoded clip frame; whole-frame completeness failed')
                        continue
                    if key in seen:raise Conflict('Duplicate decoded PTS')
                    sha = hashlib.sha256(np.ascontiguousarray(bgr[:, :, ::-1]).tobytes()).hexdigest()
                    if (sha != point['RGB_sha256'] or bgr.shape[:2] != (point['height'], point['width']) or
                            _fraction(meta['media_offset']) != _fraction(point['media_offset_seconds']) or
                            _fraction(meta['stream_origin']) != _fraction(point['stream_origin_seconds'])):
                        raise Conflict('Actual local RGB/PTS/origin differs from readback mapping')
                    seen.add(key)
                cursor = end
            if seen != set(expected):raise Conflict('Mapped source/clip frames were not all decoded')
            counts[kind] = dict(decoded=decoded, matched_unique_frames=len(seen))
    receipt = dict(verification='local_pixel_verified', decoder_provider=decoder_provider,
        decoded=counts, whole_clip_frames_verified=True, original_source_frames_verified=True,
        elapsed_seconds=time.monotonic()-started, world_clock_verified=False,
        quality_or_business_fact_verified=False)
    return LocalPixelProof(fingerprint(doc), doc['source_ref']['sha256'], doc['clip_ref']['sha256'],
                           counts['clip']['matched_unique_frames'], receipt, _AUTHORITY)


def query_frames(document, start_seconds, end_seconds):
    doc = validate_frame_map(document);lo, hi = _fraction(start_seconds), _fraction(end_seconds)
    if not 0 <= lo < hi:raise ValueError('Positive source-relative clip query required')
    points = [p for p in doc['frames'] if lo <= p['clip']['pts']*_fraction(p['clip']['time_base']) < hi]
    return dict(schema_version=SCHEMA, requested_clip_range=[str(lo), str(hi)], mapped_frames=points,
                unmapped_playback_extents=[g for g in doc['unmapped_playback_extents']
                    if _fraction(g['start_seconds']) < hi and _fraction(g['end_seconds']) > lo],
                world_clock=doc['world_clock'], continuous_time_coverage_proven=False,
                content_boundaries=doc['content_boundaries'], event_boundaries=doc['event_boundaries'],
                business_fact_confirmed=False)


class FrameMappedClips:
    def __init__(self, timeline, settings):
        self.timeline, self.settings = timeline, settings
        self.core, self.customer = timeline.core, timeline.customer

    def _verified_media_file(self, media_id, pin):
        """Existing location identities, read through a held verified directory.

        Do not call Media.locate here: its path-based read would precede this
        adapter's mandatory NAS directory-fd guard.
        """
        with self.core.db.transaction(readonly=True) as conn:
            asset = conn.execute('SELECT * FROM media_asset WHERE customer_id=%s AND id=%s', (self.customer, media_id)).fetchone()
            if not asset:raise Missing('Media outside customer scope')
            locations = conn.execute('SELECT * FROM media_location WHERE customer_id=%s AND media_id=%s ORDER BY storage', (self.customer, media_id)).fetchall()
        if asset['evidence_status'] != 'available' or (asset['verified_sha256'],asset['byte_size']) != (pin['sha256'],pin['size_bytes']):
            raise Conflict('Registered media differs from pinned pixel receipt')
        roots = dict(cache=self.settings.cache, incoming=self.settings.incoming)
        for location in locations:
            if location['storage'] not in roots:continue
            relative = location['relative_path'];parts = PurePosixPath(relative).parts
            if not parts or PurePosixPath(relative).is_absolute() or '\\' in relative or any(p in {'', '.', '..'} for p in relative.split('/')):
                raise ValueError('Registered media location is not normalized')
            path = roots[location['storage']].joinpath(*parts)
            try:
                with _held_media([path], self.settings) as held:
                    fd, st = held[0]
                    if st.st_size != pin['size_bytes'] or _hash_fd(fd, st.st_size, FrameMapLimits().max_source_bytes) != pin['sha256']:
                        raise Conflict('Current media bytes differ from the registered readback')
                return
            except FileNotFoundError:
                continue
        raise Conflict('Frame-map media unavailable')

    def register(self, key, clip_media_id, source_media_id, document, selection_key, *, actor, reason, proof=None):
        doc = validate_frame_map(document)
        if not all(isinstance(v, str) and v.strip() and len(v) <= bound
                   for v, bound in [(selection_key,200),(actor,200),(reason,4096)]):
            raise ValueError('Explicit selection, actor and reason required')
        if ident(clip_media_id) == ident(source_media_id):raise ValueError('Clip cannot be its own source')
        if proof is not None and (not isinstance(proof, LocalPixelProof) or proof._authority is not _AUTHORITY or
                proof.document_sha256 != fingerprint(doc) or proof.source_sha256 != doc['source_ref']['sha256'] or
                proof.clip_sha256 != doc['clip_ref']['sha256'] or proof.decoded_clip_frames != len(doc['frames'])):
            raise ValueError('App-owned local pixel proof does not match this exact frame map')
        descriptor = dict(doc, verification='local_pixel_verified' if proof else 'supplied_not_independently_verified',
                          local_verification_receipt=proof.receipt if proof else None)
        ids = [ident(source_media_id), ident(clip_media_id)]
        for mid, pin in zip(ids, (doc['source_ref'], doc['clip_ref'])):
            self._verified_media_file(mid, pin)
        def action(conn):
            rows = conn.execute('SELECT m.*,a.version,a.verified_sha256,a.byte_size,a.evidence_status FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id WHERE m.customer_id=%s AND m.media_id=ANY(%s)', (self.customer, ids)).fetchall()
            by = {str(r['media_id']):r for r in rows}
            if len(by) != 2:raise Missing('Frame-map media outside customer scope')
            source, clip = by[str(ids[0])], by[str(ids[1])]
            if source['kind'] != 'video' or clip['kind'] != 'video' or source['stream_key'] != clip['stream_key']:
                raise ValueError('Frame map requires same-camera source and clip video')
            if doc['playback'] is not None and Fraction(clip['duration_seconds']) != _fraction(doc['playback']['duration_seconds']):
                raise Conflict('Registered clip duration differs from explicit playback extent; use an accurately registered generation')
            for row, pin in [(source, doc['source_ref']), (clip, doc['clip_ref'])]:
                if row['evidence_status'] != 'available' or (row['verified_sha256'],row['byte_size']) != (pin['sha256'],pin['size_bytes']):
                    raise Conflict('Registered media differs from pinned pixel receipt')
            if max(_fraction(p['source']['media_offset_seconds']) for p in doc['frames']) >= Fraction(source['duration_seconds']):
                raise ValueError('Mapped source frames escape registered original media')
            existing = conn.execute('SELECT transform FROM media_derivation WHERE customer_id=%s AND parent_id=%s AND child_id=%s', (self.customer, ids[0], ids[1])).fetchone()
            output = clean(dict(descriptor, selection_key=selection_key, actor=actor, reason=reason,
                media_id=ids[1], media_version=clip['version'], sha256=clip['verified_sha256'],
                sources=[dict(media_id=ids[0], media_version=source['version'], sha256=source['verified_sha256'])]))
            if existing:
                stable_old = {k:v for k,v in existing['transform'].items() if k != 'local_verification_receipt'}
                stable_new = {k:v for k,v in output.items() if k != 'local_verification_receipt'}
                if stable_old != stable_new:raise Conflict('Existing derivation is immutable; use a new media generation')
                return existing['transform']
            cycle = conn.execute('WITH RECURSIVE ancestors(id) AS (SELECT %s::uuid UNION SELECT d.parent_id FROM media_derivation d JOIN ancestors a ON a.id=d.child_id WHERE d.customer_id=%s) SELECT 1 FROM ancestors WHERE id=%s', (ids[0], self.customer, ids[1])).fetchone()
            if cycle:raise Conflict('Frame-map derivation cycle')
            conn.execute('INSERT INTO media_derivation(customer_id,parent_id,child_id,transform) VALUES(%s,%s,%s,%s)', (self.customer, ids[0], ids[1], Jsonb(output)))
            self.timeline._dirty(conn, source['start_at'],source['end_at'],clip['start_at'],clip['end_at'])
            return output
        request_map = {k:v for k,v in descriptor.items() if k != 'local_verification_receipt'}
        return self.timeline._command(key, 'register_retained_frame_map_v2', dict(clip=clip_media_id, source=source_media_id,
            document=request_map, selection_key=selection_key, actor=actor, reason=reason), action)

    def get(self, conn, media_id):
        rows = conn.execute("SELECT transform FROM media_derivation WHERE customer_id=%s AND child_id=%s AND transform->>'schema_version'=%s", (self.customer, ident(media_id), SCHEMA)).fetchall()
        if not rows:return None
        if len(rows) != 1:raise Conflict('Ambiguous frame-map source derivations')
        return rows[0]['transform']

    @staticmethod
    def validate(conn, customer, document):
        """Consumer readiness check; historical get() remains readable as-is."""
        validate_frame_map(document)
        sources = document.get('sources')
        if not isinstance(sources, list) or len(sources) != 1:raise ValueError('One pinned source media identity required')
        for reference in [document, *sources]:
            row = conn.execute('SELECT version,verified_sha256,evidence_status FROM media_asset WHERE customer_id=%s AND id=%s',
                               (customer, ident(reference['media_id']))).fetchone()
            if not row or (row['version'],row['verified_sha256'],row['evidence_status']) != (reference['media_version'],reference['sha256'],'available'):
                raise Conflict('Frame-map provenance unavailable or superseded')
        return document
