"""Bounded, opt-in display evidence fusion; never confirms a business fact.

Light is positive evidence. Missing light is UNKNOWN unless a separately
versioned local observation-opportunity model and actual local quality support
OFF. OCR text repetition, fused images and historical values cannot supply that
missing evidence. Results are JSON; retained arrays remain in the bounded ROI
buffer and are never published as invented observations.
"""
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import itertools
import json
import math
import re
import time

from .panel_vision import libraries
from .temporal_registration import TemporalRegistrar, RegistrationConfig, normalized_box

SCHEMA = 'vc-lab-temporal-display/1'
SEGMENTS = {
    '0': 'abcdef', '1': 'bc', '2': 'abdeg', '3': 'abcdg', '4': 'bcfg',
    '5': 'acdfg', '6': 'acdefg', '7': 'abc', '8': 'abcdefg', '9': 'abcdfg',
}
SEGMENT_BOXES = {
    'a': [.23, .04, .77, .17], 'b': [.78, .17, .94, .44],
    'c': [.78, .56, .94, .83], 'd': [.23, .83, .77, .96],
    'e': [.06, .56, .22, .83], 'f': [.06, .17, .22, .44],
    'g': [.23, .44, .77, .56],
}


@dataclass(frozen=True)
class TemporalConfig:
    windows: tuple = (3, 5, 8, 15, 30)
    max_window_seconds: float = 1.0
    max_gap_seconds: float = .2
    min_independent_frames: int = 2
    max_buffer_bytes: int = 64*1024**2
    max_sessions: int = 32
    max_roi_pixels: int = 262144
    min_native_digit_height: int = 12
    registration_mode: str = 'rigid'
    compare_baselines: bool = False
    session_ttl_seconds: float = 86400

    def __post_init__(self):
        if self.windows != (3, 5, 8, 15, 30):
            raise ValueError('Explicit research windows are 3/5/8/15/30')
        if not 0 < self.max_window_seconds <= 1 or not 0 < self.max_gap_seconds <= self.max_window_seconds:
            raise ValueError('PTS span must be bounded to one second')
        if not 0 < self.max_buffer_bytes <= 64*1024**2 or not 0 < self.max_sessions <= 128:
            raise ValueError('Bounded ROI memory/session configuration required')
        if not 0 < self.max_roi_pixels <= 1048576 or not 0 < self.session_ttl_seconds <= 86400:
            raise ValueError('Bounded ROI size/24h retention required')
        if self.min_independent_frames < 2 or self.min_native_digit_height < 8:
            raise ValueError('Independent support/resolution gates cannot be disabled')


@dataclass
class _Frame:
    image: object
    source: dict
    timestamp: Fraction
    digest: str
    observations: dict
    quality: dict
    registration: dict
    text: dict = field(default_factory=dict)


@dataclass
class _State:
    registrar: object
    layout_sha: str
    source_sha: str
    touched: float
    frames: list = field(default_factory=list)
    input_count: int = 0
    last_timestamp: object = None
    last_signature: object = None


def _identity(value):
    keys = ('device_id', 'screen_id', 'field_id', 'track_session_id')
    if not isinstance(value, dict) or any(not isinstance(value.get(k), str) or not 0 < len(value[k]) <= 256 for k in keys):
        raise ValueError('Device/screen/field/track-session identity required')
    return tuple(value[k] for k in keys), {k: value[k] for k in keys}


def _reference(value):
    _, np = libraries()
    if not re.fullmatch('[0-9a-f]{64}', value.get('source_sha256', '')) or type(value.get('pts')) is not int:
        raise ValueError('Pinned original source SHA and integer original PTS required')
    base = Fraction(value['time_base'])
    if base <= 0 or value['pts'] < 0:
        raise ValueError('Positive original time base/nonnegative PTS required')
    matrix = np.asarray(value.get('source_to_roi', np.eye(3)), dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError('Finite invertible source-to-ROI mapping required')
    clean = dict(value, source_to_roi=matrix.tolist())
    encoded = json.dumps(clean, allow_nan=False)
    if len(encoded.encode()) > 16*1024:
        raise ValueError('Source reference exceeds 16KiB metadata bound')
    clean = json.loads(encoded)
    return clean, value['pts']*base


def _digit_box(spec):
    return spec['box'] if isinstance(spec, dict) else spec


def local_quality(image, color):
    """Actual local glare/low-contrast/row-discontinuity estimates, not probability.

    Brightness alone defines neither validity nor OFF. The validity mask is for
    local fusion/ranking and cannot authorize a segment's negative observation.
    """
    cv, np = libraries()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    channels = image.astype(np.int16)
    if color == 'red':
        signal = channels[:, :, 2]-np.maximum(channels[:, :, 0], channels[:, :, 1])
    elif color == 'green':
        signal = channels[:, :, 1]-np.maximum(channels[:, :, 0], channels[:, :, 2])
    else:
        signal = (255-gray).astype(np.int16)
    bright_white = (channels.min(axis=2) > 235) & (channels.max(axis=2)-channels.min(axis=2) < 20)
    emission = color in {'red', 'green'}
    # A white LCD background is ordinary display content. Saturated white may
    # be reported as possible clipping, but is not declared glare/OFF by the
    # emission-display model.
    glare = bright_white if emission else np.zeros(gray.shape, dtype=bool)
    # White glare and severe local clipping have no trustworthy stroke support.
    spread = cv.dilate(gray, np.ones((3, 3), np.uint8)).astype(float)-cv.erode(gray, np.ones((3, 3), np.uint8))
    informative = (spread >= 8) | (signal >= 35)
    rows = np.mean(np.maximum(signal, 0), axis=1)
    row_jumps = np.abs(np.diff(rows, prepend=rows[0]))
    threshold = max(35., float(np.median(row_jumps))*6)
    stripe = (row_jumps > threshold) if emission else np.zeros(rows.shape, dtype=bool)
    unknown = glare | (~informative)
    unknown[stripe, :] = True
    valid = ~unknown
    # Explicitly exclude image borders; a uniform all-one validity mask would
    # conceal unassessed regions and must not masquerade as a degradation model.
    valid[:1, :] = False; valid[-1:, :] = False
    valid[:, :1] = False; valid[:, -1:] = False
    return signal, valid, dict(method='local_emission_contrast_glare_row_change/1' if emission else 'local_lcd_contrast_unknown_regions/1',
        calibrated_probability=False, glare_fraction=float(glare.mean()),
        possible_bright_clipping_fraction=float(bright_white.mean()),
        unknown_fraction=float(unknown.mean()), row_transition_indices=np.flatnonzero(stripe).tolist(),
        laplacian_variance=float(cv.Laplacian(gray, cv.CV_64F).var()),
        validity_definition='observed local contrast/emission outside glare, row discontinuity and unassessed flat regions; never OFF authority')


def _opportunity(visibility, layout, image, matrix):
    cv, np = libraries()
    if not visibility:
        return None, {}, dict(accepted=False, reason='no_calibrated_local_opportunity_model')
    required = ('model_revision', 'evidence_ref', 'method')
    if any(not isinstance(visibility.get(k), str) or not visibility[k] for k in required):
        return None, {}, dict(accepted=False, reason='missing_opportunity_provenance')
    if (visibility['method'] != 'local_segment_visibility/1' or
            visibility['model_revision'] != layout.get('opportunity_model_revision')):
        return None, {}, dict(accepted=False, reason='unregistered_opportunity_model')
    mask = np.asarray(visibility.get('valid_mask'))
    if mask.shape != image.shape[:2] or mask.dtype != np.bool_ or not 0 < np.count_nonzero(mask) < mask.size:
        return None, {}, dict(accepted=False, reason='nonlocal_or_all_one_opportunity_mask')
    mask = cv.warpPerspective(mask.astype(np.uint8), np.asarray(matrix, float),
                             (image.shape[1], image.shape[0]), flags=cv.INTER_NEAREST) > 0
    opportunities = visibility.get('opportunities', {})
    if not isinstance(opportunities, dict):
        return None, {}, dict(accepted=False, reason='invalid_opportunity_regions')
    return mask, opportunities, dict(accepted=True, method=visibility['method'],
        model_revision=visibility['model_revision'], evidence_ref=visibility['evidence_ref'],
        provenance='supplied calibration plus actual local pixel/region gates; not independently attested by this function')


def _observations(image, layout, visibility, matrix, config):
    _, np = libraries()
    signal, local_mask, quality = local_quality(image, layout.get('color', 'dark'))
    opportunity, cells, provenance = _opportunity(visibility, layout, image, matrix)
    regions = []
    emission_gates = {}
    digits = layout.get('digits', [])
    if len(digits) > 12:
        raise ValueError('At most twelve reviewed digit slots')
    for i, digit in enumerate(digits):
        x1, y1, x2, y2 = normalized_box(_digit_box(digit), image.shape)
        if layout.get('color') in {'red', 'green'}:
            # A colored optical lens can have stronger chroma than the lit
            # stroke. Positive emission therefore also needs actual brightness
            # contrast within this digit slot. Darkness still cannot prove OFF.
            channel = 2 if layout['color'] == 'red' else 1
            slot = image[y1:y2, x1:x2, channel]
            background = float(np.percentile(slot, 25))
            upper = float(np.percentile(slot, 98))
            span = upper-background
            emission_gates[str(i)] = dict(method='same_slot_positive_emission_contrast/1',
                background_percentile=25, upper_percentile=98,
                background_channel_level=background, upper_channel_level=upper,
                channel_contrast_span=span, minimum_contrast_span=35.,
                minimum_channel_level=background+max(25., .55*span),
                accepted=span >= 35., calibrated_probability=False,
                purpose='positive stroke support only; never an OFF opportunity')
        overrides = digit.get('segments', SEGMENT_BOXES) if isinstance(digit, dict) else SEGMENT_BOXES
        if set(overrides) != set('abcdefg'):
            raise ValueError('Each digit requires seven segment regions')
        for segment, box in overrides.items():
            sx1, sy1, sx2, sy2 = normalized_box(box, (y2-y1, x2-x1))
            regions.append((f'{i}:{segment}', [x1+sx1, y1+sy1, x1+sx2, y1+sy2], y2-y1, 'segment'))
    for i, symbol in enumerate(layout.get('symbols', [])):
        if symbol.get('kind') not in {'decimal', 'sign', 'unit'}:
            raise ValueError('Decimal/sign/unit symbol evidence required')
        regions.append((f"{symbol['kind']}:{i}", list(normalized_box(symbol['box'], image.shape)), 1000, symbol['kind']))
    observed = {}
    for key, box, digit_height, kind in regions:
        x1, y1, x2, y2 = box
        patch = signal[y1:y2, x1:x2]
        valid = local_mask[y1:y2, x1:x2]
        area = patch.size
        bright = (patch > (35 if layout.get('color') in {'red', 'green'} else 65)) & valid
        emission_gate = emission_gates.get(key.split(':')[0]) if kind == 'segment' else None
        if emission_gate is not None:
            channel = 2 if layout['color'] == 'red' else 1
            bright &= (image[y1:y2, x1:x2, channel] >= emission_gate['minimum_channel_level'])
            if not emission_gate['accepted']:
                bright[:] = False
        on_fraction = float(bright.mean()) if area else 0.
        on = area >= 2 and int(bright.sum()) >= 2 and on_fraction >= (.18 if kind == 'decimal' else .30)
        # Unit semantics cannot be read from a colored blob. Its glyph proof and
        # exact text still have to come from the OCR/completeness contract.
        if kind == 'unit':
            on = False
        off = False; basis = cells.get(key, {})
        opportunity_fraction = 0.
        if opportunity is not None and area:
            opportunity_fraction = float(opportunity[y1:y2, x1:x2].mean())
            # Unlike an all-pixel quality flag, this names the actual scan slot
            # opportunity and its calibrated explanation. Darkness alone fails.
            off = (opportunity_fraction >= .95 and isinstance(basis, dict) and
                basis.get('observable') is True and bool(basis.get('basis')) and
                float(np.max(patch)) <= 20 and quality['glare_fraction'] < .15 and
                digit_height >= config.min_native_digit_height)
        status = 'ON' if on else ('OFF' if off else 'UNKNOWN')
        if on and off:
            status = 'UNKNOWN'
        observed[key] = dict(state=status, box=box, on_fraction=on_fraction,
            observed_active_pixels=int(bright.sum()), native_pixels=int(area),
            native_digit_height=int(digit_height), opportunity_fraction=opportunity_fraction,
            off_basis=basis if off else None, positive_emission_gate=emission_gate)
    quality['positive_emission_model'] = dict(method='same_slot_positive_emission_contrast/1',
        digit_slots=emission_gates, calibrated_probability=False,
        limitation='uncalibrated local brightness gate; positive evidence does not establish OFF or a complete glyph')
    quality['opportunity_model'] = provenance
    quality['state_witness'] = _state_witness(image, layout)
    return observed, quality


def _state_witness(image, layout):
    """Optional calibrated *visible latched-mode* indicator, never a value hint.

    Consistent partial digits alone cannot rule out a changing display. A
    caller cannot authorize same-state fusion merely by passing stable=True.
    This optional contract requires actual matching indicator pixels and a
    reviewed device-mode calibration explaining that this indicator latches
    the displayed field. No such calibration is supplied by default.
    """
    cv, np = libraries()
    spec = layout.get('state_witness')
    rejected = dict(accepted=False, reason='no_calibrated_same_state_visual_witness')
    if not isinstance(spec, dict) or spec.get('kind') != 'latched_display_indicator/1':
        return rejected
    if (not spec.get('calibration_ref') or not spec.get('model_revision') or
            spec.get('semantics') != 'this_field_value_latched_while_indicator_visible'):
        return dict(rejected, reason='unqualified_indicator_semantics')
    x1, y1, x2, y2 = normalized_box(spec['box'], image.shape)
    patch = cv.cvtColor(image[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)
    template = np.asarray(spec.get('template'), dtype=float)
    if (template.shape != patch.shape or template.size < 16 or not np.isfinite(template).all() or
            template.min() < 0 or template.max() > 255 or template.std() < 12):
        return dict(rejected, reason='noninformative_indicator_template')
    error = float(np.mean(np.abs(patch.astype(float)-template)))
    if error > 8 or float((patch > 245).mean()) > .1:
        return dict(rejected, reason='indicator_pixel_or_glare_mismatch', mean_absolute_error=error)
    return dict(accepted=True, method=spec['kind'], calibration_ref=spec['calibration_ref'],
        model_revision=spec['model_revision'], template_sha256=hashlib.sha256(template.astype(np.uint8).tobytes()).hexdigest(),
        box=[x1, y1, x2, y2], mean_absolute_error=error,
        semantics=spec['semantics'], calibration_provenance='supplied reviewed mode semantics; no reading value supplied')


def _candidates(observations, layout):
    choices = []
    for i, _ in enumerate(layout.get('digits', [])):
        compatible = []
        for digit, lit in SEGMENTS.items():
            if all(observations[f'{i}:{s}']['state'] == 'UNKNOWN' or
                   (observations[f'{i}:{s}']['state'] == 'ON') == (s in lit) for s in 'abcdefg'):
                compatible.append(digit)
        choices.append(compatible)
    if not choices or any(not c for c in choices):
        return [], 0, ['no_compatible_complete_digit_sequence']
    symbols = layout.get('symbols', [])
    decimal = [(i, s) for i, s in enumerate(symbols) if s['kind'] == 'decimal']
    sign = [(i, s) for i, s in enumerate(symbols) if s['kind'] == 'sign']
    reasons = []
    format_ref = layout.get('format_evidence_ref')
    if not decimal and not (format_ref and layout.get('decimal_mode') == 'none'):
        reasons.append('decimal_observation_missing')
    if not sign and not (format_ref and layout.get('supports_negative') is False):
        reasons.append('sign_observation_missing')
    for i, symbol in enumerate(symbols):
        if observations[f"{symbol['kind']}:{i}"]['state'] == 'UNKNOWN':
            reasons.append(symbol['kind']+'_unresolved')
    if layout.get('expected_unit'):
        reasons.append('unit_glyph_completeness_required')
    count = math.prod(len(c) for c in choices)
    values = []
    for chars in itertools.islice(itertools.product(*choices), 64):
        current = [''.join(chars)]
        for i, symbol in reversed(decimal):
            state = observations[f'decimal:{i}']['state']
            if state != 'OFF':
                after = symbol.get('after_digit')
                if type(after) is not int or not 0 <= after < len(chars):
                    raise ValueError('Decimal evidence must identify its digit position')
                dotted = [v[:after+1]+'.'+v[after+1:] for v in current]
                current = (dotted if state == 'ON' else current+dotted)[:64]
                if state == 'UNKNOWN':
                    count *= 2 if not values else 1
        for i, symbol in sign:
            state = observations[f'sign:{i}']['state']
            if state != 'OFF':
                negative = ['-'+v for v in current]
                current = (negative if state == 'ON' else current+negative)[:64]
        values.extend(current)
        if len(values) >= 64:
            break
    return list(dict.fromkeys(values))[:64], max(count, len(values)), sorted(set(reasons))


def _ocr(callback, image):
    if callback is None:
        return dict(raw_text='', candidates=[])
    value = callback(image)
    if not isinstance(value, dict) or not isinstance(value.get('raw_text', ''), str):
        raise ValueError('OCR callback must return original raw_text and optional candidates')
    if value.get('score') is not None and not math.isfinite(float(value['score'])):
        raise ValueError('Finite uncalibrated OCR score required')
    if (len(value.get('raw_text', '')) > 4096 or not isinstance(value.get('candidates', []), list) or
            len(value.get('candidates', [])) > 64 or any(not isinstance(v, str) or len(v) > 4096 for v in value.get('candidates', []))):
        raise ValueError('Bounded original OCR text/candidates required')
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded.encode()) > 128*1024:
        raise ValueError('OCR callback exceeds 128KiB metadata bound')
    return json.loads(encoded)


class TemporalDisplay:
    def __init__(self, config=None):
        self.config = config or TemporalConfig()
        self.states = {}
        self.numeric_reader = None

    @property
    def buffer_bytes(self):
        return sum(s.registrar.memory_bytes+sum(f.image.nbytes for f in s.frames) for s in self.states.values()) + (self.numeric_reader.buffer_bytes if self.numeric_reader else 0)

    def reset(self, identity, reason='explicit_track_reset'):
        key, _ = _identity(identity)
        self.states.pop(key, None)
        if self.numeric_reader:
            self.numeric_reader.reset(key)
        return dict(reset=True, reason=reason)

    def _new(self, key, image, layout, ref):
        if self.numeric_reader:
            self.numeric_reader.reset(key)
        boxes = list(layout.get('dynamic_boxes') or [_digit_box(d) for d in layout.get('digits', [])])
        boxes += [s['box'] for s in layout.get('symbols', [])]
        if layout.get('state_witness'):
            boxes.append(layout['state_witness']['box'])
        registrar = TemporalRegistrar(image, boxes, RegistrationConfig(mode=layout.get('registration_mode', self.config.registration_mode)))
        state = _State(registrar, hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest(),
                       ref['source_sha256'], time.monotonic())
        if len(self.states) >= self.config.max_sessions:
            oldest = min(self.states, key=lambda k: self.states[k].touched)
            self.states.pop(oldest)
            if self.numeric_reader:
                self.numeric_reader.reset(oldest)
        self.states[key] = state
        return state

    def observe(self, image_bgr, *, identity, source_ref, layout, ocr=None, visibility=None, source_registration=None):
        started = time.perf_counter()
        clocks = dict(ocr_callback_seconds=0., ocr_invocations=0,
                      registration_seconds=None, local_quality_and_segments_seconds=None,
                      local_fusion_and_decoding_enclosing_seconds=None)
        def timed_ocr(image, **kwargs):
            tick = time.perf_counter()
            try:
                return ocr(image, **kwargs)
            finally:
                clocks['ocr_callback_seconds'] += time.perf_counter()-tick
                clocks['ocr_invocations'] += 1
        result = self._observe(image_bgr, identity=identity, source_ref=source_ref, layout=layout,
                               ocr=timed_ocr if ocr else None, visibility=visibility, clocks=clocks,
                               source_registration=source_registration)
        clocks['whole_observe_seconds'] = time.perf_counter()-started
        result['stage_seconds'] = clocks
        result['stage_clock_semantics'] = 'OCR inside local fusion/baseline controls is nested; enclosing stages must not be added'
        result['external_stage_seconds'] = dict(video_decode=None, queue_wait=None, database_write=None, publication=None)
        if len(json.dumps(result, allow_nan=False).encode()) > 2*1024**2:
            raise ValueError('Temporal evidence result exceeds 2MiB bounded checkpoint')
        return result

    def _observe(self, image_bgr, *, identity, source_ref, layout, ocr, visibility, clocks, source_registration=None):
        cv, np = libraries()
        key, identity = _identity(identity)
        ref, timestamp = _reference(source_ref)
        image = np.asarray(image_bgr)
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or
                min(image.shape[:2]) < 8 or image.shape[0]*image.shape[1] > self.config.max_roi_pixels):
            raise ValueError('Bounded uint8 BGR field ROI required')
        if layout.get('display_type') not in {'seven_segment', 'lcd'} or not layout.get('revision'):
            raise ValueError('Versioned explicit display layout required')
        if (len(layout.get('digits', [])) > 12 or len(layout.get('symbols', [])) > 16 or
                len(layout.get('dynamic_boxes', [])) > 64 or
                len(json.dumps(layout, allow_nan=False).encode()) > 128*1024):
            raise ValueError('Bounded display layout: 12 digits/16 symbols/64 exclusions/128KiB')
        now = time.monotonic(); resets = []
        for expired in [k for k, v in self.states.items() if now-v.touched > self.config.session_ttl_seconds]:
            self.states.pop(expired)
            if self.numeric_reader:
                self.numeric_reader.reset(expired)
        state = self.states.get(key)
        layout_sha = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
        if state and (state.source_sha != ref['source_sha256'] or state.layout_sha != layout_sha):
            resets.append('source_or_layout_revision_changed'); self.states.pop(key); state = None
        if state and state.last_timestamp is not None:
            delta = timestamp-state.last_timestamp
            if delta <= 0:
                self.reset(identity)
                return self._rejected(identity, ref, 'duplicate_pts' if delta == 0 else 'reverse_pts', resets+['timestamp_state_reset'])
            if delta > self.config.max_gap_seconds:
                resets.append('source_pts_gap'); self.states.pop(key); state = None
        registration_tick = time.perf_counter()
        registration = dict(status='reference', accepted=True, method='reference_identity',
                            matrix_current_to_reference=np.eye(3).tolist(), dynamic_features_excluded=True)
        reference_shape = None
        if source_registration is not None:
            if not layout.get('numeric_native_ocr') or not layout.get('numeric_anchor'):
                raise ValueError('Tracked source geometry requires the native numeric path')
            from .source_tracking import align_native_roi
            aligned, reference_shape = align_native_roi(image, source_registration, ref)
            registration = dict(source_registration)
            if state is None:
                state = self._new(key, aligned, layout, ref)
        elif state:
            aligned, registration = state.registrar.register(image)
            if aligned is None:
                clocks['registration_seconds'] = time.perf_counter()-registration_tick
                self.reset(identity)
                return self._rejected(identity, ref, 'registration_failed', resets+[registration['reason']], registration)
        else:
            state = self._new(key, image, layout, ref); aligned = image.copy()
        clocks['registration_seconds'] = time.perf_counter()-registration_tick
        state.touched = now; state.input_count += 1; state.last_timestamp = timestamp
        if layout.get('numeric_anchor') is True:
            # Opt-in source-frame reading reuses this original registration and
            # exact caller identity. It never changes the legacy fusion default.
            from dataclasses import replace
            from .video_numeric import NumericVideoReader
            other_bytes = self.buffer_bytes - (self.numeric_reader.buffer_bytes if self.numeric_reader else 0)
            budget = self.config.max_buffer_bytes - other_bytes
            if budget <= 0:
                self.reset(identity)
                return self._rejected(identity, ref, 'roi_byte_budget_exceeded', resets, registration)
            if self.numeric_reader is None:
                self.numeric_reader = NumericVideoReader(replace(self.config, max_buffer_bytes=budget))
            else:
                self.numeric_reader.config = replace(self.config, max_buffer_bytes=budget)
            anchor_ref = dict(ref, source_to_roi=(np.asarray(registration['matrix_current_to_reference']) @ np.asarray(ref['source_to_roi'])).tolist(),
                              registration=registration)
            native = None
            if layout.get('numeric_native_ocr') is True:
                from .native_numeric import prepare, source
                try:
                    native = prepare(image, layout, registration['matrix_current_to_reference'],
                                     reference_shape=reference_shape)
                    native['source'] = source(native, ref, anchor_ref)
                except ValueError as exc:
                    return self._rejected(identity, ref, 'native_digit_crop_rejected', resets+[str(exc)], registration)
            result = self.numeric_reader.observe(aligned, key=key, identity=identity, source_ref=anchor_ref, layout=layout, ocr=ocr, native=native)
            result['registration'] = result['source_refs'][0].get('registration', registration)
            return result
        quality_tick = time.perf_counter()
        observations, quality = _observations(aligned, layout, visibility,
            registration['matrix_current_to_reference'], self.config)
        clocks['local_quality_and_segments_seconds'] = time.perf_counter()-quality_tick
        while state.frames and timestamp-state.frames[0].timestamp > self.config.max_window_seconds:
            state.frames.pop(0)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        signature = tuple((k, v['state']) for k, v in sorted(observations.items()))
        repeated = any(f.digest == digest for f in state.frames)
        near_repeat = bool(state.frames and signature == state.last_signature and
            float(np.mean(np.abs(aligned.astype(float)-state.frames[-1].image.astype(float)))) < 1.)
        if repeated or near_repeat:
            return self._rejected(identity, ref, 'duplicate_or_near_identical_pixels_do_not_add_support', resets, registration,
                                  input_count=state.input_count)
        state.last_signature = signature
        current = _Frame(aligned.copy(), ref, timestamp, digest, observations, quality, registration)
        while state.frames and len(state.frames) >= 30:
            state.frames.pop(0)
        # A reliable opposite segment observation is a state change or an
        # unmodelled degradation. Both forbid cross-state accumulation.
        conflicts = []
        for old in state.frames:
            conflicts.extend(k for k, value in observations.items() if value['state'] != 'UNKNOWN' and
                old.observations.get(k, {}).get('state') not in {None, 'UNKNOWN', value['state']})
        if conflicts:
            resets.append('content_change_or_degradation_conflict'); state.frames.clear()
        state.frames.append(current)
        while self.buffer_bytes > self.config.max_buffer_bytes and len(state.frames) > 1:
            state.frames.pop(0); resets.append('roi_byte_budget_eviction')
        if self.buffer_bytes > self.config.max_buffer_bytes:
            self.states.pop(key)
            return self._rejected(identity, ref, 'roi_byte_budget_exceeded', resets, registration)
        current_values, _, current_reasons = _candidates(observations, layout) if layout['display_type'] == 'seven_segment' else ([], 0, [])
        complete_single = bool(len(current_values) == 1 and not current_reasons)
        checkpoint = len(state.frames) in self.config.windows
        if not state.frames[:-1] or checkpoint or complete_single:
            current.text = _ocr(ocr, aligned)
        fusion_tick = time.perf_counter()
        result = self._evaluate(state, identity, layout, current, resets, conflicts, complete_single, ocr)
        clocks['local_fusion_and_decoding_enclosing_seconds'] = time.perf_counter()-fusion_tick
        if self.config.compare_baselines and checkpoint:
            arrays = np.stack([f.image for f in state.frames])
            result['baseline_controls'] = {name: _ocr(ocr, derived) for name, derived in [
                ('best', max(state.frames, key=lambda f: f.quality['laplacian_variance']).image),
                ('mean', arrays.mean(axis=0).astype(np.uint8)),
                ('median', np.median(arrays, axis=0).astype(np.uint8)),
                ('maximum', arrays.max(axis=0))]}
            result['baseline_controls_are_independent_evidence'] = False
        # No arrays leak into results. Fail locally if a callback violated the
        # JSON contract rather than letting a writer invent serialization.
        if len(json.dumps(result, allow_nan=False).encode()) > 2*1024**2:
            raise ValueError('Temporal evidence result exceeds 2MiB bounded checkpoint')
        return result

    def _rejected(self, identity, ref, reason, resets, registration=None, input_count=1):
        return dict(schema_version=SCHEMA, identity=identity, status='uncertain', value_text=None,
            raw_text='', candidates=[], method='rejected_observation', source_refs=[ref],
            observation_time_start=dict(pts=ref['pts'], time_base=ref['time_base']),
            observation_time_end=dict(pts=ref['pts'], time_base=ref['time_base']),
            input_frame_count=input_count, selected_frame_count=0, support_map=[],
            selected_span_seconds=0., buffer_bytes=self.buffer_bytes,
            registration=registration, reset_reasons=resets, rejection_reasons=[reason],
            unresolved_regions=[], conflicting_evidence=[], gates=dict(confirmed=False),
            quality_score_calibrated=False, confirmed_scope='visual_reading_only_not_business_fact')

    def _evaluate(self, state, identity, layout, current, resets, conflicts, complete_single, ocr):
        cv, np = libraries()
        selected = [current] if complete_single else list(state.frames)
        combined = {}; support = []; fused = current.image.copy()
        for key in current.observations:
            supported = [(i, f, f.observations[key]) for i, f in enumerate(selected)
                         if f.observations[key]['state'] != 'UNKNOWN']
            kinds = {v['state'] for _, _, v in supported}
            obs = dict(current.observations[key])
            obs['state'] = next(iter(kinds)) if len(kinds) == 1 else 'UNKNOWN'
            combined[key] = obs
            support.append(dict(region=key, box=obs['box'], state=obs['state'],
                source_ref_indices=[i for i, _, _ in supported], conflicting=len(kinds) > 1,
                unknown=not supported, coordinate_frame='registered_field_roi'))
            if supported and len(kinds) == 1:
                _, best, _ = max(supported, key=lambda p: (p[2]['on_fraction'], p[1].quality['laplacian_variance']))
                x1, y1, x2, y2 = obs['box']; fused[y1:y2, x1:x2] = best.image[y1:y2, x1:x2]
        values, count, reasons = _candidates(combined, layout) if layout['display_type'] == 'seven_segment' else ([], 0, [])
        if layout['display_type'] == 'lcd':
            raw = current.text.get('raw_text', '')
            values = list(dict.fromkeys(current.text.get('candidates', [])+[raw] if raw else current.text.get('candidates', [])))[:64]
            reasons.append('lcd_glyph_completeness_unproven')
        raw_text = current.text.get('raw_text', '')
        fused_text = None
        if not complete_single and len(selected) in self.config.windows:
            fused_text = _ocr(ocr, fused)
            raw_text = fused_text.get('raw_text', '') or raw_text
        # OCR only supplies candidate text. A score never replaces segment or
        # glyph completeness, decimal/sign/unit or state gates.
        if raw_text and raw_text not in values:
            values.append(raw_text)
            reasons.append('ocr_segment_or_format_conflict' if layout['display_type'] == 'seven_segment' else 'lcd_candidate_only')
        native_resolution = all(v['native_digit_height'] >= self.config.min_native_digit_height
            for k, v in combined.items() if ':' in k and k.split(':')[0].isdigit())
        independent = len({tuple((k, v['state']) for k, v in sorted(f.observations.items())) for f in selected})
        only_positive_multi = len(selected) > 1 and any(not any(combined[f'{i}:{s}']['state'] == 'OFF' for s in 'abcdefg')
                                                       for i, _ in enumerate(layout.get('digits', [])))
        if only_positive_multi:
            reasons.append('state_change_vs_complementary_missing_segments_unresolved')
        same_state_witness = all(f.quality['state_witness']['accepted'] for f in selected)
        if not complete_single and len(selected) > 1 and not same_state_witness:
            reasons.append('state_change_vs_degradation_not_excluded')
        if conflicts:
            reasons.append('state_boundary_or_visibility_conflict')
        if not complete_single and len(selected) < 3:
            reasons.append('minimum_adaptive_window_not_reached')
        if not complete_single and independent < self.config.min_independent_frames:
            reasons.append('insufficient_nonrepeated_local_information')
        if not native_resolution:
            reasons.append('insufficient_native_digit_resolution')
        unknown = [s['region'] for s in support if s['unknown']]
        gates = dict(unique_supported_reading=count == 1 and len(values) == 1,
            native_resolution=native_resolution, original_time_strict=True,
            registration_accepted=current.registration['accepted'],
            no_observed_state_conflict=not conflicts, symbols_and_format_observed=not any(
                'decimal' in r or 'sign_' in r or 'unit_' in r for r in reasons),
            sufficient_independent_support=complete_single or independent >= self.config.min_independent_frames,
            no_positive_only_cross_state_union=not only_positive_multi,
            same_state_visual_witness=complete_single or same_state_witness,
            whole_field_glyph_proof=layout['display_type'] == 'seven_segment',
            confirmed=False)
        confirmed = len(values) == 1 and count == 1 and not reasons and all(v for k, v in gates.items() if k != 'confirmed')
        gates['confirmed'] = confirmed
        status = 'confirmed' if confirmed else ('uncertain' if values or any(v['state'] == 'ON' for v in combined.values()) else 'unreadable')
        references = []
        for f in selected:
            reference = dict(f.source)
            matrix = np.asarray(f.registration['matrix_current_to_reference'])@np.asarray(f.source['source_to_roi'])
            reference['source_to_registered_roi'] = matrix.tolist()
            references.append(reference)
        return dict(schema_version=SCHEMA, identity=identity, status=status,
            value_text=values[0] if confirmed else None, raw_text=raw_text,
            candidates=values[:64], candidate_count=count,
            candidate_expansion_truncated=count > 64, unit=None,
            method=('reliable_single_frame_segments' if complete_single else 'local_temporal_segment_evidence')
                if layout['display_type'] == 'seven_segment' else 'temporal_ocr_candidate_without_glyph_proof',
            source_refs=references, observation_time_start=dict(pts=selected[0].source['pts'], time_base=selected[0].source['time_base']),
            observation_time_end=dict(pts=selected[-1].source['pts'], time_base=selected[-1].source['time_base']),
            observation_time_semantics='evidence support range; not an invented instantaneous reading at window end',
            input_frame_count=state.input_count, selected_frame_count=len(selected),
            independent_information_patterns=independent, adaptive_windows=list(self.config.windows),
            selected_span_seconds=float(selected[-1].timestamp-selected[0].timestamp),
            support_map=support, segment_observations=combined, unresolved_regions=unknown,
            conflicting_evidence=sorted(set(conflicts)), reset_reasons=sorted(set(resets)),
            rejection_reasons=sorted(set(reasons)), registration=current.registration,
            quality=current.quality, gates=gates, quality_score_calibrated=False,
            ocr_views=dict(current_original=current.text, local_fusion=fused_text,
                           same_source_images_are_not_independent_evidence=True),
            fusion=dict(method='one_supported_source_patch_per_local_segment',
                derived_image_sha256=hashlib.sha256(fused.tobytes()).hexdigest(),
                derived_pixels_are_new_observations=False, unknown_regions_not_filled_as_evidence=True),
            buffer_bytes=self.buffer_bytes, confirmed_scope='visual_reading_only_not_business_fact')
