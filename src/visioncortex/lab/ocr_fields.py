"""Field semantics over raw OCR, reusing explicit FR display-type rules."""
from __future__ import annotations

from decimal import Decimal
import json
import re

from .gauge import exact

NUMBER = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*([^\d\s.+-].*)?')
ROLES = {'displayed_value', 'process_value', 'setpoint', 'program_parameter',
         'status', 'clock', 'gauge_indication', 'requires_customer_confirmation'}


def _fr_rule(region, value, texts, display_rules, issue):
    """Reuse FR's explicitly confirmed display-type rule, without identity."""
    if (display_rules is None or region.get('unit') != 'g'
            or not (region.get('quantity_kind') == 'mass'
                    or str(region.get('field_key', '')).casefold() in {'mass','weight','质量','重量'})):
        return None
    from field_recognition.recognition.vision_display_rules import configured_rules, apply_rule
    rules = configured_rules(json.dumps(display_rules,ensure_ascii=False,allow_nan=False))
    return apply_rule(dict(label='质量',unit='g',text=' | '.join(texts),value=value,
        **({'quality_issue':issue} if issue else {})),rules)


def validate_registration(document, source_sha256):
    if document.get('schema_version') != 'vc-lab-ocr-frame-registration/1' or document.get('source_sha256') != source_sha256:
        raise ValueError('OCR registration must match the exact original image')
    if document.get('status') != 'verified' or not document.get('layout_revision') or not document.get('instrument_id'):
        raise ValueError('Verified layout and instrument identity required')
    regions = document.get('regions', [])
    if not isinstance(regions, list) or len(regions) > 128:
        raise ValueError('At most 128 explicit fields per photograph')
    keys = set()
    for region in regions:
        key = (region.get('field_key'), region.get('role'), region.get('channel', ''))
        if not isinstance(key[0], str) or not key[0] or key[1] not in ROLES or key in keys:
            raise ValueError('Unique field, role and channel required')
        keys.add(key)
        box = region.get('box')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('Normalized source-frame rectangle required')
        x1, y1, x2, y2 = [exact(v) for v in box]
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError('Field rectangle escaped the image')
        if region.get('data_type', 'number') not in {'number', 'text', 'clock'}:
            raise ValueError('Unknown OCR field data type')
        if region.get('unit') and region.get('unit_basis') not in {'verified_on_panel', 'verified_instrument_specification'}:
            raise ValueError('Units require a verified basis, not inferred instrument type')
        score = exact(region.get('minimum_score', '0.8'))
        if not 0 <= score <= 1:
            raise ValueError('Invalid OCR score threshold')
        for bound in ('min_value', 'max_value'):
            if region.get(bound) is not None:
                exact(region[bound])
        if region.get('min_value') is not None and region.get('max_value') is not None and exact(region['min_value']) > exact(region['max_value']):
            raise ValueError('Invalid verified field range')
    if not regions and not document.get('gauges'):
        raise ValueError('Registration must contain fields or calibrated gauges')
    if len(document.get('gauges', [])) > 16:
        raise ValueError('Too many gauges in one photograph')
    return document


def tokens(result):
    """Retain one OCR token per prediction; do not concatenate unrelated digits."""
    output = []
    for page in result.get('results', []):
        texts, scores = page.get('rec_texts', []), page.get('rec_scores', [])
        if len(texts) != len(scores) or len(texts) > 4096:
            raise ValueError('Malformed or unbounded raw OCR tokens')
        for text, score in zip(texts, scores, strict=False):
            numeric_score = Decimal(str(score))
            if not isinstance(text, str) or len(text) > 4096 or not numeric_score.is_finite() or not 0 <= numeric_score <= 1:
                raise ValueError('Invalid raw OCR token')
            output.append(dict(text=text, score=str(numeric_score)))
    return output


def field_candidate(region, result, *, display_rules=None):
    raw = tokens(result)
    texts = [t['text'].strip().replace('−', '-') for t in raw if t['text'].strip()]
    issues = []
    low_score = any(Decimal(t['score']) < exact(region.get('minimum_score', '0.8')) for t in raw)
    value, unit = None, region.get('unit')
    dtype = region.get('data_type', 'number')
    normalization = None
    raw_value = None
    if not texts:
        issues.append('blank_or_unreadable')
    elif dtype in {'text', 'clock'}:
        # Preserve table/status text. Clock text is never the capture clock.
        value = ' '.join(texts)
        if dtype == 'clock' and not re.fullmatch(r'\d{1,2}:\d{2}(?::\d{2})?', value):
            issues.append('ambiguous_clock_text')
    else:
        numbers, signs, residual = [], [], []
        for text in texts:
            if text in {'-', '+'}:
                signs.append(text)
                continue
            match = NUMBER.fullmatch(text)
            if match:
                number, observed_unit = match.groups()
                numbers.append(number)
                if observed_unit and observed_unit != unit:
                    residual.append(text)
                    issues.append('unit_mismatch_or_unverified')
            elif text != unit:
                residual.append(text)
        if len(numbers) == 1 and len(signs) <= 1 and not residual:
            value = numbers[0]
            if signs:
                if value.startswith(('-', '+')):
                    issues.append('ambiguous_sign')
                else:
                    value = signs[0] + value
            raw_value = value
            normalization = _fr_rule(region,value,texts,display_rules,
                issues[0] if issues else 'low_ocr_score' if low_score else None)
            if normalization and normalization.get('display_rule'):
                value = normalization['value']
                issues.extend(normalization.get('quality_issues',[]))
            numeric = exact(value)
            if region.get('min_value') is not None and numeric < exact(region['min_value']):
                issues.append('outside_verified_range')
            if region.get('max_value') is not None and numeric > exact(region['max_value']):
                issues.append('outside_verified_range')
        else:
            issues.append('ambiguous_or_non_numeric_tokens')
        # Only FR's explicit confirmed template may place a decimal. Never
        # infer a scale multiplier, Hz->rpm or pressure conversion.
    if low_score:
        issues.append('low_ocr_score')
    candidate = value if not issues else None
    candidate_result = dict(field_key=region['field_key'], role=region['role'], channel=region.get('channel', ''),
                value=candidate, tentative_value=value, unit=unit, unit_basis=region.get('unit_basis'),
                status='candidate' if candidate is not None else 'needs_review', issues=sorted(set(issues)),
                raw_tokens=raw, source_box=region['box'], data_type=dtype,
                use_as_capture_time=False, accuracy='NOT_PROVEN')
    if normalization is not None:
        candidate_result['raw_value'] = raw_value
    if normalization and normalization.get('display_rule'):
        candidate_result.update({k:normalization[k] for k in
            ('display_rule','normalization','raw_value','normalized_value') if k in normalization})
    return candidate_result


def consensus(fields):
    """Group like-for-like candidates; equal burst readings still need confirmation."""
    grouped = {}
    for field in fields:
        key = (field['field_key'], field.get('role', 'unspecified'), field.get('channel', ''), field.get('unit'))
        grouped.setdefault(key, []).append(field)
    output = []
    for (name, role, channel, unit), values in sorted(grouped.items(), key=lambda item: str(item[0])):
        distinct = set()
        for value in values:
            if value.get('value') is None:
                continue
            raw = str(value['value'])
            try:
                normalized = str(exact(raw).normalize()) if value.get('data_type', 'number') == 'number' else raw
            except ValueError:
                normalized = raw
            distinct.add(normalized)
        valid = sum(v.get('value') is not None for v in values)
        status = ('conflicting_candidates' if len(distinct) > 1 else 'unreadable' if not distinct else
                  'partial_candidate' if valid != len(values) else 'consistent_candidate')
        output.append(dict(field_key=name, role=role, channel=channel, unit=unit, status=status,
                           readable_candidates=valid, candidates=values, auto_confirmed=False))
    return output
