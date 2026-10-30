"""Spatial field candidates from detected display crops, never asset identity."""
from decimal import Decimal
import math
import re

from .ocr_fields import field_candidate

REVISION = 'detected-panel-fields/2'
UNITS = {'g': ('质量', 'g'), 'mg': ('质量', 'mg'), 'kg': ('质量', 'kg'),
         '℃': ('温度', '°C'), '°C': ('温度', '°C'), '℉': ('温度', '°F'), '°F': ('温度', '°F'),
         'rpm': ('转速', 'rpm'), 'r/min': ('转速', 'rpm'),
         'Pa': ('压力', 'Pa'), 'kPa': ('压力', 'kPa'), 'MPa': ('压力', 'MPa'),
         '%': ('百分比', '%'), '%RH': ('湿度', '%'), 'V': ('电压', 'V'), 'A': ('电流', 'A')}
LABELS = {'PV': 'process_value', 'SV': 'setpoint'}
QUANTITIES = {'温度':'温度', 'Temperature':'温度', 'TEMPERATURE':'温度',
              '湿度':'湿度', 'Humidity':'湿度', 'HUMIDITY':'湿度', 'RH':'湿度',
              '质量':'质量', '重量':'质量', '转速':'转速'}
VALUE = re.compile(r'(?:(PV|SV)\s*[:：]?\s*)?([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([^\d\s.+-].*)?')


def _rows(page, width, height):
    texts, scores, boxes = page.get('rec_texts', []), page.get('rec_scores', []), page.get('rec_boxes', [])
    if len(texts) != len(scores) or len(texts) != len(boxes) or len(texts) > 4096:
        raise ValueError('Panel tokens require one source box and score per text')
    result = []
    for index, (text, score, box) in enumerate(zip(texts, scores, boxes, strict=True)):
        if (not isinstance(text, str) or len(text) > 4096 or type(score) not in (float, int)
                or not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4
                or any(type(x) not in (float, int) or not math.isfinite(x) for x in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError('Invalid panel OCR geometry/confidence')
        result.append(dict(index=index, text=text.strip().replace('−', '-'), score=score, box=box))
    return result


def _near(row, other):
    a, b = row['box'], other['box']; h = max(a[3]-a[1], b[3]-b[1])
    if abs((a[1]+a[3]-b[1]-b[3])/2) > .6*h:
        return False
    gap = max(b[0]-a[2], a[0]-b[2], 0)
    return gap <= 2*h


def _overlap(a, b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
    return intersection/max(area,1e-9)


def _candidate_pages(result, width, height):
    """Keep full-frame unit/label context; detector misses remain explicit.

    Text-only regions are *not* verified screens: bottle labels can also carry
    quantities. This fallback creates reviewable text candidates, never facts.
    """
    detected=[p for p in result.get('results',[]) if p.get('panel_id')]
    for page in detected:
        bounds=page.get('panel_box')
        if (not isinstance(bounds,list) or len(bounds)!=4
                or any(type(v) is not int for v in bounds)
                or not 0<=bounds[0]<bounds[2]<=width or not 0<=bounds[1]<bounds[3]<=height):
            raise ValueError('Invalid source panel extent')
    derived=[]
    for page in result.get('results',[]):
        if page.get('panel_id'):continue
        rows=_rows(page,width,height)
        for row in rows:
            match=VALUE.fullmatch(row['text'])
            if not match:continue
            _,_,unit=match.groups()
            if unit is not None and unit not in UNITS:continue
            neighbours=[r for r in rows if r is not row and _near(row,r)
                        and (r['text'] in UNITS or r['text'] in LABELS or r['text'] in QUANTITIES or r['text'] in {'-','+'})]
            supporting=[p for p in detected if _overlap(row['box'],p['panel_box'])>.5]
            if unit is None and not any(r['text'] in UNITS for r in neighbours) and len(supporting)!=1:
                continue
            group=[row]+neighbours
            competing_unit=any(VALUE.fullmatch(other['text']) and _near(other,neighbour)
                for neighbour in neighbours if neighbour['text'] in UNITS
                for other in rows if other is not row and other is not neighbour)
            bounds=[math.floor(min(r['box'][0] for r in group)),math.floor(min(r['box'][1] for r in group)),
                    math.ceil(max(r['box'][2] for r in group)),math.ceil(max(r['box'][3] for r in group))]
            derived.append(dict(panel_id=supporting[0]['panel_id'] if len(supporting)==1 else 'text-region-'+str(len(derived)+1),
                panel_box=bounds,geometry_basis='source_ocr_numeric_unit_cluster',
                screen_localization_supported=len(supporting)==1,
                rec_texts=[r['text'] for r in group],rec_scores=[r['score'] for r in group],rec_boxes=[r['box'] for r in group],
                source_token_indices=[r['index'] for r in group],view='whole_frame_context'))
            derived[-1]['competing_unit_neighbour']=competing_unit
    return derived+detected


def detected_fields(result, *, display_rules=None):
    """Use one displayed token at a time; nearby numbers are never joined.

    Explicit units/PV/SV may name a candidate. A naked integer remains pending
    decimal interpretation. A naked decimal can be saved as a displayed-number
    candidate with unknown unit. None of these is a confirmed business reading.
    """
    localization = result.get('panel_localization') or {}
    size = localization.get('source_size')
    if not size:
        return [], dict(revision=REVISION, status='no_panel_locator', panels=0, text_regions=0, fields=0)
    if (not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size)
            or size[0]*size[1] > 16_000_000):
        raise ValueError('Invalid source image extent for panel fields')
    width, height = size; fields = []; panels = []
    for page in _candidate_pages(result,width,height):
        panel = page.get('panel_id')
        if not panel:
            continue  # Whole-image fallback can include bottle labels and dial ticks.
        if page.get('geometry_basis') not in {'model_detected_lcd_screen_not_instrument_identity','source_ocr_numeric_unit_cluster'}:
            raise ValueError('Panel candidate has no supported spatial basis')
        bounds = page.get('panel_box')
        if (not isinstance(panel, str) or len(panel) > 80 or not isinstance(bounds, list) or len(bounds) != 4
                or any(type(v) is not int for v in bounds)
                or not 0 <= bounds[0] < bounds[2] <= width or not 0 <= bounds[1] < bounds[3] <= height):
            raise ValueError('Invalid source panel extent')
        rows = _rows(page, width, height); panels.append(panel)
        for row in rows:
            box, text = row['box'], row['text']
            match = VALUE.fullmatch(text)
            if not (bounds[0] <= box[0] < box[2] <= bounds[2] and bounds[1] <= box[1] < box[3] <= bounds[3]):
                raise ValueError('OCR token crosses its source panel crop')
            if match is None:
                continue
            label, number, unit_text = match.groups(); support = [row]; issues = []
            if page.get('competing_unit_neighbour') and unit_text is None:
                issues.append('ambiguous_unit_neighbours')
            if unit_text is not None and unit_text not in UNITS:
                continue  # Clock text, model numbers and mixed glyphs stay raw.
            signs = [r for r in rows if r is not row and r['text'] in {'-', '+'}
                     and r['box'][2] <= box[0] and _near(row, r)]
            if signs:
                if len(signs) != 1 or number.startswith(('-', '+')):
                    issues.append('ambiguous_sign')
                else:
                    number = signs[0]['text'] + number; support.append(signs[0])
            # Only textual neighbours within this same panel and line may add
            # semantics. An unrelated unit on another panel is never borrowed.
            units = [r for r in rows if r is not row and r['text'] in UNITS and r['box'][0] >= box[2] and _near(row, r)]
            if any(any(v is not row and VALUE.fullmatch(v['text']) and _near(v, u) for v in rows) for u in units):
                units = []; issues.append('ambiguous_unit_neighbours')
            if unit_text is None and len(units) == 1:
                unit_text = units[0]['text']; support.append(units[0])
            elif unit_text is None and len(units) > 1:
                issues.append('ambiguous_unit_neighbours')
            labels = [r for r in rows if r is not row and r['text'] in LABELS and _near(row, r)]
            if label is None and len(labels) == 1:
                label = labels[0]['text']; support.append(labels[0])
            elif label is None and len(labels) > 1:
                issues.append('ambiguous_field_role')
            semantic = UNITS.get(unit_text)
            name, unit = semantic if semantic else ('显示读数', None)
            quantities=[r for r in rows if r is not row and r['text'] in QUANTITIES and _near(row,r)]
            if unit=='%' and {QUANTITIES[r['text']] for r in quantities}=={'湿度'}:
                name='湿度';support.extend(quantities)
            if unit_text and not semantic:
                issues.append('unrecognized_unit_or_trailing_text')
            if unit is None:
                issues.append('unit_not_observed')
            if not label:
                issues.append('field_role_not_observed')
            region = dict(field_key=name, role=LABELS.get(label, 'displayed_value'), channel=panel,
                box=[str(box[0]/width), str(box[1]/height), str(box[2]/width), str(box[3]/height)],
                unit=unit, unit_basis='observed_ocr_unit_candidate' if semantic else None, minimum_score='.8',
                quantity_kind='mass' if name == '质量' else None)
            # Canonicalize only known unit spellings, preserving all raw tokens
            # and exact source boxes separately below. Never patch numeric glyphs.
            transcript = number + (unit or '')
            # A generic LCD locator cannot establish that this is a balance
            # with the registered fixed precision. Keep visible decimal text;
            # the existing registered-ROI path still owns display-type rules.
            candidate = field_candidate(region, dict(results=[dict(rec_texts=[transcript],
                rec_scores=[min(r['score'] for r in support)])]))
            if (not semantic or name=='质量') and re.fullmatch(r'[+-]?\d+', number):
                candidate.update(value=None, status='needs_review')
                issues.append('integer_or_missing_decimal_unresolved')
            if (unit_text and not semantic) or any(i in issues for i in ['ambiguous_unit_neighbours', 'ambiguous_field_role', 'ambiguous_sign']):
                candidate.update(value=None, status='needs_review')
            candidate.update(panel_id=panel, panel_box=bounds, source_token_index=row['index'],
                raw_tokens=[dict(text=r['text'], score=str(r['score']), source_box=r['box']) for r in support],
                displayed_glyphs=number, observed_unit_text=unit_text, observed_role_label=label,
                field_mapping_basis='detected_display_and_local_ocr_not_physical_identity',
                physical_instrument_identity=None, business_confirmation=False, auto_confirmed=False)
            candidate['source_view']=page.get('view','panel_crop')
            candidate['source_pixel_box']=box
            candidate['screen_localization_supported']=page.get('screen_localization_supported',True)
            if not candidate['screen_localization_supported']:
                issues.append('screen_not_verified_text_quantity_candidate')
            candidate['issues'] = sorted(set(candidate.get('issues', []) + issues))
            if candidate.get('value') is not None:
                # The visual candidate remains distinct from a validated field
                # calibration or an inventory/experiment fact.
                candidate['status'] = 'candidate'
                Decimal(str(candidate['value']))
            fields.append(candidate)
            if len(fields) > 512:
                raise ValueError('Panel field candidate bound exceeded')
    # The same glyphs in full-frame and crop views are one candidate. Conflicts
    # retain both originals and suppress numeric confirmation/selection.
    merged=[]
    for candidate in fields:
        same=[f for f in merged if _overlap(f['source_pixel_box'],candidate['source_pixel_box'])>.65]
        if same:
            first=same[0]
            if (first['displayed_glyphs']==candidate['displayed_glyphs']
                    and (first['unit'] in {None,candidate['unit']} or candidate['unit'] is None)):
                first.setdefault('additional_views',[]).append(candidate)
                continue
            for conflicting in same+[candidate]:
                conflicting['value']=None;conflicting['status']='needs_review'
                conflicting['issues']=sorted(set(conflicting['issues']+['ocr_view_conflict']))
        merged.append(candidate)
    detected_panels={p['panel_id'] for p in result.get('results',[]) if p.get('panel_id')}
    return merged, dict(revision=REVISION, status='spatial_ocr_candidates' if merged else 'raw_ocr_only',
                        panels=len(detected_panels), text_regions=len(set(panels)-detected_panels),
                        fields=len(merged), identity_inferred=False, auto_confirmed=False)
