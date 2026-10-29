"""LED pixels are evidence, not permission to insert a missing decimal point."""
from copy import deepcopy
import math

from .panel_vision import libraries


def active_display(image, color):
    cv,np=libraries()
    if color not in {'red','green'}:
        raise ValueError('Explicit display color required')
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    hue,saturation,brightness=cv.split(hsv)
    valid=((hue<15)|(hue>165)) if color=='red' else ((hue>32)&(hue<95))
    mask=(valid&(saturation>80)&(brightness>95)).astype(np.uint8)*255
    count,labels,stats,_=cv.connectedComponentsWithStats(mask)
    kept=np.zeros_like(mask)
    for index in range(1,count):
        if stats[index,cv.CC_STAT_AREA]>=2:
            kept[labels==index]=255
    y,x=np.where(kept>0)
    if len(x)<8:
        return dict(status='unreadable',reason='no_active_display_pixels'),None
    x1,x2,y1,y2=int(x.min()),int(x.max()+1),int(y.min()),int(y.max()+1)
    height=y2-y1
    dots=[]
    for index in range(1,count):
        sx,sy,w,h,area=stats[index]
        if area>=2 and .5<=w/max(1,h)<=1.8 and 1<=h<=height*.30 and w<=height*.30 and sy+h>=y1+height*.75:
            dots.append(dict(box=[int(sx),int(sy),int(sx+w),int(sy+h)],area=int(area)))
    foreground=255-kept[y1:y2,x1:x2]
    foreground=cv.resize(foreground,None,fx=3,fy=3,interpolation=cv.INTER_NEAREST)
    foreground=cv.copyMakeBorder(foreground,24,24,24,24,cv.BORDER_CONSTANT,value=255)
    return dict(status='candidate',method='active_led_color_pixels/1',color=color,
                active_box=[x1,y1,x2,y2],visible_decimal_candidates=dots),foreground


def active_tokens(result, pixels):
    """Exclude separate brand/ghost tokens outside the measured lit-digit box.

    Missing geometry keeps the token, so a competing token still forces review.
    Retain the complete response in the inference receipt and excluded tokens here.
    """
    document = deepcopy(result)
    excluded = []
    active = pixels.get('active_box')
    if not active:
        return document, excluded
    ax1, ay1, ax2, ay2 = active
    for page in document.get('results', []):
        texts, scores, boxes = page.get('rec_texts', []), page.get('rec_scores', []), page.get('rec_boxes', [])
        if len(texts) != len(scores):
            raise ValueError('OCR token/score length mismatch')
        if len(boxes) != len(texts):
            continue
        keep = []
        for i, box in enumerate(boxes):
            if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                raise ValueError('Invalid OCR token geometry')
            x1, y1, x2, y2 = map(float, box)
            if x1 >= x2 or y1 >= y2:
                raise ValueError('Invalid OCR token rectangle')
            overlap = max(0, min(x2, ax2)-max(x1, ax1))*max(0, min(y2, ay2)-max(y1, ay1))
            if overlap/(min(x2-x1, ax2-ax1)*min(y2-y1, ay2-ay1)) >= .35:
                keep.append(i)
            else:
                excluded.append(dict(text=texts[i], score=scores[i], box=box, reason='outside_active_led_box'))
        for key in ('rec_texts', 'rec_scores', 'rec_boxes', 'rec_polys'):
            if key in page and len(page[key]) == len(texts):
                page[key] = [page[key][i] for i in keep]
    return document, excluded


def guard_decimal(candidate, pixels, region=None):
    region = region or {}
    value = candidate.get('value')
    issues = list(candidate.get('issues', []))
    if value is not None and candidate.get('data_type', 'number') == 'number':
        text = str(value)
        places = region.get('decimal_places')
        if '.' not in text:
            # A connected/blurred dot may not form a separate component. Absence
            # of a detected dot never proves that an LED displays an integer.
            if pixels.get('visible_decimal_candidates') and places != 0:
                issues.append('visible_decimal_missing_in_ocr')
            if places is None:
                issues.append('integer_or_missing_decimal_unresolved')
        if places is not None and (len(text.rsplit('.', 1)[1]) if '.' in text else 0) != places:
            issues.append('verified_display_precision_mismatch')
        if issues:
            return dict(candidate, value=None, status='needs_review', issues=sorted(set(issues)), pixel_evidence=pixels)
    return dict(candidate,pixel_evidence=pixels)
