"""Automatic image -> layout -> rectified fields/needle candidates -> observations."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from .gauge import read_automatic_gauge
from .ocr_fields import field_candidate
from .panel_vision import libraries
from .needle_vision import detect_dials, detect_needle
from .display_pixels import active_display, active_tokens, guard_decimal


class AutomaticReadout:
    def __init__(self, profiles):
        self.profiles=profiles

    def run(self, image, source_sha256, infer, *, instrument_id=None, instrument_resolver=None):
        located=self.profiles.locate(image)
        if located['status']!='candidate':
            return dict(status=located['status'],fields=[],registration=located,invocations=[],
                        raw_fallback_required=True,accuracy='NOT_PROVEN')
        cv,np=libraries()
        match=located['match'];canonical,matrix,profile=self.profiles.extract(image,match)
        fields,receipts=[],[]
        inverse=np.linalg.inv(matrix)
        height,width=canonical.shape[:2]
        identity_method='externally_verified_for_this_panel' if instrument_id else 'unassigned'
        if instrument_resolver and profile.get('identity_qr_box'):
            x1,y1,x2,y2=map(float,profile['identity_qr_box'])
            # Decode only a reviewed panel-attached QR area. A QR elsewhere in
            # the photograph could belong to another instrument/sample.
            instrument_id=instrument_resolver(canonical[int(y1*height):int(y2*height),int(x1*width):int(x2*width)])
            identity_method='registered_panel_qr_region' if instrument_id else 'unassigned'
        for region in profile.get('regions',[]):
            x1,y1,x2,y2=[float(x) for x in region['box']]
            xlo,ylo,xhi,yhi=int(x1*width),int(y1*height),min(width,int(x2*width+.999)),min(height,int(y2*height+.999))
            patch=canonical[ylo:yhi,xlo:xhi]
            ok,encoded=cv.imencode('.png',patch)
            if not ok or not 0<encoded.size<=12*1024**2:
                raise ValueError('Rectified OCR field exceeds payload bound')
            raw=encoded.tobytes();result,receipt=infer(raw)
            value=field_candidate(region,result)
            views=[dict(view='rectified_color',candidate=value)]
            extra=[]
            if region.get('display_color'):
                pixels,foreground=active_display(patch,region['display_color'])
                filtered, excluded = active_tokens(result, pixels)
                value=guard_decimal(field_candidate(region,filtered),pixels,region)
                value['excluded_tokens']=excluded
                views[0]['candidate']=value
                if foreground is None:
                    value=dict(value,value=None,status='needs_review',issues=value['issues']+['no_active_display_pixels'])
                else:
                    ok,binary=cv.imencode('.png',foreground)
                    if not ok:
                        raise ValueError('LED foreground encoding failed')
                    second,second_receipt=infer(binary.tobytes())
                    alternative=guard_decimal(field_candidate(region,second),pixels,region)
                    views.append(dict(view='active_led_pixels',candidate=alternative))
                    valid=[v['candidate'] for v in views if v['candidate']['value'] is not None]
                    distinct={str(Decimal(v['value']).normalize()) for v in valid}
                    if len(distinct)==1:
                        value=valid[-1]
                    elif len(distinct)>1:
                        value=dict(value,value=None,status='needs_review',issues=value['issues']+['ocr_view_conflict'])
                    # A rejected view may still contain competing, high-score
                    # numeric text. Never let filtering/low-score/decimal guards
                    # turn that disagreement into artificial consensus.
                    tentative={str(Decimal(v['candidate']['tentative_value']).normalize()) for v in views
                               if v['candidate'].get('tentative_value') is not None and region.get('data_type','number')=='number'}
                    if len(tentative)>1:
                        value=dict(value,value=None,status='needs_review',issues=sorted(set(value['issues']+['ocr_view_conflict'])))
                    extra.append(dict(execution=second_receipt,derivation=dict(kind='active_led_pixels',original_sha256=source_sha256,
                        derived_sha256=hashlib.sha256(binary.tobytes()).hexdigest(),derived_bytes=int(binary.size),pixel_evidence=pixels)))
                value=dict(value,ocr_views=views)
            corners=np.float32([[xlo,ylo],[xhi,ylo],[xhi,yhi],[xlo,yhi]])
            source_quad=cv.perspectiveTransform(corners.reshape(1,-1,2),inverse).reshape(-1,2)
            normalized=source_quad/np.array([image.shape[1],image.shape[0]],np.float32)
            value.update(canonical_box=region['box'],source_quad=normalized.tolist(),
                         source_box=[str(float(normalized[:,0].min())),str(float(normalized[:,1].min())),
                                     str(float(normalized[:,0].max())),str(float(normalized[:,1].max()))],
                         layout_profile_id=profile['id'])
            native_height=min(float(np.linalg.norm(source_quad[3]-source_quad[0])),float(np.linalg.norm(source_quad[2]-source_quad[1])))
            value['native_field_height_pixels']=native_height
            if native_height<12:
                value.update(value=None,status='needs_review',issues=value.get('issues',[])+['insufficient_native_field_resolution'])
            fields.append(value)
            receipts.append(dict(execution=receipt,derivation=dict(kind='registered_perspective_field',
                 original_sha256=source_sha256,derived_sha256=hashlib.sha256(raw).hexdigest(),derived_bytes=len(raw),
                 source_quad=normalized.tolist(),canonical_box=region['box'],profile_id=profile['id'],profile_fingerprint=self.profiles.fingerprint)))
            receipts.extend(extra)
            if len(json.dumps(dict(fields=fields,invocations=receipts),ensure_ascii=False))>3*1024**2:
                raise ValueError('Automatic field evidence exceeds its bounded checkpoint budget')
        for spec in profile.get('gauges',[]):
            center,radius=spec.get('center'),spec.get('radius')
            if center is None or radius is None:
                dials=detect_dials(canonical)['dials']
                if len(dials)!=1:
                    fields.append(dict(field_key=spec['field_key'],role='gauge_indication',channel=spec.get('channel',''),value=None,unit=spec.get('unit'),status='ambiguous_dial',dial_candidates=dials))
                    continue
                center=dials[0]['center'];radius=min(dials[0]['axes'])/2*.94
            needle=detect_needle(canonical,center,float(radius),source_sha256=source_sha256,coordinate_frame='rectified_dial_plane')
            calibration=spec.get('calibration')
            value=dict(field_key=spec['field_key'],role='gauge_indication',channel=spec.get('channel',''),value=None,
                       unit=spec.get('unit'),status='requires_verified_calibration',needle=needle)
            if needle['status']!='candidate':
                value['status']=needle['status']
            elif calibration and calibration.get('status')=='verified' and instrument_id==calibration.get('instrument_id'):
                # Reuse strict scale interpolation, while keeping the distinct
                # automatic-candidate geometry provenance in the returned field.
                computed=read_automatic_gauge(calibration,needle,source_sha256=source_sha256)
                value.update(computed,needle=needle)
            fields.append(value)
        return dict(status='candidate',fields=fields,registration=located,invocations=receipts,instrument_id=instrument_id,
                    physical_instrument_identity=identity_method,
                    raw_fallback_required=False,accuracy='NOT_PROVEN')
