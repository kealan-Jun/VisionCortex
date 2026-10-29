"""Synthetic geometry/transport regressions; no customer/model accuracy claim."""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest

from visioncortex.lab.automatic_readout import AutomaticReadout
from visioncortex.lab.display_pixels import active_display, active_tokens, guard_decimal
from visioncortex.lab.gauge import read_automatic_gauge
from visioncortex.lab.needle_vision import detect_dials, detect_needle
from visioncortex.lab.ocr_fields import field_candidate
from visioncortex.lab.panel_vision import PanelProfiles, read_image


def template(folder):
    folder.mkdir(parents=True, exist_ok=True)
    image = np.full((384,512,3), 235, np.uint8)
    rng = np.random.default_rng(27)
    for i in range(150):
        x, y = rng.integers([60,50],[450,330])
        cv2.circle(image, (int(x),int(y)), int(rng.integers(2,6)), (int(rng.integers(0,190)),)*3, 1)
    cv2.putText(image, 'SYNTHETIC PANEL A', (62,95),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),2)
    cv2.putText(image, '26.2', (210,190),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
    path = folder/'template.png'
    cv2.imwrite(str(path), image)
    doc = dict(schema_version='vc-lab-panel-profiles/1',status='research',revision='fixture/1',profiles=[
        dict(id='layout',size=[400,288],reference=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             quad=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]),
             regions=[dict(field_key='temperature',role='process_value',channel='upper',box=['.35','.35','.65','.6'],
                           unit='°C',unit_basis='source_visible_research_annotation')])])
    manifest = folder/'Profiles.json'
    manifest.write_text(json.dumps(doc))
    return image, manifest, doc


def load(path, doc=None, scope='fixture'):
    if doc is not None:
        path.write_text(json.dumps(doc))
    return PanelProfiles(path,hashlib.sha256(path.read_bytes()).hexdigest(),data_scope=scope,model_root=path.parent)


def test_registration_masks_dynamic_digits_and_survives_perspective(tmp_path):
    image,path,_=template(tmp_path)
    profiles=load(path)
    # Change all pixels of the digit field before a perspective camera movement.
    cv2.rectangle(image,(200,150),(300,215),(10,10,10),-1)
    target=np.float32([[30,20],[490,8],[475,367],[10,350]])
    transform=cv2.getPerspectiveTransform(np.float32([[0,0],[511,0],[511,383],[0,383]]),target)
    moved=cv2.warpPerspective(image,transform,(512,384),borderValue=(255,255,255))
    located=profiles.locate(moved)
    assert located['status']=='candidate',located
    assert located['physical_instrument_identity']=='unassigned'
    canonical,_,_=profiles.extract(moved,located['match'])
    assert canonical.shape[:2]==(288,400)
    assert located['match']['inliers']>=12


def test_registration_abstains_on_blank_or_competing_layouts(tmp_path):
    image,path,doc=template(tmp_path)
    assert load(path).locate(np.full_like(image,255))['status']=='unmatched'
    duplicate=deepcopy(doc['profiles'][0]);duplicate['id']='different-layout-same-appearance'
    doc['profiles'].append(duplicate)
    assert load(path,doc).locate(image)['status']=='ambiguous_layout'


@pytest.mark.parametrize('mutation', ['hash','physical_identity','unit','precision','field','path'])
def test_profiles_require_pinned_layout_and_field_semantics(tmp_path,mutation):
    _,path,doc=template(tmp_path)
    p=doc['profiles'][0];r=p['regions'][0]
    if mutation=='hash':p['reference']['sha256']='0'*64
    if mutation=='physical_identity':p['instrument_id']='must-not-be-inferred'
    if mutation=='unit':r['unit_basis']='guessed'
    if mutation=='precision':r.update(decimal_places=2,decimal_places_basis='guessed')
    if mutation=='field':r['role']='guess'
    if mutation=='path':p['reference']['path']='../template.png'
    with pytest.raises(ValueError):load(path,doc)


def test_research_profiles_cannot_enable_customer_processing(tmp_path):
    _,path,_=template(tmp_path)
    with pytest.raises(ValueError,match='Research'):load(path,scope='customer')


def test_automatic_fields_are_source_pinned_candidates_and_fallback_is_explicit(tmp_path):
    image,path,_=template(tmp_path)
    profiles=load(path)
    calls=[]
    def infer(raw):
        calls.append(hashlib.sha256(raw).hexdigest())
        return dict(results=[dict(rec_texts=['26.2'],rec_scores=[.99])]),dict(fixture=True,source_sha256=calls[-1])
    result=AutomaticReadout(profiles).run(image,'a'*64,infer)
    assert result['fields'][0]['value']=='26.2'
    assert result['physical_instrument_identity']=='unassigned'
    assert result['invocations'][0]['derivation']['original_sha256']=='a'*64
    assert len(calls)==1 and result['accuracy']=='NOT_PROVEN'
    result=AutomaticReadout(profiles).run(np.full_like(image,255),'b'*64,infer)
    assert result['raw_fallback_required'] and not result['fields'] and len(calls)==1


def numeric(text):
    return field_candidate(dict(field_key='reading',role='displayed_value',box=['0','0','1','1']),
                           dict(results=[dict(rec_texts=[text],rec_scores=[.99])]))


def test_led_missing_or_connected_decimal_never_becomes_a_large_integer():
    for pixels in [dict(status='candidate'),dict(visible_decimal_candidates=[dict(box=[10,10,12,12])])]:
        result=guard_decimal(numeric('5969'),pixels)
        assert result['value'] is None and result['tentative_value']=='5969'
    assert guard_decimal(numeric('59.69'),dict(status='candidate'))['value']=='59.69'
    assert guard_decimal(numeric('140'),{},dict(decimal_places=0))['value']=='140'
    assert guard_decimal(numeric('5969'),{},dict(decimal_places=2))['value'] is None
    assert guard_decimal(numeric('59.6'),{},dict(decimal_places=2))['value'] is None


def test_led_masks_do_not_select_separate_ghost_digits_or_brand_tokens():
    image=np.zeros((60,200,3),np.uint8)
    cv2.putText(image,'8',(2,42),cv2.FONT_HERSHEY_SIMPLEX,1,(25,25,25),2)
    cv2.putText(image,'36',(65,42),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    pixels,_=active_display(image,'red')
    raw=dict(results=[dict(rec_texts=['8','36'],rec_scores=[.99,.98],rec_boxes=[[0,10,25,50],[65,10,110,50]])])
    kept,excluded=active_tokens(raw,pixels)
    assert kept['results'][0]['rec_texts']==['36'] and excluded[0]['text']=='8'
    assert raw['results'][0]['rec_texts']==['8','36']
    assert active_display(np.zeros_like(image),'red')[1] is None


def dial(*,symmetric=False,blank=False):
    image=np.full((400,400,3),255,np.uint8)
    cv2.circle(image,(200,200),150,(0,0,0),2)
    if not blank:
        theta=np.deg2rad(35)
        tip=tuple(np.rint([200+130*np.cos(theta),200+130*np.sin(theta)]).astype(int))
        opposite=tuple(np.rint([200-(130 if symmetric else 70)*np.cos(theta),200-(130 if symmetric else 70)*np.sin(theta)]).astype(int))
        cv2.line(image,(200,200),opposite,(0,0,0),4 if symmetric else 16)
        cv2.line(image,(200,200),tip,(0,0,0),4)
        cv2.circle(image,(200,200),8,(0,0,0),2)
    return image


def test_needle_selects_long_thin_tip_instead_of_thick_counterweight():
    result=detect_needle(dial(),[200,200],145,source_sha256='a'*64)
    assert result['status']=='candidate',result
    assert abs(float(result['angle_degrees'])-35)<2
    assert result['tip_status']=='algorithm_candidate' and result['accuracy']=='NOT_PROVEN'


@pytest.mark.parametrize('symmetric,blank',[(True,False),(False,True)])
def test_needle_rejects_equal_ends_and_blank_dial(symmetric,blank):
    result=detect_needle(dial(symmetric=symmetric,blank=blank),[200,200],145,source_sha256='a'*64)
    assert result['status'] in {'ambiguous','unreadable'}


def test_multiple_dials_remain_separate_candidates():
    image=np.concatenate([dial(blank=True),dial(blank=True)],axis=1)
    found=detect_dials(image)
    assert len(found['dials'])==2,found


def test_automatic_scale_never_masquerades_as_manually_verified_geometry():
    cal=dict(schema_version='vc-lab-gauge-calibration/1',status='verified',revision='fixture/1',instrument_id='fixture',
        verified_by='synthetic fixture',unit='MPa',center=['200','200'],direction='clockwise',tip_radius_range=['50','160'],
        max_angle_uncertainty_degrees='3',report_increment='.001',anchors=[dict(point=['350','200'],value='-.1'),
        dict(point=['200','350'],value='-.05'),dict(point=['50','200'],value='0')])
    obs=detect_needle(dial(),[200,200],145,source_sha256='a'*64,coordinate_frame='rectified_dial_plane')
    # Use exactly the independently detected center as a fixture calibration.
    delta=np.array(list(map(float,obs['center'])))-200
    cal['center']=obs['center']
    for a in cal['anchors']:a['point']=[str(float(v)+d) for v,d in zip(a['point'],delta)]
    value=read_automatic_gauge(cal,obs,source_sha256='a'*64)
    assert value['geometry_verification']=='automatic_candidate_not_manual_verification'
    assert not value['stable_pressure_proven']
    cal['center']=['230','230']
    assert read_automatic_gauge(cal,obs,source_sha256='a'*64)['status']=='requires_geometry_review'


def test_image_reader_bounds_pixels_before_cv_decode(tmp_path):
    from PIL import Image
    from unittest.mock import patch
    image,path,_=template(tmp_path)
    decoded,sha=read_image(path.parent/'template.png')
    assert decoded.shape==image.shape and len(sha)==64
    oversized=tmp_path/'oversized.png'
    Image.new('L',(4100,4000),255).save(oversized)
    with patch('cv2.imdecode',side_effect=AssertionError('Decoder must not see oversized image')):
        with pytest.raises(ValueError,match='oversized'):read_image(oversized)
