"""Locator geometry and fallback contracts; fake model outputs are not quality evidence."""
import importlib.util

import numpy as np
import pytest

from ._support import ROOT

spec=importlib.util.spec_from_file_location('panel_ocr_test',ROOT/'tools/inference/panel_ocr.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def locator(backend, boxes):
    instance=object.__new__(module.PanelOCR)
    instance.backend=backend
    instance.configuration=module.settings({'panel_locator':{'checkpoint':'fixture'}})
    instance.locate=lambda image:(boxes,dict(source_size=[image.shape[1],image.shape[0]]))
    return instance


class Backend:
    def __init__(self, fail=()):self.calls=[];self.fail=fail
    def predict(self,image):
        self.calls.append(image.copy())
        if len(self.calls) in self.fail:raise ValueError('fixture OCR failure')
        return dict(results=[dict(rec_texts=['1.23g'],rec_scores=[.99],rec_boxes=[[1,2,8,9]],
                                  rec_polys=[[[1,2],[8,2],[8,9],[1,9]]])])


def test_no_locator_hit_still_runs_whole_image_without_instrument_gate():
    backend=Backend();image=np.zeros((80,100,3),dtype=np.uint8)
    result=locator(backend,[]).predict(image)
    assert len(backend.calls)==1 and np.array_equal(backend.calls[0],image)
    assert result['results'][0].get('panel_id') is None
    assert result['panel_localization']['fallback']=='whole_image_raw_ocr_always_retained'


def test_native_crop_coordinates_and_pixels_are_source_relative():
    backend=Backend();image=np.arange(80*100*3,dtype=np.uint8).reshape(80,100,3)
    model=locator(backend,[dict(box=[20,30,40,50],score=.8)])
    model.configuration['padding']=0
    result=model.predict(image);page=result['results'][1]
    assert np.array_equal(backend.calls[1],image[30:50,20:40])
    assert page['rec_boxes']==[[21,32,28,39]] and page['panel_box']==[20,30,40,50]
    assert page['rec_polys'][0][0]==[21,32]
    assert page['geometry_basis']=='model_detected_lcd_screen_not_instrument_identity'


def test_partial_crop_failure_preserves_successful_whole_view():
    backend=Backend(fail=(2,));model=locator(backend,[dict(box=[20,30,40,50],score=.8)])
    result=model.predict(np.zeros((80,100,3),dtype=np.uint8))
    assert len(result['results'])==1 and result['panel_executions'][1]['status']=='needs_attention'
    backend=Backend(fail=(1,2));model=locator(backend,[dict(box=[20,30,40,50],score=.8)])
    with pytest.raises(ValueError,match='No OCR page'):model.predict(np.zeros((80,100,3),dtype=np.uint8))


def test_tiled_detections_restore_source_offsets_and_deduplicate():
    model=object.__new__(module.PanelOCR)
    model.configuration=module.settings({'panel_locator':{'checkpoint':'fixture','tiled_detection':True}})
    calls=[]
    def view(image):
        calls.append(image.shape)
        boxes=([dict(box=[40,40,50,50],score=.9)] if len(calls)==1 else
               [dict(box=[5,5,15,15],score=.8)] if len(calls)==5 else [])
        return boxes,dict(candidate_count=len(boxes),excluded_count=0,seconds=.01)
    model._locate_single=view
    boxes,receipt=model.locate(np.zeros((100,100,3),dtype=np.uint8))
    assert len(calls)==5 and len(boxes)==1 and boxes[0]['box']==[40,40,50,50]
    assert receipt['excluded_count']==1 and len(receipt['views'])==5


@pytest.mark.parametrize('specification',[{'checkpoint':'x','tiled_detection':'yes'},
    {'checkpoint':'x','max_panels':1000},{'checkpoint':'x','confidence':float('nan')}])
def test_invalid_unbounded_locator_configuration_is_rejected(specification):
    with pytest.raises(ValueError):module.settings({'panel_locator':specification})
