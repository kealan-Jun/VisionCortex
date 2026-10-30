"""Real local files: FR readability updates navigation without changing Results."""
import hashlib
import json
import pytest
from visioncortex.lab.comment_fr_presentation import CommentFRPublication


def test_fr_navigation_checks_new_display_bytes_and_preserves_result_and_source(tmp_path):
    root=tmp_path/'archive';folder='2026-10-29_camera/Comment/OCR/VoicePhotoReadings/event'
    target=root/folder;target.mkdir(parents=True)
    readings=dict(schema='field-numeric-observations/1',readings=[dict(value='0.2287',unit='g',measurement_name='质量',result_stage='reviewed')])
    original=(json.dumps(readings)+'\n').encode();(target/'Readings.json').write_bytes(original)
    evidence=dict(schema='field-recognition-event/2',readings=dict(path=folder+'/Readings.json',sha256=hashlib.sha256(original).hexdigest()),
                  measurement_files=[dict(result=folder+'/Result.json',sha256='a'*64)],source='original-source')
    (target/'Evidence.json').write_text(json.dumps(evidence))
    (target/'Result.json').write_bytes(b'original exact six-field result')
    class Guard:
        @staticmethod
        def guarded_atomic(path,blocks):return False
    publisher=CommentFRPublication(history=tmp_path/'history')
    raw=publisher._json(readings);publisher._write(root,folder+'/Readings.json',raw,Guard)
    evidence['readings']['sha256']=hashlib.sha256(raw).hexdigest()
    publisher._write(root,folder+'/Evidence.json',publisher._json(evidence),Guard)
    assert (target/'Result.json').read_bytes()==b'original exact six-field result'
    assert json.loads((target/'Readings.json').read_bytes())['readings']==readings['readings']
    assert json.loads((target/'Evidence.json').read_bytes())['source']=='original-source'
    evidence['source']='changed-source'
    with pytest.raises(ValueError,match='source evidence'):
        publisher._write(root,folder+'/Evidence.json',publisher._json(evidence),Guard)
