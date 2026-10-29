"""Only the explicit private compatibility descriptor crosses the worker boundary."""
import json
from types import SimpleNamespace
from uuid import uuid4

from visioncortex.lab.processing import Worker
import visioncortex.lab.processing as processing


def test_archive_compatibility_path_forwarded_without_expanding_environment(tmp_path,monkeypatch):
    descriptor=tmp_path/'private.json';descriptor.write_text('{}')
    monkeypatch.setenv('VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST',str(descriptor))
    monkeypatch.setenv('VC_LAB_UNAUTHORIZED_SECRET','must-not-cross')
    seen={}
    def run(command,request,**options):
        seen.update(options);seen['request']=json.loads(request)
        return json.dumps({'model_revision':'fixture/1','analysis_outcome':'no_transcript'})
    monkeypatch.setattr(processing,'run_bounded',run)
    worker=Worker.__new__(Worker)
    worker.settings=SimpleNamespace(allow_external_inference=False,runtime_root=tmp_path,cache=tmp_path,nas_marker_sha256=None)
    result=worker.execute_model(dict(id=uuid4(),capture_id=uuid4(),worker_kind='asr'),
        dict(start_seconds=0,end_seconds=1,occurred_at=None),tmp_path/'audio.wav',{'revision':'fixture/1','python':'/fixture/python'})
    assert result['model_revision']=='fixture/1'
    assert seen['env']['VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST']==str(descriptor)
    assert 'VC_LAB_UNAUTHORIZED_SECRET' not in seen['env'] and 'must-not-cross' not in json.dumps(seen)
