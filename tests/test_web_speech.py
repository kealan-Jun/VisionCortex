import shutil
import subprocess
from pathlib import Path

import pytest


def test_browser_upload_maps_audio_to_physical_segments():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for upload planner characterization")
    source = (Path(__file__).parents[1] / "src/visioncortex/web/app.js").read_text()
    prefix = "function buildUploadPlan("
    function = prefix + source.split(prefix, 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    script = (
        """const assert = require('node:assert/strict');
const file = name => ({name, size: 10, lastModified: 1});
const state = {sources: [
 {viewId:'fp', role:'first_person', segments:[{video:file('one.mp4'),audio:file('one.opus'),audioOffsetSeconds:1.25}]},
 {viewId:'tp', role:'third_person', segments:[{video:file('two.mp4')},{video:file('three.mp4'),audio:file('three.opus')}]}
]};
"""
        + function
        + """
const plan = buildUploadPlan({title:'experiment'});
assert.equal(plan.files.filter(x=>x.kind==='audio').length,2);
assert.equal(plan.view_specs[0].audio_index,0);
assert.equal(plan.view_specs[0].audio_offset_ms,1250);
assert.equal(plan.view_specs[1].segments[1].audio_index,1);
assert.equal(plan.view_specs[1].segments[1].audio_offset_ms,null);
assert.equal(plan.view_specs[1].segments[0].audio_index,undefined);
assert.equal(plan.files.reduce((n,x)=>n+x.size,0),50);
assert.ok(!JSON.parse(plan.signature).files.some(x=>'browser_file' in x));
"""
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shared_timeline_inverse_mapping_gaps_and_controller_cleanup():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    source = (Path(__file__).parents[1]/'src/visioncortex/web/speech-sync.js').read_text()
    harness = '''
const assert=require('node:assert/strict');
class Media extends EventTarget {
 constructor(){super(); this.currentTime=0; this.paused=true; this.ended=false; this.readyState=1; this.playbackRate=1; this.src=''; this.plays=0;}
 pause(){this.paused=true;}
 play(){this.paused=false; this.plays++; return Promise.resolve();}
 load(){}
}
const window=new EventTarget();
'''+source+'''
const {mapTime,locateAudio,bind}=window.VisionCortexSpeechSync;
const mapping={anchors:[[0,0],[1,500],[2,1000],[6,1500]],max_gap_seconds:2,chunk_id:'one'};
assert.equal(mapTime(mapping,1.5),750);
assert.equal(mapTime(mapping,750,true),1.5);
assert.equal(mapTime(mapping,4),null);
assert.equal(mapTime(mapping,1250,true),null);
assert.equal(mapTime(mapping,-1),null);
assert.equal(mapTime({anchors:[]},0),null);
assert.equal(mapTime(mapping,null),null);
assert.equal(locateAudio([mapping],[{id:'one',start_seconds:0,end_seconds:2}],1500),null);
assert.equal(locateAudio([mapping],[{id:'one',start_seconds:0,end_seconds:6}],750).seconds,1.5);
const audio=new Media(), video=new Media(), status={}; let current=true, observed=null;
const controller=bind({audio,video,mapping,videos:[{view_id:'fp',url:'verified-video',anchors:[[0,0],[10,10000]]}],selectedView:()=> 'fp',status,current:()=>current,onTime:t=>observed=t});
audio.currentTime=1.5; audio.paused=false; audio.dispatchEvent(new Event('seeking'));
assert.equal(video.currentTime,.75); assert.equal(observed,750); assert.equal(video.paused,false);
audio.currentTime=4; audio.dispatchEvent(new Event('timeupdate'));
assert.equal(video.hidden,true); assert.equal(video.paused,true); assert.equal(observed,null);
audio.currentTime=1; audio.dispatchEvent(new Event('timeupdate')); assert.equal(video.hidden,false);
current=false; audio.dispatchEvent(new Event('timeupdate')); assert.equal(audio.paused,true); assert.equal(video.paused,true);
const count=video.plays; audio.dispatchEvent(new Event('play')); assert.equal(video.plays,count);
controller.dispose();
'''
    result = subprocess.run([node,'-e',harness],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_staging_detail_load_survives_unrelated_archive_directory_refresh():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    source = (Path(__file__).parents[1]/'src/visioncortex/web/app.js').read_text()
    prefix = 'async function loadArchive('
    function = prefix + source.split(prefix,1)[1].split('\n}\n',1)[0] + '\n}\n'
    script = '''const assert=require('node:assert/strict');
const state={archiveCacheEpoch:0,archiveCache:new Map()};
const api=async()=>{state.archiveCacheEpoch++; return {name:'retained',staging_run_id:'run'};};
'''+function+'''(async()=>{const result=await loadArchive('run',true); assert.equal(result.staging_run_id,'run');})();'''
    result = subprocess.run([node,'-e',script],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
