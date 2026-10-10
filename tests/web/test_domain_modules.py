"""Exercise actual browser factories with explicit ports, without live services."""
import shutil
import subprocess

import pytest

from repo_paths import WEB


def run_node(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for browser module tests")
    modules = "\n".join((WEB / name).read_text() for name in (
        "stores.js", "uploads.js", "archives.js", "materials.js", "tasks.js",
    ))
    completed = subprocess.run(
        [node], input='const assert=require("node:assert/strict"); const window={};\n'
         + modules + "\n" + script, text=True, capture_output=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_store_compatibility_view_updates_the_single_domain_owner():
    run_node('''
const {state,stores}=window.VisionCortexStores.createStores();
state.uploadCancelled=true; assert.equal(stores.uploads.uploadCancelled,true);
stores.tasks.activeRun={run_id:"R"}; assert.equal(state.activeRun.run_id,"R");
state.archiveCache.set("A",{}); assert.equal(stores.archives.archiveCache.size,1);
assert.notEqual(window.VisionCortexStores.createStores().state.archiveCache,state.archiveCache);
''')


def test_upload_factory_reuses_only_the_matching_live_session_and_preserves_audio():
    run_node('''
(async()=>{
const records=new Map(); global.localStorage={getItem:k=>records.get(k),
  setItem:(k,v)=>records.set(k,v),removeItem:k=>records.delete(k)};
const file=name=>({name,size:10,lastModified:1});
const state={sources:[{viewId:"fp",role:"first_person",segments:[
  {video:file("v.mp4"),audio:file("a.opus"),audioOffsetSeconds:1.25}]}]};
const calls=[];const api=async(url,options)=>{calls.push([url,options]);
  return options ? {session_id:"new",status:"open"} : {session_id:"old",status:"open"};};
const uploads=window.VisionCortexUploads.createUploads({state,api});
const plan=uploads.buildUploadPlan({title:"experiment"});
assert.equal(plan.view_specs[0].audio_offset_ms,1250);
assert.equal(plan.files[1].kind,"audio");
uploads.rememberUploadSession(plan.signature,"old");
assert.equal((await uploads.createOrResumeUploadSession(plan)).session_id,"old");
assert.equal(calls.length,1);
assert.equal((await uploads.createOrResumeUploadSession({...plan,signature:"changed"})).session_id,"new");
assert.equal(calls.length,2);
assert.ok(!JSON.parse(calls[1][1].body).files.some(file=>"browser_file" in file));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_archive_factory_discards_a_stale_listing_response():
    run_node('''
(async()=>{
let finish;const state={archiveRequestId:0,archiveCacheEpoch:0,archives:[]};
const archive=window.VisionCortexArchives.createArchives({state,
  archiveSearchQuery:()=>"current",api:()=>new Promise(resolve=>{finish=resolve;})});
const pending=archive.loadArchiveListing("old");
state.archiveRequestId+=1;finish({archives:[{name:"stale"}],total:1});
assert.equal(await pending,false);assert.deepEqual(state.archives,[]);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')
