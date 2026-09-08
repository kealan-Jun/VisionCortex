import shutil
import subprocess
from pathlib import Path

import pytest


def test_playback_has_real_poster_and_retained_filters_preserve_access_to_all_candidates():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for frontend characterization")
    source = (Path(__file__).parents[1] / "src/visioncortex/web/app.js").read_text()
    script = r'''
const assert = require("node:assert/strict");
const location = {href:"http://test/",origin:"http://test"};
const esc = x=>String(x??""), number=x=>Number(x||0), timecode=x=>String(x), icon=()=>"";
const productObjectLabel = x=>x, ACTION_LABELS = {object_movement:"物体移动"};
const state = {selectedMaterials:new Map()};
'''
    for name in ["videoPosterUrl", "videoPreview", "experimentRecordRoute", "retainedMaterialsView", "automaticReviewLabel", "movementScreeningView", "materialsView", "ensureMaterialSelection", "materialSelectionScope"]:
        prefix = f"function {name}("
        script += prefix + source.split(prefix, 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    script += r'''
const url="/api/archive-file?archive=demo&path=Experiment-Clips/a.mp4&release=one";
const poster=videoPosterUrl(url);
assert.ok(poster.includes("release=one"));
assert.ok(poster.includes("poster=true"));
assert.equal(videoPosterUrl("http://outside/a.mp4"),"");
const player=videoPreview(url,"同步双视角");
assert.ok(player.includes('preload="none"'));
assert.ok(player.includes('poster="'));
assert.ok(player.includes('aria-label="播放同步双视角"'));
assert.ok(!player.includes("autoplay"));
const data={name:"demo",staging_run_id:"run",key_events:[],quarantined_materials:[
 {event_id:"full",group_folder:"001",cv_action_type:"object_movement",cv_objects:["tube"],clip_url:url,preview_review:{priority:true,duration_ms:2000}},
 {event_id:"short",group_folder:"002",cv_action_type:"object_movement",clip_url:url,preview_review:{priority:false,duration_ms:150,reason_codes:["short_candidate"]}}
]};
let html=retainedMaterialsView(data);
assert.ok(html.includes('素材记录：full'));
assert.ok(html.includes('无需人工审批'));
data.partial_delivery={pending_semantic_results:[{status:"disabled"}]};
assert.ok(retainedMaterialsView(data).includes('本次运行未启用自动语义核验'));
assert.ok(!html.includes('素材记录：short'));
assert.ok(html.includes('#/stage/run/experiments?group=001'));
state.retainedMaterialFilters.scope="all";
html=retainedMaterialsView(data);
assert.ok(html.includes('素材记录：short'));
assert.ok(html.includes('候选持续不足 1 秒'));
state.retainedMaterialFilters.group="002";
html=retainedMaterialsView(data);
assert.ok(html.includes('素材记录：short'));
assert.ok(!html.includes('素材记录：full'));
assert.equal(data.key_events.length,0);
assert.equal(data.quarantined_materials.length,2);
data.movement_screening={counts:{contradicted:3,unverified:2,supported:1},total_candidates:6,
  candidates:[{view_id:"fp",start_ms:1000,end_ms:1500,objects:["pipette"],status:"contradicted"}]};
html=materialsView(data);
assert.ok(html.includes("物体移动筛选结果"));
assert.ok(html.includes("画面不支持移动"));
assert.ok(html.includes("查看完整实验片段"));
assert.ok(html.includes("候选持续不足 1 秒"));
data.quarantined_materials=[];
assert.ok(materialsView(data).includes("物体移动筛选结果"));
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
