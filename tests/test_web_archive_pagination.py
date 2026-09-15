import shutil
import subprocess
from pathlib import Path

import pytest


def test_experiment_list_includes_retained_runs_without_promoting_them():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for frontend characterization")
    source = (Path(__file__).parents[1] / "src/visioncortex/web/app.js").read_text()
    names = ("experimentRecords", "experimentRecordRoute", "filteredArchives",
             "archiveProductStatus", "archiveRows", "rerunArchiveAction")
    functions = []
    for name in names:
        prefix = f"function {name}("
        functions.append(prefix + source.split(prefix, 1)[1].split("\n}\n", 1)[0] + "\n}\n")
    script = r'''
const assert = require("node:assert/strict");
const state = {search:"", archiveFilters:{status:"all",date:"all",owner:"all",tag:"all"},
  archives:[{name:"formal",path:"/nas/formal",pipeline_stage:"completed",key_event_count:6}],
  runs:[
    {run_id:"local-result",experiment_id:"real-local",state:"failed",nas_staging:"/local/result",
     observability:{status:{updated_at:"2026-09-07"},retained_experiment_count:4,
       stage_receipts:[{stage:"experiment_clips",status:"completed",
         artifacts:[{relative_path:"Experiment-Clips",available:true}]}]}},
    {run_id:"running",experiment_id:"active",state:"key_materials",nas_staging:"/nas/stage",
     observability:{status:{},retained_experiment_count:2,stage_receipts:[{}]}},
    {run_id:"waiting",state:"queued",nas_staging:"/nas/empty"},
    {run_id:"finished",state:"completed",nas_staging:"/nas/formal"},
    {run_id:"missing",state:"failed",nas_staging:"/local/missing",result_available:false},
  ]};
const experimentMetadata=()=>({}), productExperimentName=x=>x, archiveWithinDate=()=>true;
const esc=x=>String(x??""), number=x=>Number(x||0), formatDate=x=>x||"", icon=()=>"";
'''
    script += "\n".join(functions)
    script += r'''
const records=experimentRecords();
assert.equal(records.length,3);
assert.equal(state.archives.length,1);
const retained=records.find(x=>x.staging_run_id==="local-result");
assert.equal(retained.experiment_count,4);
assert.equal(retained.has_experiment_videos,true);
assert.equal(records.find(x=>x.staging_run_id==="running").has_experiment_videos,false);
assert.equal(retained.key_event_count,0);
assert.equal(retained.formal_accuracy_claim_allowed,false);
assert.equal(experimentRecordRoute(retained),"#/stage/local-result/experiments");
state.archiveFilters.status="attention";
assert.deepEqual(filteredArchives(true).map(x=>x.name),["real-local"]);
const rows=archiveRows(filteredArchives(true));
assert.ok(rows.includes('href="#/stage/local-result/experiments"'));
assert.ok(rows.includes("阶段产出 · 待补全"));
assert.ok(rows.includes('class="archive-video-availability">4 组视频片段可查看'));
assert.ok(rows.includes('class="row-link">查看视频'));
assert.ok(!archiveRows([records.find(x=>x.staging_run_id==="running")]).includes("组视频片段可查看"));
retained.has_experiment_videos=false;
assert.ok(!archiveRows([retained]).includes("组视频片段可查看"),"unavailable clips must not advertise playback");
assert.ok(!archiveRows([records.find(x=>x.name==="formal")],"reports").includes("组视频片段可查看"));
state.archiveFilters.status="completed";
assert.deepEqual(filteredArchives(true).map(x=>x.name),["formal"]);
state.archiveFilters.status="all"; state.search="real-local";
assert.equal(filteredArchives(true).length,1);
assert.ok(!rerunArchiveAction({retry_available:false}).includes("data-rerun-archive"));
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_sectioned_archive_loaders_preserve_pagination_and_local_library_state():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for frontend loader characterization")
    source = (Path(__file__).parents[1] / "src/visioncortex/web/app.js").read_text()
    names = (
        "loadArchive", "loadArchiveSection", "loadArchiveMaterials",
        "loadArchiveExperiments", "loadArchiveView", "materialQueryParameters",
        "ensureMaterialFilters", "archiveProcessStopped", "loadLibraryDetail",
        "libraryQueryKey", "cachedArchiveDetail", "invalidateArchiveCache",
    )
    functions = []
    for name in names:
        prefix = f"async function {name}(" if f"async function {name}(" in source else f"function {name}("
        body = source.split(prefix, 1)[1].split("\n}\n", 1)[0]
        functions.append(prefix + body + "\n}\n")
    script = r'''
const assert = require("node:assert/strict");
const state = {
  archiveCache: new Map(), materialCache: new Map(),
  materialFilters: {}, globalMaterialFilters: {action:"all",object:"all",support:"all"},
  search: "", selectedMaterials: new Set(),
};
let release = "v1";
const requests = [];
async function api(url) {
  requests.push(url);
  const query = new URL(url,"http://test").searchParams;
  if (url.startsWith("/api/key-events?")) return {
    items: [{event_id:query.has("cursor")?"E2":"E1",release_id:release}],
    next_cursor:query.has("cursor")?null:"next",total_count:2,
  };
  assert.ok(query.has("section"), "complete archives must use bounded sections");
  const summary = {name:"Archive",release_id:release,counts:{experiments:2,key_events:2},observability:{status:{stage:"completed"}}};
  if (query.get("section")==="summary") return summary;
  if (query.get("section")==="reports") return {...summary,daily_report:{report_id:"R1"}};
  return {...summary,experiment_groups:[],experiments:[{name:query.has("cursor")?"G2":"G1"}],next_cursor:query.has("cursor")?null:"next"};
}
'''
    script += "\n".join(functions)
    script += r'''
(async()=>{
  const first = await loadArchiveView("Archive","experiments");
  assert.equal(first.key_events.length,0);
  assert.equal(first.counts.key_events,2);
  await Promise.all([loadArchiveExperiments("Archive",first.next_cursor),loadArchiveExperiments("Archive",first.next_cursor)]);
  assert.deepEqual((await loadArchiveView("Archive","experiments")).experiments.map(x=>x.name),["G1","G2"]);
  await loadArchiveMaterials("Archive");
  await Promise.all([loadArchiveMaterials("Archive","next"),loadArchiveMaterials("Archive","next")]);
  assert.equal((await loadArchiveView("Archive","materials")).key_events.length,2);
  await loadLibraryDetail("Archive","reports");
  await loadLibraryDetail("Archive","materials");
  await loadLibraryDetail("Archive","materials","next");
  assert.equal(cachedArchiveDetail("Archive").daily_report.report_id,"R1");
  assert.equal(cachedArchiveDetail("Archive").key_events.length,2);
  assert.equal(cachedArchiveDetail("Archive").libraryKeys.reports,"reports");
  await loadLibraryDetail("Archive","reports");
  assert.equal(cachedArchiveDetail("Archive").key_events.length,2,"report summary placeholders must not erase material pages");
  state.globalMaterialFilters.support="formal";
  await loadLibraryDetail("Archive","materials");
  assert.ok(!new URL(requests.at(-1),"http://test").searchParams.has("cross_view"));
  state.materialFilters.query="开盖";
  state.materialFilters.object="tube";
  assert.equal(materialQueryParameters("Archive").get("q"),"开盖 tube");
  release="v2";
  state.archiveCache.get("Archive:summary").loadedAt=0;
  assert.equal((await loadArchive("Archive")).release_id,"v2");
  assert.equal(cachedArchiveDetail("Archive"),null);
  assert.equal(state.materialCache.size,0);
  assert.ok(![...state.archiveCache.keys()].some(k=>k.includes(":v1:")));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_material_library_loads_candidates_by_run_and_keeps_formal_totals():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for frontend characterization")
    source = (Path(__file__).parents[1] / "src/visioncortex/web/app.js").read_text()
    names = ("experimentRecords", "experimentRecordRoute", "libraryRecordKey",
             "cachedLibraryDetail", "cachedArchiveDetail", "libraryRecordQueryKey",
             "libraryQueryKey", "loadLibraryRecord", "loadArchive", "ensureLibraryDetails",
             "materialLibraryEntries", "automaticReviewLabel", "globalMaterialCard", "materialFocusRoute", "eventObjectValues",
             "renderMaterialsLibrary", "archiveSyncNotice", "libraryLoadState", "libraryRequestKey", "libraryFailureNotice", "bindLibraryRetry")
    script = r'''
const assert = require("node:assert/strict");
const state = {archives:[{name:"shared",key_event_count:30}], runs:[],archiveCache:new Map(),
  archiveTotals:{key_events:130}, libraryLoadErrors:new Set(),
  globalMaterialFilters:{date:"all",action:"all",object:"all",support:"all"},
  search:"",globalMaterialLimit:18};
for (const [id,count] of [["old",2],["latest",1]]) state.runs.push({run_id:id,
  experiment_id:"shared",state:"failed",nas_staging:`/local/${id}`,
  observability:{stage_receipts:[{stage:"key_materials",status:"completed"}],
    partial_delivery:{created_at:id,quarantined_event_count:count}}});
state.archiveCache.set("archive/shared", {key_events:[{event_id:"formal",action_type:"object_movement"}],
  libraryKeys:{materials:JSON.stringify([state.globalMaterialFilters,""])}});
const main={innerHTML:""};
const document={querySelectorAll:()=>[],querySelector:()=>null};
const routeParts=()=>["materials"], refreshProgressiveSurface=()=>{}, setChrome=()=>{};
const bindLibraryImageFallbacks=()=>{}, bindArchivePagination=()=>{};
const esc=x=>String(x??""),number=x=>Number(x||0),formatDate=()=>"date",icon=()=>"";
const productExperimentName=x=>x,productObjectLabel=x=>x,archiveWithinDate=()=>true;
const productEvidenceText=(x,y)=>x||y,productState=()=>"empty",libraryCardSkeleton=()=>"loading";
const eventHasAlignedDualViewMaterial=()=>true,eventHasDualViewSupport=()=>true;
const ACTION_LABELS={object_movement:"物体移动"};
const requests=[];
async function api(url) {
  requests.push(url);
  assert.ok(url.startsWith("/api/staging-runs/"));
  const id=url.split("/")[3];
  const event={event_id:"same-event",cv_action_type:"object_movement",cv_objects:["pipette"],
    frame_url:`/${id}.jpg`,preview_review:{priority:false}};
  const extra={...event,event_id:"second"};
  return {name:"shared",staging_run_id:id,key_events:[],
    preliminary_materials:[event],quarantined_materials:id==="old"?[event,extra]:[event]};
}
'''
    for name in names:
        prefix = f"async function {name}(" if f"async function {name}(" in source else f"function {name}("
        script += prefix + source.split(prefix, 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    script += r'''
(async()=>{
  await ensureLibraryDetails();
  assert.equal(requests.length,2);
  await ensureLibraryDetails();
  assert.equal(requests.length,2,"unchanged records must not reload");
  const entries=materialLibraryEntries();
  assert.equal(entries.filter(x=>!x.preliminary).length,1);
  assert.equal(entries.filter(x=>x.preliminary).length,3,"deduplicate within a run only");
  const latest=entries.find(x=>x.archive.staging_run_id==="latest");
  assert.deepEqual(eventObjectValues(latest.event),["pipette"]);
  const card=globalMaterialCard(latest);
  assert.ok(card.includes('#/stage/latest/materials?candidate=same-event'));
  assert.ok(card.includes('/latest.jpg'));
  assert.ok(card.includes('未满足自动收录条件'));
  assert.ok(!card.includes('双视角印证'));
  renderMaterialsLibrary();
  assert.ok(main.innerHTML.includes('<small>正式素材（全部档案）</small><strong>130</strong>'));
  assert.ok(main.innerHTML.includes('<small>候选素材</small><strong>3</strong>'));
  state.globalMaterialFilters.support="candidate";
  renderMaterialsLibrary();
  assert.ok(!main.innerHTML.includes('?focus=formal'));
  assert.ok(main.innerHTML.includes('?candidate=same-event'));
  await state.libraryLoadPromise;
  state.globalMaterialFilters.support="dual";
  renderMaterialsLibrary();
  assert.ok(!main.innerHTML.includes('?candidate=same-event'));
  await state.libraryLoadPromise;
  state.runs[1].observability.partial_delivery.created_at="updated";
  await ensureLibraryDetails();
  assert.equal(requests.at(-1),"/api/staging-runs/latest/archive?section=library-materials");
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    # Formal pagination is covered separately; keep the existing cached page here.
    script = script.replace('const requests=[];', '''async function loadLibraryDetail(name,section) {
      const data=state.archiveCache.get(`archive/${name}`);
      data.libraryKeys[section]=libraryQueryKey(section); return data;
    }
const requests=[];''')
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_material_counts_refresh_on_new_receipts_and_formal_completion():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for frontend characterization")
    source = (Path(__file__).parents[1] / "src/visioncortex/web/app.js").read_text()
    script = r'''
const assert=require("node:assert/strict");
const state={archiveListingQuery:"",runs:[],archives:[],libraryLoadErrors:new Set(),archiveCache:new Map(),materialCache:new Map()};
const document={hidden:false}, routeParts=()=>["materials"];
const updateServiceChrome=()=>{}, archiveSearchQuery=()=>"";
let renders=0,archiveLoads=0;
const renderMaterialsLibrary=()=>{renders++};
let run={run_id:"run",experiment_id:"same",state:"failed",nas_staging:"/local/run",
  observability:{stage_receipts:[{stage:"key_materials"}],
    partial_delivery:{created_at:"one",quarantined_event_count:61}}};
async function api(url) {
  if(url==="/api/runs") return {runs:[structuredClone(run)]};
  archiveLoads++;
  return {archives:[{name:"same",path:"/formal/same",key_event_count:2}],totals:{key_events:2}};
}
'''
    for name in ("refreshTaskSnapshots", "refreshLibraryOverview", "experimentRecords", "libraryRecordKey", "invalidateArchiveCache", "loadArchiveListing", "applyArchiveListingPage"):
        prefix = f"async function {name}(" if f"async function {name}(" in source else f"function {name}("
        script += prefix + source.split(prefix, 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    script += r'''
(async()=>{
  await refreshTaskSnapshots();
  assert.equal(renders,1);
  assert.equal(experimentRecords()[0].candidate_material_count,61);
  await refreshTaskSnapshots();
  assert.equal(renders,1,"unchanged polling must not rerender");
  run.observability.partial_delivery.created_at="two";
  run.observability.partial_delivery.quarantined_event_count=62;
  await refreshTaskSnapshots();
  assert.equal(renders,2);
  assert.equal(experimentRecords()[0].candidate_material_count,62);
  state.archiveCache.set("archive/same",{release_id:"old"});
  state.archiveCache.set("same:summary",{release_id:"old"});
  state.materialCache.set("same:old:page",{events:["old"]});
  run.state="completed";
  await refreshTaskSnapshots();
  assert.equal(archiveLoads,1);
  assert.equal(state.archiveCache.has("archive/same"),false);
  assert.equal(state.archiveCache.has("same:summary"),false);
  assert.equal(state.materialCache.size,0);
  assert.equal(renders,3);
  assert.equal(state.archiveTotals.key_events,2);
  assert.equal(experimentRecords().length,1);
  assert.equal(experimentRecords()[0].staging_run_id,undefined);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
