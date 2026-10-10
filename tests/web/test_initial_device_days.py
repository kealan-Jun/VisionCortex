"""Keep device-day delivery independent of slow reads and detached pages."""

import shutil
import subprocess

import pytest

from repo_paths import WEB


def app_function(name):
    source = (WEB / "app.js").read_text(encoding="utf-8")
    prefix = f"async function {name}(" if f"async function {name}(" in source else f"function {name}("
    return prefix + source.split(prefix, 1)[1].split("\n}\n", 1)[0] + "\n}\n"


def run_node(script, functions=(), timeline=False):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for browser characterization")
    source = "\n".join(app_function(name) for name in functions)
    if timeline:
        source += (WEB / "day-timeline.js").read_text(encoding="utf-8")
    completed = subprocess.run(
        [node], input='const assert=require("node:assert/strict"); const window={};\n'
        + source + "\n" + script, text=True, capture_output=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


TIMELINE_DOM = r'''
const location={hash:"#/day-timeline/2026-10-10"},chip={textContent:"unchanged"};
const document={hidden:false,querySelector:()=>chip};
const timers=[];global.setTimeout=(callback,ms)=>{timers.push({callback,ms});return timers.length;};
global.clearTimeout=()=>{};
const pages=[];let html="",nodes;
const main={get innerHTML(){return html;},set innerHTML(value){
  if(nodes)Object.values(nodes).forEach(node=>{node.isConnected=false;});
  html=value;
  const node=()=>({isConnected:true,innerHTML:"",textContent:"",insertAdjacentHTML:()=>{},
    querySelector:()=>null});
  nodes=Object.fromEntries(["[data-timeline-date]","[data-timeline-refresh]",
    "[data-processing-overview]","[data-timeline-body]","[data-material-query]",
    "[data-material-result]"].map(selector=>[selector,node()]));
  const time={value:"00:00:00"},duration={value:"60"};
  nodes["[data-material-query]"].querySelector=selector=>selector==="[data-material-time]"?time:duration;
  pages.push(nodes);
},querySelector:selector=>nodes[selector]};
const requests=[];
const api=(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject}));
const ctx={main,api,esc:value=>String(value??""),setChrome:()=>{},state:{deviceDayArchives:[]}};
const emptyDay={entries:[],camera_count:0,indexed_recordings:0,pending_recordings:0,
  activity_seconds:0,errors:[],discovery_errors:[],periods:[],cross_view_links:[]};
const progress=focus=>({focus_date:focus,observed_at:1,running:[],days:{},errors:[]});
const tick=()=>new Promise(setImmediate);
'''


def test_timeline_shell_and_day_output_do_not_wait_for_progress():
    run_node(TIMELINE_DOM + r'''
(async()=>{
  const rendering=window.VisionCortexDayTimeline.render(ctx,"2026-10-10");
  assert.ok(main.innerHTML.includes("2026-10-10 时间线"));
  assert.deepEqual(requests.map(request=>request.url),[
    "/api/device-day-progress","/api/day-timeline/2026-10-10"]);
  assert.equal(requests[0].options.readTimeoutMs,10000);
  assert.equal(requests[1].options.readTimeoutMs,15000);
  requests[1].resolve(emptyDay);await rendering;
  assert.ok(pages[0]["[data-timeline-body]"].innerHTML.includes("当天尚无已处理结果"));
  assert.equal(chip.textContent,"unchanged","pending progress cannot block the day's output");
})().catch(error=>{console.error(error);process.exitCode=1;});
''', timeline=True)


def test_timeline_refresh_ignores_old_progress_and_day_output_even_with_same_hash():
    run_node(TIMELINE_DOM + r'''
(async()=>{
  const old=window.VisionCortexDayTimeline.render(ctx,"2026-10-10");
  const current=window.VisionCortexDayTimeline.render(ctx,"2026-10-10");
  requests[3].resolve({...emptyDay,camera_count:2});await current;
  const currentBody=pages[1]["[data-timeline-body]"].innerHTML;
  requests[0].resolve(progress("2026-09-16"));requests[1].resolve({...emptyDay,camera_count:99});
  await old;await tick();
  assert.equal(chip.textContent,"unchanged");
  assert.equal(pages[0]["[data-timeline-body]"].innerHTML,"");
  assert.equal(pages[1]["[data-timeline-body]"].innerHTML,currentBody);
  requests[2].resolve(progress("2026-10-10"));await tick();
  assert.ok(chip.textContent.includes("后台处理：0"));
})().catch(error=>{console.error(error);process.exitCode=1;});
''', timeline=True)


def test_timeline_default_date_still_follows_verified_progress_focus_when_it_arrives():
    run_node(TIMELINE_DOM + r'''
(async()=>{
  location.hash="#/day-timeline";
  ctx.state.deviceDayArchives=[{archive:"2026-10-09/camera"}];
  const first=window.VisionCortexDayTimeline.render(ctx);
  assert.ok(main.innerHTML.includes("2026-10-09 时间线"));
  requests[0].resolve(progress("2026-09-16"));await tick();
  assert.ok(main.innerHTML.includes("2026-09-16 时间线"));
  assert.equal(requests[3].url,"/api/day-timeline/2026-09-16");
  requests[1].resolve(emptyDay);await first;
  assert.equal(pages[0]["[data-timeline-body]"].innerHTML,"");
  requests[3].resolve(emptyDay);await tick();
  assert.ok(pages[1]["[data-timeline-body]"].innerHTML.includes("当天尚无已处理结果"));
})().catch(error=>{console.error(error);process.exitCode=1;});
''', timeline=True)


@pytest.mark.parametrize("fail", [False, True])
def test_timeline_detached_time_query_cannot_rewrite_a_refreshed_page(fail):
    run_node(TIMELINE_DOM + f"\nconst fail={str(fail).lower()};\n" + r'''
(async()=>{
  const old=window.VisionCortexDayTimeline.render(ctx,"2026-10-10");
  requests[1].resolve(emptyDay);await old;
  pages[0]["[data-material-query]"].onsubmit({preventDefault:()=>{}});
  assert.ok(requests[2].url.includes("/at?"));
  const current=window.VisionCortexDayTimeline.render(ctx,"2026-10-10");
  requests[4].resolve(emptyDay);await current;
  if(fail)requests[2].reject(Error("offline"));
  else requests[2].resolve({start_us:0,end_us:1,devices:[]});
  await tick();
  assert.equal(pages[0]["[data-material-result]"].innerHTML,"");
  assert.equal(pages[1]["[data-material-result]"].innerHTML,"");
  assert.equal(timers.length,0,"detached queries must not schedule more reads");
})().catch(error=>{console.error(error);process.exitCode=1;});
''', timeline=True)


@pytest.mark.parametrize("route", ["device-days", "device-day", "day-timeline", "knowledge"])
def test_direct_delivery_route_renders_without_unrelated_initial_reads(route):
    run_node(f'const route="{route}";\n' + r'''
const location={hash:`#/${route}`},main={innerHTML:"skeleton"},routeParts=()=>[route],events=[];
const state={},document={hidden:false,querySelector:()=>null};window.scrollTo=()=>{};
const router=()=>{events.push("render");main.innerHTML="delivery shell";};
const loadAll=()=>{throw Error("unrelated initial directory reads");};
const loadDeviceDayArchives=()=>{events.push("own listing");return new Promise(()=>{});};
const refreshTaskSnapshots=()=>{throw Error("unrelated periodic snapshots");};
const api=(url,options)=>{events.push(url);assert.equal(options.readTimeoutMs,15000);return new Promise(()=>{});};
const updateServiceChrome=()=>{},productState=()=>"error";
void initializePage();
assert.equal(main.innerHTML,"delivery shell");
assert.deepEqual(events,route==="device-days"?["render","own listing","/api/health"]:["render","/api/health"]);
(async()=>{await refreshPageSnapshots();
  assert.deepEqual(events,route==="device-days"?["render","own listing","/api/health"]:["render","/api/health"]);
})().catch(error=>{console.error(error);process.exitCode=1;});
''', functions=("initializePage", "routeFromNavigation", "refreshPageSnapshots"))


def test_general_snapshot_bundle_loads_once_on_later_navigation_and_ignores_old_route():
    run_node(r'''
let route="home",finish,loads=0;const location={hash:"#/home"},state={};
const routeParts=()=>[route],renders=[];window.scrollTo=()=>{};
const router=()=>{renders.push(route);};
const loadAll=()=>{loads++;return new Promise(resolve=>{finish=()=>{loadAll.initialized=true;resolve();};});};
(async()=>{
  const first=routeFromNavigation();route="tasks";location.hash="#/tasks";
  const second=routeFromNavigation();assert.equal(loads,1);assert.deepEqual(renders,[]);
  route="day-timeline";location.hash="#/day-timeline";await routeFromNavigation();
  assert.deepEqual(renders,["day-timeline"]);finish();await Promise.all([first,second]);
  assert.deepEqual(renders,["day-timeline"],"old bundle responses cannot reset the current page");
  route="tasks";location.hash="#/tasks";await routeFromNavigation();
  assert.equal(loads,1);assert.deepEqual(renders,["day-timeline","tasks"]);
})().catch(error=>{console.error(error);process.exitCode=1;});
''', functions=("routeFromNavigation",))


def test_device_day_polling_is_bounded_and_cannot_render_a_departed_page():
    run_node(r'''
let route="device-days",finish,reads=0,taskReads=0,renders=0;
const location={hash:"#/device-days"},state={deviceDayUpdatedAt:Date.now()},document={hidden:false};
const routeParts=()=>[route],router=()=>{renders++;};
const loadDeviceDayArchives=()=>{reads++;return new Promise(resolve=>{finish=resolve;});};
const refreshTaskSnapshots=()=>{taskReads++;};
(async()=>{
  await refreshPageSnapshots();assert.equal(reads,0,"recent directory reads are reused");
  document.hidden=true;await refreshPageSnapshots(true);assert.equal(reads,0);
  document.hidden=false;const current=refreshPageSnapshots(true);await refreshPageSnapshots(true);
  assert.equal(reads,1,"concurrent polling shares the in-flight directory read");
  route="day-timeline";location.hash="#/day-timeline";finish();await current;
  assert.equal(renders,0);await refreshPageSnapshots();assert.equal(taskReads,0);
  route="tasks";await refreshPageSnapshots();assert.equal(taskReads,1);
})().catch(error=>{console.error(error);process.exitCode=1;});
''', functions=("refreshPageSnapshots",))


def test_timeline_progress_uses_each_stages_eligible_denominator():
    run_node(TIMELINE_DOM + r'''
(async()=>{
  const rendering=window.VisionCortexDayTimeline.render(ctx,"2026-10-10");
  const stages=Object.fromEntries(["retention","vision","stt","understanding","report"].map(stage=>[
    stage,{completed:stage==="stt"?3:1,failed:0,running:0,pending:0,processing_total:stage==="stt"?3:2}
  ]));
  requests[0].resolve({...progress("2026-10-10"),retry_repair_html:"<p>bounded local repair owner</p>",days:{"2026-10-10":{
    total:3,processing_total:2,missing_input_count:1,stages
  }}});await tick();
  const overview=pages[0]["[data-processing-overview]"].innerHTML;
  assert.ok(overview.includes("<strong>3<small> / 3"));assert.ok(overview.includes('value="3" max="3"'));
  assert.equal(overview.includes("<strong>3<small> / 2"),false);
  assert.ok(overview.includes("bounded local repair owner"));
  requests[1].resolve(emptyDay);await rendering;
})().catch(error=>{console.error(error);process.exitCode=1;});
''', timeline=True)


@pytest.mark.parametrize("navigate_away", [False, True])
def test_device_day_listing_renders_while_unrelated_initial_reads_are_pending(navigate_away):
    run_node(f"const navigateAway={str(navigate_away).lower()};\n" + r'''
let route="device-days",renders=0;
const routeParts=()=>[route],router=async()=>{renders++;};
const state={runPollRequestId:0,deviceDayArchives:[],deviceDayErrors:[],deviceDayQueue:{}};
const requests=[];
const api=(url,options)=>new Promise(resolve=>requests.push({url,options,resolve}));
const loadNasRecordings=()=>new Promise(()=>{}),archiveSearchQuery=()=>"all";
const loadArchiveListing=()=>new Promise(()=>{}),updateServiceChrome=()=>{};
(async()=>{
  let complete=false;void loadAll().then(()=>{complete=true;});
  if(navigateAway)route="knowledge";
  requests.find(request=>request.url==="/api/device-days").resolve({archives:[{archive:"current"}],queue:{vision:1}});
  await new Promise(setImmediate);
  assert.equal(complete,false,"unrelated health/archive reads remain pending");
  assert.equal(renders,navigateAway?0:1);
  assert.equal(state.deviceDayArchives[0].archive,"current");
})().catch(error=>{console.error(error);process.exitCode=1;});
''', functions=("loadAll", "loadDeviceDayArchives"))


@pytest.mark.parametrize("old_fails", [False, True])
def test_device_day_listing_discards_obsolete_success_and_failure(old_fails):
    run_node(f"const oldFails={str(old_fails).lower()};\n" + r'''
const state={deviceDayArchives:[],deviceDayErrors:[],deviceDayQueue:{}};
const requests=[],api=()=>new Promise((resolve,reject)=>requests.push({resolve,reject}));
(async()=>{
  const old=loadDeviceDayArchives(),current=loadDeviceDayArchives();
  requests[1].resolve({archives:[{archive:"new"}],queue:{vision:2}});assert.equal(await current,true);
  if(oldFails)requests[0].reject(Error("old failure"));else requests[0].resolve({archives:[{archive:"old"}]});
  assert.equal(await old,false);assert.equal(state.deviceDayArchives[0].archive,"new");
  assert.equal(state.deviceDaySyncError,false);
  const update=loadDeviceDayArchives();requests[2].resolve({archives:[{archive:"new"}],queue:{vision:3}});
  assert.equal(await update,true,"new queue progress must update the library even when files are unchanged");
})().catch(error=>{console.error(error);process.exitCode=1;});
''', functions=("loadDeviceDayArchives",))


def test_existing_snapshot_refresh_updates_the_device_day_library_when_records_change():
    run_node(r'''
const state={libraryLoadErrors:new Set()},document={querySelector:()=>null};
const routeParts=()=>["device-days"];let renders=0;const router=async()=>{renders++;};
refreshLibraryOverview(false);assert.equal(renders,0);
refreshLibraryOverview(true);assert.equal(renders,1);
''', functions=("refreshLibraryOverview",))


def test_explicit_read_deadline_preserves_caller_cancellation_and_leaves_other_calls_unbounded():
    run_node(r'''
const timers=new Map();let nextTimer=0;
global.setTimeout=(callback,ms)=>{const id=++nextTimer;timers.set(id,{callback,ms});return id;};
global.clearTimeout=id=>timers.delete(id);
const requests=[];
global.fetch=(url,options)=>{requests.push({url,options});return new Promise((resolve,reject)=>{
  const abort=()=>reject(options.signal.reason);
  if(options.signal?.aborted)abort();else options.signal?.addEventListener("abort",abort,{once:true});
});};
(async()=>{
  const caller=new AbortController(),options={signal:caller.signal,readTimeoutMs:100};
  const timed=api("/api/device-days",options);const timedResult=assert.rejects(timed,{name:"AbortError"});
  assert.notEqual(requests[0].options.signal,caller.signal);
  assert.equal(requests[0].options.readTimeoutMs,undefined);assert.equal(options.readTimeoutMs,100);
  timers.values().next().value.callback();await timedResult;
  assert.equal(caller.signal.aborted,false,"deadline aborts only its own read");assert.equal(timers.size,0);
  const cancel=new AbortController();const cancelled=api("/api/device-day-progress",{signal:cancel.signal,readTimeoutMs:100});
  const reason=Error("caller cancelled");const cancelledResult=assert.rejects(cancelled,error=>error===reason);
  cancel.abort(reason);await cancelledResult;assert.equal(timers.size,0);
  const upload=new AbortController();void api("/api/uploads",{method:"POST",signal:upload.signal,readTimeoutMs:100});
  assert.equal(requests[2].options.signal,upload.signal);assert.equal(timers.size,0);
  void api("/api/models/schema",{signal:caller.signal});
  assert.equal(requests[3].options.signal,caller.signal);assert.equal(timers.size,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
''', functions=("api",))


def test_read_deadline_covers_response_body_and_releases_its_caller_listener():
    run_node(r'''
let timer;global.setTimeout=callback=>{timer=callback;return 1;};global.clearTimeout=()=>{};
const controller=new AbortController(),signal=controller.signal;
let added=0,removed=0;
const originalAdd=signal.addEventListener.bind(signal),originalRemove=signal.removeEventListener.bind(signal);
signal.addEventListener=(...args)=>{added++;return originalAdd(...args);};
signal.removeEventListener=(...args)=>{removed++;return originalRemove(...args);};
global.fetch=async(url,options)=>({ok:true,json:()=>new Promise((resolve,reject)=>{
  options.signal.addEventListener("abort",()=>reject(options.signal.reason),{once:true});
})});
(async()=>{
  const pending=api("/api/runs",{signal,readTimeoutMs:100});
  const result=assert.rejects(pending,{name:"AbortError"});await new Promise(setImmediate);
  timer();await result;assert.equal(added,1);assert.equal(removed,1);
})().catch(error=>{console.error(error);process.exitCode=1;});
''', functions=("api",))
