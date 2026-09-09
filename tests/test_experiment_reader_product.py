"""Reader navigation must preserve evidence boundaries and output provenance."""
from fastapi import HTTPException
from fastapi.testclient import TestClient

from test_web_library_reliability import run_javascript
from visioncortex import api


def test_reader_groups_only_unambiguously_contained_steps_without_losing_records():
    run_javascript(("readerStepSections",), r'''
const data={workflow_units:[{name:"称量",start_ms:10,end_ms:30},{name:"配液",start_ms:40,end_ms:60}],steps:[
 {start_global_ms:11,end_global_ms:20},{start_global_ms:29,end_global_ms:41},
 {start_global_ms:42,end_global_ms:50},{start_global_ms:null,end_global_ms:15},
 {start_global_ms:70,end_global_ms:80}]};
const before=JSON.stringify(data),groups=readerStepSections(data);
assert.deepEqual(groups.map(x=>x.indices),[[0],[2],[1,3,4]]);
assert.equal(groups[2].name,"单元间及待归属记录");
assert.equal(JSON.stringify(data),before,"presentation never edits source records");
assert.deepEqual(readerStepSections({steps:data.steps})[0].indices,[0,1,2,3,4]);
const ambiguous=readerStepSections({workflow_units:[...data.workflow_units,{name:"冲突",start_ms:10,end_ms:25}],steps:[data.steps[0]]});
assert.deepEqual(ambiguous.at(-1).indices,[0],"overlapping unit ownership stays unresolved");
assert.equal(new Set(groups.flatMap(x=>x.indices)).size,data.steps.length);
''')


def test_playback_never_invents_a_step_in_gaps_or_drops_overlapping_steps():
    run_javascript(("readerPlaybackMatches",), r'''
const intervals=[{start:100,end:200},{start:150,end:220},{start:300,end:330},{start:NaN,end:400}];
assert.deepEqual(readerPlaybackMatches(intervals,160),[0,1]);
assert.deepEqual(readerPlaybackMatches(intervals,200),[1]);
assert.deepEqual(readerPlaybackMatches(intervals,220),[]);
assert.deepEqual(readerPlaybackMatches(intervals,270),[]);
assert.deepEqual(readerPlaybackMatches(intervals,300),[2]);
assert.deepEqual(readerPlaybackMatches(intervals,NaN),[]);
assert.deepEqual(readerPlaybackMatches([{start:4,end:3}],3),[]);
''')


def test_result_tools_keep_staging_links_local_paths_and_report_status_distinct():
    run_javascript(("experimentResultTools",), r'''
const esc=x=>String(x??""),icon=()=>"";
const local={name:"A",staging_run_id:"R",path:"D:/Lab Results/R",links:{partial_report:"/stage-report",partial_json:"/stage.json"}};
const html=experimentResultTools(local,true);
assert.ok(html.includes('data-open-staging-folder="R"'));
assert.ok(!html.includes('data-open-folder='));
assert.ok(html.includes('href="/stage-report"'));
assert.ok(experimentResultTools({...local,links:{...local.links,partial_pdf:'/branded.pdf'}},true).includes('href="/branded.pdf"'));
assert.ok(html.includes('href="/stage.json"'));
assert.ok(html.includes('D:/Lab Results/R'));
assert.ok(!html.includes('NAS'));
assert.ok(experimentResultTools({...local,links:{}},true).includes("报告尚未生成"));
assert.ok(!experimentResultTools({...local,links:{}},true).includes("导出 JSON"));
assert.ok(experimentResultTools({name:"A",path:"/lab/archive/A"}).includes('data-open-folder="A"'));
''')


def test_open_staging_folder_uses_resolver_without_promoting_or_creating_outputs(tmp_path, monkeypatch):
    root = tmp_path / "existing-stage"
    root.mkdir()
    marker = root / "partial.json"
    marker.write_text('{"stage":"failed"}')
    calls = []

    def resolve(run_id):
        if run_id != "R":
            raise HTTPException(404, "unknown run")
        return root

    monkeypatch.setattr(api, "_resolve_staging_run", resolve)
    monkeypatch.setattr(api, "_folder_open_command", lambda path: ["file-manager", str(path)])
    monkeypatch.setattr(api.subprocess, "Popen", lambda command, **kwargs: calls.append((command, kwargs)))
    client = TestClient(api.app)
    assert client.post("/api/staging-runs/unknown/open").status_code == 404
    assert not calls
    response = client.post("/api/staging-runs/R/open")
    assert response.status_code == 200
    assert response.json() == {"status": "opened", "path": str(root)}
    assert calls == [(["file-manager", str(root)], {"start_new_session": True})]
    assert list(root.iterdir()) == [marker]
    assert marker.read_text() == '{"stage":"failed"}'


def test_open_staging_folder_reports_missing_system_opener(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_resolve_staging_run", lambda _: tmp_path)

    def missing(_):
        raise RuntimeError("No supported desktop folder opener is installed")

    monkeypatch.setattr(api, "_folder_open_command", missing)
    response = TestClient(api.app).post("/api/staging-runs/R/open")
    assert response.status_code == 501


def test_step_seek_survives_play_control_reload_and_new_selection_cancels_old_seek():
    run_javascript(("readerSeekVideo",), r'''
const handlers=new Set();let loaded=0,observed=0;
const video={readyState:1,duration:300,currentTime:0,error:true,
 addEventListener:(event,fn)=>handlers.add(fn),removeEventListener:(event,fn)=>handlers.delete(fn),
 load(){loaded++;this.readyState=0;this.currentTime=0;this.error=null;}};
const first=readerSeekVideo(video,180,()=>video.load(),()=>observed++);
assert.equal(video.currentTime,0);assert.equal(handlers.size,1);
first();assert.equal(handlers.size,0);
readerSeekVideo(video,191,null,()=>observed++);
video.readyState=1;[...handlers].forEach(fn=>fn());
assert.equal(video.currentTime,191,"latest requested step survives metadata reload");
assert.equal(observed,1);
readerSeekVideo(video,999,null,()=>{});assert.equal(video.currentTime,300);
readerSeekVideo(video,-2,null,()=>{});assert.equal(video.currentTime,0);
assert.equal(typeof readerSeekVideo(null,NaN,null,()=>{}),"function");
''')


def test_video_first_workspace_keeps_checks_reachable_and_performance_on_its_tab():
    run_javascript(('resultWorkspaceContent', 'stageDeliveryView'), r'''
const esc=String,number=Number,icon=()=>'',STAGE_LABELS={};
const archiveProcessStopped=()=>true,resultHeader=()=>'<nav>RESULT NAV</nav>';
const experimentAttentionPanel=()=>'<p>FAILURE REASON</p>',resultReviewPanel=()=>'<button data-result-check>CHECK</button>';
const componentResultCards=()=>'<p>COMPONENTS</p>',experimentExecutiveSummary=()=>'<p>OVERVIEW</p>';
const experimentResultTools=()=>'<aside>EXPORT</aside>',professionalReportsView=()=>'<p>REPORTS</p>';
const experimentRecordRoute=()=>'/metrics',movementScreeningView=()=>'<p>SCREENING</p>',metricsView=()=>'<p>QUALITY</p>';
const dailyReportView=()=>'<p>DAILY</p>';
globalThis.VisionCortexRunInsights={render:()=>'<section>GPU TOKEN METRICS</section>'};
const data={name:'A',quality_acceptance:{passed:false},result_review:{findings:[{}]}};
const html=resultWorkspaceContent(data,'experiments',true,'','<video>SYNC</video><video>FP</video><video>TP</video>');
assert.equal((html.match(/<video>/g)||[]).length,3);
assert.ok(!html.includes('GPU TOKEN METRICS'));
assert.ok(!html.includes('<details open'));
assert.ok(html.indexOf('RESULT NAV')<html.indexOf('SYNC'));
assert.ok(html.indexOf('SYNC')<html.indexOf('EXPORT'));
assert.ok(html.includes('处理已结束 · 结果待核对'));
assert.ok(!html.includes('完整分析尚未完成'));
assert.ok(!html.includes('data-result-check'));
assert.ok(!html.includes('<details'));
assert.ok(html.includes('href="/metrics"'));
const reports=resultWorkspaceContent(data,'reports',true,'','');
assert.ok(reports.includes('REPORTS') && reports.includes('DAILY'));
assert.ok(!reports.includes('GPU TOKEN METRICS'));
const metrics=resultWorkspaceContent(data,'metrics',true,'','');
assert.ok(metrics.includes('GPU TOKEN METRICS'));
assert.ok(metrics.includes('SCREENING'));
assert.ok(metrics.includes('data-result-check'));
assert.ok(!metrics.includes('REPORTS'));
assert.ok(!metrics.includes('<video>'));
''')
