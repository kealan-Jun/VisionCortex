"""Characterize library queries, failure recovery, and task synchronization."""

import shutil
import subprocess
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[1] / "src/visioncortex/web/app.js"


def test_auxiliary_activity_is_labeled_without_suggesting_unfinished_experiment():
    run_javascript(('workflowCompletionLabel','workflowLabel'), r'''
const item={completion_status:"unresolved",activity_assessment:{is_auxiliary:true,label:"器材整理"}};
assert.equal(workflowLabel(item),"器材整理");
assert.equal(workflowCompletionLabel(item),"辅助活动 · 不计入实验");
''')


def test_stage_version_links_are_read_only_and_legacy_snapshot_is_not_invented():
    run_javascript(('stageDeliveryView',), r'''
const esc=String,STAGE_LABELS={clips:"视频片段",speech:"录音转写"};
const html=stageDeliveryView({observability:{stage_receipts:[
 {stage:"clips",status:"completed",receipt_url:"/receipt",version_url:"/version",artifacts:[{name:"clips.json",url:"/clips"}]},
 {stage:"speech",status:"skipped",receipt_url:"/speech",reason:"没有录音"}]}});
assert.ok(html.includes("/version"));assert.ok(html.includes("没有录音"));
assert.ok(html.includes("此历史阶段的文件与完成记录可查看"));assert.ok(!html.includes("/retry"));
''')


def test_global_library_retains_staging_events_without_promoting_or_losing_their_links():
    run_javascript(("materialLibraryEntries", "globalMaterialCard", "materialFocusRoute"), r'''
const archive={name:"A",staging_run_id:"R"},experimentRecords=()=>[archive];
const libraryRecordQueryKey=()=>"current",eventHasAlignedDualViewMaterial=e=>e.ready;
const detail={libraryKeys:{materials:"current"},key_events:[{event_id:"E",ready:true,aligned_frame_url:"/frame"}],
  quarantined_materials:[{event_id:"Q",frame_url:"/candidate"}],preliminary_materials:[]};
const cachedLibraryDetail=()=>detail;
const rows=materialLibraryEntries();assert.equal(rows.length,2);
assert.equal(rows.filter(e=>!e.preliminary).length,0,"failed run cannot promote formal results");
assert.equal(rows.filter(e=>e.staged).length,1);assert.equal(rows.filter(e=>e.preliminary&&!e.staged).length,1);
const esc=String,icon=()=>"",productExperimentName=String,formatDate=()=>"today",ACTION_LABELS={};
const experimentRecordRoute=()=>"#/stage/R/materials";
const html=globalMaterialCard(rows[0]);
assert.ok(html.includes("?focus=E"));assert.ok(!html.includes("?candidate=E"));
assert.ok(html.includes("阶段素材"));assert.ok(html.includes("尚未通过整体验收"));
assert.equal(detail.key_events.length,1,"library rendering must preserve source records");
''')


def test_stopped_run_summary_counts_retained_materials_and_candidates_separately():
    run_javascript(("experimentAttentionPanel",), r'''
const esc=String,number=Number,icon=()=>"",STAGE_LABELS={semantic_refinement:"完成步骤核验",speech:"录音"};
const experimentRecords=()=>[],archiveProductStatus=()=>({key:"attention"});
const friendlyFailureReason=()=>"质量检查未通过",rerunArchiveAction=()=>"";
const data={name:"run",experiments:[{}],counts:{experiments:5,key_events:32},
  key_events:[],preliminary_materials:[{},{}],quarantined_materials:Array(28).fill({}),
  observability:{stage_receipts:[
    {stage:"semantic_refinement",status:"completed",completed_at:"2026-09-08T13:00:00Z"},
    {stage:"speech",status:"completed",completed_at:"2026-09-08T12:00:00Z"}]}};
const html=experimentAttentionPanel(data,"materials");
assert.ok(html.includes("5 个实验片段、32 份关键素材、30 份候选素材"));
assert.ok(html.includes("尚未生成"));
assert.ok(html.includes("完成步骤核验"),"last completion follows timestamps, not alphabetical file order");
''')


def run_javascript(names, script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for frontend characterization")
    source = SOURCE.read_text()
    functions = []
    for name in names:
        prefix = f"async function {name}(" if f"async function {name}(" in source else f"function {name}("
        functions.append(prefix + source.split(prefix, 1)[1].split("\n}\n", 1)[0] + "\n}\n")
    result = subprocess.run(
        [node, "-e", 'const assert=require("node:assert/strict");\n' + "\n".join(functions) + script],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_material_title_and_support_do_not_promote_partial_cv_pairing():
    run_javascript(("materialOperationTitle", "eventHasDualViewSupport"), r'''
const ACTION_LABELS={hand_object_contact:"手部接触"};
const productEvidenceText=value=>value;
assert.equal(materialOperationTitle({action_type:"hand_object_contact",provenance:{mllm:{operation_title:"握持移液器"}}}),"握持移液器");
assert.ok(materialOperationTitle({action_type:"hand_object_contact"}).includes("对象待核对"));
assert.equal(eventHasDualViewSupport({cross_view_associations:[{both_views_support_action:true,consistency:"partial"}]}),false);
assert.equal(eventHasDualViewSupport({cross_view_associations:[{both_views_support_action:true,consistency:"consistent"}]}),true);
assert.equal(eventHasDualViewSupport({}),false);
''')


def test_partial_analysis_is_terminal_and_never_links_to_formal_archive():
    run_javascript(("archiveProductStatus", "elapsedForRun", "stageResultRoute", "friendlyFailureReason", "guidedStageState"), r'''
const run={state:"partial",run_id:"R",experiment_id:"A",observability:{
  status:{stage:"partial",elapsed_seconds:12,updated_at:"2000-01-01"},
  partial_delivery:{quality_gaps:[{message:"GROUP-2：缺少共同支持"}]}}};
assert.equal(archiveProductStatus({pipeline_stage:"partial"}).key,"attention");
assert.equal(elapsedForRun(run,run.observability.status),12,"finished duration must stop growing");
assert.equal(stageResultRoute(run,"reports"),"#/stage/R/reports");
assert.ok(friendlyFailureReason(run).includes("GROUP-2"));
assert.ok(!friendlyFailureReason(run).includes("网络"));
assert.equal(guidedStageState(run,{stages:["package"],completedBy:[]},new Map(),0).state,"partial");
assert.equal(guidedStageState(run,{stages:["daily_report"],completedBy:[]},new Map(),1).state,"withheld");
''')


def test_long_experiment_evidence_stays_available_without_filling_the_overview():
    run_javascript(("readerClock", "readerStepSections", "workflowCompletionLabel", "workflowLabel", "experimentCard", "experimentExecutiveSummary", "nextStepLabel", "readableEvidenceParts", "evidenceParagraphs", "experimentStepDescription", "experimentLimitations", "evidenceSentences", "experimentObjects", "experimentStepTitle"), r'''
const esc=x=>String(x??""),icon=()=>"",number=x=>Number(x||0),timecode=x=>String(x);
const ACTION_LABELS={object_movement:"物体移动"},OBJECT_LABELS={},productObjectLabel=()=>"样品瓶";
const videoPreview=(url,label)=>`<video src="${url}" aria-label="${label}"></video>`;
const archiveProcessStopped=()=>true;
const long="未观察到开盖，不能确认容器状态变化。".repeat(120);
const group={name:"sample handling",folder:"G1",start_ms:10,end_ms:100,
  summary:long,uncertainties:["先后顺序尚未确认。"],continuity_type:"uncertain",
  aligned_video_url:"/aligned",first_person_video_url:"/first",third_person_video_url:"/third",
  steps:[{action_type:"object_movement",current_step:long,next_step:"无法判断后续。",
    next_step_status:"unknown",objects:["sample_bottle"]}]};
const card=experimentCard(group);
assert.ok(card.includes("样品瓶操作记录"));
assert.ok(!card.includes('data-experiment-step-panel="0" hidden'));
assert.ok(card.includes('<details class="experiment-step-record"'));
assert.ok(card.includes('data-expand-steps aria-expanded="false"'));
assert.equal(experimentStepTitle({operation_title:"拿起试管架",action_type:"object_movement"}),"拿起试管架");
assert.ok(!card.includes("独立实验"));
assert.ok(card.includes('class="experiment-reader-audit"><summary>'));
assert.ok((card.match(/未观察到开盖，不能确认容器状态变化。/g)||[]).length>=360,"all evidence presentations retain the original negations alongside the directory excerpt");
assert.ok(card.includes('class="step-evidence"><summary>'));
assert.ok(card.includes("后续未知"));
assert.ok(card.includes("操作过程"));
const evidence=experimentStepDescription({current_step:"未观察到开盖；物体移动。CV 标签 spearhead 身份不可辨。",next_step:"可能继续操作",next_step_status:"inferred"});
assert.ok(evidence.includes("<li>未观察到开盖；</li>"));
assert.ok(evidence.includes("<li>物体移动。</li>"));
assert.ok(evidence.indexOf("CV 标签")>evidence.indexOf("<details"));
assert.ok(evidence.includes("推测，画面尚未确认"));
const scoped=experimentStepDescription({current_step:"手指按触面板",physical_change:"读数变化未确认",
  observed_result:"读数变化未确认",time_scope:{complete_operation_boundaries_proven:false}});
assert.ok(scoped.includes("完整操作的起止位置尚未核对"));
assert.ok(scoped.includes("当前结果"));
assert.ok(!scoped.includes("精确原片时间"));
const candidate=readableEvidenceParts("未观察到开盖。候选 container_state_change 未被证明。");
assert.deepEqual(candidate.narrative,["未观察到开盖。"]);
assert.deepEqual(candidate.technical,["候选 container_state_change 未被证明。"]);
assert.deepEqual(evidenceSentences("手持试管架（管盖盖合；未开盖）。随后放下。"),["手持试管架（管盖盖合；未开盖）。","随后放下。"]);
assert.ok(!card.includes('class="experiment-view-options"'),"the original three-player area is directly visible");
assert.ok(card.includes('experiment-media--aligned experiment-primary-video'));
assert.ok(card.indexOf('src="/third"')<card.indexOf('<details'),"individual-view players are not hidden inside a disclosure");
assert.equal((card.match(/<video /g)||[]).length,3,"all existing views remain playable");
const overview=experimentExecutiveSummary({experiments:[group,group],counts:{experiments:2},
  observability:{status:{stage:"failed"}}});
assert.ok(overview.includes("2 条操作记录"));
assert.ok(!overview.includes("物体移动"),"the overview does not substitute CV categories for an operation narrative");
assert.ok(!overview.includes(long),"a single group's audit prose is not a run overview");
assert.ok(overview.includes("非完整结论"));
assert.ok(experimentCard({...group,aligned_video_url:null}).includes('src="/first"'));
''')


def test_experiment_browser_selects_one_group_and_preserves_all_navigation_targets():
    run_javascript(("readerClock", "experimentGroupBrowser",), r'''
const esc=String,number=x=>Number(x||0),timecode=String;
const experimentRecordRoute=()=>"#/stage/R/experiments";
const archiveProcessStopped=()=>true,productState=(_a,_b,title)=>title;
const experimentCard=group=>`<article data-test-group="${group.folder}"></article>`;
const groups=Array.from({length:5},(_,index)=>({folder:`G ${index+1}`,start_ms:index*1000,end_ms:(index+1)*1000,steps:[{}]}));
const data={experiments:groups};
const html=experimentGroupBrowser(data,false,"G 3");
assert.equal((html.match(/data-test-group=/g)||[]).length,1,"render only the selected group's players and prose");
assert.ok(html.includes('data-test-group="G 3"'));
assert.equal((html.match(/<option /g)||[]).length,5,"every loaded group remains selectable");
assert.ok(html.includes('value="#/stage/R/experiments?group=G%203" selected'));
assert.ok(html.includes('href="#/stage/R/experiments?group=G%202"'));
assert.ok(html.includes('href="#/stage/R/experiments?group=G%204"'));
assert.ok(experimentGroupBrowser(data).includes('data-test-group="G 1"'));
assert.ok(experimentGroupBrowser({...data,next_cursor:"next"},false,"missing").includes("指定片段尚未载入"));
assert.ok(experimentGroupBrowser({experiments:[]}).includes("本次处理未生成实验片段"));
assert.equal(groups.length,5,"selection must not remove source records");
''')


def test_both_report_tabs_link_saved_partial_report_without_claiming_formal_reports():
    run_javascript(("dailyReportView", "professionalReportsView", "incompleteReportView"), r'''
const esc=x=>String(x??""),icon=()=>"",number=x=>Number(x||0),timecode=x=>String(x);
const archiveProcessStopped=()=>true,friendlyFailureReason=()=>"双视角证据不足";
const experimentRecordRoute=()=>"#/stage/run/experiments",metricsView=()=>"metrics";
const data={links:{partial_report:"/api/staging-file?partial"},experiments:[{
  folder:"G1",start_ms:0,end_ms:60000,steps:[{}],uncertainties:["未确认"]}]};
for (const render of [dailyReportView,professionalReportsView]) {
  const html=render(data);
  assert.ok(html.includes('href="/api/staging-file?partial"'));
  assert.ok(html.includes("双视角证据不足"));
  assert.ok(html.includes("完整日报与正式 PDF / JSON 尚未生成"));
  assert.ok(html.includes("1 个模型步骤"));
  assert.ok(!html.includes('class="report-file-card"'),"partial HTML is never a formal PDF");
}
''')


def test_failure_reason_uses_historical_semantic_receipt_without_guessing_account_failure():
    run_javascript(("friendlyFailureReason",), r'''
const partial=statuses=>({status:"awaiting_semantic_recovery",
  pending_semantic_results:statuses.map(status=>({status}))});
const disabled=friendlyFailureReason({observability:{partial_delivery:partial(["disabled","disabled"])}});
assert.ok(disabled.includes("该任务运行时未启用智能理解"));
assert.ok(disabled.includes("当前 AI 配置"));
assert.ok(!disabled.includes("账户或服务恢复"));
const missing=friendlyFailureReason({partial_delivery:partial(["disabled","skipped_missing_api_key"])});
assert.ok(missing.includes("该任务运行时缺少可用的 AI 密钥"));
for (const statuses of [[],["error"],["disabled","error"]]) {
  const reason=friendlyFailureReason({partial_delivery:partial(statuses)});
  assert.ok(reason.includes("具体原因"));
  assert.ok(!reason.includes("运行时未启用"));
  assert.ok(!reason.includes("账户或服务恢复"));
}
assert.ok(friendlyFailureReason({observability:{status:{error:"connection timeout"}}}).includes("连接暂时中断"));
assert.ok(friendlyFailureReason({observability:{status:{error:"CUDA out of memory"}}}).includes("计算资源暂时不足"));
for (const error of ["FileNotFoundError: [WinError 206] sam2/frames 文件名或扩展名太长", "ENAMETOOLONG frames"]) {
  const reason=friendlyFailureReason({observability:{status:{error}},partial_delivery:partial(["error"])});
  assert.ok(reason.includes("路径过长"));assert.ok(!reason.includes("无法读取"));
}
assert.ok(friendlyFailureReason({}).includes("没有完整结束"));
const qualityReason=friendlyFailureReason({partial_delivery:partial(["error"]),quality_acceptance:{passed:false,
 segmentation_integrity:{canonical_pair_coverage_passed:false},step_action_consistency:{passed:false}}});
assert.ok(qualityReason.includes("跨") || qualityReason.includes("第一与第三人称共同支持"));
assert.ok(qualityReason.includes("步骤文字"));
assert.ok(!qualityReason.includes("AI 配置"));
assert.ok(friendlyFailureReason({error:"Automatic experiment/material quality acceptance failed"}).includes("质量检查"));
''')


def test_preflight_failure_does_not_claim_saved_analysis_or_offer_empty_preview():
    run_javascript(("friendlyFailureReason", "runObservabilityCard", "guidedStageState"), r'''
const esc=x=>String(x??""),icon=()=>"",productExperimentName=String;
const newestFreshness=()=>({stale:false}),guidedPipelineView=()=>"",STAGE_LABELS={failed:"分析失败"};
const run={run_id:"original-run",state:"failed",nas_staging:"/local/staging",
 observability:{status:{},partial_delivery:{},metrics:{nas_index_ingest:{failure_stage:"input_preflight"}},
 stage_receipts:[{stage:"original_ingest"}]}};
const html=runObservabilityCard(run);
assert.ok(!html.includes("阶段产出已保存"));
assert.ok(!html.includes("预览已完成内容"));
assert.ok(html.includes("继续未完成环节"));
assert.ok(friendlyFailureReason(run).includes("分析尚未开始"));
const definition={stages:["original_ingest","input_preflight"],completedBy:["original_ingest"]};
assert.equal(guidedStageState(run,definition,new Map([["original_ingest",{}]]),0).state,"failed");
run.observability.stage_receipts.push({stage:"experiment_clips"});
assert.ok(runObservabilityCard(run).includes("预览已完成内容"));
''')


def test_duration_rounding_carries_seconds_into_minutes_and_hours():
    declaration = "const duration = " + SOURCE.read_text().split("const duration = ", 1)[1].split("\n};", 1)[0] + "\n};\n"
    run_javascript((), declaration + r'''
assert.equal(duration(899.965),"15分00秒");
assert.equal(duration(59.9),"1分00秒");
assert.equal(duration(3599.9),"1时0分0秒");
''')


def test_zero_priority_candidates_are_visible_on_first_visit_without_changing_user_filter():
    run_javascript(("retainedMaterialsView",), r'''
const state={},esc=String,number=x=>Number(x||0),timecode=String,ACTION_LABELS={};
const automaticReviewLabel=()=>"自动核验未完成",productObjectLabel=String,experimentRecordRoute=()=>"#/stage/run/experiments";
const data={preliminary_materials:[{event_id:"E1",timestamp_ms:1234}]};
assert.ok(retainedMaterialsView(data).includes("当前显示 1 / 1 个候选"));
assert.equal(state.retainedMaterialFilters.scope,"all");
state.retainedMaterialFilters.scope="priority";
assert.ok(retainedMaterialsView(data).includes("当前显示 0 / 1 个候选"),"explicit user filtering remains authoritative");
''')


def test_running_retry_cannot_present_retained_findings_as_current_completion():
    run_javascript(("experimentExecutiveSummary",), r'''
const icon=()=>"";
const old={observability:{status:{stage:"experiment_understanding"}},
  experiments:[{summary:"旧轮次没有通过验收的动作"}],
  daily_report:{overview:{key_event_count:61}}};
const html=experimentExecutiveSummary(old);
assert.ok(html.includes("本次分析仍在进行"));
assert.ok(html.includes("复跑时也会保留之前的阶段内容"));
assert.ok(!html.includes("本次完成"));
assert.ok(!html.includes("旧轮次没有通过验收的动作"));
''')


def test_unknown_token_usage_is_distinct_from_zero_and_reused_results():
    run_javascript(("metricTokenValue",), r'''
const number=value=>String(value);
assert.equal(metricTokenValue({total_tokens:null,executed_call_count:2},"total_tokens"),"未知");
assert.equal(metricTokenValue({total_tokens:0,executed_call_count:2},"total_tokens"),"0");
assert.equal(metricTokenValue({total_tokens:null,executed_call_count:0,reused_call_count:5},"total_tokens"),"0");
assert.equal(metricTokenValue({total_tokens:120},"total_tokens"),"120");
''')


@pytest.mark.parametrize("route", [
    "archive/A/reports", "archive/A/materials", "stage/R/materials",
    "tasks", "new", "operations", "ai-settings",
])
def test_initial_directory_failure_does_not_block_independent_pages(route):
    source = SOURCE.read_text()
    startup = "loadAll().then(routeFromNavigation)" + source.split(
        "loadAll().then(routeFromNavigation)", 1
    )[1].split("\n\nlet refreshingNasRecordings", 1)[0]
    run_javascript(("loadAll", "loadNasRecordings", "loadArchiveListing", "router", "routeFromNavigation"), r'''
const state={archives:[],archiveListingQuery:null,runs:[]},main={innerHTML:""};
const document={activeElement:null,querySelector:()=>({addEventListener:()=>{}})};
const archiveSearchQuery=()=>"",updateServiceChrome=()=>{},setChrome=()=>{};
const productState=(_,__,title)=>title;
''' + f'const location={{hash:{("#/" + route)!r}}};\n' + r'''
const routeParts=()=>location.hash.slice(2).split("/");
const renderArchive=(name,tab,staging=false)=>{main.innerHTML=`${staging?"stage":"archive"}/${name}/${tab}`;};
const renderTasks=()=>{main.innerHTML="tasks";},renderNew=()=>{main.innerHTML="new";};
const refreshNasPickers=()=>{};
const renderOperations=()=>{main.innerHTML="operations";};
const window={scrollTo:()=>{},VisionCortexAISettings:{render:()=>{main.innerHTML="ai-settings";}}};
const esc=x=>x,icon=()=>"";
async function api(url){
  if(url.startsWith("/api/archives?"))throw Error("directory unavailable");
  if(url==="/api/runs")return {runs:[{run_id:"R",state:"completed"}]};
  return {};
}
const startup=
''' + startup + r'''
(async()=>{
  await startup;
  assert.equal(main.innerHTML,location.hash.slice(2),"a directory outage must not replace an independently readable page");
  assert.equal(state.archiveRefreshPending,true,"directory recovery must remain scheduled");
  assert.equal(state.runs[0].run_id,"R","available task results must still load");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("task_offline", [False, True])
def test_manual_refresh_keeps_reading_the_current_page_when_directory_is_unavailable(task_offline):
    source = SOURCE.read_text()
    listener = source.split('document.querySelector("#refresh-button").addEventListener("click",', 1)[1].split(
        'const mobileMoreDialog', 1
    )[0]
    run_javascript(("loadAll", "loadNasRecordings", "loadArchiveListing"), r'''
const state={archiveCache:new Map(),materialCache:new Map(),libraryLoadErrors:new Set(),runs:[]};
let refresh,reads=0;
const messages=[],button={disabled:false,classList:{add:()=>{},remove:()=>{}},addEventListener:(_,fn)=>refresh=fn};
const document={activeElement:null,querySelector:()=>button};
const routeParts=()=>["archive","A","reports"],archiveSearchQuery=()=>"",updateServiceChrome=()=>{};
async function router(){reads++;}
const toast=(message,tone)=>messages.push({message,tone});
''' + f'const taskOffline={str(task_offline).lower()};\n' + r'''
async function api(url){
  if(url.startsWith("/api/archives?"))throw Error("directory unavailable");
  if(url==="/api/runs"){if(taskOffline)throw Error("tasks unavailable");return {runs:[]};}
  return {};
}
document.querySelector("#refresh-button").addEventListener("click",
''' + listener + r'''
(async()=>{
  await refresh();
  assert.equal(reads,1,"refresh must still read the current archive page after a directory failure");
  assert.equal(messages.at(-1).tone,"error","partial refresh must not claim everything refreshed");
  assert.ok(messages.at(-1).message.includes("实验目录"));
  if(taskOffline)assert.ok(messages.at(-1).message.includes("任务状态"));
  assert.equal(state.archiveRefreshPending,true);assert.equal(button.disabled,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("route", ["home", "experiments", "materials", "reports"])
@pytest.mark.parametrize("has_completed_run", [False, True])
def test_initial_directory_outage_recovers_without_new_task_or_false_empty_results(route, has_completed_run):
    runs = '[{run_id:"R",state:"completed",experiment_id:"A"}]' if has_completed_run else "[]"
    run_javascript(("loadAll", "loadNasRecordings", "loadArchiveListing", "applyArchiveListingPage",
                    "refreshTaskSnapshots", "refreshLibraryOverview", "archiveSyncNotice"), r'''
const state={archives:[],archiveCache:new Map(),archiveListingQuery:null,archiveTotals:{key_events:0},
 runs:[],libraryLoadErrors:new Set()};
const document={hidden:false,activeElement:null};
const archiveSearchQuery=()=>"",updateServiceChrome=()=>{},invalidateArchiveCache=()=>{};
const experimentRecords=()=>state.archives,libraryRecordKey=x=>x.name;
let available=false,directoryReads=0,filterSyncs=0;
const frames=[],render=()=>frames.push({count:state.archiveTotals.key_events,notice:archiveSyncNotice()});
const renderHome=render,renderExperiments=render,renderMaterialsLibrary=render,renderReportsLibrary=render;
const syncArchiveFiltersFromRoute=()=>filterSyncs++;
''' + f'const routeParts=()=>[{route!r}];\nconst runs={runs};\n' + r'''
async function api(url){
  if(url==="/api/runs")return {runs};
  if(url.startsWith("/api/archives")){
    directoryReads++;if(!available)throw Error("temporary directory outage");
    return {archives:[{name:"A",key_event_count:7}],total_count:1,totals:{key_events:7}};
  }
  return {};
}
(async()=>{
  assert.equal(await loadAll(),false,"directory failure must be reported without blocking independent pages");
  assert.equal(state.archiveListingQuery,null);assert.deepEqual(state.runs,runs);
  assert.equal(state.archiveRefreshPending,true,"initial directory failure must schedule automatic recovery");
  await refreshTaskSnapshots();await refreshTaskSnapshots();
  assert.equal(directoryReads,3,"unchanged task snapshots must not block directory retries");
  assert.equal(frames.length,0,"an unread directory must not render an empty library with zero counts");
  available=true;await refreshTaskSnapshots();
  assert.equal(directoryReads,4);assert.equal(state.archiveRefreshPending,false);
  assert.equal(state.archiveListingQuery,"");assert.deepEqual(frames,[{count:7,notice:""}]);
  assert.equal(filterSyncs,routeParts()[0]==="experiments"?1:0,"recovered experiment deep links must restore route filters");
  await refreshTaskSnapshots();assert.equal(directoryReads,4);assert.equal(frames.length,1);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_obsolete_directory_failure_cannot_requeue_a_newer_successful_refresh():
    run_javascript(("loadArchiveListing", "applyArchiveListingPage", "refreshTaskSnapshots", "refreshLibraryOverview"), r'''
const state={archives:[],archiveListingQuery:"",archiveCache:new Map(),libraryLoadErrors:new Set(),
 runs:[{run_id:"R",state:"key_materials",experiment_id:"A"}]};
const document={hidden:false},archiveSearchQuery=()=>"",routeParts=()=>["tasks"];
const experimentRecords=()=>state.archives,libraryRecordKey=x=>x.name;
const updateServiceChrome=()=>{},renderTasks=()=>{},invalidateArchiveCache=()=>{};
let rejectOld,started,reads=0;
const oldStarted=new Promise(resolve=>started=resolve);
async function api(url){
  if(url==="/api/runs")return {runs:[{run_id:"R",state:"completed",experiment_id:"A"}]};
  if(++reads===1){started();return new Promise((_,reject)=>rejectOld=reject);}
  return {archives:[{name:"A",key_event_count:7}],totals:{key_events:7}};
}
(async()=>{
  const poll=refreshTaskSnapshots();await oldStarted;
  assert.equal(state.archiveRefreshPending,true);
  await loadArchiveListing();assert.equal(state.archiveRefreshPending,false);
  rejectOld(Error("obsolete outage"));await poll;
  assert.equal(state.archiveRefreshPending,false,"late failure must not undo a successful directory refresh");
  assert.equal(state.taskSyncError,false);assert.equal(state.archiveTotals.key_events,7);
  await refreshTaskSnapshots();assert.equal(reads,2,"no redundant retry after the newer request succeeded");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_obsolete_directory_query_failure_does_not_start_retries_for_current_page():
    run_javascript(("loadArchiveListing",), r'''
const state={archiveRefreshPending:false,archiveListingQuery:""};
let query="old query",rejectOld;
const archiveSearchQuery=()=>query,api=()=>new Promise((_,reject)=>rejectOld=reject);
(async()=>{
  const pending=loadArchiveListing();query="";
  rejectOld(Error("old query unavailable"));assert.equal(await pending,false);
  assert.equal(state.archiveRefreshPending,false);assert.equal(state.archiveListingQuery,"");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_failed_search_directory_retries_without_rendering_previous_query_counts():
    run_javascript(("loadArchiveListing", "applyArchiveListingPage", "router",
                    "refreshTaskSnapshots", "refreshLibraryOverview"), r'''
const state={archives:[{name:"A",key_event_count:100}],archiveListingQuery:"old",archiveTotals:{key_events:100},
 runs:[],archiveCache:new Map(),libraryLoadErrors:new Set()};
const document={hidden:false,querySelector:()=>({addEventListener:()=>{}})},location={hash:"#/experiments"};
const routeParts=()=>["experiments"],archiveSearchQuery=()=>"current",experimentRecords=()=>state.archives,libraryRecordKey=x=>x.name;
const updateServiceChrome=()=>{},invalidateArchiveCache=()=>{},setChrome=()=>{},syncArchiveFiltersFromRoute=()=>{};
const main={innerHTML:""},pageSkeleton=()=>"loading",productState=()=>"directory unavailable";
const counts=[],renderExperiments=()=>counts.push(state.archiveTotals.key_events);
let available=false,reads=0;
async function api(url){
  if(url==="/api/runs")return {runs:[]};
  assert.ok(url.endsWith("&q=current"));reads++;
  if(!available)throw Error("query unavailable");
  return {archives:[{name:"B",key_event_count:3}],totals:{key_events:3}};
}
(async()=>{
  await router();assert.equal(main.innerHTML,"directory unavailable");
  assert.equal(state.archiveRefreshPending,true);
  await refreshTaskSnapshots();assert.deepEqual(counts,[],"old query totals must not appear as current query results");
  assert.equal(main.innerHTML,"directory unavailable");
  available=true;await refreshTaskSnapshots();assert.deepEqual(counts,[3]);
  assert.equal(state.archiveListingQuery,"current");assert.equal(state.archiveRefreshPending,false);
  await refreshTaskSnapshots();assert.equal(reads,3);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("route", ["home", "experiments", "materials", "reports"])
def test_library_directory_sync_failure_is_visible_without_count_change_and_recovers(route):
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview", "archiveSyncNotice"), r'''
const archiveSearchQuery=()=>"",syncArchiveFiltersFromRoute=()=>{};
const saved={name:"A",key_event_count:2};
const state={archiveListingQuery:"",runs:[{run_id:"R",state:"key_materials",experiment_id:"A"}],archives:[saved],
 archiveTotals:{key_events:2},libraryLoadErrors:new Set(["existing-detail-failure"])};
const document={hidden:false},experimentRecords=()=>state.archives,libraryRecordKey=x=>x.name;
const updateServiceChrome=()=>{},invalidateArchiveCache=()=>{},frames=[];
const render=()=>frames.push({notice:archiveSyncNotice(),count:state.archiveTotals.key_events});
const renderHome=render,renderExperiments=render,renderMaterialsLibrary=render,renderReportsLibrary=render;
''' + f'const routeParts=()=>[{route!r}];\n' + r'''
const api=async()=>({runs:[{run_id:"R",state:"completed",experiment_id:"A"}]});
let available=false,requests=0;
async function loadArchiveListing(){requests++;if(!available)throw Error("directory unavailable");state.archives=[{name:"A",key_event_count:7}];state.archiveTotals={key_events:7};state.archiveRefreshPending=false;return true;}
(async()=>{
  await refreshTaskSnapshots();
  assert.equal(state.runs[0].state,"completed");assert.equal(state.taskSyncError,false);
  assert.equal(frames.length,1);assert.ok(frames[0].notice.includes("当前列表与数量尚未确认包含最新成果"));
  assert.equal(frames[0].count,2,"last known count must stay available while explicitly marked stale");
  assert.equal(state.archives[0],saved);assert.equal(state.libraryLoadErrors.size,1,"sync notice must not restart unrelated failed detail loads");
  await refreshTaskSnapshots();assert.equal(requests,2);assert.equal(frames.length,1,"unchanged failure must not repeatedly replace the page");
  available=true;await refreshTaskSnapshots();
  assert.equal(state.archiveRefreshPending,false);assert.equal(frames.length,2);assert.equal(frames[1].notice,"");
  assert.equal(frames[1].count,7);assert.equal(state.libraryLoadErrors.size,0);
  await refreshTaskSnapshots();assert.equal(requests,3);assert.equal(frames.length,2);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("route", ["home", "experiments", "materials", "reports"])
@pytest.mark.parametrize("initial_load", [False, True])
def test_directory_retry_recovers_while_task_status_remains_unavailable(route, initial_load):
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview", "loadArchiveListing",
                    "applyArchiveListingPage", "archiveSyncNotice"), r'''
const saved={run_id:"R",state:"completed",experiment_id:"A"};
const state={runs:[saved],activeRun:saved,archives:[{name:"A",key_event_count:2}],archiveListingQuery:"",
 archiveTotals:{key_events:2},archiveRefreshPending:true,libraryReleaseRefreshPending:true,
 archiveCache:new Map(),libraryLoadErrors:new Set()};
const document={hidden:false},archiveSearchQuery=()=>"",syncArchiveFiltersFromRoute=()=>{};
const experimentRecords=()=>state.archives,libraryRecordKey=x=>x.name;
const updateServiceChrome=()=>{},invalidateArchiveCache=()=>{},frames=[];
const render=()=>frames.push({count:state.archiveTotals.key_events,notice:archiveSyncNotice()});
const renderHome=render,renderExperiments=render,renderMaterialsLibrary=render,renderReportsLibrary=render;
''' + f'const routeParts=()=>[{route!r}];\nconst initialLoad={str(initial_load).lower()};\n' + r'''
if(initialLoad){state.archives=[];state.archiveListingQuery=null;state.archiveTotals={key_events:0};}
let tasksAvailable=false,directoryAvailable=false,directoryReads=0;
async function api(url){
  if(url==="/api/runs"){if(!tasksAvailable)throw Error("tasks unavailable");return {runs:[saved]};}
  directoryReads++;if(!directoryAvailable)throw Error("directory unavailable");
  return {archives:[{name:"A",key_event_count:7}],total_count:1,totals:{key_events:7}};
}
(async()=>{
  await refreshTaskSnapshots();
  assert.equal(directoryReads,1,"task endpoint failure must not skip the queued directory retry");
  assert.equal(state.taskSyncError,true);assert.equal(state.archiveRefreshPending,true);
  assert.equal(state.libraryReleaseRefreshPending,true);assert.equal(state.runs[0],saved);
  assert.equal(frames.length,initialLoad?0:1,"an unread directory must not be rendered as empty");
  directoryAvailable=true;await refreshTaskSnapshots();
  assert.equal(directoryReads,2);assert.equal(state.archiveTotals.key_events,7);
  assert.equal(state.archiveRefreshPending,false);assert.equal(state.libraryReleaseRefreshPending,false);
  assert.equal(state.taskSyncError,true,"directory recovery must not imply task status recovered");
  assert.equal(state.activeRun,saved);assert.equal(state.runs[0],saved);
  assert.equal(frames.at(-1).count,7);assert.ok(frames.at(-1).notice.includes("任务状态同步中断"));
  const rendered=frames.length;await refreshTaskSnapshots();
  assert.equal(directoryReads,2);assert.equal(frames.length,rendered,"unchanged task failures must not restart directory or page reads");
  tasksAvailable=true;await refreshTaskSnapshots();
  assert.equal(state.taskSyncError,false);assert.equal(directoryReads,2);
  assert.equal(frames.at(-1).notice,"");assert.equal(frames.at(-1).count,7);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("route", ["home", "experiments", "materials", "reports"])
def test_library_task_sync_disconnect_and_recovery_update_notice_without_erasing_results(route):
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview", "archiveSyncNotice"), r'''
const archiveSearchQuery=()=>"",syncArchiveFiltersFromRoute=()=>{};
const saved={run_id:"R",state:"completed"},record={name:"A",key_event_count:2};
const state={archiveListingQuery:"",runs:[saved],archives:[record],libraryLoadErrors:new Set(["detail-failure"])};
const document={hidden:false},experimentRecords=()=>state.archives,libraryRecordKey=x=>x.name,updateServiceChrome=()=>{};
const notices=[],render=()=>notices.push(archiveSyncNotice());
const renderHome=render,renderExperiments=render,renderMaterialsLibrary=render,renderReportsLibrary=render;
''' + f'const routeParts=()=>[{route!r}];\n' + r'''
let disconnected=true;
const api=async()=>{if(disconnected)throw Error("task endpoint unavailable");return {runs:[saved]};};
(async()=>{
  await refreshTaskSnapshots();
  assert.equal(notices.length,1);assert.ok(notices[0].includes('role="alert"'));assert.ok(notices[0].includes("任务状态同步中断"));
  assert.equal(state.runs[0],saved);assert.equal(state.archives[0],record);assert.equal(state.libraryLoadErrors.size,1);
  await refreshTaskSnapshots();assert.equal(notices.length,1);
  disconnected=false;await refreshTaskSnapshots();
  assert.deepEqual(notices.slice(1),[""]);assert.equal(state.archives[0],record);assert.equal(state.libraryLoadErrors.size,1);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_pending_directory_sync_set_by_another_poll_becomes_visible_on_next_snapshot():
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview", "archiveSyncNotice"), r'''
const archiveSearchQuery=()=>"";
const state={archiveListingQuery:"",runs:[{run_id:"R",state:"completed"}],archiveRefreshPending:true,libraryLoadErrors:new Set()};
const document={hidden:false},experimentRecords=()=>[],updateServiceChrome=()=>{},routeParts=()=>["materials"];
let renders=0;const renderMaterialsLibrary=()=>{renders++;assert.ok(archiveSyncNotice().includes("目录仍在同步"))};
const api=async()=>({runs:state.runs}),loadArchiveListing=async()=>{throw Error("directory unavailable")};
(async()=>{await refreshTaskSnapshots();assert.equal(renders,1);await refreshTaskSnapshots();assert.equal(renders,1);})().catch(error=>{console.error(error);process.exitCode=1});
''')


def run_material_library_query(script):
    run_javascript(("renderMaterialsLibrary", "archiveSyncNotice", "materialLibraryEntries", "libraryRecordQueryKey",
                    "libraryQueryKey", "cachedLibraryDetail", "cachedArchiveDetail", "libraryRecordKey",
                    "eventObjectValues", "eventHasAlignedDualViewMaterial", "closeGlobalSearch"), r'''
const archive={name:"A"},state={archives:[archive],runs:[],search:"开盖",globalMaterialLimit:18,
 globalMaterialFilters:{date:"all",action:"all",object:"all",support:"all"},archiveTotals:{key_events:100},archiveCache:new Map()};
const formal={event_id:"E17",action_type:"container_state_change",dual_view_material_ready:true};
const candidate={event_id:"C1",cv_action_type:"container_state_change"};
const detail={material_total_count:1,key_events:[formal,{event_id:"UNREADY",action_type:"container_state_change"}],preliminary_materials:[candidate]};
function publish(){detail.libraryKeys={materials:libraryQueryKey("materials")};state.archiveCache.set("archive/A",detail)}
publish();
const main={innerHTML:""},handlers={},input={value:state.search,setAttribute(name,value){this[name]=value}},panel={hidden:false};
const document={querySelectorAll:()=>[],querySelector:selector=>selector==="#global-search"?input:selector==="#global-search-results"?panel:
 selector==="[data-reset-global-materials]"&&main.innerHTML.includes("data-reset-global-materials")?{addEventListener:(_,fn)=>handlers.reset=fn}:null};
const experimentRecords=()=>state.archives,setChrome=()=>{},ensureLibraryDetails=()=>{};
const libraryLoadState=()=>({records:state.archives,ready:state.archives,failed:[],pending:[]});
let withinDate=true;const archiveWithinDate=()=>withinDate,eventHasDualViewSupport=()=>false;
const productExperimentName=String,ACTION_LABELS={container_state_change:"容器状态变化"};
const experimentRecordRoute=()=>"#",globalMaterialCard=entry=>`CARD:${entry.event.event_id}:${entry.preliminary?"candidate":"formal"}`;
const libraryFailureNotice=()=>"",number=String,icon=()=>"",esc=String,productObjectLabel=String;
const productState=(...args)=>args.join(" "),bindLibraryImageFallbacks=()=>{},bindLibraryRetry=()=>{},bindArchivePagination=()=>{};
''' + script)


@pytest.mark.parametrize("query", ["开盖", "E17", "cap open", "container_state_change"])
def test_material_library_keeps_server_alias_identifier_and_multiword_matches(query):
    run_material_library_query(f'state.search={query!r};publish();\n' + r'''
renderMaterialsLibrary();
assert.ok(main.innerHTML.includes("CARD:E17:formal"),"server search results must not be narrowed to literal card text");
assert.ok(!main.innerHTML.includes("CARD:UNREADY"),"search cannot promote incomplete media to formal material");
assert.ok(!main.innerHTML.includes("CARD:C1"),"server result count does not prove an unfiltered candidate matched");
assert.ok(main.innerHTML.includes('<small>正式素材（全部档案）</small><strong>100</strong>'));
state.search="another query";renderMaterialsLibrary();
assert.ok(!main.innerHTML.includes("CARD:E17"),"a prior query's indexed response must not bypass the current query");
''')


def test_material_search_still_applies_local_filters_and_candidate_text_matching():
    run_material_library_query(r'''
state.globalMaterialFilters.support="candidate";publish();renderMaterialsLibrary();
assert.ok(!main.innerHTML.includes("CARD:E17"));assert.ok(!main.innerHTML.includes("CARD:C1"));
candidate.provenance={mllm:{current_step:"开盖候选，仅供查阅"}};renderMaterialsLibrary();
assert.ok(main.innerHTML.includes("CARD:C1:candidate"));assert.ok(!main.innerHTML.includes("CARD:E17"));
withinDate=false;renderMaterialsLibrary();assert.ok(!main.innerHTML.includes("CARD:C1"));
withinDate=true;state.globalMaterialFilters.support="formal";publish();renderMaterialsLibrary();
assert.ok(main.innerHTML.includes("CARD:E17:formal"));assert.ok(!main.innerHTML.includes("CARD:C1"));
state.globalMaterialFilters.object="unmatched-object";publish();renderMaterialsLibrary();assert.ok(!main.innerHTML.includes("CARD:E17"));
state.globalMaterialFilters.object="all";state.globalMaterialFilters.support="dual";publish();renderMaterialsLibrary();assert.ok(!main.innerHTML.includes("CARD:E17"));
state.globalMaterialFilters.support="all";publish();delete detail.material_total_count;renderMaterialsLibrary();
assert.ok(!main.innerHTML.includes("CARD:E17"),"unfiltered fallback data still needs local text matching");
''')


def test_reset_global_material_filters_clears_search_and_restores_unfiltered_query():
    run_material_library_query(r'''
state.search="absent";input.value="absent";state.globalMaterialLimit=60;
state.globalMaterialFilters={date:"7d",action:"object_movement",object:"pipette",support:"dual"};
detail.key_events=[];detail.preliminary_materials=[];detail.material_total_count=0;publish();
renderMaterialsLibrary();assert.equal(typeof handlers.reset,"function");handlers.reset();
assert.equal(state.search,"");assert.equal(input.value,"");assert.equal(panel.hidden,true);
assert.equal(input["aria-expanded"],"false");assert.equal(state.globalMaterialLimit,18);
assert.deepEqual(state.globalMaterialFilters,{date:"all",action:"all",object:"all",support:"all"});
assert.ok(!main.innerHTML.includes("CARD:"),"filtered response must not be reused as the restored complete result");
detail.key_events=[formal];detail.material_total_count=1;publish();renderMaterialsLibrary();
assert.ok(main.innerHTML.includes("CARD:E17:formal"));
assert.ok(main.innerHTML.includes('<small>正式素材（全部档案）</small><strong>100</strong>'));
''')


@pytest.mark.parametrize("page", ["first", "next"])
@pytest.mark.parametrize("cached_section", ["library", "summary"])
def test_archive_pages_refresh_global_totals_and_discard_off_page_old_release(page, cached_section):
    run_javascript(("loadArchiveListing", "loadMoreArchives", "applyArchiveListingPage", "invalidateArchiveCache"), r'''
const a={name:"A",release_id:"a1",modified_at:"today",key_event_count:1};
const currentA={release_id:"a1",key_events:[{event_id:"keep"}]};
const state={archives:[a],archiveListingQuery:"",archiveRequestId:0,archiveNextCursor:"next",archiveTotal:2,
 archiveTotals:{experiments:2,key_events:2},archiveCache:new Map([["archive/A",currentA]]),materialCache:new Map([["A:a1:page",{}],["B:b1:page",{}]])};
const archiveSearchQuery=()=>"";
''' + f'const page={page!r},cachedSection={cached_section!r};\n' + r'''
const oldKey=cachedSection==="library"?"archive/B":"B:summary";
state.archiveCache.set(oldKey,{release_id:"b1",key_events:[{event_id:"old"}]});
state.archiveCache.set("B:b1:reports",{daily_report:{report_id:"old-report"}});
const api=async()=>({archives:[{name:"B",release_id:"b2",key_event_count:6}],total_count:2,
 totals:{experiments:3,key_events:7},next_cursor:page==="first"?"page-2":null});
(async()=>{
  if(page==="first")await loadArchiveListing();else await loadMoreArchives();
  assert.equal(state.archiveTotals.key_events,7,"global material total must use the received page's aggregate");
  assert.equal(state.archiveTotals.experiments,3);assert.equal(state.archiveTotal,2);
  assert.deepEqual(state.archives.map(x=>x.name),page==="first"?["B"]:["A","B"]);
  assert.equal(state.archiveCache.has(oldKey),false,"off-page old release must not be reused as current material");
  assert.equal(state.archiveCache.has("B:b1:reports"),false);assert.equal(state.materialCache.has("B:b1:page"),false);
  assert.equal(state.archiveCache.get("archive/A"),currentA,"unchanged archives retain their loaded material");
  assert.equal(state.materialCache.has("A:a1:page"),true);assert.equal(state.archiveCacheEpoch,1);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_overlapping_archive_page_updates_changed_record_once_and_preserves_unchanged_cache():
    run_javascript(("loadMoreArchives", "applyArchiveListingPage", "invalidateArchiveCache"), r'''
const a={name:"A",release_id:"r1",key_event_count:1,modified_at:"today"},cached={release_id:"r1"};
const state={archives:[a],archiveListingQuery:"",archiveRequestId:0,archiveNextCursor:"next",archiveTotal:2,
 archiveTotals:{key_events:2},archiveCache:new Map([["archive/A",cached]]),materialCache:new Map()};
const archiveSearchQuery=()=>"";let changed=false;
const api=async()=>({archives:[{...a,key_event_count:changed?3:1},{name:"B",release_id:"r1"}],total_count:2,
 totals:{key_events:changed?4:2},next_cursor:"another"});
(async()=>{
  await loadMoreArchives();assert.equal(state.archiveCache.get("archive/A"),cached);
  assert.equal(state.archiveCacheEpoch,undefined,"unchanged overlap must not invalidate other in-flight work");
  changed=true;await loadMoreArchives();
  assert.deepEqual(state.archives.map(x=>x.name),["A","B"],"overlap must not duplicate the experiment");
  assert.equal(state.archives[0].key_event_count,3);assert.equal(state.archiveTotals.key_events,4);
  assert.equal(state.archiveCache.has("archive/A"),false);assert.equal(state.archiveCacheEpoch,1);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_empty_archive_page_keeps_explicit_zero_totals_and_missing_aggregate_does_not_reset_counts():
    run_javascript(("loadMoreArchives", "applyArchiveListingPage"), r'''
const state={archives:[{name:"A"}],archiveListingQuery:"",archiveRequestId:0,archiveNextCursor:"next",
 archiveTotal:2,archiveTotals:{experiments:2,key_events:4},archiveCache:new Map()};
const archiveSearchQuery=()=>"";let payload={archives:[],next_cursor:"next-again"};
const api=async()=>payload;
(async()=>{
  await loadMoreArchives();assert.equal(state.archiveTotals.key_events,4,"absent aggregates must not fabricate zero");
  payload={archives:[],total_count:0,totals:{experiments:0,key_events:0},next_cursor:null};
  await loadMoreArchives();assert.equal(state.archiveTotal,0,"an explicit server zero must not fall back to loaded row count");
  assert.equal(state.archiveTotals.key_events,0);assert.equal(state.archiveNextCursor,null);
  assert.deepEqual(state.archives,[],"an empty directory must not retain ghost rows from earlier pages");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_obsolete_archive_page_cannot_change_new_totals_or_invalidate_new_release_cache():
    run_javascript(("loadArchiveListing", "loadMoreArchives", "applyArchiveListingPage", "invalidateArchiveCache"), r'''
const state={archives:[{name:"B",release_id:"b1"}],archiveListingQuery:"",archiveRequestId:0,archiveNextCursor:"old-next",
 archiveTotal:2,archiveTotals:{key_events:2},archiveCache:new Map(),materialCache:new Map()};
const archiveSearchQuery=()=>"";let finish;
const api=url=>url.includes("cursor=")?new Promise(resolve=>finish=resolve):Promise.resolve({archives:[{name:"B",release_id:"b2"}],total_count:1,totals:{key_events:8}});
(async()=>{
  const pending=loadMoreArchives();await loadArchiveListing();
  const current={release_id:"b2",key_events:[{event_id:"new"}]};state.archiveCache.set("archive/B",current);
  const epoch=state.archiveCacheEpoch;
  finish({archives:[{name:"B",release_id:"b1"}],total_count:2,totals:{key_events:2},next_cursor:"wrong"});await pending;
  assert.equal(state.archiveTotals.key_events,8);assert.equal(state.archiveTotal,1);assert.equal(state.archiveNextCursor,null);
  assert.equal(state.archiveCache.get("archive/B"),current);assert.equal(state.archiveCacheEpoch,epoch);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_experiment_metadata_normalizes_invalid_shapes_and_reuses_unchanged_storage():
    run_javascript(("loadExperimentMetadata", "experimentMetadata", "saveExperimentMetadata"), r'''
const state={},EXPERIMENT_META_KEY="test-only";
let stored="null",writesFail=false,parses=0;
const localStorage={getItem:()=>stored,setItem:(key,value)=>{if(writesFail)throw Error("full");stored=value}};
const parse=JSON.parse;JSON.parse=(...args)=>{parses++;return parse(...args)};
for(const value of ["null","[]","5",'"old-format"','{"A":null,"B":[]}']){
  stored=value;assert.deepEqual(experimentMetadata("A"),{});
}
stored='{"A":{"displayName":"本机名称","owner":5,"tags":["质控",null,{}],"note":[]}}';
const a=loadExperimentMetadata();assert.equal(a.A.displayName,"本机名称");
assert.deepEqual(a.A.tags,["质控"]);assert.equal(a.A.owner,"");assert.equal(a.A.note,"");
const before=parses;assert.equal(loadExperimentMetadata(),a);assert.equal(parses,before,"unchanged metadata must not be parsed for every card");
assert.deepEqual(experimentMetadata("constructor"),{},"archive names must not resolve inherited properties");
writesFail=true;assert.throws(()=>saveExperimentMetadata("A",{displayName:"unsaved"}));
assert.equal(experimentMetadata("A").displayName,"本机名称","failed storage write must not mutate the cached display");
writesFail=false;saveExperimentMetadata("A",{displayName:"已保存",tags:["更新"]});
assert.equal(experimentMetadata("A").displayName,"已保存");
stored="invalid-json";assert.deepEqual(loadExperimentMetadata(),{});
''')


def test_local_display_name_owner_and_tag_search_keep_paginated_server_listing_unfiltered():
    run_javascript(("archiveSearchQuery", "loadExperimentMetadata", "experimentMetadata", "filteredArchives"), r'''
const state={search:"",archiveFilters:{},archives:[{name:"RAW-A"}]},EXPERIMENT_META_KEY="test-only";
const localStorage={getItem:()=>JSON.stringify({"RAW-A":{displayName:"称量质控",owner:"张老师",tags:["第二轮"]}})};
let route="experiments";const routeParts=()=>[route],experimentRecords=()=>state.archives;
const productExperimentName=name=>experimentMetadata(name).displayName||name;
for(const query of ["称量质控","张老师","第二轮"]){
  state.search=query;assert.equal(archiveSearchQuery(),"","browser-only metadata must not be sent as a raw-name server filter");
  assert.deepEqual(filteredArchives().map(x=>x.name),["RAW-A"]);
}
state.search="RAW-A";assert.equal(archiveSearchQuery(),"RAW-A","raw identifiers retain bounded server search");
state.search="unmatched";assert.equal(archiveSearchQuery(),"unmatched");
route="materials";assert.equal(archiveSearchQuery(),"");
''')


def test_clearing_search_on_unchanged_route_reloads_listing_and_restores_pagination():
    run_javascript(("bindArchiveFilters", "updateArchiveFilterRoute", "archiveLibraryHash"), r'''
const state={search:"A",archiveListingQuery:"A",archiveNextCursor:"old",archives:[{name:"A"}],
  archiveFilters:{status:"all",date:"all",owner:"all",tag:"all",view:"list"}};
const location={hash:"#/experiments"};let clear;
const input={value:"A"},button={addEventListener:(name,fn)=>clear=fn};
const document={querySelectorAll:selector=>selector==="[data-clear-archive-filters]"?[button]:[],querySelector:()=>input};
let reloads=0;const router=()=>{reloads++;state.archiveListingQuery="";state.archives=[{name:"A"},{name:"B"}];state.archiveNextCursor="all-page"};
const renderExperiments=()=>{throw Error("render alone does not revalidate the listing")};
bindArchiveFilters();clear();
assert.equal(reloads,1);assert.equal(state.search,"");assert.equal(input.value,"");
assert.equal(state.archiveListingQuery,"");assert.equal(state.archiveNextCursor,"all-page");
assert.deepEqual(state.archives.map(x=>x.name),["A","B"]);
''')


def test_filtered_empty_results_are_not_presented_as_an_empty_workspace():
    run_javascript(("archiveRows",), r'''
const state={search:"",archiveNextCursor:null},icon=()=>"";
const productState=(...args)=>args.join(" ");
let html=archiveRows([],"experiments","list",true);
assert.ok(html.includes("没有符合条件的实验"));assert.ok(html.includes("清空筛选"));assert.ok(!html.includes("创建实验"));
state.archiveNextCursor="more";html=archiveRows([],"experiments","list",true);
assert.ok(html.includes("已载入的记录中暂无匹配项"));assert.ok(html.includes("当前空结果不代表没有匹配实验"));
state.archiveNextCursor=null;html=archiveRows([],"experiments","list",false);
assert.ok(html.includes("从第一个实验开始"));assert.ok(html.includes("创建实验"));
''')


def test_selected_owner_and_tag_remain_visible_when_their_record_is_on_another_page():
    run_javascript(("renderExperiments", "archiveSyncNotice"), r'''
const state={search:"",archives:[{name:"A"}],archiveTotal:80,archiveNextCursor:"more",archiveFilters:{owner:"未载入负责人",tag:"未载入标签",status:"all",date:"all",view:"list"}};
const main={innerHTML:""},setChrome=()=>{},experimentRecords=()=>state.archives,experimentMetadata=()=>({}),filteredArchives=()=>[];
const isNasMode=()=>false,number=String,icon=()=>"",esc=String,archiveRows=()=>"empty",bindArchiveFilters=()=>{},bindArchivePagination=()=>{};
renderExperiments();
assert.ok(main.innerHTML.includes('<option value="未载入负责人" selected>未载入负责人</option>'));
assert.ok(main.innerHTML.includes('<option value="未载入标签" selected>未载入标签</option>'));
assert.ok(main.innerHTML.includes("当前仅匹配已载入的记录"));assert.ok(main.innerHTML.includes("1 / 80"));
state.archiveFilters.owner="all";state.archiveFilters.tag="all";renderExperiments();
assert.ok(main.innerHTML.includes("已载入记录<span"));assert.ok(!main.innerHTML.includes("全部记录<span"));
state.archiveNextCursor=null;renderExperiments();assert.ok(main.innerHTML.includes("全部记录<span"));
''')


def test_playback_controls_target_their_own_card_when_event_ids_repeat():
    run_javascript(("bindMaterialInteractions", "materialVideo", "materialSelectionKey"), r'''
const state={},data={key_events:[{event_id:"E",event_uid:"u1"},{event_id:"E",event_uid:"u2"}]};
const first={dataset:{materialVideo:"E"},currentTime:0,playbackRate:1,loop:false};
const second={dataset:{materialVideo:"E"},currentTime:0,duration:10,playbackRate:1,loop:false,requestFullscreen:async()=>{second.fullscreen=true}};
const card={dataset:{materialSelectionKey:materialSelectionKey(data.key_events[1])},querySelector:()=>second};
const controls=new Map();
for(const [selector,dataset] of [
  ["[data-material-focus]",{materialFocus:"E"}],["[data-video-seek]",{videoSeek:"E",seconds:"4"}],
  ["[data-video-speed]",{videoSpeed:"E"}],["[data-video-loop]",{videoLoop:"E"}],
  ["[data-video-fullscreen]",{videoFullscreen:"E"}],
])controls.set(selector,{dataset,value:"1.5",checked:true,closest:()=>card,handlers:{},addEventListener(name,fn){this.handlers[name]=fn}});
let focused;const openMaterialFocus=(...args)=>focused=args,updateMaterialSelectionBar=()=>{};
const document={querySelector:()=>null,querySelectorAll:selector=>selector==="video[data-material-video]"?[first,second]:controls.has(selector)?[controls.get(selector)]:[]};
(async()=>{
  bindMaterialInteractions(data);
  controls.get("[data-video-seek]").handlers.click();assert.equal(second.currentTime,4);assert.equal(first.currentTime,0);
  controls.get("[data-video-speed]").handlers.change();assert.equal(second.playbackRate,1.5);assert.equal(first.playbackRate,1);
  controls.get("[data-video-loop]").handlers.change();assert.equal(second.loop,true);assert.equal(first.loop,false);
  await controls.get("[data-video-fullscreen]").handlers.click();assert.equal(second.fullscreen,true);assert.equal(first.fullscreen,undefined);
  controls.get("[data-material-focus]").handlers.click();assert.equal(focused[3],materialSelectionKey(data.key_events[1]));
  assert.equal(materialVideo("E"),null,"ambiguous unscoped lookup must not pick the first video");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_focus_navigation_preserves_identity_and_pauses_hidden_media():
    run_javascript(("openMaterialFocus", "materialFocusMarkup", "materialSelectionKey", "pauseMaterialVideos", "videoPosterUrl", "videoPreview"), r'''
const location={href:"http://test/",origin:"http://test"},bindVideoPreviews=()=>{};
const state={},events=[1,2].map(i=>({event_id:"E",event_uid:`u${i}`,parent_event_id:`G${i}`,
 action_type:"move",start_us:i*1000000,end_us:(i+1)*1000000,aligned_clip_url:`/clip${i}.mp4`}));
const data={name:"A",key_events:events},filteredMaterialEvents=()=>events,notices=[];
const toast=(...args)=>notices.push(args),number=String,esc=String,timecode=String,icon=()=>"";
const ACTION_LABELS={move:"move"},productExperimentName=String,nextStepLabel=()=>"unknown",productEvidenceText=(value,fallback)=>value||fallback;
const background={paused:false,pause(){this.paused=true}},main={querySelectorAll:()=>[background]};
let dialog;function control(dataset={}){return {dataset,handlers:{},addEventListener(name,fn){this.handlers[name]=fn},click(){this.handlers.click()}}}
const host={querySelectorAll:()=>dialog?[dialog.video]:[],set innerHTML(value){
  this.html=value;dialog=control();dialog.video={paused:false,pause(){this.paused=true}};
  dialog.closeButton=control();dialog.nav=[control({focusNav:"previous"}),control({focusNav:"next"})];
  dialog.showModal=()=>{};dialog.close=()=>dialog.handlers.close();
  dialog.querySelector=selector=>selector==="[data-close-material-focus]"?dialog.closeButton:selector==="[data-focus-video]"?dialog.video:null;
  dialog.querySelectorAll=selector=>selector==="[data-focus-nav]"?dialog.nav:selector==="video"?[dialog.video]:[];
}};
const document={querySelector:selector=>selector==="#material-focus-host"?host:selector==="#material-focus-dialog"?dialog:null};
openMaterialFocus(data,"E",null,materialSelectionKey(events[1]));
assert.ok(host.html.includes('聚焦查看 · 2 / 2'));assert.ok(host.html.includes('/clip2.mp4'));
assert.equal(background.paused,true,"entering focus stops playback behind the dialog");
const second=dialog;second.nav[0].click();
assert.ok(host.html.includes('聚焦查看 · 1 / 2'));assert.ok(host.html.includes('/clip1.mp4'));
assert.equal(second.video.paused,true,"switching material stops the replaced video");
dialog.nav[1].click();assert.ok(host.html.includes('/clip2.mp4'));
const last=dialog;last.closeButton.click();assert.equal(last.video.paused,true);assert.equal(state.focusedMaterialId,null);
openMaterialFocus(data,"E",null,materialSelectionKey(events[0]));dialog.handlers.close();
assert.equal(dialog.video.paused,true,"native Escape close follows the same cleanup");
const previous=host.html;openMaterialFocus(data,"E");assert.equal(host.html,previous);assert.equal(notices.at(-1)[1],"error");
''')


def test_collapsing_either_material_details_layer_pauses_its_video():
    run_javascript(("bindMaterialInteractions", "pauseMaterialVideos"), r'''
function detail(){return {open:false,video:{paused:false,pause(){this.paused=true}},addEventListener(name,fn){this.toggle=fn},querySelectorAll(){return [this.video]}};}
const outer=detail(),inner=detail();
const document={querySelector:()=>null,querySelectorAll:selector=>selector===".material-product-details"?[outer]:selector===".material-clip-details"?[inner]:[]};
const updateMaterialSelectionBar=()=>{};
bindMaterialInteractions({key_events:[]});outer.toggle();inner.toggle();
assert.equal(outer.video.paused,true);assert.equal(inner.video.paused,true);
outer.video.paused=false;outer.open=true;outer.toggle();assert.equal(outer.video.paused,false);
''')


def test_material_shortcuts_operate_on_the_open_focus_video_and_respect_other_dialogs():
    run_javascript(("handleMaterialReviewShortcut",), r'''
const routeParts=()=>["archive","A","materials"];
function video(){return {currentTime:5,duration:20,paused:true,play(){this.paused=false;return Promise.resolve()},pause(){this.paused=true}}}
const background=video(),focused=video();let navigation=0,pageNavigation=0,otherDialog=false;
const next={disabled:false,click(){navigation++}};
let dialog={matches:()=>!otherDialog,querySelector:selector=>selector==="[data-focus-video]"?focused:next};
const document={activeElement:{tagName:"VIDEO"},querySelector:selector=>selector==="dialog[open]"?dialog:selector.startsWith(".material-product-details[open]")?null:background};
const activateAdjacentMaterial=()=>pageNavigation++;
const key=value=>({key:value,preventDefault(){this.defaultPrevented=true}});
handleMaterialReviewShortcut(key("ArrowRight"));assert.equal(focused.currentTime,10);assert.equal(background.currentTime,5);
handleMaterialReviewShortcut(key(" "));assert.equal(focused.paused,false);assert.equal(background.paused,true);
handleMaterialReviewShortcut(key("n"));assert.equal(navigation,1);assert.equal(pageNavigation,0);
next.disabled=true;handleMaterialReviewShortcut(key("n"));assert.equal(navigation,1);
otherDialog=true;handleMaterialReviewShortcut(key("ArrowRight"));assert.equal(focused.currentTime,10);
otherDialog=false;document.activeElement.isContentEditable=true;
handleMaterialReviewShortcut(key("ArrowRight"));assert.equal(focused.currentTime,10);
document.activeElement={tagName:"INPUT"};handleMaterialReviewShortcut(key("ArrowRight"));assert.equal(focused.currentTime,10);
document.activeElement={tagName:"VIDEO"};dialog=null;
handleMaterialReviewShortcut(key("ArrowRight"));assert.equal(background.currentTime,5,"collapsed parent must not expose a hidden player to shortcuts");
''')


def test_adjacent_material_navigation_uses_the_source_card_not_a_repeated_id():
    run_javascript(("activateAdjacentMaterial",), r'''
let scrolled;const cards=[0,1,2].map(index=>({dataset:{materialCard:index<2?"E":"F"},
  querySelector:()=>({focus(){}}),scrollIntoView(){scrolled=index}}));
const document={querySelectorAll:selector=>selector==="[data-material-card]"?cards:[]};
const setTimeout=fn=>fn(),toast=()=>{};
activateAdjacentMaterial("E","next",cards[1]);assert.equal(scrolled,2);
scrolled=null;activateAdjacentMaterial("E","next");assert.equal(scrolled,null,"ambiguous fallback must not navigate from the wrong card");
''')


@pytest.mark.parametrize("entry", ["retry", "archive", "benchmark", "staging"])
def test_acknowledged_run_remains_queued_without_secondary_directory_reads(entry):
    run_javascript(("submitRecoveryAction", "beginStageFollow", "rerunArchive", "rerunBenchmark", "showAcceptedRun",
                    "rememberRunSnapshot"), r'''
const state={runs:[{run_id:"old",state:"failed",experiment_id:"A",source_collection_id:"C",error:"old failure"}],archiveCache:new Map(),runPollRequestId:10};
const notices=[],requests=[];const button={disabled:false,innerHTML:"retry",dataset:{}};
const document={querySelector:()=>button},location={hash:"#/stage/old/experiments"};
const icon=()=>"",toast=(...args)=>notices.push(args),updateServiceChrome=()=>{};
let renders=0;const renderTasks=()=>renders++;
const loadAll=async()=>{throw Error("directory unavailable")};
let accept;const api=(url,options)=>{requests.push({url,options});return new Promise(resolve=>accept=resolve)};
''' + f'const entry={entry!r};\n' + r'''
(async()=>{
  const invoke=()=>["retry","staging"].includes(entry)?submitRecoveryAction("old","retry",{revision:"checked"},button):entry==="archive"?rerunArchive({name:"A"},button):rerunBenchmark();
  const pending=invoke();await invoke();assert.equal(requests.length,1,"duplicate click must not issue another POST");
  const id=["retry","staging"].includes(entry)?"old":"new";
  accept({run_id:id,state:"queued",archive_name:"A"});await pending;
  assert.equal(button.disabled,true,"accepted submission must not be presented as retryable failure");
  assert.equal(location.hash,"#/tasks");assert.equal(renders,1);
  assert.equal(state.activeRun.run_id,id);assert.equal(state.activeRun.state,"queued");
  assert.equal(state.activeRun.experiment_id,"A");assert.equal(state.activeRun.error,undefined);
  assert.equal(state.runs.filter(x=>x.run_id===id).length,1);assert.equal(state.archiveRefreshPending,true);
  assert.equal(state.runPollRequestId,11,"accepted rerun invalidates the previous attempt's pending poll");
  assert.ok(notices.every(([,tone])=>tone!=="error"));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_rejected_run_submission_can_be_retried_without_changing_existing_task():
    run_javascript(("retryRetainedRun",), r'''
const state={activeRun:{run_id:"R",state:"failed"}},button={disabled:false},notices=[];
const api=async()=>{throw Error("submission rejected")},toast=(...args)=>notices.push(args);
(async()=>{
  await retryRetainedRun("R",button);
  assert.equal(button.disabled,false);assert.equal(state.activeRun.state,"failed");
  assert.deepEqual(notices,[["submission rejected","error"]]);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("terminal", ["completed", "failed", "interrupted"])
def test_run_polling_stops_at_terminal_receipt_and_directory_failure_cannot_reverse_it(terminal):
    run_javascript(("pollRun", "rememberRunSnapshot"), r'''
const state={activeRun:{run_id:"R",state:"queued"},runs:[]};
const location={hash:"#/new"},routeParts=()=>[location.hash.slice(2)];
const stages=[],progress=[],notices=[],invalidated=[];
const setTimeout=fn=>fn(),delay=async()=>{},setPhase=()=>{},updateServiceChrome=()=>{};
const renderStages=(...args)=>stages.push(args),setProgress=(...args)=>progress.push(args),toast=(...args)=>notices.push(args);
const friendlyFailureReason=()=>"stopped",invalidateArchiveCache=name=>invalidated.push(name);
const document={querySelector:()=>({removeAttribute(){}})};
const loadAll=async()=>{throw Error("directory unavailable")};
let reads=0;
''' + f'const terminal={terminal!r};\n' + r'''
const api=async()=>{reads++;if(reads>1)throw Error("must stop polling terminal task");return {run_id:"R",state:terminal,progress:1}};
(async()=>{
  await pollRun("R","Archive A");
  assert.equal(reads,1);assert.equal(state.activeRun.state,terminal);assert.equal(state.runs[0].state,terminal);
  if(terminal==="completed"){
    assert.equal(state.archiveRefreshPending,true);assert.deepEqual(invalidated,["Archive A"]);
    assert.equal(location.hash,"#/archive/Archive%20A/experiments");
    assert.ok(notices.every(([,tone])=>tone!=="error"));
  }else assert.equal(location.hash,"#/new");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("same_id", [False, True])
def test_old_run_poll_cannot_overwrite_a_new_active_run(fails, same_id):
    run_javascript(("pollRun",), r'''
const state={activeRun:{run_id:"old",state:"queued"}};
const setTimeout=fn=>fn();let finish,reject,started;
const ready=new Promise(resolve=>started=resolve);
const api=()=>{started();return new Promise((resolve,fail)=>{finish=resolve;reject=fail})};
''' + f'const fails={str(fails).lower()},sameId={str(same_id).lower()};\n' + r'''
(async()=>{
  const pending=pollRun("old","A");await ready;
  const current={run_id:sameId?"old":"new",state:"queued"};state.activeRun=current;
  if(sameId)state.runPollRequestId++;
  if(fails)reject(Error("old request failed"));else finish({run_id:"old",state:"completed"});
  await pending;assert.deepEqual(state.activeRun,current);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_run_read_failure_keeps_snapshot_and_recovers_without_claiming_it_is_running():
    run_javascript(("pollRun", "rememberRunSnapshot"), r'''
const state={activeRun:{run_id:"R",state:"running",progress:.5},runs:[]},messages=[];
const setTimeout=fn=>fn(),delay=async()=>{},setPhase=()=>{},renderStages=()=>{},toast=()=>{},updateServiceChrome=()=>{};
const routeParts=()=>["tasks"],renderTasks=()=>{},document={querySelector:()=>null};
const friendlyFailureReason=()=>"stopped",setProgress=(value,message)=>messages.push(message);
let reads=0;
const api=async()=>{if(++reads===1)throw Error("offline");return {run_id:"R",state:"interrupted",progress:.5}};
(async()=>{
  await pollRun("R","A");assert.equal(reads,2);assert.equal(state.activeRun.state,"interrupted");
  assert.ok(messages[0].includes("已保留上次进度"));assert.ok(!messages[0].includes("仍在后台运行"));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_interrupted_run_is_not_counted_as_active_in_global_chrome():
    run_javascript(("updateServiceChrome", "setPhase"), r'''
const state={health:{status:"ok"},runs:[{run_id:"R",state:"interrupted"}],activeRun:{run_id:"R",state:"interrupted"}};
const STAGE_LABELS={interrupted:"服务重启后待续跑"};
const nodes=new Map(),document={querySelectorAll:()=>[],querySelector:selector=>{
  if(!nodes.has(selector))nodes.set(selector,{classList:{toggle(){}},querySelector:()=>document.querySelector(selector+" span")});
  return nodes.get(selector);
}};
updateServiceChrome();
assert.equal(nodes.get("#running-badge").hidden,true);assert.equal(nodes.get("#running-badge").textContent,0);
assert.equal(nodes.get("#phase-chip").className,"phase-chip failed");
''')


def test_interrupted_run_does_not_block_a_valid_new_submission():
    run_javascript(("updateReview",), r'''
const state={health:{analysis_ready:true},activeRun:{run_id:"R",state:"interrupted"}};
const nodes=new Map(),document={querySelector:selector=>{if(!nodes.has(selector))nodes.set(selector,{});return nodes.get(selector)}};
const isNasMode=()=>false,reviewState=()=>({ready:true}),number=String;
updateReview();assert.equal(nodes.get("#start-run").disabled,false);
state.activeRun.state="queued";updateReview();assert.equal(nodes.get("#start-run").disabled,true);
state.activeRun.state="interrupted";state.health.analysis_ready=false;updateReview();
assert.equal(nodes.get("#start-run").disabled,true,"runtime readiness still gates new submissions");
''')


@pytest.mark.parametrize("fails", [False, True])
def test_task_list_read_started_before_acceptance_cannot_replace_acknowledged_run(fails):
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview",), r'''
const state={runs:[{run_id:"R",state:"failed"}],runPollRequestId:1},document={hidden:false};
const experimentRecords=()=>[],libraryRecordKey=()=>"";
let finish,reject;const api=()=>new Promise((resolve,fail)=>{finish=resolve;reject=fail});
''' + f'const fails={str(fails).lower()};\n' + r'''
(async()=>{
  const reading=refreshTaskSnapshots();
  state.runPollRequestId++;state.runs=[{run_id:"R",state:"queued"}];
  if(fails)reject(Error("old list failed"));else finish({runs:[{run_id:"R",state:"failed"}]});
  await reading;assert.equal(state.runs[0].state,"queued");assert.equal(state.refreshingTasks,false);
  assert.notEqual(state.taskSyncError,true,"old failure must not mark the acknowledged task disconnected");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("old_fails", [False, True])
def test_nas_read_from_an_older_full_refresh_cannot_replace_newer_results(old_fails):
    run_javascript(("loadAll", "loadNasRecordings",), r'''
const state={},document={activeElement:null},routeParts=()=>["tasks"];
const archiveSearchQuery=()=>"",loadArchiveListing=async()=>true,updateServiceChrome=()=>{};
const pending=[],api=async url=>url==="/api/nas-recordings"?new Promise((resolve,reject)=>pending.push({resolve,reject})):{};
''' + f'const oldFails={str(old_fails).lower()};\n' + r'''
(async()=>{
  await loadAll();await loadAll();assert.equal(pending.length,2);
  pending[1].resolve({recordings:[{recording_id:"NEW"}],batches:[{batch_id:"NEW"}],monitor:{observed_at:"new"}});
  await new Promise(setImmediate);
  if(oldFails)pending[0].reject(Error("old failure"));else pending[0].resolve({recordings:[{recording_id:"OLD"}]});
  await new Promise(setImmediate);
  assert.equal(state.nasRecordings[0]?.recording_id,"NEW","an obsolete read must not replace the latest recording list");
  assert.equal(state.nasBatches[0]?.batch_id,"NEW");assert.equal(state.nasMonitor.observed_at,"new");
  assert.equal(state.nasError,"");assert.equal(state.nasLoading,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_fast_nas_failure_is_handled_before_other_initial_reads_finish():
    run_javascript(("loadAll", "loadNasRecordings",), r'''
const state={},document={activeElement:null},routeParts=()=>["tasks"];
const archiveSearchQuery=()=>"",loadArchiveListing=async()=>true,updateServiceChrome=()=>{};
let finishHealth;const unhandled=[];process.on("unhandledRejection",error=>unhandled.push(error));
const api=async url=>{
  if(url==="/api/nas-recordings")throw Error("recording list unavailable");
  if(url==="/api/health")return new Promise(resolve=>finishHealth=resolve);
  return {};
};
(async()=>{
  const initial=loadAll();await new Promise(setImmediate);
  const earlyErrors=[...unhandled];finishHealth({status:"ok"});await initial;await new Promise(setImmediate);
  assert.deepEqual(earlyErrors,[],"independent NAS failure needs an immediate rejection handler");
  assert.equal(state.nasLoading,false);assert.ok(state.nasError);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_failed_nas_refresh_retains_recordings_batches_and_user_selection():
    run_javascript(("loadAll", "loadNasRecordings",), r'''
const state={nasRecordings:[{recording_id:"A"}],nasBatches:[{batch_id:"B"}],nasSelection:{A:"first_person"},nasMonitor:{status:"watching"}};
const document={activeElement:null},routeParts=()=>["tasks"];
const archiveSearchQuery=()=>"",loadArchiveListing=async()=>true,updateServiceChrome=()=>{};
const api=async url=>{if(url==="/api/nas-recordings")throw Error("temporary outage");return {};};
(async()=>{
  await loadAll();await new Promise(setImmediate);
  assert.equal(state.nasRecordings[0]?.recording_id,"A");assert.equal(state.nasBatches[0]?.batch_id,"B");
  assert.deepEqual(state.nasSelection,{A:"first_person"});assert.equal(state.nasSyncError,true);
  assert.equal(state.nasLoading,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("old_fails", [False, True])
def test_obsolete_nas_read_cannot_end_a_newer_loading_state(old_fails):
    run_javascript(("loadNasRecordings",), r'''
const state={},pending=[];
const api=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
''' + f'const oldFails={str(old_fails).lower()};\n' + r'''
(async()=>{
  const old=loadNasRecordings(),current=loadNasRecordings();
  if(oldFails)pending[0].reject(Error("old outage"));else pending[0].resolve({});
  assert.equal(await old,null);assert.equal(state.nasLoading,true);
  assert.notEqual(state.nasSyncError,true);
  pending[1].resolve({recordings:[],batches:[]});await current;
  assert.equal(state.nasLoading,false);assert.equal(state.nasSyncError,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_late_initial_nas_read_updates_only_pickers_without_replacing_the_new_experiment_form():
    run_javascript(("loadAll", "loadNasRecordings"), r'''
const state={},document={activeElement:null},routeParts=()=>["new"];
const archiveSearchQuery=()=>"",loadArchiveListing=async()=>true,updateServiceChrome=()=>{};
let finish,fullRenders=0,pickerRenders=0;
const renderNew=()=>fullRenders++,refreshNasPickers=()=>pickerRenders++;
const api=async url=>url==="/api/nas-recordings"?new Promise(resolve=>finish=resolve):{};
(async()=>{
  await loadAll();finish({batches:[{batch_id:"A"}]});await new Promise(setImmediate);
  assert.equal(fullRenders,0,"late recording data must not reconstruct a form the user already started filling");
  assert.equal(pickerRenders,1);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_typing_outside_nas_pickers_does_not_stop_background_catalog_reads():
    run_nas_poll(r'''
(async()=>{
  active={tagName:"INPUT",outsidePicker:true,value:"UNSAVED TITLE"};
  const poll=refreshNasRecordings();assert.equal(pending.length,1,"editing the title must not suspend independent batch discovery");
  pending[0].resolve({batches:[{batch_id:"NEW"}]});await poll;
  assert.equal(rendered.length,2);assert.equal(active.value,"UNSAVED TITLE");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def run_nas_poll(script):
    run_javascript(("loadNasRecordings", "refreshNasRecordings", "refreshNasPickers", "flushNasPickersAfterFocus"), r'''
const state={health:{collection_ingest:{mode:"directory_metadata"}},nasRecordings:[{recording_id:"A"}],
 nasBatches:[{batch_id:"B"}],nasSelection:{A:"first_person"}};
let refreshingNasRecordings=false,route="new",active=null;
const routeParts=()=>[route],rendered=[];
const document={hidden:false,get activeElement(){return active;},querySelector:selector=>({contains:element=>element!=null&&!element.outsidePicker,set outerHTML(value){rendered.push({selector,value});}})};
const nasBatchPicker=()=>state.nasSyncError?"BATCH SYNC ERROR":"BATCH READY";
const nasRecordingPicker=()=>state.nasSyncError?"RECORDINGS SYNC ERROR":"RECORDINGS READY";
const bindNasBatchPicker=()=>{},bindNasRecordingPicker=()=>{},pending=[];
const api=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
''' + script)


@pytest.mark.parametrize("next_focus", ["outside", "inside", "navigation"])
def test_nas_picker_focus_exit_applies_deferred_data_without_another_read(next_focus):
    run_nas_poll(f'const nextFocus={next_focus!r};\n' + r'''
const scheduled=[],setTimeout=callback=>scheduled.push(callback);
(async()=>{
  active={tagName:"SELECT"};
  const poll=refreshNasRecordings();assert.equal(pending.length,1,"picker interaction must not stop discovery");
  pending[0].resolve({batches:[{batch_id:"NEW"}]});await poll;
  assert.deepEqual(rendered,[]);assert.equal(state.nasRenderPending,true);
  flushNasPickersAfterFocus({target:{closest:()=>({})}});assert.equal(scheduled.length,1);
  if(nextFocus==="outside")active={tagName:"INPUT",outsidePicker:true};
  else if(nextFocus==="navigation")route="tasks";
  scheduled.shift()();
  assert.equal(rendered.length,nextFocus==="outside"?2:0);
  assert.equal(pending.length,1,"focus exit must use the result already received");
  assert.equal(state.nasRenderPending,nextFocus!=="outside");
  if(nextFocus!=="outside"){
    route="new";active=null;
    flushNasPickersAfterFocus({target:{closest:()=>({})}});scheduled.shift()();
    assert.equal(rendered.length,2);
  }
  refreshNasPickers();assert.equal(rendered.length,2,"an applied update must not render twice");
  assert.equal(pending.length,1);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_deferred_nas_picker_waits_for_a_newer_inflight_request():
    run_nas_poll(r'''
const scheduled=[],setTimeout=callback=>scheduled.push(callback);
(async()=>{
  active={tagName:"SELECT"};
  const first=refreshNasRecordings();pending[0].resolve({batches:[{batch_id:"OLD"}]});await first;
  const second=refreshNasRecordings();assert.equal(state.nasLoading,true);
  active=null;flushNasPickersAfterFocus({target:{closest:()=>({})}});scheduled.shift()();
  assert.deepEqual(rendered,[],"focus exit must not publish data superseded by an in-flight refresh");
  pending[1].resolve({batches:[{batch_id:"NEW"}]});await second;
  assert.equal(state.nasBatches[0].batch_id,"NEW");assert.equal(rendered.length,2);
  assert.equal(state.nasRenderPending,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("focused_picker", ["#nas-batches", "#nas-recordings"])
def test_nas_picker_updates_are_scoped_to_the_actual_focused_container(focused_picker):
    run_javascript(("refreshNasPickers",), r'''
const state={nasRenderPending:true},routeParts=()=>["new"],rendered=[];
const nasBatchPicker=()=>"BATCHES",nasRecordingPicker=()=>"RECORDINGS",bindNasBatchPicker=()=>{},bindNasRecordingPicker=()=>{};
''' + f'const focusedPicker={focused_picker!r};\n' + r'''
const input={value:"UNSAVED SEARCH"};
const document={activeElement:input,querySelector:selector=>({contains:element=>element===input&&selector===focusedPicker,
 set outerHTML(value){rendered.push(selector);}})};
refreshNasPickers();assert.equal(rendered.length,1);assert.notEqual(rendered[0],focusedPicker);
assert.equal(document.activeElement,input);assert.equal(input.value,"UNSAVED SEARCH");
assert.equal(state.nasRenderPending,true);
''')


@pytest.mark.parametrize("old_fails", [False, True])
def test_manual_nas_read_supersedes_poll_without_stale_picker_updates(old_fails):
    run_nas_poll(f'const oldFails={str(old_fails).lower()};\n' + r'''
(async()=>{
  const poll=refreshNasRecordings();await refreshNasRecordings();assert.equal(pending.length,1);
  const manual=loadNasRecordings();
  pending[1].resolve({recordings:[{recording_id:"NEW"}],batches:[{batch_id:"NEW"}]});await manual;
  if(oldFails)pending[0].reject(Error("old poll error"));else pending[0].resolve({recordings:[],batches:[]});
  await poll;
  assert.equal(state.nasRecordings[0].recording_id,"NEW");assert.equal(state.nasBatches[0].batch_id,"NEW");
  assert.deepEqual(rendered,[],"an obsolete poll must not replace the current picker");
  assert.equal(refreshingNasRecordings,false);assert.equal(state.nasLoading,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_nas_poll_failure_and_recovery_refresh_visible_status_without_erasing_selection():
    run_nas_poll(r'''
(async()=>{
  const failed=refreshNasRecordings();pending[0].reject(Error("temporary disconnect"));await failed;
  assert.equal(state.nasRecordings[0].recording_id,"A");assert.equal(state.nasBatches[0].batch_id,"B");
  assert.deepEqual(state.nasSelection,{A:"first_person"});assert.equal(rendered.length,2);
  assert.ok(rendered.every(item=>item.value.includes("SYNC ERROR")));
  const recovered=refreshNasRecordings();pending[1].resolve({recordings:[{recording_id:"A"},{recording_id:"C"}],batches:[{batch_id:"B"}]});await recovered;
  assert.equal(state.nasRecordings.length,2);assert.equal(state.nasSyncError,false);assert.equal(state.nasError,"");
  assert.deepEqual(state.nasSelection,{A:"first_person"});assert.equal(rendered.length,4);
  assert.ok(rendered.slice(2).every(item=>item.value.includes("READY")));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("change", ["navigation", "editing"])
def test_nas_poll_rechecks_page_and_editing_before_replacing_picker(change):
    run_nas_poll(f'const change={change!r};\n' + r'''
(async()=>{
  const poll=refreshNasRecordings();
  if(change==="navigation")route="tasks";else active={tagName:"INPUT"};
  pending[0].resolve({recordings:[{recording_id:"C"}]});await poll;
  assert.equal(state.nasRecordings[0].recording_id,"C");assert.deepEqual(rendered,[]);
  assert.equal(refreshingNasRecordings,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("has_cached_results", [False, True])
def test_nas_picker_failure_is_not_presented_as_empty_or_healthy_monitoring(has_cached_results):
    run_javascript(("nasBatchPicker", "nasRecordingPicker"), r'''
const state={nasSyncError:true,nasError:"connection unavailable",nasMonitor:{status:"watching"},nasRecordings:[],nasBatches:[],nasSelection:{},nasQuery:""};
const esc=String,icon=()=>"",number=Number,formatDate=String,duration=String,formatBytes=String;
''' + f'const cached={str(has_cached_results).lower()};\n' + r'''
if(cached){state.nasRecordings=[{recording_id:"A",relative_path:"SYNTHETIC.mp4",issues:[]}];state.nasBatches=[{batch_id:"B"}];}
const batches=nasBatchPicker(),recordings=nasRecordingPicker();
assert.ok(batches.includes("采集素材列表同步中断"));assert.ok(!batches.includes("持续监控中"));
assert.ok(!batches.includes("等待相机完成采集"));assert.ok(!recordings.includes("还没有采集素材"));
if(cached){assert.ok(recordings.includes("SYNTHETIC.mp4"));assert.ok(batches.includes('data-run-nas-batch="B"'));}
else {assert.ok(batches.includes("暂时无法读取采集批次"));assert.ok(recordings.includes("暂时无法读取采集素材"));}
state.nasSyncError=false;state.nasError="";
assert.ok(nasBatchPicker().includes("持续监控中"));
''')


def test_full_refresh_does_not_replace_a_newer_terminal_receipt():
    run_javascript(("loadAll", "loadNasRecordings",), r'''
const state={runPollRequestId:1,runs:[{run_id:"R",state:"running"}]},document={activeElement:null};
const archiveSearchQuery=()=>"",loadArchiveListing=async()=>true,updateServiceChrome=()=>{},routeParts=()=>["tasks"];
let finish;const api=async url=>url==="/api/runs"?new Promise(resolve=>finish=resolve):{};
(async()=>{
  const reading=loadAll();state.runPollRequestId++;state.runs=[{run_id:"R",state:"completed"}];
  finish({runs:[{run_id:"R",state:"running"}]});await reading;
  assert.equal(state.runs[0].state,"completed");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_material_selection_survives_filters_and_distinguishes_repeated_event_ids():
    run_javascript(("bindMaterialInteractions", "selectedMaterialEvents", "materialSelectionKey",
                    "materialSelectionScope", "ensureMaterialSelection"), r'''
const state={selectedMaterials:new Map()}, notices=[];
const toast=(...args)=>notices.push(args),updateMaterialSelectionBar=()=>{};
const filteredMaterialEvents=data=>data.key_events;
const navigator={clipboard:{writeText:async()=>{throw Error("denied")}}};
function control(dataset={}){return {dataset,handlers:{},addEventListener(name,fn){this.handlers[name]=fn}};}
let checkbox,selectVisible,bar;
const document={
  querySelector:selector=>selector==="#material-selection-bar"?bar:selector==="[data-select-visible]"?selectVisible:null,
  querySelectorAll:selector=>selector==="[data-material-select]"?[checkbox]:[],
};
function page(event){
  const data={name:"A",release_id:"r1",key_events:[event]};
  ensureMaterialSelection(data);
  checkbox=control({materialSelect:event.event_id});
  checkbox.closest=()=>({dataset:{materialSelectionKey:materialSelectionKey(event)}});
  selectVisible=control();bar=control();bar.buttons=new Map();
  bar.querySelector=selector=>{if(!bar.buttons.has(selector))bar.buttons.set(selector,control());return bar.buttons.get(selector)};
  bindMaterialInteractions(data);return data;
}
(async()=>{
  const first={event_id:"E",event_uid:"uid1",parent_event_id:"G1"};
  const second={event_id:"E",event_uid:"uid2",parent_event_id:"G2"};
  const a=page(first);checkbox.checked=true;checkbox.handlers.change();
  const b=page(second);checkbox.checked=true;checkbox.handlers.change();
  assert.deepEqual(selectedMaterialEvents(b).map(x=>x.event_uid),["uid1","uid2"]);
  assert.equal(selectedMaterialEvents({...b,key_events:[]}).length,2,"empty filter must not lose selections");
  assert.equal(state.selectedMaterials.size,2,"repeated event IDs in different groups are distinct materials");
  selectVisible.handlers.click();selectVisible.handlers.click();assert.equal(state.selectedMaterials.size,2);
  await bar.buttons.get("[data-copy-selected]").handlers.click();
  assert.equal(notices.at(-1)[1],"error");assert.equal(state.selectedMaterials.size,2);
  page(first);checkbox.checked=false;checkbox.handlers.change();
  assert.deepEqual(selectedMaterialEvents(a).map(x=>x.event_uid),["uid2"]);
  bar.buttons.get("[data-clear-selected]").handlers.click();assert.equal(state.selectedMaterials.size,0);
  assert.notEqual(materialSelectionKey({event_id:"E",parent_event_id:"G1"}),
                  materialSelectionKey({event_id:"E",parent_event_id:"G2"}),"legacy events retain group identity");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_material_selection_is_scoped_to_archive_run_and_release():
    run_javascript(("ensureMaterialFilters", "ensureMaterialSelection", "materialSelectionScope",
                    "selectedMaterialEvents"), r'''
const state={materialFilters:{},selectedMaterials:new Map()},notices=[];
const toast=message=>notices.push(message);
const data={name:"A",release_id:"r1"};
ensureMaterialSelection(data);state.selectedMaterials.set("E",{event_id:"E"});
ensureMaterialFilters({name:"A"});ensureMaterialSelection({...data,key_events:[]});
assert.equal(state.selectedMaterials.size,1,"filter setup without a summary must not erase selection");
assert.deepEqual(selectedMaterialEvents({...data,release_id:"r2"}),[],"never export another release's selection");
ensureMaterialSelection({...data,release_id:"r2"});assert.equal(state.selectedMaterials.size,0);
assert.equal(notices.length,1,"tell the user why a populated selection was cleared");
for(const scope of [{name:"B",release_id:"r2"},{name:"B",staging_run_id:"run1"},{name:"B",staging_run_id:"run2"}]){
  state.selectedMaterials.set("E",{event_id:"E"});ensureMaterialSelection(scope);
  assert.equal(state.selectedMaterials.size,0);
}
''')


@pytest.mark.parametrize(("loaded", "total"), [(0, 0), (24, 40), (40, 40)])
def test_material_counts_distinguish_loaded_pages_from_matching_total(loaded, total):
    run_javascript(("workflowLabel", "workflowCompletionLabel", "materialsView", "materialResults"), r'''
const state={materialFilters:{group:"all",action:"all",object:"all",support:"all",query:""}};
const ensureMaterialSelection=()=>{},ensureMaterialFilters=()=>{},movementScreeningView=()=>"",retainedMaterialsView=()=>"";
const eventHasAlignedDualViewMaterial=event=>event.ready,filteredMaterialEvents=data=>data.key_events.filter(eventHasAlignedDualViewMaterial);
const eventObjectValues=()=>[],ACTION_LABELS={move:"移动",liquid:"液体移动"};
const number=String,esc=String,icon=()=>"",timecode=String,materialCard=()=>"<article>card</article>";
const materialSelectionBar=()=>"",productState=(_,__,title)=>title;
''' + f'const loaded={loaded},total={total};\n' + r'''
const data={name:"A",counts:{key_events:99},material_total_count:total,material_next_cursor:loaded<total?"next":null,
 experiment_groups:[{group_id:"G1",name:"Group",key_event_count:99}],
 key_events:[...Array.from({length:loaded},()=>({ready:true,action_type:"move",experiment_group:{group_id:"G1"}})),{ready:false}]};
const html=materialsView(data);
assert.ok(html.includes(`符合筛选 ${total} 份`),"the header must use the filtered server total, including explicit zero");
assert.ok(html.includes('>全部实验</option>'),"the all-groups option must not claim that the loaded page is the archive total");
assert.ok(html.includes('>液体移动</option>'),"an unloaded category has no measured count; do not label it zero");
assert.ok(!html.includes('Group（99）'),"unfiltered group totals must not masquerade as matching counts");
if(loaded){
 assert.ok(html.includes(`已加载 ${loaded} / ${total} 份关键素材`));
 assert.ok(html.includes(`已加载 ${loaded} 份</span>`),"each group badge describes visible materials");
}else assert.ok(html.includes("没有符合条件的关键素材"));
''')


@pytest.mark.parametrize("filter_kind", ["group", "action", "object", "support", "query"])
def test_material_counts_use_filtered_ready_items_for_unpaginated_results(filter_kind):
    run_javascript(("workflowLabel", "workflowCompletionLabel", "materialsView", "materialResults", "filteredMaterialEvents"), r'''
const state={materialFilters:{group:"all",action:"all",object:"all",support:"all",query:""}};
const ensureMaterialSelection=()=>{},ensureMaterialFilters=()=>{},movementScreeningView=()=>"",retainedMaterialsView=()=>"";
const eventHasAlignedDualViewMaterial=event=>event.ready,eventHasDualViewSupport=event=>event.dual;
const eventObjectValues=event=>event.objects.target,ACTION_LABELS={move:"移动",liquid:"液体移动"};
const number=String,esc=String,icon=()=>"",timecode=String,materialCard=()=>"<article>card</article>";
const productObjectLabel=String,materialSelectionBar=()=>"",productState=(_,__,title)=>title;
const matching={event_id:"MATCH",ready:true,action_type:"move",objects:{target:["tube"]},dual:true,experiment_group:{group_id:"G1"}};
const other={event_id:"OTHER",ready:true,action_type:"liquid",objects:{target:["bottle"]},dual:false,experiment_group:{group_id:"G2"}};
const data={name:"A",staging_run_id:"run1",counts:{key_events:99},key_events:[matching,matching,other,{...matching,ready:false}],
 experiment_groups:[{group_id:"G1",name:"One",key_event_count:3},{group_id:"G2",name:"Two",key_event_count:1}]};
''' + f'const filterKind={filter_kind!r};\n' + r'''
state.materialFilters[filterKind]={group:"G1",action:"move",object:"tube",support:"dual",query:"MATCH"}[filterKind];
const html=materialsView(data);
assert.ok(html.includes("符合筛选 2 份"));
assert.ok(html.includes("已加载 2 / 2 份关键素材"),"local filters must affect both the numerator and denominator");
assert.equal((html.match(/<article>card<\/article>/g)||[]).length,2,"quarantined and filtered-out events do not count");
''')


def test_empty_material_results_keep_selected_action_and_object_visible():
    run_javascript(("materialsView",), r'''
const state={materialFilters:{group:"all",action:"custom_action",object:"selected_object",support:"all",query:"no matches"}};
const ensureMaterialSelection=()=>{},ensureMaterialFilters=()=>{},movementScreeningView=()=>"",retainedMaterialsView=()=>"";
const eventHasAlignedDualViewMaterial=()=>true,eventObjectValues=()=>[],ACTION_LABELS={move:"移动"};
const number=String,esc=String,productObjectLabel=String,materialResults=()=>"",materialSelectionBar=()=>"";
const html=materialsView({name:"A",key_events:[],material_total_count:0});
assert.ok(html.includes('<option value="custom_action" selected>custom_action</option>'));
assert.ok(html.includes('<option value="selected_object" selected>selected_object</option>'));
assert.equal(state.materialFilters.object,"selected_object");assert.equal(state.materialFilters.action,"custom_action");
''')


def test_empty_server_filter_keeps_selection_and_filter_controls_accessible():
    run_javascript(("materialsView", "ensureMaterialFilters", "ensureMaterialSelection",
                    "materialSelectionScope", "materialSelectionBar"), r'''
const data={name:"A",release_id:"r1",key_events:[],material_total_count:0};
const state={materialFilters:{},materialSelectionSource:materialSelectionScope(data),
  selectedMaterials:new Map([["uid1",{event_id:"E"}]])};
const movementScreeningView=()=>"<section>screening</section>",retainedMaterialsView=()=>"<section>retained</section>";
const eventHasAlignedDualViewMaterial=()=>true,eventObjectValues=()=>[],ACTION_LABELS={};
const number=String,esc=String,icon=()=>"",materialResults=()=>"<p>empty results</p>";
const html=materialsView(data);
assert.ok(html.includes('id="material-filters"'),"empty server results must still allow changing the filter");
assert.ok(html.includes('data-selected-count>1</b>'),"hidden selections must remain exportable");
assert.ok(html.includes("empty results"));assert.equal(state.selectedMaterials.size,1);
''')


def test_material_export_keeps_all_selected_provenance_and_recovers_from_download_errors():
    run_javascript(("exportSelectedMaterials", "selectedMaterialEvents", "materialSelectionScope",
                    "materialFocusRoute", "experimentRecordRoute"), r'''
const data={name:"A",release_id:"r1",key_events:[]};
const events=[1,2].map(i=>({event_id:`E${i}`,event_uid:`uid${i}`,parent_event_id:`G${i}`,
  release_id:"r1",action_type:"move",start_us:1000000,end_us:2000000,
  aligned_frame_url:`/frame${i}?release=r1`,aligned_clip_url:`/clip${i}?release=r1`}));
const state={materialSelectionSource:materialSelectionScope(data),selectedMaterials:new Map(events.map(x=>[x.event_uid,x]))};
const ACTION_LABELS={move:"移动"},timecode=String,eventHasDualViewSupport=()=>false;
let captured,fail="",clicked=0,removed=0;const notices=[],timers=[],revoked=[];
const toast=(...args)=>notices.push(args);
const Blob=class{constructor(parts){captured=JSON.parse(parts[0]);if(fail==="blob")throw Error("failed");}};
URL.createObjectURL=()=>"blob:selection";URL.revokeObjectURL=url=>revoked.push(url);
const setTimeout=(fn,ms)=>timers.push({fn,ms});
const document={body:{appendChild(){}},createElement:()=>({remove(){removed++},click(){if(fail==="click")throw Error("blocked");clicked++}})};
exportSelectedMaterials(data);
assert.equal(captured.selected_count,2);assert.equal(captured.selected_materials.length,2);
assert.equal(captured.evidence_status,"DERIVED_SELECTION_NOT_GROUND_TRUTH");
assert.equal(captured.source_release_id,"r1");assert.equal(clicked,1);assert.equal(removed,1);
for(const [i,item] of captured.selected_materials.entries()){
  assert.equal(item.event_uid,events[i].event_uid);assert.equal(item.parent_event_id,events[i].parent_event_id);
  assert.equal(item.release_id,"r1");assert.equal(item.start_us,1000000);assert.equal(item.action_type,"move");
  const query=new URLSearchParams(item.material_route.split("?")[1]);
  assert.equal(query.get("release"),"r1");assert.equal(query.get("event_uid"),events[i].event_uid);
  assert.equal(item.clip_url,events[i].aligned_clip_url);
}
assert.equal(revoked.length,0,"download URL must outlive the click");
assert.ok(timers[0].ms>0);timers[0].fn();assert.deepEqual(revoked,["blob:selection"]);
for(const failure of ["blob","click"]){
  fail=failure;exportSelectedMaterials(data);
  assert.equal(notices.at(-1)[1],"error");assert.equal(state.selectedMaterials.size,2);
}
fail="";exportSelectedMaterials(data);assert.equal(clicked,2,"retry uses the intact selection");
''')


@pytest.mark.parametrize("page", ["first", "next"])
@pytest.mark.parametrize("summary_release,event_release", [("r1", "r2"), ("r2", "r1"), ("r1", None)])
def test_global_library_rejects_mixed_release_pages_and_waits_for_directory_recovery(page, summary_release, event_release):
    event_release_js = "null" if event_release is None else repr(event_release)
    cursor_js = "null" if page == "first" else '"next"'
    run_javascript(("loadLibraryDetail", "libraryQueryKey", "cachedArchiveDetail",
                    "libraryRequestKey", "libraryRecordKey", "libraryRecordQueryKey", "libraryLoadState",
                    "cachedLibraryDetail", "invalidateArchiveCache", "queueLibraryReleaseRefresh"), r'''
const state={archives:[{name:"A"}],archiveCache:new Map(),materialCache:new Map(),libraryLoadErrors:new Set(),
 globalMaterialFilters:{action:"all",object:"all",support:"all"},search:""};
const archiveProcessStopped=()=>false,experimentRecords=()=>state.archives;
''' + f'const summaryRelease={summary_release!r},eventRelease={event_release_js};\n' + r'''
const old={name:"A",release_id:summaryRelease,key_events:[{event_id:"old",release_id:summaryRelease}],
 daily_report:{report_id:"old-report"},libraryKeys:{materials:libraryQueryKey("materials"),reports:"reports"}};
state.archiveCache.set("archive/A",old);state.archiveCache.set("A:summary",{release_id:summaryRelease});
const unrelated={release_id:"keep"};state.archiveCache.set("archive/B",unrelated);
const loadArchive=async()=>({name:"A",release_id:summaryRelease,counts:{key_events:2}});
const api=async()=>({items:[{event_id:"new",release_id:eventRelease}],total_count:7});
''' + f'const cursor={cursor_js};\n' + r'''
(async()=>{
  await assert.rejects(loadLibraryDetail("A","materials",cursor),/版本/);
  assert.equal(cachedArchiveDetail("A"),null,"do not publish new material under old summary or mix paginated releases");
  assert.equal(state.archiveCache.has("A:summary"),false);
  assert.equal(state.archiveCache.get("archive/B"),unrelated,"unrelated loaded archives remain available");
  assert.equal(state.archiveRefreshPending,true,"release mismatch must trigger automatic directory reconciliation");
  assert.equal(libraryLoadState("materials").failed.length,1,"block immediate repeated detail reads until directory reconciliation");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_library_release_conflicts_recover_automatically_without_a_request_loop_or_new_task():
    run_javascript(("loadLibraryDetail", "libraryQueryKey", "cachedArchiveDetail", "cachedLibraryDetail",
                    "queueLibraryReleaseRefresh", "invalidateArchiveCache", "libraryLoadState", "libraryRequestKey",
                    "libraryRecordKey", "libraryRecordQueryKey", "ensureLibraryDetails", "refreshTaskSnapshots",
                    "loadArchiveListing", "applyArchiveListingPage", "refreshLibraryOverview"), r'''
const state={archives:[{name:"A",release_id:"r1",key_event_count:1},{name:"B",release_id:"r1",key_event_count:1}],
 archiveListingQuery:"",archiveTotals:{key_events:2},runs:[],archiveCache:new Map(),materialCache:new Map(),
 libraryLoadErrors:new Set(),globalMaterialFilters:{action:"all",object:"all",support:"all"},search:""};
let route="materials";
const document={hidden:false},archiveSearchQuery=()=>"",routeParts=()=>[route];
const archiveProcessStopped=()=>false,experimentRecords=()=>state.archives,updateServiceChrome=()=>{},renderMaterialsLibrary=()=>{};
const refreshProgressiveSurface=()=>{},loadLibraryRecord=(record,section)=>loadLibraryDetail(record.name,section);
let recovered=false,detailReads=0,directoryReads=0;
const loadArchive=async name=>({name,release_id:recovered?"r2":"r1",counts:{key_events:recovered?(name==="A"?2:5):1}});
const loadArchiveSection=async()=>({release_id:recovered?"r2":"r1",daily_report:{report_id:"available-report"}});
async function api(url){
  if(url==="/api/runs")return {runs:[]};
  if(url.startsWith("/api/archives?")){
    directoryReads++;if(!recovered)throw Error("directory unavailable");
    return {archives:[{name:"A",release_id:"r2",key_event_count:2},{name:"B",release_id:"r2",key_event_count:5}],totals:{key_events:7}};
  }
  detailReads++;const name=new URL(url,"http://fixture").searchParams.get("archive");
  return {items:Array.from({length:name==="A"?2:5},(_,i)=>({event_id:name+i,release_id:"r2"})),total_count:name==="A"?2:5};
}
(async()=>{
  await ensureLibraryDetails();assert.equal(detailReads,2);
  assert.equal(state.archiveRefreshPending,true);assert.equal(state.libraryReleaseRefreshPending,true);
  assert.equal(state.archiveCache.size,0,"neither conflict nor obsolete sibling response may enter the cache");
  await ensureLibraryDetails();assert.equal(detailReads,2,"multiple archives must not trigger a fast retry loop");
  await refreshTaskSnapshots();assert.equal(directoryReads,1);assert.equal(state.archiveTotals.key_events,2);
  await ensureLibraryDetails();assert.equal(detailReads,2,"directory outage must keep detail retries paused");
  route="reports";await ensureLibraryDetails();assert.equal(libraryLoadState("reports").ready.length,2,"material release conflict must not block independently readable reports");
  route="materials";await ensureLibraryDetails();assert.equal(detailReads,2);
  recovered=true;await refreshTaskSnapshots();await ensureLibraryDetails();
  assert.equal(state.archiveRefreshPending,false);assert.equal(state.libraryReleaseRefreshPending,false);
  assert.equal(state.archiveTotals.key_events,7);assert.equal(detailReads,4);
  assert.equal(libraryLoadState("materials").ready.length,2);assert.equal(libraryLoadState("materials").failed.length,0);
  for(const name of ["A","B"]){
    const detail=cachedArchiveDetail(name);assert.equal(detail.release_id,"r2");
    assert.ok(detail.key_events.every(event=>event.release_id===detail.release_id));
  }
  await refreshTaskSnapshots();await ensureLibraryDetails();assert.equal(directoryReads,2);assert.equal(detailReads,4);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("status", [409, 503])
def test_library_http_version_conflict_queues_recovery_but_other_failures_do_not(status):
    run_javascript(("loadLibraryDetail", "libraryQueryKey", "cachedArchiveDetail", "queueLibraryReleaseRefresh",
                    "invalidateArchiveCache", "libraryRequestKey", "libraryRecordKey", "libraryRecordQueryKey"), r'''
const state={archiveCache:new Map(),materialCache:new Map(),libraryLoadErrors:new Set(),
 globalMaterialFilters:{action:"all",object:"all",support:"all"},search:""};
const archiveProcessStopped=()=>false,loadArchive=async()=>({name:"A",release_id:"r1"});
''' + f'const status={status};\n' + r'''
const api=async()=>{throw Object.assign(Error("unavailable"),{status});};
(async()=>{
  await assert.rejects(loadLibraryDetail("A","materials"),/unavailable/);
  assert.equal(Boolean(state.archiveRefreshPending),status===409);
  assert.equal(Boolean(state.libraryReleaseRefreshPending),status===409);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_legacy_library_materials_without_release_ids_remain_readable():
    run_javascript(("loadLibraryDetail", "libraryQueryKey", "cachedArchiveDetail"), r'''
const state={archiveCache:new Map(),globalMaterialFilters:{action:"all",object:"all",support:"all"},search:""};
const archiveProcessStopped=()=>false,loadArchive=async()=>({name:"A",counts:{key_events:1}});
const api=async()=>({items:[{event_id:"legacy"}],total_count:1});
(async()=>{
  const detail=await loadLibraryDetail("A","materials");assert.equal(detail.key_events[0].event_id,"legacy");
  assert.equal(Boolean(state.archiveRefreshPending),false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_material_request_keeps_its_filter_and_does_not_publish_stale_results():
    run_javascript(("loadLibraryDetail", "libraryQueryKey", "cachedArchiveDetail"), r'''
const state={archiveCache:new Map(),globalMaterialFilters:{action:"old",object:"all",support:"all"},search:"first"};
const archiveProcessStopped=()=>false;
let releaseSummary;
const summary=new Promise(resolve=>releaseSummary=resolve);
const loadArchive=()=>summary;
const requests=[];
async function api(url){requests.push(url);return {items:[{event_id:"old"}],total_count:1};}
(async()=>{
  const pending=loadLibraryDetail("A","materials");
  state.globalMaterialFilters.action="new"; state.search="second";
  releaseSummary({name:"A",release_id:"r1",counts:{key_events:8},experiments:[],key_events:[]});
  await pending;
  if(requests.length){
    const query=new URL(requests[0],"http://test").searchParams;
    assert.equal(query.get("action_type"),"old","request identity must match its captured filter");
    assert.equal(query.get("q"),"first");
  }
  assert.equal(cachedArchiveDetail("A"),null,"old query must not overwrite current library data");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_library_refresh_keeps_other_sections_but_replaces_summary_counts():
    run_javascript(("loadLibraryDetail", "libraryQueryKey", "cachedArchiveDetail"), r'''
const state={archiveCache:new Map(),globalMaterialFilters:{action:"all",object:"all",support:"all"},search:""};
const archiveProcessStopped=()=>false;
state.archiveCache.set("archive/A",{release_id:"r1",counts:{key_events:1},
  daily_report:{report_id:"report"},key_events:[{event_id:"old"}],libraryKeys:{reports:"reports"}});
async function loadArchive(){return {name:"A",release_id:"r1",counts:{key_events:8},experiments:[],key_events:[]};}
async function api(){return {items:[{event_id:"new",release_id:"r1"}],total_count:8};}
(async()=>{
  const result=await loadLibraryDetail("A","materials");
  assert.equal(result.counts.key_events,8,"fresh summary must replace cached totals");
  assert.equal(result.daily_report.report_id,"report");
  assert.deepEqual(result.key_events.map(x=>x.event_id),["new"]);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_library_failures_are_section_scoped_and_retryable_without_false_loaded_counts():
    run_javascript(("libraryRecordKey", "cachedLibraryDetail", "libraryRecordQueryKey", "libraryRequestKey",
                    "libraryQueryKey", "libraryLoadState", "libraryFailureNotice", "bindLibraryRetry",
                    "ensureLibraryDetails"), r'''
const state={archives:[{name:"A"}],archiveCache:new Map(),libraryLoadErrors:new Set(),
  globalMaterialFilters:{action:"all",support:"all",object:"all"},search:""};
const experimentRecords=()=>state.archives;
let route="materials", fail=true, requests=0, retry;
const routeParts=()=>[route],refreshProgressiveSurface=()=>{},number=x=>x;
const document={querySelector:()=>({addEventListener:(name,callback)=>{retry=callback;}})};
async function loadLibraryRecord(record,section){
  requests++;
  if(section==="materials" && fail)throw Error("temporary unavailable");
  const detail=state.archiveCache.get("archive/A")||{};
  detail.libraryKeys={...detail.libraryKeys,[section]:libraryRecordQueryKey(record,section)};
  state.archiveCache.set("archive/A",detail);return detail;
}
(async()=>{
  await ensureLibraryDetails();
  let status=libraryLoadState("materials");
  assert.equal(status.ready.length,0);assert.equal(status.failed.length,1);assert.equal(status.pending.length,0);
  assert.ok(libraryFailureNotice(status,"materials").includes("当前结果不完整"));
  await ensureLibraryDetails();assert.equal(requests,1,"failed request must settle instead of looping");
  route="reports";await ensureLibraryDetails();
  assert.equal(requests,2,"material failure must not block report loading");
  assert.equal(libraryLoadState("reports").ready.length,1);
  assert.equal(libraryLoadState("materials").ready.length,0,"report data does not count as loaded materials");
  route="materials";fail=false;bindLibraryRetry("materials");retry();await ensureLibraryDetails();
  status=libraryLoadState("materials");assert.equal(status.ready.length,1);assert.equal(status.failed.length,0);
  assert.equal(requests,3);
  state.globalMaterialFilters.action="changed";
  assert.equal(libraryLoadState("materials").ready.length,0,"old query must not count as loaded");
  await ensureLibraryDetails();assert.equal(requests,4);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_global_library_navigation_restores_unfiltered_listing_and_rejects_old_pages():
    run_javascript(("loadArchiveListing", "loadMoreArchives", "applyArchiveListingPage", "router", "archiveSearchQuery", "loadExperimentMetadata"), r'''
const state={archives:[{name:"only-A"}],archiveListingQuery:"A",archiveNextCursor:"filtered-page",
  archiveTotal:1,archiveTotals:{key_events:1},search:"A",archiveRequestId:0,archiveCache:new Map()};
const location={hash:"#/materials"};
const routeParts=()=>location.hash.slice(2).split("/");
const invalidated=[];const invalidateArchiveCache=name=>invalidated.push(name);
let rendered=0, requests=[],resolveOldPage,resolveOldListing;
const setChrome=()=>{},pageSkeleton=()=>"loading",main={innerHTML:""};
const renderMaterialsLibrary=()=>{rendered++};
const renderReportsLibrary=()=>{},syncArchiveFiltersFromRoute=()=>{},renderExperiments=()=>{};
async function api(url){
  requests.push(url);
  if(url.includes("cursor="))return new Promise(resolve=>resolveOldPage=resolve);
  if(url.includes("q=slow"))return new Promise(resolve=>resolveOldListing=resolve);
  return {archives:[{name:"A"},{name:"B"}],total_count:60,totals:{key_events:130},next_cursor:"all-page"};
}
(async()=>{
  await router();
  assert.equal(requests[0],"/api/archives?limit=50");
  assert.equal(state.archiveListingQuery,"");assert.equal(state.archiveTotals.key_events,130);
  assert.equal(state.archives.length,2);assert.equal(rendered,1);
  const page=loadMoreArchives();
  location.hash="#/experiments";state.search="different";
  await loadArchiveListing();
  resolveOldPage({archives:[{name:"stale"}],total_count:100,next_cursor:"wrong"});await page;
  assert.ok(!state.archives.some(x=>x.name==="stale"));
  assert.equal(state.archiveNextCursor,"all-page");
  state.search="slow";const old=loadArchiveListing();
  location.hash="#/materials";await router();
  resolveOldListing({archives:[{name:"slow"}],total_count:1,totals:{key_events:1}});await old;
  assert.equal(state.archiveListingQuery,"");assert.equal(state.archiveTotals.key_events,130);
  assert.ok(!state.archives.some(x=>x.name==="slow"));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_new_release_discards_inflight_old_materials():
    run_javascript(("loadLibraryDetail", "libraryQueryKey", "cachedArchiveDetail", "invalidateArchiveCache"), r'''
const state={archiveCache:new Map(),materialCache:new Map(),globalMaterialFilters:{action:"all",object:"all",support:"all"},search:""};
const archiveProcessStopped=()=>false;
let resolvePage,requestStarted;
const started=new Promise(resolve=>requestStarted=resolve);
async function loadArchive(){return {name:"A",release_id:"old",counts:{key_events:1}};}
async function api(){requestStarted();return new Promise(resolve=>resolvePage=resolve);}
(async()=>{
  const pending=loadLibraryDetail("A","materials");await started;
  invalidateArchiveCache("A");
  const current={name:"A",release_id:"new",key_events:[{event_id:"current"}]};
  state.archiveCache.set("archive/A",current);
  resolvePage({items:[{event_id:"stale"}],total_count:1});await pending;
  assert.equal(cachedArchiveDetail("A"),current);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_search_refreshes_only_relevant_data_and_does_not_reopen_after_navigation():
    source = SOURCE.read_text()
    listener = source.split('globalSearchInput.addEventListener("input",', 1)[1].split(
        'globalSearchInput.addEventListener("focus",', 1
    )[0]
    run_javascript((), r'''
const state={search:"",archiveListingQuery:""},location={hash:"#/materials"};
let scheduled,inputHandler,resolveListing;
const requests=[],renders=[],popups=[];
const globalSearchInput={addEventListener:(_,callback)=>inputHandler=callback};
const document={activeElement:globalSearchInput};
const routeParts=()=>location.hash.slice(2).split("/");
const archiveSearchQuery=()=>state.search.trim();
const clearTimeout=()=>{},setTimeout=callback=>{scheduled=callback;return 1;};
let archiveSearchTimer;
async function loadAll(){requests.push("health","runs","collections","nas-recordings","archives");}
async function loadArchiveListing(){
  requests.push("archives");
  if(state.search==="slow")await new Promise(resolve=>resolveListing=resolve);
  return true;
}
async function router(){renders.push(location.hash);}
const renderGlobalSearchResults=query=>popups.push(query),toast=()=>{};
globalSearchInput.addEventListener("input",
''' + listener + r'''
(async()=>{
  inputHandler({target:{value:"tube"}});await scheduled();
  assert.deepEqual(requests,[],"material filtering must not rescan NAS or reload unrelated state");
  assert.deepEqual(renders,["#/materials"]);assert.deepEqual(popups,["tube"]);
  location.hash="#/experiments";inputHandler({target:{value:"Archive"}});await scheduled();
  assert.deepEqual(requests,["archives"],"experiment search only needs an archive listing");
  assert.equal(popups.at(-1),"Archive");
  inputHandler({target:{value:"slow"}});const pending=scheduled();
  location.hash="#/tasks";document.activeElement=null;
  resolveListing();await pending;
  assert.equal(renders.length,2,"old search must not render the destination page");
  assert.equal(popups.length,2,"old search must not reopen a closed search panel");
  location.hash="#/reports";inputHandler({target:{value:"report"}});await scheduled();
  assert.equal(popups.length,2,"typing then blurring must keep the search popup closed");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_task_sync_failure_is_visible_and_recovery_keeps_the_last_snapshot():
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview", "renderTasks", "rememberRunDisclosures"), r'''
const saved={run_id:"R",experiment_id:"A",state:"key_materials"};
const state={runs:[saved],libraryLoadErrors:new Set()};
const document={hidden:false,querySelectorAll:()=>[]},main={innerHTML:"",querySelectorAll:()=>[]};
const experimentRecords=()=>[],routeParts=()=>["tasks"],updateServiceChrome=()=>{};
const setChrome=()=>{},statusCard=()=>"",number=x=>x,icon=()=>"",bindArchiveActions=()=>{};
const runObservabilityCard=run=>`<article>${run.run_id}:${run.state}</article>`;
let failing=true;
async function api(){if(failing)throw Error("offline");return {runs:[saved]};}
(async()=>{
  await refreshTaskSnapshots();
  assert.equal(state.taskSyncError,true,"a failed refresh must not look current");
  assert.equal(state.runs[0],saved,"keep the last snapshot instead of reporting zero tasks");
  assert.ok(main.innerHTML.includes('role="alert"'));
  assert.ok(main.innerHTML.includes("任务状态同步中断"));
  assert.ok(main.innerHTML.includes("R:key_materials"));
  failing=false;await refreshTaskSnapshots();
  assert.equal(state.taskSyncError,false);
  assert.ok(!main.innerHTML.includes("任务状态同步中断"));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_completed_run_refreshes_an_open_report_library_even_with_unchanged_counts():
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview",), r'''
const archiveSearchQuery=()=>"";
const state={archiveListingQuery:"",runs:[{run_id:"R",state:"key_materials",experiment_id:"A"}],libraryLoadErrors:new Set()};
const document={hidden:false};
const experimentRecords=()=>[{name:"A",key_event_count:4}],libraryRecordKey=x=>x.name;
const routeParts=()=>["reports"],updateServiceChrome=()=>{},invalidateArchiveCache=()=>{};
let reportRenders=0,listingRequests=0;
const renderReportsLibrary=()=>reportRenders++;
async function api(){return {runs:[{run_id:"R",state:"completed",experiment_id:"A"}]};}
async function loadArchiveListing(){listingRequests++;state.archiveRefreshPending=false;return true;}
(async()=>{
  await refreshTaskSnapshots();
  assert.equal(listingRequests,1);assert.equal(reportRenders,1,"completed report must become visible without manual refresh");
  await refreshTaskSnapshots();
  assert.equal(listingRequests,1);assert.equal(reportRenders,1,"unchanged polling must not reload the report library");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_archive_refresh_failure_does_not_block_completed_task_status_and_retries():
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview",), r'''
const prior={run_id:"R",state:"key_materials",experiment_id:"A"};
const state={runs:[prior],activeRun:prior,libraryLoadErrors:new Set()};
const document={hidden:false};
const experimentRecords=()=>[],routeParts=()=>["tasks"],invalidateArchiveCache=()=>{};
const updateServiceChrome=()=>{};
const rendered=[];const renderTasks=()=>rendered.push(state.activeRun.state);
let listingRequests=0;
async function api(){return {runs:[{run_id:"R",state:"completed",experiment_id:"A"}]};}
async function loadArchiveListing(){listingRequests++;if(listingRequests===1)throw Error("archive offline");state.archiveRefreshPending=false;return true;}
(async()=>{
  await refreshTaskSnapshots();
  assert.equal(state.activeRun.state,"completed","archive failure must not freeze task status");
  assert.deepEqual(rendered,["completed"]);assert.equal(state.archiveRefreshPending,true);
  assert.equal(state.taskSyncError,false,"archive failure and task polling failure are distinct");
  await refreshTaskSnapshots();
  assert.equal(listingRequests,2);assert.equal(state.archiveRefreshPending,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_manual_refresh_reports_partial_failure_instead_of_false_success():
    source = SOURCE.read_text()
    listener = source.split('document.querySelector("#refresh-button").addEventListener("click",', 1)[1].split(
        'const mobileMoreDialog', 1
    )[0]
    run_javascript((), r'''
const state={archiveCache:new Map(),materialCache:new Map(),libraryLoadErrors:new Set()};
let refresh,failed=true;
const messages=[];
const button={disabled:false,classList:{add:()=>{},remove:()=>{}},addEventListener:(_,fn)=>refresh=fn};
const document={querySelector:()=>button};
async function loadAll(){state.taskSyncError=failed;return true;}
async function router(){}
const toast=(message,tone)=>messages.push({message,tone});
document.querySelector("#refresh-button").addEventListener("click",
''' + listener + r'''
(async()=>{
  await refresh();
  assert.equal(messages.at(-1).tone,"error","partial refresh must not claim full success");
  assert.ok(messages.at(-1).message.includes("任务状态"));assert.equal(button.disabled,false);
  failed=false;await refresh();assert.equal(messages.at(-1).tone,"success");
  loadAll=async()=>{state.archiveRefreshPending=true;return false;};
  router=async()=>{state.archiveRefreshPending=false;};
  await refresh();assert.equal(messages.at(-1).tone,"success","a directory recovered by the current route must not retain an obsolete failure toast");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_archive_pagination_does_not_reopen_a_page_after_navigation():
    run_javascript(("bindArchivePagination",), r'''
const state={archiveNextCursor:"next",archives:[{}],archiveTotal:2,archiveRequestId:1};
const location={hash:"#/experiments"};
let handler,finish;
const button={disabled:false,addEventListener:(_,callback)=>handler=callback};
const document={querySelector:selector=>selector==="#load-more-archives"?button:{insertAdjacentHTML:()=>{}}};
const number=x=>x,rendered=[],errors=[];
const renderExperiments=target=>rendered.push(target),toast=message=>errors.push(message);
const loadMoreArchives=()=>new Promise(resolve=>finish=resolve);
(async()=>{
  bindArchivePagination();const pending=handler({currentTarget:button});
  location.hash="#/tasks";finish();await pending;
  assert.deepEqual(rendered,[],"late pagination must not replace the task page");
  assert.deepEqual(errors,[]);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


MATERIAL_FILTER_HARNESS = r'''
const state={materialFilters:{archive:"A",group:"all",action:"all",object:"all",support:"all",query:""},archiveViewRequestId:1,archiveCacheEpoch:0};
const location={hash:"#/archive/A/materials"},nodes=new Map(),timers=new Map(),reads=[],applied=[],fullRenders=[];
let timerId=0;
function node(selector){if(!nodes.has(selector))nodes.set(selector,{value:"",innerHTML:"",textContent:"",hidden:false,disabled:false,
  handlers:{},attrs:{},addEventListener(type,callback){this.handlers[type]=callback},
  setAttribute(key,value){this.attrs[key]=value},removeAttribute(key){delete this.attrs[key]},
  insertAdjacentHTML(){},remove(){},querySelector(child){return node(selector+" "+child)}});return nodes.get(selector);}
const input=node("#material-query"),document={activeElement:input,querySelector:node};
input.selectionStart=2;input.selectionEnd=4;
const clearTimeout=id=>timers.delete(id),setTimeout=callback=>{timers.set(++timerId,callback);return timerId};
async function flush(){const pending=[...timers.values()];timers.clear();return Promise.all(pending.map(fn=>fn()));}
let stopped=false;
const archiveProcessStopped=()=>stopped,bindMaterialInteractions=()=>{},materialQueryParameters=()=>new URLSearchParams();
const renderArchive=(...args)=>fullRenders.push(args),toast=()=>{},esc=String;
const updateMaterialResults=data=>applied.push(data),materialResults=()=>"local results";
const loadArchiveMaterials=(name,cursor)=>new Promise((resolve,reject)=>reads.push({name,cursor,query:state.materialFilters.query,resolve,reject}));
const initial={name:"A",release_id:"r1",key_events:[],material_next_cursor:"next"};
function type(value,extra={}){input.value=value;return input.handlers.input({target:input,...extra});}
function response(query){return {...initial,query,material_next_cursor:null};}
'''


def test_material_search_keeps_the_input_available_during_a_request():
    run_javascript(("bindMaterialFilters",), MATERIAL_FILTER_HARNESS + r'''
(async()=>{
 bindMaterialFilters(initial);type("tube");const first=flush();
 assert.deepEqual(fullRenders,[],"search must not replace the archive page or its input");
 assert.equal(reads.length,1);assert.equal(document.activeElement,input);
 type("tube cap");const second=flush();assert.equal(reads.length,2,"typing remains available while the old read is pending");
 reads[1].resolve(response("tube cap"));await second;
 reads[0].resolve(response("tube"));await first;
 assert.deepEqual(applied.map(x=>x.query),["tube cap"]);
 assert.equal(input.value,"tube cap");assert.equal(input.selectionStart,2);assert.equal(input.selectionEnd,4);
 assert.equal(document.activeElement,input);assert.equal(state.archiveViewRequestId,1);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_material_search_waits_for_composition_and_collapses_the_final_input_event():
    run_javascript(("bindMaterialFilters",), MATERIAL_FILTER_HARNESS + r'''
(async()=>{
 bindMaterialFilters(initial);type("yi");
 assert.equal(typeof input.handlers.compositionstart,"function");
 input.handlers.compositionstart();type("yiye",{isComposing:true});await flush();assert.equal(reads.length,0);
 input.value="移液";input.handlers.compositionend({target:input});type("移液");
 assert.equal(timers.size,1,"compositionend and final input must share one debounce");
 const pending=flush();assert.equal(reads.length,1);assert.equal(reads[0].query,"移液");
 reads[0].resolve(response("移液"));await pending;assert.equal(applied.length,1);
 assert.equal(input.value,"移液");assert.deepEqual(fullRenders,[]);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("change", ["query", "page", "composition"])
@pytest.mark.parametrize("fails", [False, True])
def test_material_search_ignores_superseded_results_and_errors(change, fails):
    run_javascript(("bindMaterialFilters",), MATERIAL_FILTER_HARNESS + f'const change={change!r},fails={str(fails).lower()};\n' + r'''
(async()=>{
 bindMaterialFilters(initial);type("old");const pending=flush();
 if(change==="query")type("new");
 if(change==="page")state.archiveViewRequestId++;
 if(change==="composition")input.handlers.compositionstart();
 const notice=node("#material-filter-status span").textContent;
 if(fails)reads[0].reject(Error("obsolete failure"));else reads[0].resolve(response("old"));
 await pending;
 assert.deepEqual(applied,[]);assert.deepEqual(fullRenders,[]);
 assert.equal(node("#material-filter-status span").textContent,notice,"old responses must not alter current status");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("change", ["epoch", "release"])
def test_material_search_does_not_mix_archive_versions(change):
    run_javascript(("bindMaterialFilters",), MATERIAL_FILTER_HARNESS + f'const change={change!r};\n' + r'''
(async()=>{
 bindMaterialFilters(initial);type("tube");const pending=flush();
 if(change==="epoch")state.archiveCacheEpoch++;
 reads[0].resolve({...response("tube"),release_id:change==="release"?"r2":"r1"});await pending;
 assert.deepEqual(applied,[]);
 assert.ok(node("#material-filter-status span").textContent.includes("档案版本已更新"));
 assert.equal(node("#material-filter-status button").hidden,false);
 assert.equal(input.value,"tube");assert.equal(document.activeElement,input);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_material_search_failure_retains_input_and_retries_the_latest_query():
    run_javascript(("bindMaterialFilters",), MATERIAL_FILTER_HARNESS + r'''
(async()=>{
 bindMaterialFilters(initial);node("#material-results").innerHTML="previous cards";
 type("tube");const pending=flush();reads[0].reject(Error("offline"));await pending;
 assert.deepEqual(applied,[]);assert.equal(node("#material-results").innerHTML,"previous cards");
 assert.equal(input.value,"tube");assert.equal(document.activeElement,input);
 assert.equal(node("#material-filter-status button").hidden,false);
 assert.equal(node("#material-results").attrs["aria-busy"],"false");
 type("tube cap");node("#material-filter-status button").handlers.click();
 assert.equal(timers.size,0,"retry must absorb any pending debounce");assert.equal(reads[1].query,"tube cap");
 reads[1].resolve(response("tube cap"));await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(applied.map(x=>x.query),["tube cap"]);
 assert.equal(node("#material-filter-status").hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("source", ["staging", "stopped"])
def test_material_search_updates_complete_local_results_without_network_reads(source):
    run_javascript(("bindMaterialFilters",), MATERIAL_FILTER_HARNESS + f'const source={source!r};\n' + r'''
(async()=>{
 if(source==="staging")initial.staging_run_id="run1";else stopped=true;
 bindMaterialFilters(initial);type("离心管");await flush();
 assert.deepEqual(reads,[]);assert.deepEqual(applied,[initial]);assert.deepEqual(fullRenders,[]);
 assert.equal(document.activeElement,input);assert.equal(input.value,"离心管");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_material_search_pagination_uses_only_the_current_result_cursor():
    run_javascript(("bindMaterialFilters",), MATERIAL_FILTER_HARNESS + r'''
(async()=>{
 bindMaterialFilters(initial);type("new query");
 const first=node("#load-more-materials").handlers.click({currentTarget:node("#load-more-materials")});
 assert.equal(timers.size,0);assert.equal(reads[0].cursor,null,"typing invalidates the previous query's cursor");
 reads[0].resolve({...response("new query"),material_next_cursor:"new-next"});await first;
 const second=node("#load-more-materials").handlers.click({currentTarget:node("#load-more-materials")});
 assert.equal(reads[1].cursor,"new-next");reads[1].resolve(response("new query"));await second;
 assert.equal(applied.length,2);assert.deepEqual(fullRenders,[]);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_material_result_patch_preserves_input_and_active_dropdown_nodes():
    run_javascript(("updateMaterialResults",), r'''
const input={value:"移液器",selectionStart:1,selectionEnd:2},select={innerHTML:"active dropdown"};
const nodes=new Map([["#material-query",input],["#material-action-filter",select]]),writes=[],paused=[],bound=[];
const allowed=["#material-results",".material-toolbar-heading .badge","#material-group-filter","#material-action-filter","#material-object-filter"];
for(const key of allowed)if(!nodes.has(key))nodes.set(key,{set innerHTML(value){writes.push([key,value])}});
const template={content:{querySelector:selector=>({innerHTML:"new "+selector})}};
const document={activeElement:select,createElement:type=>{assert.equal(type,"template");return template},
 querySelector:selector=>{assert.ok(allowed.includes(selector));return nodes.get(selector)}};
const data={name:"A"},materialsView=value=>{assert.equal(value,data);return "updated workspace"};
const pauseMaterialVideos=node=>paused.push(node),bindMaterialInteractions=value=>bound.push(value);
updateMaterialResults(data);
assert.equal(template.innerHTML,"updated workspace");assert.equal(writes.length,4);
assert.equal(select.innerHTML,"active dropdown");assert.equal(document.activeElement,select);
assert.equal(input.value,"移液器");assert.equal(input.selectionStart,1);assert.equal(input.selectionEnd,2);
assert.deepEqual(paused,[nodes.get("#material-results")]);assert.deepEqual(bound,[data]);
''')


def test_detail_search_timer_and_pagination_do_not_reopen_a_departed_archive():
    run_javascript(("bindMaterialFilters",), MATERIAL_FILTER_HARNESS + r'''
(async()=>{
 bindMaterialFilters(initial);type("tube");location.hash="#/reports";await flush();
 assert.equal(reads.length,0,"a departed page must not start its delayed query");
 location.hash="#/archive/A/materials";
 const pending=node("#load-more-materials").handlers.click({currentTarget:node("#load-more-materials")});
 location.hash="#/tasks";reads[0].resolve(response("tube"));await pending;
 assert.deepEqual(applied,[]);assert.deepEqual(fullRenders,[]);
 assert.equal(node("#load-more-materials").disabled,false);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("change", ["route", "query", "refresh"])
def test_global_material_pagination_stops_old_batches(change):
    source = SOURCE.read_text()
    listener = source.split('document.querySelector("[data-library-next]")?.addEventListener("click",', 1)[1].split(
        '\n    });', 1
    )[0]
    run_javascript((), r'''
const state={globalMaterialLimit:18,archiveCacheEpoch:0};
const location={hash:"#/materials"};let query="old",handler;
const more=Array.from({length:6},(_,index)=>({name:`A${index}`}));
const libraryQueryKey=()=>query,cachedArchiveDetail=()=>({material_next_cursor:"next"});
const document={querySelector:()=>({addEventListener:(_,callback)=>handler=callback})};
const requests=[],pending=[],rendered=[];
const loadLibraryDetail=(name,section,cursor)=>{requests.push({name,cursor});return requests.length<=3 ? new Promise(resolve=>pending.push(resolve)) : Promise.resolve({});};
const renderMaterialsLibrary=()=>rendered.push(true),toast=()=>{};
document.querySelector("[data-library-next]").addEventListener("click",
''' + listener + '\n});\n' + f'const change={change!r};\n' + r'''
(async()=>{
  const button={disabled:false};
  const loading=handler({currentTarget:button});
  assert.equal(requests.length,3);
  if(change==="route")location.hash="#/tasks";
  if(change==="query")query="new";
  if(change==="refresh")state.archiveCacheEpoch++;
  pending.splice(0).forEach(resolve=>resolve({}));
  await loading;
  assert.equal(requests.length,3,"old pagination must stop before another batch");
  assert.deepEqual(rendered,[]);assert.equal(state.globalMaterialLimit,18);
  assert.equal(button.disabled,false,"canceled pagination must release its busy state");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_detail_material_request_cannot_switch_filters_while_loading_groups():
    run_javascript(("loadArchiveMaterials", "materialQueryParameters", "ensureMaterialFilters"), r'''
const state={materialFilters:{archive:"A",group:"all",action:"old",object:"all",support:"all",query:""},
  materialCache:new Map(),archiveCache:new Map()};
let groupsReady,started;
const groupStarted=new Promise(resolve=>started=resolve);
const loadArchive=async()=>({name:"A",release_id:"r1"}),archiveProcessStopped=()=>false;
const loadArchiveSection=()=>{started();return new Promise(resolve=>groupsReady=resolve)};
const requests=[];
const api=async url=>{requests.push(url);return {items:[{event_id:"E",release_id:"r1"}]};};
(async()=>{
  const loading=loadArchiveMaterials("A");await groupStarted;
  state.materialFilters.action="new";groupsReady({experiment_groups:[]});
  await loading.catch(()=>{});
  assert.equal(state.materialCache.size,0,"stale request must not populate a different filter cache");
  assert.equal(requests.length,0,"cancel stale work before querying events");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_invalidated_archive_section_cannot_repopulate_old_cache():
    run_javascript(("loadArchiveSection", "invalidateArchiveCache"), r'''
const state={archiveCache:new Map(),materialCache:new Map()};
let finish,started;
const requestStarted=new Promise(resolve=>started=resolve);
const loadArchive=async()=>({name:"A",release_id:"r1"});
const api=()=>{started();return new Promise(resolve=>finish=resolve);};
(async()=>{
  const loading=loadArchiveSection("A","reports");await requestStarted;
  invalidateArchiveCache("A");const current={name:"A",release_id:"r2"};state.archiveCache.set("A:summary",current);
  finish({release_id:"r1",daily_report:{report_id:"old"}});await loading.catch(()=>{});
  assert.equal(state.archiveCache.size,1,"invalidated report must not return to cache");
  assert.equal(state.archiveCache.get("A:summary"),current);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("view_name,staging,expected", [("A", False, True), ("B", False, False), ("A", True, False)])
def test_archive_invalidation_marks_only_the_matching_formal_detail_for_sync(view_name, staging, expected):
    run_javascript(("invalidateArchiveCache",), r'''
const state={archiveCache:new Map(),materialCache:new Map()};
''' + f'const expected={str(expected).lower()};state.archiveView={{name:{view_name!r},staging:{str(staging).lower()}}};\n' + r'''
invalidateArchiveCache("A");
assert.equal(Boolean(state.archiveView.refreshPending),expected);
''')


def test_background_archive_read_failure_preserves_content_and_schedules_retry():
    run_javascript(("renderArchive", "stageSnapshotVersion",), r'''
const state={},location={hash:"#/archive/A/reports"},main={innerHTML:"LAST REPORT"};
let reject;const notices=[];
const loadArchiveView=()=>new Promise((_,fail)=>reject=fail);
const setChrome=()=>{},productExperimentName=String,pageSkeleton=()=>"loading",archiveLabel=()=>"archive";
const archiveRefreshNotice=message=>notices.push(message),archiveViewBusy=()=>false;
const productState=()=>"ERROR",document={querySelector:()=>null};
(async()=>{
  const read=renderArchive("A","reports",false,true);
  assert.equal(main.innerHTML,"LAST REPORT","a background read must keep the currently visible report");
  reject(Error("temporary report outage"));await read;
  assert.equal(main.innerHTML,"LAST REPORT");assert.equal(state.archiveView.refreshPending,true);
  assert.equal(state.archiveView.loading,false);assert.ok(notices.at(-1).includes("自动重试"));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def run_archive_live_sync(script):
    run_javascript(("renderArchive", "stageSnapshotVersion", "invalidateArchiveCache", "refreshLibraryOverview",
                    "refreshOpenArchive", "archiveViewBusy", "archiveRefreshNotice"), r'''
const state={archiveCache:new Map(),materialCache:new Map()},location={hash:"#/archive/A/reports"},window={};
let active=null,dialog=null,media=[];
const notice={hidden:true,textContent:""},main={innerHTML:"",contains:element=>element!=null,querySelectorAll:()=>media};
const document={get activeElement(){return active;},querySelector:selector=>selector==="dialog[open]"?dialog:selector==="#archive-sync-notice"?notice:null};
const pending=[],loadArchiveView=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
const setChrome=()=>{},productExperimentName=String,pageSkeleton=()=>"loading",archiveLabel=()=>"archive";
const archiveProcessStopped=()=>false,resultHeader=()=>"",resultReviewPanel=()=>"",dailyReportView=data=>data.marker;
const componentResultCards=()=>"",experimentResultTools=()=>"";
const resultWorkspaceContent=data=>data.marker;
const materialsView=dailyReportView,professionalReportsView=dailyReportView,experimentExecutiveSummary=dailyReportView;
const experimentGroupBrowser=dailyReportView;
const bindArchiveActions=()=>{},bindVideoPreviews=()=>{},bindExperimentReaders=()=>{},bindDetailAnchors=()=>{},bindRerunActions=()=>{};
const bindMaterialFilters=()=>{},bindRetainedMaterialFilters=()=>{},updateResultNavDensity=()=>{};
const requestAnimationFrame=callback=>callback(),routeQuery=()=>new URLSearchParams("focus=E1&candidate=C1&group=G1");
const productState=()=>"ERROR",openLinkedMaterial=()=>{throw Error("background refresh must not reopen an old focus link")};
const routeParts=()=>["archive"];
const data=marker=>({name:"A",release_id:marker,experiments:[],marker});
const tick=()=>new Promise(setImmediate);
''' + script)


@pytest.mark.parametrize("tab", ["experiments", "materials", "reports", "metrics"])
def test_formal_archive_sync_updates_only_invalidated_detail_without_duplicate_reads_or_deep_link_replay(tab):
    run_archive_live_sync(f'const tab={tab!r};\n' + r'''
(async()=>{
  location.hash="#/archive/A/"+tab;
  // Initial navigation is tested separately; begin with an already displayed detail.
  main.innerHTML="REPORT R1";
  state.archiveView={name:"A",tab,hash:location.hash,staging:false,loading:false};
  invalidateArchiveCache("B");refreshLibraryOverview();assert.equal(pending.length,0);
  invalidateArchiveCache("A");refreshLibraryOverview();assert.equal(pending.length,1);
  assert.equal(main.innerHTML,"REPORT R1");assert.equal(notice.hidden,false);
  refreshLibraryOverview();assert.equal(pending.length,1,"polls must not duplicate an in-flight detail read");
  pending[0].resolve(data("REPORT R2"));await tick();
  assert.ok(main.innerHTML.includes("REPORT R2"));assert.ok(!main.innerHTML.includes("REPORT R1"));
  assert.equal(state.archiveView.refreshPending,false);assert.equal(state.archiveView.loading,false);
  refreshLibraryOverview();assert.equal(pending.length,1,"unchanged polls must not reload a current detail");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("activity", ["video", "audio", "dialog", "INPUT", "TEXTAREA", "SELECT", "editable"])
def test_archive_sync_waits_for_playback_or_editing_then_continues_automatically(activity):
    run_archive_live_sync(f'const activity={activity!r};\n' + r'''
(async()=>{
  main.innerHTML="REPORT R1";
  state.archiveView={name:"A",tab:"reports",hash:location.hash,refreshPending:true};
  if(["video","audio"].includes(activity))media=[{paused:false,ended:false,tagName:activity.toUpperCase()}];
  else if(activity==="dialog")dialog={open:true};
  else active={tagName:activity,isContentEditable:activity==="editable"};
  refreshLibraryOverview();assert.equal(pending.length,0);assert.equal(main.innerHTML,"REPORT R1");
  assert.ok(notice.textContent.includes("播放或编辑结束后自动同步"));
  active=null;dialog=null;media=[];
  refreshLibraryOverview();assert.equal(pending.length,1);
  pending[0].resolve(data("REPORT R2"));await tick();assert.ok(main.innerHTML.includes("REPORT R2"));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_archive_sync_rechecks_activity_when_the_background_response_arrives():
    run_archive_live_sync(r'''
(async()=>{
  main.innerHTML="REPORT R1";
  state.archiveView={name:"A",tab:"reports",hash:location.hash,refreshPending:true};
  refreshLibraryOverview();active={tagName:"INPUT"};
  pending[0].resolve(data("REPORT R2"));await tick();
  assert.equal(main.innerHTML,"REPORT R1","editing begun during a read must also be protected");
  assert.equal(state.archiveView.refreshPending,true);assert.equal(state.archiveView.loading,false);
  refreshLibraryOverview();assert.equal(pending.length,1);
  active=null;refreshLibraryOverview();assert.equal(pending.length,2);
  pending[1].resolve(data("REPORT R2"));await tick();assert.ok(main.innerHTML.includes("REPORT R2"));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("old_fails", [False, True])
def test_archive_sync_retries_failures_and_discards_responses_after_navigation(old_fails):
    run_archive_live_sync(f'const oldFails={str(old_fails).lower()};\n' + r'''
(async()=>{
  main.innerHTML="REPORT R1";
  state.archiveView={name:"A",tab:"reports",hash:location.hash,refreshPending:true};
  refreshLibraryOverview();pending[0].reject(Error("offline"));await tick();
  assert.equal(main.innerHTML,"REPORT R1");assert.ok(notice.textContent.includes("自动重试"));
  refreshLibraryOverview();assert.equal(pending.length,2);
  location.hash="#/archive/B/reports";main.innerHTML="REPORT B";
  if(oldFails)pending[1].reject(Error("late failure"));else pending[1].resolve(data("REPORT A2"));
  await tick();assert.equal(main.innerHTML,"REPORT B");
  refreshLibraryOverview();assert.equal(pending.length,2,"leaving the detail must stop its background retries");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("old_fails", [False, True])
def test_same_route_detail_requests_only_publish_the_latest_response(old_fails):
    run_javascript(("renderArchive", "stageSnapshotVersion",), r'''
const state={},location={hash:"#/archive/A/reports"},main={innerHTML:""};
const pending=[];
const loadArchiveView=()=>new Promise((resolve,reject)=>pending.push({resolve,reject}));
const setChrome=()=>{},productExperimentName=x=>x,pageSkeleton=()=>"loading",archiveLabel=()=>"archive";
const archiveProcessStopped=()=>false,resultHeader=()=>"",resultReviewPanel=()=>"",dailyReportView=data=>data.marker;
const componentResultCards=()=>"",experimentResultTools=()=>"";
const resultWorkspaceContent=data=>data.marker;
const bindArchiveActions=()=>{},bindVideoPreviews=()=>{},bindExperimentReaders=()=>{},bindDetailAnchors=()=>{},bindRerunActions=()=>{};
const requestAnimationFrame=()=>{},routeQuery=()=>new URLSearchParams();
const productState=()=>"ERROR",document={querySelector:()=>null};
''' + f'const oldFails={str(old_fails).lower()};\n' + r'''
(async()=>{
  const old=renderArchive("A","reports"),current=renderArchive("A","reports");
  pending[1].resolve({name:"A",experiments:[],marker:"NEW"});await current;
  assert.ok(main.innerHTML.includes("NEW"));
  if(oldFails)pending[0].reject(Error("old failure"));else pending[0].resolve({name:"A",experiments:[],marker:"OLD"});
  await old;assert.ok(main.innerHTML.includes("NEW"),"old response must not replace a newer render on the same URL");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("staging", [False, True])
def test_linked_material_uid_lookup_is_not_limited_by_text_search_collisions(staging):
    run_javascript(("openLinkedMaterial", "materialFocusRoute", "experimentRecordRoute"), r'''
const state={archiveCacheEpoch:0,materialFilters:{query:"keep this filter"}},requests=[],opened=[],errors=[];
const target={event_id:"E1",event_uid:"uid:目标-29",parent_event_id:"G1",release_id:"r1",ready:true};
const data={name:"实验 A & B",release_id:"r1",key_events:[]};
''' + f'const staging={str(staging).lower()};\n' + r'''
if(staging)data.staging_run_id="run:29";
const route=materialFocusRoute(data,target),routeQuery=()=>new URLSearchParams(route.split("?")[1]);
const eventHasAlignedDualViewMaterial=event=>event.ready,openMaterialFocus=(page,id,event)=>opened.push(event);
const toast=message=>errors.push(message);
async function api(url){
  requests.push(url);
  if(url.startsWith("/api/key-events/"+encodeURIComponent(target.event_uid)) || url.startsWith("/api/staging-runs/"))return target;
  return {items:Array.from({length:24},(_,i)=>({...target,event_uid:"collision-"+i})),next_cursor:"more-matches"};
}
(async()=>{
  await openLinkedMaterial(data,target.event_id,()=>true);
  assert.deepEqual(opened,[target],"a unique material link must resolve even when the first search page contains 24 other matches");
  assert.equal(requests.length,1,"resolve by identity without scanning search-result pages");
  const expected=staging?`/api/staging-runs/${encodeURIComponent(data.staging_run_id)}/key-events/${encodeURIComponent(target.event_uid)}`
    :`/api/key-events/${encodeURIComponent(target.event_uid)}?archive=${encodeURIComponent(data.name)}`;
  assert.equal(requests[0],expected);assert.deepEqual(errors,[]);
  assert.equal(state.materialFilters.query,"keep this filter");assert.deepEqual(data.key_events,[]);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("use_uid", [False, True])
def test_linked_material_loads_an_exact_item_beyond_the_page_without_changing_filters(use_uid):
    run_javascript(("openLinkedMaterial", "materialFocusRoute", "experimentRecordRoute"), r'''
const state={archiveCacheEpoch:0,materialFilters:{action:"different",query:"previous search"}};
const target={event_id:"E29",event_uid:"uid-29",parent_event_id:"G2",release_id:"r1",ready:true};
''' + f'const useUid={str(use_uid).lower()};\n' + r'''
const route=materialFocusRoute({name:"A"},{...target,event_uid:useUid?target.event_uid:null}),routeQuery=()=>new URLSearchParams(route.split("?")[1]);
const data={name:"A",release_id:"r1",key_events:Array.from({length:24},(_,i)=>({event_id:`E${i}`}))};
const filters=state.materialFilters,requests=[],opened=[],errors=[];
const eventHasAlignedDualViewMaterial=event=>event.ready;
const openMaterialFocus=(page,id,event)=>opened.push({page,id,event}),toast=message=>errors.push(message);
async function api(url){requests.push(url);return useUid?target:{items:[{...target,event_id:"E290"},target]};}
(async()=>{
  await openLinkedMaterial(data,"E29",()=>true);
  assert.equal(requests.length,1);const params=new URL(requests[0],"http://test").searchParams;
  assert.equal(params.get("archive"),"A");
  if(useUid)assert.equal(requests[0],"/api/key-events/uid-29?archive=A");
  else {
    assert.equal(params.get("q"),"E29");assert.equal(params.get("parent_event_id"),"G2");
    assert.equal(params.get("material_ready"),"true");assert.equal(params.get("limit"),"24");
  }
  assert.equal(params.has("cursor"),false);
  assert.equal(opened[0].event,target);assert.equal(opened[0].page,data);assert.equal(opened[0].id,"E29");
  assert.equal(data.key_events.length,24);assert.equal(state.materialFilters,filters);assert.deepEqual(errors,[]);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("problem", ["uid", "event_id", "group", "release", "unready", "missing", "old_link"])
def test_linked_material_rejects_wrong_identity_release_or_unready_evidence(problem):
    run_javascript(("openLinkedMaterial",), r'''
const state={};const routeQuery=()=>new URLSearchParams(`event_uid=uid-29&parent_event_id=G2&release=${problem==="old_link"?"old":"r1"}`);
const data={name:"A",release_id:"r1",key_events:[]};
const target={event_id:"E29",event_uid:"uid-29",parent_event_id:"G2",release_id:"r1",ready:true};
const eventHasAlignedDualViewMaterial=event=>event.ready;
const opened=[],errors=[];let requests=0;
const openMaterialFocus=()=>opened.push(true),toast=message=>errors.push(message);
async function api(){requests++;if(problem==="missing")throw Error("not found");return {...target,
  event_id:problem==="event_id"?"E290":target.event_id,parent_event_id:problem==="group"?"other":target.parent_event_id,
  event_uid:problem==="uid"?"other":target.event_uid,release_id:problem==="release"?"old":"r1",ready:problem!=="unready"};}
''' + f'const problem={problem!r};\n' + r'''
(async()=>{
  await openLinkedMaterial(data,"E29",()=>true);
  assert.deepEqual(opened,[]);assert.equal(errors.length,1);
  if(problem==="old_link")assert.equal(requests,0,"an old pinned link must not open a new release");
  else assert.equal(requests,1,"a failed identity lookup must not fall back to an unrelated search match");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_loaded_material_identity_uses_the_current_page_without_another_request():
    run_javascript(("openLinkedMaterial",), r'''
const state={},routeQuery=()=>new URLSearchParams("event_uid=uid-29&parent_event_id=G2&release=r1");
const target={event_id:"E29",event_uid:"uid-29",parent_event_id:"G2",release_id:"r1",ready:true};
const data={name:"A",release_id:"r1",key_events:[{...target,event_uid:"other"},target]};
const opened=[],errors=[],api=async()=>{throw Error("unexpected extra read")};
const eventHasAlignedDualViewMaterial=event=>event.ready,openMaterialFocus=(page,id,event)=>opened.push(event),toast=message=>errors.push(message);
(async()=>{await openLinkedMaterial(data,"E29",()=>true);assert.deepEqual(opened,[target]);assert.deepEqual(errors,[]);})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("change", ["navigation", "refresh"])
@pytest.mark.parametrize("use_uid", [False, True])
@pytest.mark.parametrize("old_fails", [False, True])
def test_linked_material_does_not_open_after_its_page_is_superseded(change, use_uid, old_fails):
    run_javascript(("openLinkedMaterial",), r'''
const state={archiveCacheEpoch:0},routeQuery=()=>new URLSearchParams(useUid?"event_uid=uid-29":"");
let current=true,finish,reject;const opened=[],errors=[];
const api=()=>new Promise((resolve,fail)=>{finish=resolve;reject=fail}),eventHasAlignedDualViewMaterial=()=>true;
const openMaterialFocus=()=>opened.push(true),toast=message=>errors.push(message);
''' + f'const change={change!r},useUid={str(use_uid).lower()},oldFails={str(old_fails).lower()};\n' + r'''
(async()=>{
  const loading=openLinkedMaterial({name:"A",release_id:"r1",key_events:[]},"E29",()=>current);
  if(change==="navigation")current=false;else state.archiveCacheEpoch++;
  const target={event_id:"E29",event_uid:"uid-29",release_id:"r1"};
  if(oldFails)reject(Error("late failure"));else finish(useUid?target:{items:[target]});
  await loading;
  assert.deepEqual(opened,[]);assert.deepEqual(errors,[]);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("route", ["home", "experiments", "materials", "reports", "tasks", "archive"])
@pytest.mark.parametrize("directory_pending", [False, True])
def test_open_global_search_tracks_completed_materials_and_version_links(route, directory_pending):
    run_javascript(("refreshTaskSnapshots", "refreshLibraryOverview", "invalidateArchiveCache",
                    "loadGlobalMaterialSearch", "globalMaterialSearchKey", "globalSearchGroups",
                    "renderGlobalSearchResults", "materialFocusRoute", "experimentRecordRoute"), r'''
const state={search:"tube",archiveCacheEpoch:0,archiveCache:new Map(),materialCache:new Map(),
 runs:[{run_id:"R",experiment_id:"A",state:"key_materials"}],libraryLoadErrors:new Set(),archiveListingQuery:"tube"};
const panel={hidden:false,innerHTML:"",querySelectorAll:()=>[],querySelector:()=>null};
const input={setAttribute:()=>{}};
const document={hidden:false,querySelector:selector=>selector==="#global-search-results"?panel:input};
const experimentRecords=()=>[],archiveSearchQuery=()=>"tube",updateServiceChrome=()=>{};
const renderHome=()=>{},renderExperiments=()=>{},renderMaterialsLibrary=()=>{},renderReportsLibrary=()=>{},renderTasks=()=>{};
const syncArchiveFiltersFromRoute=()=>{},eventHasAlignedDualViewMaterial=()=>true,productExperimentName=String;
const ACTION_LABELS={},number=Number,esc=String,icon=()=>"",closeGlobalSearch=()=>panel.hidden=true;
''' + f'const routeParts=()=>[{route!r}],directoryPending={str(directory_pending).lower()};\n' + r'''
let revision=1,searchReads=0;
async function api(url){
  if(url==="/api/runs")return {runs:[{run_id:"R",experiment_id:"A",state:"completed"}]};
  searchReads++;return {items:[{archive_name:"A",event_id:"E1",event_uid:"uid-"+revision,
    parent_event_id:"G1",release_id:"r"+revision}],total_count:revision===1?2:7};
}
async function loadArchiveListing(){
  if(directoryPending){state.archiveListingQuery="old-query";throw Error("directory offline");}
  state.archiveRefreshPending=false;return true;
}
(async()=>{
  renderGlobalSearchResults(state.search);await new Promise(setImmediate);
  assert.ok(panel.innerHTML.includes("共找到 2 份"));assert.ok(panel.innerHTML.includes("release=r1"));
  revision=2;await refreshTaskSnapshots();await new Promise(setImmediate);
  assert.equal(searchReads,2,"an open search must refresh when a completed task invalidates its material results");
  assert.ok(panel.innerHTML.includes("共找到 7 份"));assert.ok(panel.innerHTML.includes("release=r2"));
  assert.ok(panel.innerHTML.includes("event_uid=uid-2"));assert.ok(!panel.innerHTML.includes("release=r1"));
  const html=panel.innerHTML;state.searchActiveIndex=1;
  await refreshTaskSnapshots();await new Promise(setImmediate);
  assert.equal(searchReads,2);assert.equal(panel.innerHTML,html);
  assert.equal(state.searchActiveIndex,1,"unchanged polls must preserve keyboard selection");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


@pytest.mark.parametrize("panel_state", ["open", "closed", "missing"])
def test_global_search_epoch_refresh_without_record_changes_respects_panel_visibility(panel_state):
    run_javascript(("refreshLibraryOverview", "globalMaterialSearchKey"), r'''
const state={search:"tube",archiveCacheEpoch:2,globalMaterialSearch:{key:'["tube",1]'}};
''' + f'const panelState={panel_state!r};\n' + r'''
const panel=panelState==="missing"?null:{hidden:panelState==="closed"};
const document={querySelector:()=>panel};
let renders=0;
const renderGlobalSearchResults=query=>{renders++;state.globalMaterialSearch.key=globalMaterialSearchKey(query)};
refreshLibraryOverview();assert.equal(renders,panelState==="open"?1:0);
refreshLibraryOverview();assert.equal(renders,panelState==="open"?1:0);
if(panel)assert.equal(panel.hidden,panelState==="closed","sync must not reopen a dismissed search");
''')


def test_global_material_search_covers_unloaded_archives_and_preserves_link_identity():
    run_javascript(("loadGlobalMaterialSearch", "globalMaterialSearchKey", "globalSearchGroups",
                    "materialFocusRoute", "experimentRecordRoute"), r'''
const state={archives:[],search:"移液器"},requests=[];
const experimentRecords=()=>state.archives;
const document={querySelector:()=>({hidden:true})};
const eventHasAlignedDualViewMaterial=event=>event.ready,productExperimentName=x=>x;
const ACTION_LABELS={object_movement:"物体移动"};
async function api(url){requests.push(url);return {items:[{archive_name:"Unloaded-B",event_id:"E29",event_uid:"uid-29",
  parent_event_id:"G2",release_id:"r2",action_type:"object_movement",ready:true}],total_count:40};}
(async()=>{
  await loadGlobalMaterialSearch(state.search);await loadGlobalMaterialSearch(state.search);
  assert.equal(requests.length,1,"repeat rendering must reuse the same search");
  const params=new URL(requests[0],"http://test").searchParams;
  assert.equal(params.has("archive"),false,"search must include archives outside the loaded listing");
  assert.equal(params.get("limit"),"5");assert.equal(params.get("material_ready"),"true");
  assert.equal(state.globalMaterialSearch.total,40);
  const groups=globalSearchGroups(state.search),item=groups.find(([name])=>name==="关键素材")[1][0];
  assert.ok(item.href.startsWith("#/archive/Unloaded-B/materials?"));
  const link=new URLSearchParams(item.href.split("?")[1]);
  assert.equal(link.get("focus"),"E29");assert.equal(link.get("event_uid"),"uid-29");
  assert.equal(link.get("parent_event_id"),"G2");assert.equal(link.get("release"),"r2");
  state.archiveCacheEpoch=1;assert.deepEqual(globalSearchGroups(state.search),[],"invalidated search must not remain visible");
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_global_search_finds_retained_experiments_without_promoting_them_to_reports():
    run_javascript(("globalSearchGroups", "experimentRecords", "experimentRecordRoute",
                    "archiveProductStatus", "cachedLibraryDetail", "libraryRecordKey",
                    "globalMaterialSearchKey"), r'''
const state={archives:[],runs:[{run_id:"retained-1",experiment_id:"Real-Experiment",state:"failed",
  nas_staging:"/nas/processing/real",observability:{stage_receipts:[{stage:"key_materials"}],
    status:{updated_at:"2026-09-07"}}}],archiveCache:new Map()};
const productExperimentName=String,experimentMetadata=()=>({});
state.archiveCache.set("stage/retained-1",{
  experiments:[{steps:[{current_step:"Real-Experiment pipette transfer"}]}],
  daily_report:{report_id:"partial-draft",report_date:"2026-09-07"}
});
const groups=globalSearchGroups("Real-Experiment");
assert.equal(groups.find(([name])=>name==="实验")[1][0].href,"#/stage/retained-1/experiments");
assert.ok(groups.find(([name])=>name==="实验")[1][0].meta.includes("待补全"));
assert.ok(groups.find(([name])=>name==="步骤")[1][0].meta.includes("阶段步骤"));
assert.ok(!groups.some(([name])=>name==="报告"),"retained drafts must not appear as published reports");
assert.ok(!groups.some(([name])=>name==="关键素材"),"retained candidates must not become formal material");
state.runs[0].result_available=false;
assert.deepEqual(globalSearchGroups("Real-Experiment"),[],"unavailable retained output must not be linked");
''')


def test_global_search_discards_old_responses_and_never_reopens_a_closed_panel():
    run_javascript(("loadGlobalMaterialSearch", "globalMaterialSearchKey"), r'''
const state={search:"old"},panel={hidden:false},requests=[],renders=[];
const document={querySelector:()=>panel},eventHasAlignedDualViewMaterial=()=>true;
const renderGlobalSearchResults=query=>renders.push(query);
const api=url=>new Promise(resolve=>requests.push({url,resolve}));
(async()=>{
  const old=loadGlobalMaterialSearch("old");state.search="new";
  const current=loadGlobalMaterialSearch("new");panel.hidden=true;
  requests[1].resolve({items:[{event_id:"new"}],total_count:1});await current;
  requests[0].resolve({items:[{event_id:"old"}],total_count:1});await old;
  assert.equal(state.globalMaterialSearch.items[0].event_id,"new");
  assert.deepEqual(renders,[]);assert.equal(panel.hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_global_material_search_failure_has_explicit_retry_and_keeps_scope_visible():
    run_javascript(("renderGlobalSearchResults", "loadGlobalMaterialSearch", "globalMaterialSearchKey"), r'''
const state={search:"tube"};let retry,fail=true,requests=0;
const input={setAttribute:()=>{},focus:()=>{}},panel={hidden:false,innerHTML:"",querySelectorAll:()=>[],
  querySelector:()=>({addEventListener:(_,callback)=>retry=callback})};
const document={querySelector:selector=>selector==="#global-search-results"?panel:input};
const globalSearchGroups=()=>[],number=x=>Number(x||0),eventHasAlignedDualViewMaterial=()=>true;
const api=async()=>{requests++;if(fail)throw Error("offline");return {items:[],total_count:0};};
(async()=>{
  renderGlobalSearchResults("tube");await new Promise(setImmediate);
  assert.ok(panel.innerHTML.includes("正式素材搜索暂不可用"));assert.ok(panel.innerHTML.includes("重新搜索"));
  assert.ok(!panel.innerHTML.includes("没有找到匹配内容"));
  assert.ok(panel.innerHTML.includes("实验、步骤和报告仅搜索已载入内容"));
  renderGlobalSearchResults("tube");await new Promise(setImmediate);assert.equal(requests,1,"failures must not loop");
  fail=false;retry();await new Promise(setImmediate);
  assert.equal(requests,2);assert.equal(state.globalMaterialSearch.status,"ready");
  assert.ok(panel.innerHTML.includes("没有找到匹配内容"));assert.ok(!panel.innerHTML.includes("暂不可用"));
})().catch(error=>{console.error(error);process.exitCode=1});
''')


def test_local_desktop_directory_input_renders_index_selection_and_upload_fallback():
    run_javascript(("renderNew",), r'''
const state={health:{storage_mode:"local",collection_ingest:{enabled:true,mode:"directory_metadata"}},nasBatches:[],collections:[],sources:[]};
const main={innerHTML:""},document={querySelector:()=>null};
const isNasMode=()=>false,setChrome=()=>{},roleGuidance=()=>({tone:"ok",text:"roles"}),selectedCollection=()=>null;
const icon=()=>"",esc=x=>String(x??""),sourceCard=()=>"",bindNewPage=()=>{},updateReview=()=>{};
const nasBatchPicker=()=>"INDEX_BATCH_SELECTION",nasRecordingPicker=()=>"USER_SELECTS_CAMERA_ROLES";
renderNew();
assert.ok(main.innerHTML.includes("INDEX_BATCH_SELECTION"));
assert.ok(main.innerHTML.includes("USER_SELECTS_CAMERA_ROLES"));
assert.ok(main.innerHTML.includes('id="batch-input"'));
state.health.collection_ingest.enabled=false;
renderNew();
assert.ok(!main.innerHTML.includes("INDEX_BATCH_SELECTION"));
assert.ok(main.innerHTML.includes('id="batch-input"'));
''')


def test_latest_check_does_not_hide_full_quality_failure_or_expand_all_observations():
    run_javascript(("resultReviewPanel", "evidenceParagraphs", "evidenceSentences"), r'''
const esc=String, number=Number, readerClock=String, archiveProcessStopped=()=>true;
const data={staging_run_id:"r",result_review:{available:true,revision:"1234567890",
  latest_check_current:true,step_consistency_passed:true,original_quality_passed:false,
  findings:[{kind:"unrecorded_interval"}],windows:[{window_id:"w",group_id:"g",start_ms:0,end_ms:30000}],
  gap_observations:[{group_id:"g",start_ms:0,end_ms:30000,sample_count:8,
    status:"supplementary_observations",observations:[{title:"拿起纸包",description:"双手拿起纸包；尚未打开。",frame_ids:["F001","F002"]}]}]}};
const html=resultReviewPanel(data);
assert.ok(html.includes("步骤引用与文字检查通过"));
assert.ok(html.includes("未通过（历史记录）"));
assert.ok(html.includes("仍需视频核验"));
assert.ok(html.includes('<details class="gap-observation">'));
assert.ok(html.includes("尚未打开。"));
assert.ok(!html.includes("<details open"));
''')


def test_stage_status_does_not_turn_skipped_or_failed_audio_green():
    run_javascript(("stageDisplayState", "guidedStageState"), r'''
const run={state:"mllm",observability:{status:{stage:"mllm",completed_stages:["alignment"]},stage_receipts:[
 {stage:"speech",status:"skipped",reason:"没有可用录音，视频分析继续"}]}};
assert.equal(stageDisplayState(run,"speech").label,"已跳过");
assert.equal(stageDisplayState(run,"mllm").label,"处理中");
assert.equal(stageDisplayState(run,"semantic_refinement").label,"待处理");
run.observability.stage_receipts[0].status="failed";
assert.equal(stageDisplayState(run,"speech").label,"未完成");
assert.equal(run.state,"mllm");
const definition={stages:["alignment","speech"],completedBy:["alignment"]};
run.state="speech";run.observability.status.stage="speech";
assert.equal(guidedStageState(run,definition,new Map([["alignment",{status:"completed"}]]),1).state,"active");
''')


def test_follow_navigation_only_uses_saved_results_and_never_cancels_analysis():
    run_javascript(("beginStageFollow", "followStageResults", "routeFromNavigation", "stageResultRoute"), r'''
const state={},location={hash:"#/new"},STAGE_LABELS={experiment_clips:"实验片段"};
let routes=0;const router=()=>{routes++},toast=()=>{},window={scrollTo(){}};
const run={run_id:"R",experiment_id:"A",state:"mllm",observability:{stage_receipts:[]}};
beginStageFollow("R");followStageResults(run);assert.equal(location.hash,"#/new");
run.observability.stage_receipts=[{stage:"speech",status:"skipped",completed_at:"1"},
 {stage:"experiment_clips",status:"completed",completed_at:"2"}];
followStageResults(run);assert.equal(location.hash,"#/stage/R/experiments");
routeFromNavigation();assert.equal(state.followRun.enabled,true);
assert.equal(run.state,"mllm","navigation cannot change the backend task state");
const before=routes;followStageResults(run);assert.equal(routes,before,"same receipt does not interrupt playback");
location.hash="#/materials";routeFromNavigation();assert.equal(state.followRun.enabled,false);
run.observability.stage_receipts.push({stage:"mllm",status:"completed",completed_at:"3"});
followStageResults(run);assert.equal(location.hash,"#/materials");
beginStageFollow("R");followStageResults({...run,run_id:"OTHER"});assert.equal(location.hash,"#/materials");
followStageResults(run);assert.equal(location.hash,"#/stage/R/materials");
''')


def test_closed_bad_nas_batch_is_actionable_instead_of_recording_forever():
    run_javascript(("nasBatchPicker", "captureQualityCopy"), r'''
const state={nasBatches:[{available:false,camera_count:2,issues:["采集程序报告数据不完整","采集相机视角待确认"],
 issue_details:[{camera_key:"cam01",issues:["采集程序报告数据不完整"],capture_quality:{reason:"rgb coverage below 98 percent",rgb_coverage_ratio:.22}}]}]};
const esc=String,number=String,duration=String,formatBytes=String,formatDate=String,icon=()=>"";
const html=nasBatchPicker();assert.ok(html.includes("需要处理"));assert.ok(html.includes("22.0%"));
assert.ok(html.includes("视频时间覆盖率低于 98%"));assert.ok(html.includes("采集相机视角待确认"));
assert.ok(!html.includes(">采集中<"));assert.ok(html.includes("disabled"));
''')


def test_run_disclosure_survives_snapshot_and_timer_changes_text_only():
    run_javascript(('rememberRunDisclosures','updateRunElapsedLabels','guidedPipelineView','guidedStageState','elapsedForRun','productRunMessage'), r'''
const state={taskDisclosures:new Map()},document={hidden:false},esc=String,icon=()=>'',duration=s=>String(Math.floor(s));
const STAGE_LABELS={},stageArtifactUrl=()=>'',stageResultRoute=()=>'/result';
const GUIDED_PIPELINE=[{id:'clips',number:'04',title:'理解实验步骤',doing:'处理中',stages:['experiment_clips'],completedBy:['experiment_clips'],resultTab:['experiments','实验片段']}];
const item={dataset:{runDisclosure:'R:outputs'},open:true};
rememberRunDisclosures({querySelectorAll:()=>[item]});
const run={run_id:'R',state:'experiment_clips',observability:{status:{stage:'experiment_clips'},stage_receipts:[]}};
assert.ok(guidedPipelineView(run).includes('data-run-disclosure="R:outputs" open'));
item.open=false;rememberRunDisclosures({querySelectorAll:()=>[item]});
assert.ok(!guidedPipelineView(run,true).includes('data-run-disclosure="R:outputs" open'));
const now=Date.now;Date.now=()=>100000;
const label={dataset:{elapsedUpdated:new Date(98000).toISOString(),elapsedSeconds:'58'},textContent:''};
let selector='';const root={querySelectorAll:s=>{selector=s;return [label];}};
updateRunElapsedLabels(root);assert.equal(label.textContent,'分析用时 60');
Date.now=()=>101000;updateRunElapsedLabels(root);assert.equal(label.textContent,'分析用时 61');
assert.ok(selector.includes("data-elapsed-running='true'"),'finished runs must not tick');
Date.now=now;
''')


def test_partial_empty_stages_are_not_published_as_experiments():
    run_javascript(('guidedStageState','guidedPipelineView','elapsedForRun','productRunMessage'), r'''
const esc=String,icon=()=>'',duration=String,STAGE_LABELS={},stageArtifactUrl=()=>'/receipt',stageResultRoute=()=>'/experiments';
const GUIDED_PIPELINE=[{id:'clips',number:'04',title:'理解实验步骤',doing:'处理中',stages:['experiment_understanding','experiment_clips'],completedBy:['experiment_clips'],resultTab:['experiments','实验片段']}];
const receipt={stage:'experiment_clips',status:'completed',receipt:'clips.json',stage_duration_seconds:.2};
const run={run_id:'R',state:'partial',progress:1,observability:{retained_experiment_count:0,status:{stage:'package'},stage_receipts:[receipt]}};
const html=guidedPipelineView(run,true);
assert.ok(html.includes('未生成实验成果'));assert.ok(html.includes('不代表视频中没有实验操作'));
assert.ok(!html.includes('已完成 100%'));assert.ok(!html.includes('href="/experiments"'));
const reports={id:'reports',stages:['daily_report','professional_pdf'],completedBy:['daily_report']};
run.observability.status.stage_outcomes={daily_report:{status:'blocked'}};
assert.equal(guidedStageState(run,reports,new Map(),6).state,'blocked');
run.state='completed';run.observability.retained_experiment_count=1;run.observability.metrics={stage_durations:[{stage:'experiment_understanding',duration_seconds:180},{stage:'experiment_clips',duration_seconds:20}]};
assert.ok(guidedPipelineView(run).includes('处理耗时 200'));
''')
