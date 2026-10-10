"use strict";

// Explicit application ports keep domain modules independently testable.
window.VisionCortexTasks = Object.freeze({
createTasks(context) {
  const { GUIDED_PIPELINE, STAGE_LABELS, api, archiveViewBusy, bindArchiveActions, delay, deviceDayQueueSection, duration, esc, experimentRecords, friendlyFailureReason, icon, invalidateArchiveCache, libraryRecordKey, loadArchiveListing, loadDeviceDayArchives, main, number, productExperimentName, refreshLibraryOverview, routeParts, router, setChrome, setPhase, setProgress, state, statusCard, toast, updateServiceChrome } = context;

async function refreshTaskSnapshots() {
  if (state.refreshingTasks || document.hidden) return;
  state.refreshingTasks = true;
  const runRequestId = state.runPollRequestId || 0;
  try {
    const deviceDaysChanged = Date.now() - state.deviceDayUpdatedAt > 30000 ? await loadDeviceDayArchives() : false;
    const previous = JSON.stringify(experimentRecords().map((record)=>[libraryRecordKey(record), record.material_revision, record.key_event_count]));
    const completed = new Set(state.runs.filter((run)=>run.state === "completed").map((run)=>run.run_id));
    try {
      const payload = await api("/api/runs");
      if (runRequestId !== (state.runPollRequestId || 0)) return;
      state.runs = payload.runs || [];
      state.taskSyncError = false;
      for (const run of state.runs.filter((run)=>run.state === "completed" && !completed.has(run.run_id))) {
        state.archiveRefreshPending = true;
        invalidateArchiveCache(run.experiment_id);
      }
    } catch {
      if (runRequestId !== (state.runPollRequestId || 0)) return;
      // Preserve the last task snapshot while continuing independent directory recovery.
      state.taskSyncError = true;
    }
    let archivesUpdated = false;
    if (state.archiveRefreshPending) {
      try {
        archivesUpdated = await loadArchiveListing();
      } catch {
        // The current directory request keeps its retry pending without blocking task status.
      }
    }
    if (state.activeRun?.run_id) {
      state.activeRun = state.runs.find((run) => run.run_id === state.activeRun.run_id) || state.activeRun;
    }
    if (state.archiveView?.staging && !state.archiveView.loading) {
      const run = state.runs.find(item=>item.run_id === state.archiveView.name);
      if (run?.observability && stageSnapshotVersion(run.observability) !== state.archiveView.stageVersion) state.archiveView.refreshPending = true;
    }
    if (state.archiveView?.staging && !state.archiveView.loading) {
      const run = state.runs.find(item=>item.run_id === state.archiveView.name);
      if (run?.observability && stageSnapshotVersion(run.observability) !== state.archiveView.stageVersion) state.archiveView.refreshPending = true;
    }
    updateServiceChrome();
    if (state.followRun) followStageResults(state.runs.find(run=>run.run_id === state.followRun.runId));
    const route = routeParts()[0] || "home";
    if (route === "tasks") renderTasks();
    const current = JSON.stringify(experimentRecords().map((record)=>[libraryRecordKey(record), record.material_revision, record.key_event_count]));
    refreshLibraryOverview(deviceDaysChanged || archivesUpdated || current !== previous);
  } catch {
    if (runRequestId !== (state.runPollRequestId || 0)) return;
    state.taskSyncError = true;
    if (routeParts()[0] === "tasks") renderTasks();
    refreshLibraryOverview();
  } finally {
    state.refreshingTasks = false;
  }
}


function renderStages(activeStage, progressValue) {
  const stages = ["preflight","alignment","speech","motion_probe","candidate_coarse","candidate_fine","candidate_audit","experiment_understanding","experiment_clips","key_materials","mllm","material_refinement","semantic_refinement","package","daily_report","professional_pdf","finalizing"];
  if (!stages.includes(activeStage) && !["completed","partial","failed","interrupted"].includes(activeStage)) stages.unshift(activeStage);
  const element = document.querySelector("#run-stages");
  if (!element) return;
  const run = state.activeRun || {state: activeStage};
  element.innerHTML = stages.map(stage => {
    const outcome = stageDisplayState(run, stage, activeStage);
    return `<div class="${outcome.state}" title="${esc(outcome.reason || "")}" ${outcome.state === "active" ? 'aria-current="step"' : ""}><i aria-hidden="true">${outcome.symbol}</i><span>${esc(STAGE_LABELS[stage] || stage)}</span><small>${outcome.label}</small>${outcome.reason ? `<p>${esc(outcome.reason)}</p>` : ""}</div>`;
  }).join("");
  setProgress(progressValue, run.message || STAGE_LABELS[activeStage] || activeStage);
}


function stageDisplayState(run, stage, activeStage = run.observability?.status?.stage || run.state) {
  const snapshot = run.observability || {}, status = snapshot.status || {};
  const receipt = (snapshot.stage_receipts || []).find(item=>item.stage === stage);
  const outcome = receipt || status.stage_outcomes?.[stage];
  if (outcome?.status === "blocked") return {state:"blocked",label:"等待依赖",symbol:"↳",reason:outcome.reason};
  if (outcome?.status === "completed" && outcome.archive_status === "pending") return {state:"pending",label:"已完成 · 待归档",symbol:"↑",reason:outcome.reason};
  if (outcome?.status === "skipped") return {state:"skipped",label:"已跳过",symbol:"−",reason:outcome.reason};
  if (outcome?.status === "failed") return {state:"failed",label:"未完成",symbol:"!",reason:outcome.reason};
  if (["failed","interrupted"].includes(run.state) && status.failed_stage === stage) return {state:"failed",label:"已停止",symbol:"!"};
  if (activeStage === stage) return {state:"active",label:stage === "queued" ? "排队中" : "处理中",symbol:"↻"};
  if (outcome?.status === "completed") return {state:"done",label:outcome.reused ? "已复用" : "已完成",symbol:"✓"};
  if ((status.completed_stages || []).includes(stage)) return {state:"passed",label:"已通过",symbol:"✓"};
  return {state:"waiting",label:["completed","partial","failed","interrupted"].includes(run.state) ? "未执行" : "待处理",symbol:"·"};
}


async function pollRun(runId, archiveName) {
  let consecutiveReadFailures = 0;
  const requestId = (state.runPollRequestId || 0) + 1;
  state.runPollRequestId = requestId;
  const isCurrent = () => state.activeRun?.run_id === runId && state.runPollRequestId === requestId;
  while (isCurrent()) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    if (!isCurrent()) return;
    let run;
    try {
      run = await api(`/api/runs/${encodeURIComponent(runId)}`);
      consecutiveReadFailures = 0;
    } catch (error) {
      if (!isCurrent()) return;
      consecutiveReadFailures += 1;
      const retrySeconds = Math.min(10, 2 ** Math.min(3, consecutiveReadFailures));
      const progress = .15 + Number(state.activeRun?.progress || 0) * .85;
      setProgress(
        progress,
        `暂时无法读取最新任务状态，已保留上次进度；连接恢复后自动继续同步`,
      );
      await delay(retrySeconds * 1000);
      continue;
    }
    if (!isCurrent()) return;
    rememberRunSnapshot(run);
    if (state.followRun) followStageResults(run);
    if (["completed", "partial", "failed", "interrupted"].includes(run.state)) state.runPollRequestId += 1;
    setPhase(run);
    renderStages(run.state, .15 + Number(run.progress || 0) * .85);
    if (run.state === "completed") {
      setProgress(1, run.parent_run_id ? "所选阶段结果已更新" : "分析完成，全部实验成果已保存");
      toast(run.parent_run_id ? "所选阶段已更新，可回到原实验查看。" : "实验分析完成，结果已保存。", "");
      state.archiveRefreshPending = true;
      invalidateArchiveCache(archiveName);
      updateServiceChrome();
      if (routeParts()[0] === "new") location.hash = `#/archive/${encodeURIComponent(archiveName)}/experiments`;
      else if (routeParts()[0] === "tasks") renderTasks();
      return;
    }
    if (run.state === "partial") {
      setProgress(1, "分析结束，阶段成果与证据缺口已保存");
      toast("分析结束，可查看已完成成果和阶段报告。", "");
      invalidateArchiveCache(archiveName);
      updateServiceChrome();
      document.querySelector("#start-run")?.removeAttribute("disabled");
      if (routeParts()[0] === "new") location.hash = `#/stage/${encodeURIComponent(runId)}/experiments`;
      else if (routeParts()[0] === "tasks") renderTasks();
      return;
    }
    if (["failed", "interrupted"].includes(run.state)) {
      const failure = friendlyFailureReason(run);
      toast(failure, "error");
      setProgress(1, `失败：${failure}`);
      document.querySelector("#start-run")?.removeAttribute("disabled");
      updateServiceChrome();
      if (routeParts()[0] === "tasks") renderTasks();
      return;
    }
  }
}


function rememberRunSnapshot(run) {
  state.activeRun = run;
  state.runs = [run, ...state.runs.filter((item)=>item.run_id !== run.run_id)];
}


function showAcceptedRun(result, archiveName = "") {
  state.runPollRequestId = (state.runPollRequestId || 0) + 1;
  const previous = state.runs.find((run)=>run.run_id === result.run_id);
  const run = {
    state: "queued", progress: 0,
    experiment_id: result.experiment_id || result.archive_name || archiveName || previous?.experiment_id,
    source_collection_id: result.source_collection_id || previous?.source_collection_id,
    ...result,
  };
  rememberRunSnapshot(run);
  state.archiveRefreshPending = true;
  updateServiceChrome();
  location.hash = "#/tasks";
  renderTasks();
}


function renderTasks() {
  rememberRunDisclosures();
  setChrome("tasks");
  const terminal = new Set(["completed", "partial", "failed", "interrupted"]);
  const runs = [...state.runs].sort((a, b) => Number(terminal.has(a.state)) - Number(terminal.has(b.state)) || String(b.updated_at || b.observability?.status?.updated_at || "").localeCompare(String(a.updated_at || a.observability?.status?.updated_at || "")));
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">实验分析</p><h1>任务进度</h1><p>查看实验处理状态和当前所在环节。</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header>${state.taskSyncError ? `<div class="freshness-warning" role="alert">任务状态同步中断，当前显示最近收到的数据；连接恢复后会自动更新。</div>` : state.archiveRefreshPending ? `<div class="freshness-warning" role="status">任务状态已更新，实验档案目录仍在同步；系统会自动重试。</div>` : ""}${deviceDayQueueSection()}<section class="status-grid">${statusCard("activity","手动分析任务",number(runs.length),"")}${statusCard("gauge","正在分析",number(runs.filter((run)=>!["completed","partial","failed","interrupted"].includes(run.state)).length),"")}${statusCard("check","已归档",number(runs.filter((run)=>run.state==="completed"&&!run.parent_run_id).length),"")}${statusCard("file","需要关注",number(runs.filter((run)=>["partial","failed","interrupted"].includes(run.state)).length),"")}</section>${runs.length ? runs.map((run)=>runObservabilityCard(run,false)).join("") : `<section class="panel"><div class="empty-state"><span class="empty-illustration" aria-hidden="true">${icon("activity")}</span><strong>暂无分析任务</strong><a class="secondary-button" href="#/new">${icon("plus")}新建实验</a></div></section>`}</div>`;
  bindArchiveActions();
  document.querySelectorAll("[data-retry-run]").forEach((button) => button.addEventListener("click", () => retryRetainedRun(button.dataset.retryRun, button)));
}


async function retryRetainedRun(runId, button) {
  if (button.disabled) return;
  button.disabled = true;
  try {
    const plan = await api(`/api/runs/${encodeURIComponent(runId)}/recovery-plan`);
    document.getElementById("recovery-dialog")?.remove();
    const dialog=document.createElement("dialog");dialog.id="recovery-dialog";dialog.className="recovery-dialog";
    dialog.innerHTML=`<header><div><p class="eyebrow">继续处理已保存的实验</p><h2>恢复与补全</h2></div><button class="dialog-close" type="button" data-recovery-close aria-label="关闭恢复方案">${icon("x")}</button></header><p>已保存 ${number(plan.retained_stage_count)} 个阶段回执。原任务模型：${esc(plan.provider||"未记录")} · ${esc(plan.model||"未记录")}。</p>${plan.model_ready?"":`<p class="recovery-model-notice">${esc(plan.model_check_message)} <a href="#/ai-settings" data-recovery-settings>前往 AI 服务设置</a></p>`}<div class="recovery-options"><button type="button" data-recovery-action="resume" ${plan.actions.resume?'':'disabled'}><strong>继续未完成环节</strong><span>保留现有视频和产物；逐环节校验后继续处理。${number((plan.checkpoint_stages||[]).length)} 个环节有恢复点，实际复用以执行记录为准。</span></button><button type="button" data-recovery-action="operations" ${plan.actions.operations?'':'disabled'}><strong>重新整理操作步骤</strong><span>使用已审核事件和保存画面，整理具体操作、减少重复记录。可能产生模型费用。</span></button><button type="button" data-recovery-action="reports" ${plan.actions.reports?'':'disabled'}><strong>更新已有结果的报告</strong><span>不调用模型；质量未通过时更新阶段报告。</span></button><button type="button" data-recovery-action="retry" ${plan.actions.retry?'':'disabled'}><strong>复跑完整流程</strong><span>使用原输入，重新检查并补全各环节；校验通过的缓存继续使用。</span></button></div><details><summary>复用范围、历史产出与费用说明</summary><p>${esc(plan.cache_policy)}</p><p>${esc(plan.resume_effect || plan.retry_effect)}</p><p>${esc(plan.model_cost)}</p><p>${esc(plan.quality_policy)}</p><p class="recovery-path">产出路径：${esc(plan.output_path)}</p></details><p role="status" data-recovery-status>请选择本次需要处理的范围。</p>`;
    document.body.append(dialog);dialog.showModal();
    dialog.querySelector('[data-recovery-close]').addEventListener('click',()=>dialog.close());
    dialog.querySelector('[data-recovery-settings]')?.addEventListener('click',()=>dialog.close());
    dialog.addEventListener('close',()=>{dialog.remove();button.focus();},{once:true});
    dialog.querySelectorAll('[data-recovery-action]').forEach(action=>action.addEventListener('click',async()=>{
      dialog.querySelectorAll('[data-recovery-action]').forEach(b=>{b.disabled=true;});
      const status=dialog.querySelector('[data-recovery-status]');status.textContent='正在校验当前版本并提交任务…';
      const scope=action.dataset.recoveryAction;
      try {
        await submitRecoveryAction(runId,scope,plan,action);
        dialog.close();
      } catch(error) {
        status.textContent=error.message+'。关闭后重新打开可获取最新方案。';
      }
    }));
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}


async function submitRecoveryAction(runId, scope, plan, button) {
  if (button.dataset?.submitting === 'true') return;
  if(button.dataset)button.dataset.submitting='true';
  button.disabled=true;
  const base=`/api/runs/${encodeURIComponent(runId)}`;
  const url=['retry','resume'].includes(scope)?`${base}/retry?revision=${encodeURIComponent(plan.revision)}&mode=${scope==='resume'?'resume':'full'}`:`${base}/refresh/${scope}${scope==='operations'?`?revision=${encodeURIComponent(plan.group_revision)}`:''}`;
  const result=await api(url,{method:'POST'});
  state.archiveCache.clear();showAcceptedRun(result);beginStageFollow(result.run_id);
  toast('任务已排队，可在任务进度中查看实际复用与新增用量。','success');
}


function normalizedViews(status) {
  const views = status?.views || {};
  if (Array.isArray(views)) return views.map((item,index)=>[item.view_id || `view-${index+1}`, item]);
  return Object.entries(views);
}


function newestFreshness(snapshot) {
  const candidates = [
    snapshot?.freshness?.telemetry_updated_at,
    snapshot?.freshness?.source_progress_updated_at,
    snapshot?.freshness?.status_updated_at,
  ].filter(Boolean).map((value)=>new Date(value)).filter((value)=>!Number.isNaN(value.getTime()));
  if (!candidates.length) return { value: null, age: null, stale: false };
  const value = new Date(Math.max(...candidates.map((item)=>item.getTime())));
  const age = Math.max(0, (Date.now() - value.getTime()) / 1000);
  return { value, age, stale: age > 20 };
}


function elapsedForRun(run, status) {
  let value = Number(status?.elapsed_seconds || 0);
  const updated = status?.updated_at ? new Date(status.updated_at) : null;
  if (!["completed","partial","failed","interrupted"].includes(run.state) && updated && !Number.isNaN(updated.getTime())) {
    value += Math.max(0, (Date.now() - updated.getTime()) / 1000);
  }
  return value;
}


function rememberRunDisclosures(root = main) {
  root.querySelectorAll("details[data-run-disclosure]").forEach(item => {
    state.taskDisclosures.set(item.dataset.runDisclosure, item.open);
  });
}


function updateRunElapsedLabels(root = main) {
  if (document.hidden) return;
  root.querySelectorAll("[data-run-elapsed][data-elapsed-running='true']").forEach(item => {
    const updated = Date.parse(item.dataset.elapsedUpdated);
    if (!Number.isFinite(updated)) return;
    const seconds = Number(item.dataset.elapsedSeconds) + Math.max(0, (Date.now() - updated) / 1000);
    item.textContent = `分析用时 ${duration(Math.floor(seconds))}`;
  });
}


function guidedStageState(run, definition, receipts, index) {
  const status = run.observability?.status || {};
  const current = status.stage || run.state;
  const outcome = stage => receipts.get(stage) || status.stage_outcomes?.[stage];
  if (run.state === "partial" && definition.stages.includes("package")) return { state: "partial", receipt: null };
  const failed = status.failed_stage || (run.state === "failed"
    ? run.observability?.metrics?.nas_index_ingest?.failure_stage || current : null);
  if (["failed","interrupted"].includes(run.state) && definition.stages.includes(failed)) return { state: run.state, receipt: null };
  const componentFailure = definition.stages.map(outcome).find(item=>["failed","blocked"].includes(item?.status));
  if (componentFailure) return {state:componentFailure.status,receipt:componentFailure};
  const completedReceipt = definition.completedBy.map(outcome).find(item=>item?.status === "completed");
  if (!["completed","partial","failed","interrupted"].includes(run.state) && definition.stages.includes(current)) return { state: "active", receipt: completedReceipt };
  if (definition.id === "reports" && run.state !== "completed") return {state:"withheld",receipt:null};
  if (["partial","failed","interrupted"].includes(run.state) && run.observability?.retained_experiment_count === 0
      && ["discovery","clips","materials"].includes(definition.id) && completedReceipt) return {state:"empty",receipt:completedReceipt};
  if (completedReceipt?.archive_status === "pending") return {state:"pending",receipt:completedReceipt};
  if (completedReceipt) return { state: "done", receipt: completedReceipt };
  if (run.state === "partial") return { state: "withheld", receipt: null };
  const currentIndex = GUIDED_PIPELINE.findIndex((item)=>item.stages.includes(current));
  if (currentIndex > index) return { state: "done-unreceipted", receipt: null };
  return { state: "waiting", receipt: null };
}


function stageArtifactUrl(run, relativePath) {
  if (!relativePath) return "";
  const query = new URLSearchParams();
  if (run.state === "completed") {
    query.set("archive", run.experiment_id || "");
    query.set("path", relativePath);
    return `/api/archive-file?${query}`;
  }
  query.set("run_id", run.run_id);
  query.set("path", relativePath);
  return `/api/staging-file?${query}`;
}


function stageResultRoute(run, tabName) {
  return run.state === "completed"
    ? `#/archive/${encodeURIComponent(run.experiment_id || "")}/${tabName}`
    : `#/stage/${encodeURIComponent(run.run_id)}/${tabName}`;
}


function guidedPipelineView(run, outputsOpen = false) {
  const disclosureKey = `${run.run_id}:outputs`;
  outputsOpen = typeof state !== "undefined" ? state.taskDisclosures?.get(disclosureKey) ?? outputsOpen : outputsOpen;
  const snapshot = run.observability || {};
  const receipts = new Map((snapshot.stage_receipts || []).map((item)=>[item.stage,item]));
  const rows = GUIDED_PIPELINE.map((definition,index)=>({ definition, ...guidedStageState(run,definition,receipts,index) }));
  const active = rows.find((item)=>["active","partial","failed","interrupted"].includes(item.state)) || rows.find((item)=>item.state === "waiting") || rows.at(-1);
  const activeIndex = rows.indexOf(active);
  const next = rows.slice(activeIndex+1).find((item)=>item.state === "waiting");
  const status = snapshot.status || {};
  const stateLabel = { active:"正在进行", done:"已完成", empty:"未生成实验成果", pending:"已完成 · 待归档", "done-unreceipted":"已通过", waiting:"等待中", partial:"证据不足", withheld:"未发布", failed:"本环节失败", blocked:"未生成 · 缺少依赖", interrupted:"等待续跑" };
  const cards = rows.map(({definition,state:stageState,receipt})=>{
    const deliveredReceipts = definition.stages.map((stage)=>receipts.get(stage)).filter(Boolean);
    const artifacts = deliveredReceipts.flatMap((item)=>item.artifacts || []).filter((item)=>item.available);
    const filesByName = new Map(artifacts.filter((item)=>item.kind === "file" && item.relative_path).map((item)=>[item.name,item]));
    const featured = (definition.outputs || []).map(([name,label])=>({ item: filesByName.get(name), label })).filter(({item})=>item);
    const latestReceipt = receipt || deliveredReceipts.at(-1);
    const receiptUrl = latestReceipt ? stageArtifactUrl(run, latestReceipt.receipt) : "";
    const resultLink = definition.resultTab && !["empty","blocked","withheld"].includes(stageState) && deliveredReceipts.some(item=>item.status === "completed")
      ? `<a href="${esc(stageResultRoute(run,definition.resultTab[0]))}">${icon("arrow")}${esc(definition.resultTab[1])}</a>` : "";
    const artifactLinks = deliveredReceipts.length ? `<div class="journey-artifacts">${resultLink}${featured.map(({item,label})=>`<a target="_blank" href="${esc(stageArtifactUrl(run,item.relative_path))}">${icon("file")}${esc(label)}</a>`).join("")}${receiptUrl ? `<a class="receipt-link" target="_blank" href="${esc(receiptUrl)}">${icon("check")}完整清单</a>` : ""}</div>` : "";
    const timings = (snapshot.metrics?.stage_durations || []).filter(item=>definition.stages.includes(item.stage));
    const seconds = timings.length ? timings.reduce((sum,item)=>sum + Number(item.duration_seconds || 0),0) : receipt?.stage_duration_seconds;
    const detail = stageState === "done"
      ? `${seconds != null ? `${definition.id === "originals" ? "输入准备（含上传与等待）" : "处理耗时"} ${duration(seconds)}` : "完成记录已保存"}`
      : stageState === "empty" ? "未得到通过分组检查的实验；保存了筛选记录，没有生成对应实验视频或步骤成果。"
      : stageState === "pending" ? "计算已完成，正在等待文件同步归档。"
      : stageState === "blocked" ? "本环节未执行，缺少必要结果；阶段报告与正式报告分开保留。"
      : stageState === "withheld" ? "完整质量检查通过后生成；当前阶段成果可正常查看。"
      : stageState === "partial" ? "分析已结束，阶段成果和具体证据缺口已保存。"
      : stageState === "active" ? definition.doing
      : stageState === "failed" ? `停止位置：${STAGE_LABELS[status.failed_stage] || "当前分析环节"}`
      : stageState === "interrupted" ? "已完成内容会保留，可以稍后重新分析。"
      : stageState === "done-unreceipted" ? "该环节已通过，后续分析已经开始。"
      : "等待前序环节完成";
    const componentNotes = deliveredReceipts.filter(item=>["skipped","failed","blocked"].includes(item.status)).map(item=>`<p class="component-outcome">${esc(STAGE_LABELS[item.stage] || item.stage)} · ${item.status === "skipped" ? "已跳过" : item.status === "blocked" ? "等待依赖" : "未完成"}：${esc(item.reason || "请查看该环节记录")}</p>`).join("");
    return `<article class="journey-step ${stageState}"><span class="journey-number">${definition.number}</span><div class="journey-copy"><header><strong>${esc(definition.title)}</strong><span>${esc(stateLabel[stageState])}</span></header><p>${esc(detail)}</p>${componentNotes}${artifactLinks}</div></article>`;
  }).join("");
  return `<section class="current-guide ${active.state}"><div><span class="guide-kicker">${esc(stateLabel[active.state])} · 第 ${esc(active.definition.number)} 步，共 7 步</span><h3>${esc(active.definition.title)}</h3><p>${esc(productRunMessage(run, active.definition.doing))}</p></div><aside><small>${run.state === "partial" ? "本次结果" : next ? "接下来" : "最终结果"}</small><strong>${esc(run.state === "partial" ? "分析记录已保存，完整结论未发布" : next?.definition.title || "实验成果已完整保存")}</strong></aside></section><div class="task-progress-line"><span>${["failed", "interrupted"].includes(run.state) ? "处理已停止" : run.state === "partial" ? "处理已结束 · 完整分析未通过" : `已完成 ${Math.round(Number(run.progress || 0) * 100)}%`}</span><span data-run-elapsed data-elapsed-seconds="${Number(status.elapsed_seconds || 0)}" data-elapsed-updated="${esc(status.updated_at || "")}" data-elapsed-running="${!["completed","partial","failed","interrupted"].includes(run.state)}">分析用时 ${duration(elapsedForRun(run, status))}</span></div><details class="technical-observability" data-run-disclosure="${esc(disclosureKey)}" ${outputsOpen ? "open" : ""}><summary>查看各环节与生成文件</summary><div class="pipeline-journey">${cards}</div></details>`;
}


function runObservabilityCard(run, outputsOpen = false) {
  if(run.parent_run_id) {
    const names={result_check:"最新结果检查",gap_review:"缺口补充分析",operations:"操作步骤整理",reports:"报告更新",understanding:"录音理解更新",timeline:"时间轴更新",capture_quality:"采集质量复核",search:"检索索引更新"};
    const receipt=run.refresh_receipt||{}, usage=receipt.additional_usage||{};
    const finished=run.state==='completed';
    const details=receipt.dependencies||{};
    return `<section class="panel live-run-card"><header class="panel-heading"><div><p class="panel-kicker">所选阶段处理</p><h2>${esc(names[run.refresh_scope||run.scope]||"实验结果更新")}</h2></div><a class="secondary-button" href="#/stage/${encodeURIComponent(run.parent_run_id)}/experiments">查看原实验</a></header><p>${esc(finished?"所选阶段结果已更新":run.state==="failed"?"本次更新未完成":"正在处理所选阶段")}</p>${finished?`<p>${Number.isInteger(details.accepted_group_count)?`${number(details.accepted_group_count)} / ${number(details.group_count)} 个片段的操作整理通过证据校验。`:''}原实验的完整性与质量状态保持原记录。</p><div class="status-grid">${statusCard("clock","本次处理用时",duration(receipt.wall_seconds),"")}${statusCard("activity","新模型请求",number(receipt.model_invocations),"含重试")}${statusCard("file","新增 Token",usage.total_tokens==null?"未记录":number(usage.total_tokens),"以回执为准")}</div>`:run.state==='failed'?`<p role="alert">${esc(friendlyFailureReason(run))}</p>`:'<p>使用原实验已保存内容；完成后可回到原实验查看结果与本次用量。</p>'}</section>`;
  }
  const snapshot = run.observability || {};
  const status = snapshot.status || {};
  const freshness = newestFreshness(snapshot);
  const current = status.stage || run.state;
  const freshnessNote = freshness.stale && !["completed","partial","failed","interrupted"].includes(run.state)
    ? `<div class="freshness-warning">任务状态已 ${duration(freshness.age)} 未更新；分析可能仍在后台继续，页面会自动刷新。</div>` : "";
  const partial = snapshot.partial_delivery?.status;
  // Legacy receipts classified optional missing audio as a capture defect.
  const optionalAudioNotes = new Set(["此路没有可用录音","此路没有音轨"]);
  const audioOptional = status.stage_outcomes?.speech?.status === "skipped";
  const captureWarnings = (snapshot.capture_quality?.records || []).flatMap(item=>(item.warnings || [])
    .filter(message=>!audioOptional || !optionalAudioNotes.has(message)).map(message=>`${item.view_id}：${message}`));
  const audioNote = audioOptional ? '<p class="component-outcome">录音已跳过 · 视频分析不受影响</p>' : "";
  const captureKey = `${run.run_id}:capture`;
  const captureOpen = typeof state !== "undefined" ? state.taskDisclosures?.get(captureKey) ?? true : true;
  const captureNotice = captureWarnings.length ? `<details class="analysis-readiness-note" data-run-disclosure="${esc(captureKey)}" ${captureOpen ? "open" : ""}><summary>采集质量预检发现问题（仅限抽样）</summary>${captureWarnings.map(message=>`<p>${esc(message)}</p>`).join("")}</details>` : "";
  const retryButton = ["partial", "failed", "interrupted"].includes(run.state) && run.nas_staging && run.retry_available !== false
    ? `<button class="secondary-button" data-retry-run="${esc(run.run_id)}" type="button">${icon("refresh")}继续未完成环节</button>` : "";
  const previewAvailable = (snapshot.stage_receipts || []).some((receipt)=>
    ["experiment_understanding", "experiment_clips", "key_materials", "mllm", "material_refinement", "semantic_refinement", "package", "daily_report"].includes(receipt.stage));
  const resultRoute = run.nas_staging && run.state !== "completed"
    ? `stage/${encodeURIComponent(run.run_id)}` : `archive/${encodeURIComponent(run.experiment_id || "")}`;
  const archiveLink = run.state === "completed" || previewAvailable
    ? `<a class="secondary-button" href="#/${resultRoute}/experiments">${run.state === "completed" ? "查看实验结果" : "预览已完成内容"}</a>` : "";
  return `<section class="panel live-run-card"><header class="panel-heading"><div><p class="panel-kicker">实验分析</p><h2>${esc(productExperimentName(run.experiment_id || run.run_id))}</h2></div><div class="run-heading-actions">${archiveLink}${retryButton}<span class="queue-status ${esc(run.state)}"><i></i>${esc(run.state === "failed" && partial ? "阶段产出已保存 · 待补全" : STAGE_LABELS[current] || (run.state === "failed" ? "处理失败" : "处理中"))}</span></div></header>${freshnessNote}${captureNotice}${audioNote}${guidedPipelineView(run,outputsOpen)}</section>`;
}


function productRunMessage(run, fallback) {
  if (run?.state === "partial" && run.observability?.retained_experiment_count === 0) return "本次未生成实验片段。筛选与诊断记录已保存；这不代表视频中没有实验操作。";
  if (run?.state === "partial") return "分析结束，已完成成果和证据缺口均已保存，可查看阶段报告。";
  if (["failed", "interrupted"].includes(run?.state)) return friendlyFailureReason(run);
  const raw = String(run?.observability?.status?.message || "").trim();
  if (!raw || raw.length > 180 || /(?:runtimeerror|traceback|exception|[a-z]:\\|\/(?:mnt|home|tmp|var)\/)/i.test(raw)) return fallback;
  return raw;
}


function beginStageFollow(runId) {
  state.followRun = {runId, enabled: true, seen: new Set(), expectedHash: location.hash};
}


function followStageResults(run) {
  const follow = state.followRun;
  if (!run || !follow?.enabled || follow.runId !== run.run_id) return;
  const receipts = (run.observability?.stage_receipts || []).filter(item=>item.status === "completed" && !item.reused)
    .sort((a,b)=>String(a.completed_at || "").localeCompare(String(b.completed_at || "")));
  const fresh = receipts.filter(item=>!follow.seen.has(`${item.stage}:${item.completed_at}`))
    .sort((a,b)=>String(a.completed_at || "").localeCompare(String(b.completed_at || "")));
  const finished = run.state === "completed" && !follow.finished;
  const latest = fresh.at(-1) || (finished ? receipts.at(-1) : null);
  if (!latest) return;
  if (state.archiveView && archiveViewBusy()) return;
  for (const item of receipts) follow.seen.add(`${item.stage}:${item.completed_at}`);
  if (finished) follow.finished = true;
  const tabs = {speech:"speech", experiment_understanding:"experiments", experiment_clips:"experiments", key_materials:"materials", mllm:"materials", material_refinement:"materials", semantic_refinement:"experiments", stage_report:"reports", package:"metrics", daily_report:"reports", professional_pdf:"reports", finalizing:"reports", capture_quality:"metrics", alignment:"metrics", motion_probe:"metrics", candidate_coarse:"metrics", candidate_fine:"materials", candidate_audit:"experiments"};
  const target = tabs[latest.stage] ? stageResultRoute(run, tabs[latest.stage]) : "#/tasks";
  toast(`${STAGE_LABELS[latest.stage] || latest.stage}已完成，产出已保存。`, "");
  if (location.hash === target) { void router(); return; }
  follow.expectedHash = target;
  location.hash = target;
}


function stageDeliveryView(data) {
  const receipts = (data.observability?.stage_receipts || []).filter(item=>item.receipt_url);
  if (!receipts.length) return "";
  const ordered = [...receipts].sort((a,b)=>(Date.parse(a.completed_at)||0)-(Date.parse(b.completed_at)||0));
  return `<section class="stage-delivery" aria-label="已保存的阶段成果"><header><div><strong>阶段成果</strong><p>各环节独立留存，后续处理不会清除已保存记录。</p></div><label>选择环节<select data-stage-version>${ordered.map((item,index)=>`<option value="${index}" ${index===ordered.length-1?"selected":""}>${esc(STAGE_LABELS[item.stage]||item.stage)} · ${item.status==="completed"?"已保存":item.status==="skipped"?"已跳过":"未完成"}</option>`).join("")}</select></label></header>${ordered.map((item,index)=>`<div data-stage-version-panel="${index}" ${index===ordered.length-1?"":"hidden"}><p>${esc(item.reason||"本阶段文件已保存，可在后续分析进行时查看。")}</p><nav aria-label="阶段文件">${item.version_url?`<a class="secondary-button" href="${esc(item.version_url)}" target="_blank" rel="noopener">当时的结果与溯源</a>`:""}<a class="text-button" href="${esc(item.receipt_url)}" target="_blank" rel="noopener">完成记录</a>${(item.artifacts||[]).filter(x=>x.url).slice(0,3).map(x=>`<a class="text-button" href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.name)}</a>`).join("")}</nav>${!item.version_url?'<small>此历史阶段的文件与完成记录可查看，未提供独立版本索引。</small>':""}</div>`).join("")}</section>`;
}


function stageSnapshotVersion(snapshot) {
  return JSON.stringify([snapshot?.status?.stage, (snapshot?.stage_receipts || []).map(item=>[item.stage,item.status,item.completed_at])]);
}


  return Object.freeze({ refreshTaskSnapshots, renderStages, stageDisplayState, pollRun, rememberRunSnapshot, showAcceptedRun, renderTasks, retryRetainedRun, submitRecoveryAction, normalizedViews, newestFreshness, elapsedForRun, rememberRunDisclosures, updateRunElapsedLabels, guidedStageState, stageArtifactUrl, stageResultRoute, guidedPipelineView, runObservabilityCard, productRunMessage, beginStageFollow, followStageResults, stageDeliveryView, stageSnapshotVersion });
}
});
