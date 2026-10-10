"use strict";

// Explicit application ports keep domain modules independently testable.
window.VisionCortexMaterials = Object.freeze({
createMaterials(context) {
  const { ACTION_LABELS, api, archiveProcessStopped, archiveSyncNotice, archiveWithinDate, bindArchivePagination, bindLibraryRetry, bindVideoPreviews, cachedArchiveDetail, cachedLibraryDetail, closeGlobalSearch, duration, ensureLibraryDetails, esc, eventObjectValues, experimentRecordRoute, experimentRecords, formatDate, icon, libraryCardSkeleton, libraryFailureNotice, libraryLoadState, libraryQueryKey, libraryRecordKey, libraryRecordQueryKey, loadArchiveMaterials, loadLibraryDetail, main, materialObjectItems, nextStepLabel, number, productEvidenceText, productExperimentName, productObjectLabel, productState, renderArchive, routeParts, routeQuery, setChrome, sourceCard, state, timecode, toast, verificationTrace, videoPosterUrl, videoPreview, workflowCompletionLabel, workflowLabel } = context;

function materialLibraryEntries() {
  const entries = [];
  experimentRecords().forEach((archive) => {
    const detail = cachedLibraryDetail(archive);
    if (!detail) return;
    if (detail.libraryKeys?.materials !== libraryRecordQueryKey(archive, "materials")) return;
    const formal = archive.staging_run_id ? [] : (detail.key_events || []).filter(eventHasAlignedDualViewMaterial);
    formal.forEach((event) => entries.push({archive, detail, event, preliminary: false}));
    const seen = new Set(formal.map((event) => event.event_id));
    if (archive.staging_run_id) (detail.key_events || []).filter(eventHasAlignedDualViewMaterial).forEach(event=>{
      if (seen.has(event.event_id)) return;
      seen.add(event.event_id);
      entries.push({archive, detail, event, preliminary: true, staged: true});
    });
    const candidates = [...(detail.quarantined_materials || []), ...(detail.preliminary_materials || [])];
    candidates.forEach((event) => {
      if (seen.has(event.event_id)) return;
      seen.add(event.event_id);
      entries.push({archive, detail, event, preliminary: true});
    });
  });
  return entries;
}


function automaticReviewLabel(event) {
  return event.disposition === "machine_quarantined_semantic_unavailable" || event.review_status === "pending_semantic_review"
    ? "自动核验未完成" : "未满足自动收录条件";
}


function materialFocusRoute(archive, event) {
  const base = experimentRecordRoute(archive, "materials");
  if (!event.event_id) return base;
  const query = new URLSearchParams({focus:String(event.event_id)});
  if (event.event_uid) query.set("event_uid", event.event_uid);
  const group = event.parent_event_id || event.experiment_group?.group_id;
  if (group) query.set("parent_event_id", group);
  if (event.release_id) query.set("release", event.release_id);
  return `${base}?${query}`;
}


function globalMaterialCard(entry) {
  const {archive,event,preliminary,staged} = entry;
  const current = staged ? "已保留动作素材，整体验收尚未通过；可查看现有画面和核验记录。" : preliminary ? "已保留候选画面，尚未满足自动核验的正式收录条件。" : productEvidenceText(event.provenance?.mllm?.current_step || event.decision?.observed_facts?.[0], "已保存关键实验画面。");
  const dual = !preliminary && eventHasDualViewSupport(event);
  const imageUrl = event.aligned_frame_url || event.frame_url;
  const eventId = String(event.event_id || "");
  const target = preliminary && !staged ? `${experimentRecordRoute(archive, "materials")}${eventId ? `?candidate=${encodeURIComponent(eventId)}` : ""}` : materialFocusRoute(archive, event);
  return `<a class="global-material-card ${preliminary ? "is-preliminary" : ""}" href="${target}"><figure>${imageUrl ? `<img loading="lazy" src="${esc(imageUrl)}" alt="${esc(productExperimentName(archive.name))}的关键画面"/>` : `<span class="material-image-fallback">${icon("image")}<small>画面暂不可用</small></span>`}<em>${staged ? "尚未通过整体验收" : preliminary ? automaticReviewLabel(event) : dual ? "双视角支持" : "动作待核对"}</em></figure><div class="global-material-copy"><small>${esc(productExperimentName(archive.name))} · ${formatDate(archive.modified_at)}</small><h2>${esc(preliminary ? `${staged ? "阶段素材" : "候选"} · ${ACTION_LABELS[event.cv_action_type || event.action_type] || "动作待确认"}` : ACTION_LABELS[event.action_type] || "关键实验动作")}</h2><p>${esc(current)}</p><span>查看素材 ${icon("arrow")}</span></div></a>`;
}


function bindLibraryImageFallbacks() {
  document.querySelectorAll(".global-material-card img").forEach((image)=>image.addEventListener("error", () => {
    image.replaceWith(Object.assign(document.createElement("span"), { className: "material-image-fallback", innerHTML: `${icon("image")}<small>画面暂不可用</small>` }));
  }, { once: true }));
}


function renderMaterialsLibrary(progressive = false) {
  setChrome("materials");
  void ensureLibraryDetails();
  const filters = state.globalMaterialFilters;
  const status = libraryLoadState("materials");
  const records = status.records;
  const all = materialLibraryEntries();
  const actionTypes = [...new Set([...(filters.action !== "all" ? [filters.action] : []), ...all.map((item)=>item.event.action_type || item.event.cv_action_type).filter(Boolean)])].sort();
  const objects = [...new Set([...(filters.object !== "all" ? [filters.object] : []), ...all.flatMap((item)=>eventObjectValues(item.event))])].sort();
  const query = state.search.trim().toLocaleLowerCase("zh-CN");
  const filtered = all.filter((entry) => {
    if (!archiveWithinDate(entry.archive, filters.date)) return false;
    if (filters.action !== "all" && (entry.event.action_type || entry.event.cv_action_type) !== filters.action) return false;
    if (filters.object !== "all" && !eventObjectValues(entry.event).includes(filters.object)) return false;
    if (filters.support === "dual" && (entry.preliminary || !eventHasDualViewSupport(entry.event))) return false;
    if (filters.support === "partial" && !entry.preliminary && eventHasDualViewSupport(entry.event)) return false;
    if (filters.support === "candidate" && (!entry.preliminary || entry.staged)) return false;
    if (filters.support === "stage" && !entry.staged) return false;
    if (filters.support === "formal" && entry.preliminary) return false;
    // Indexed formal results already matched the server's aliases, identifiers and search terms.
    if (!query || (!entry.preliminary && entry.detail.material_total_count != null)) return true;
    const text = [productExperimentName(entry.archive.name), ACTION_LABELS[entry.event.action_type || entry.event.cv_action_type], ...eventObjectValues(entry.event), entry.event.provenance?.mllm?.current_step].join(" ").toLocaleLowerCase("zh-CN");
    return text.includes(query);
  });
  const visible = filtered.slice(0,state.globalMaterialLimit);
  const loaded = status.ready.length;
  const formalCount = Number(state.archiveTotals?.key_events ?? state.archives.reduce((total, archive)=>total + Number(archive.key_event_count || 0), 0));
  const candidateCount = records.reduce((total, archive)=>total + (status.ready.includes(archive) ? all.filter((entry)=>libraryRecordKey(entry.archive) === libraryRecordKey(archive) && entry.preliminary && !entry.staged).length : Number(archive.candidate_material_count || 0)), 0);
  const stageCount = all.filter(entry=>entry.staged).length;
  const latest = all.find((entry)=>!entry.preliminary) || all[0];
  const loading = status.pending.length > 0;
  const more = status.ready.filter((archive) => !archive.staging_run_id && cachedLibraryDetail(archive)?.material_next_cursor);
  const incomplete = loading || status.failed.length > 0 || state.archiveNextCursor || more.length > 0;
  main.innerHTML = `<div class="page library-page"><header class="page-hero compact library-hero"><div><p class="eyebrow">关键素材库</p><h1>关键素材库</h1><p>候选由系统自动隔离；正式收录以自动核验结果为准，无需人工审批。</p></div>${latest ? `<div class="hero-actions"><a class="primary-button" href="${experimentRecordRoute(latest.archive, "materials")}">${icon("image")}打开最近素材</a></div>` : ""}</header>${archiveSyncNotice()}<section class="library-summary-strip" aria-label="素材概览"><span><small>正式素材（全部档案）</small><strong>${number(formalCount)}</strong></span><span><small>涉及实验</small><strong>${number(new Set(all.map((item)=>item.archive.name)).size)}</strong></span><span><small>阶段素材（已载入）</small><strong>${number(stageCount)}</strong></span><span><small>候选素材</small><strong>${number(candidateCount)}</strong></span><span><small>已载入实验</small><strong>${number(loaded)} / ${number(records.length)}</strong></span></section>${libraryFailureNotice(status,"materials")}<section class="panel library-filter-panel"><div class="product-filter-grid"><label><span>实验日期</span><select data-global-material-filter="date"><option value="all">全部日期</option><option value="today" ${filters.date==="today"?"selected":""}>今天</option><option value="7d" ${filters.date==="7d"?"selected":""}>最近 7 天</option><option value="30d" ${filters.date==="30d"?"selected":""}>最近 30 天</option></select></label><label><span>动作类型</span><select data-global-material-filter="action"><option value="all">全部动作</option>${actionTypes.map((value)=>`<option value="${esc(value)}" ${filters.action===value?"selected":""}>${esc(ACTION_LABELS[value]||value)}</option>`).join("")}</select></label><label><span>相关对象</span><select data-global-material-filter="object"><option value="all">全部对象</option>${objects.map((value)=>`<option value="${esc(value)}" ${filters.object===value?"selected":""}>${esc(productObjectLabel(value))}</option>`).join("")}</select></label><label><span>画面支持</span><select data-global-material-filter="support"><option value="all">全部素材</option><option value="formal" ${filters.support==="formal"?"selected":""}>仅正式素材</option><option value="stage" ${filters.support==="stage"?"selected":""}>仅阶段素材</option><option value="candidate" ${filters.support==="candidate"?"selected":""}>仅候选素材</option><option value="dual" ${filters.support==="dual"?"selected":""}>双视角印证</option><option value="partial" ${filters.support==="partial"?"selected":""}>主要视角清晰 / 候选</option></select></label></div></section><div class="library-results-heading" aria-live="polite"><div><strong>${query || Object.values(filters).some((value)=>value!=="all") ? (incomplete ? "筛选结果（已载入范围）" : "筛选结果") : (incomplete ? "已载入关键素材" : "全部关键素材")}</strong><span>${number(filtered.length)} 份</span></div>${loading ? `<small><i></i>正在继续载入其他实验…</small>` : ""}</div>${visible.length ? `<section class="library-card-grid">${visible.map(globalMaterialCard).join("")}</section>${visible.length < filtered.length ? `<button class="library-load-more" type="button" data-library-more>再显示 ${number(Math.min(18,filtered.length-visible.length))} 份</button>` : ""}` : loading ? libraryCardSkeleton(progressive ? 3 : 6) : status.failed.length ? "" : productState("neutral","search","没有符合条件的关键素材","可以调整日期、动作、对象或画面支持条件。",`<button class="secondary-button" type="button" data-reset-global-materials>重置筛选</button>`)}</div>`;
  document.querySelectorAll("[data-global-material-filter]").forEach((control)=>control.addEventListener("change",()=>{ state.globalMaterialFilters[control.dataset.globalMaterialFilter]=control.value; state.globalMaterialLimit=18; renderMaterialsLibrary(true); }));
  document.querySelector("[data-library-more]")?.addEventListener("click",()=>{ state.globalMaterialLimit += 18; renderMaterialsLibrary(true); });
  document.querySelector("[data-reset-global-materials]")?.addEventListener("click",()=>{
    state.globalMaterialFilters={date:"all",action:"all",object:"all",support:"all"};
    state.globalMaterialLimit=18;
    state.search="";
    const input=document.querySelector("#global-search");
    if(input) input.value="";
    closeGlobalSearch();
    renderMaterialsLibrary(true);
  });
  bindLibraryImageFallbacks();
  bindLibraryRetry("materials");
  if (more.length) {
    document.querySelector("#main-content .page")?.insertAdjacentHTML("beforeend", `<div class="pagination-actions"><button class="secondary-button" data-library-next type="button">加载下一页关键素材</button></div>`);
    document.querySelector("[data-library-next]")?.addEventListener("click", async (event) => {
      const button = event.currentTarget;
      const requestedHash = location.hash;
      const queryKey = libraryQueryKey("materials");
      const epoch = state.archiveCacheEpoch || 0;
      const isCurrent = () => location.hash === requestedHash && libraryQueryKey("materials") === queryKey && (state.archiveCacheEpoch || 0) === epoch;
      const pages = more.map((archive)=>({name:archive.name,cursor:cachedArchiveDetail(archive.name)?.material_next_cursor})).filter((page)=>page.cursor);
      button.disabled = true;
      try {
        for (let offset = 0; offset < pages.length; offset += 3) {
          if (!isCurrent()) return;
          const results = await Promise.allSettled(pages.slice(offset, offset + 3).map((page) => loadLibraryDetail(page.name, "materials", page.cursor)));
          if (!isCurrent()) return;
          results.forEach((result) => { if (result.status === "rejected") toast(result.reason.message, "error"); });
        }
        state.globalMaterialLimit += 24;
        renderMaterialsLibrary(true);
      } finally {
        button.disabled = false;
      }
    });
  }
  bindArchivePagination("materials");
}


function materialOperationTitle(event) {
  const model = event.provenance?.mllm || {};
  const title = String(model.operation_title || "").trim();
  if (title) return productEvidenceText(title, "操作对象待核对");
  return `${ACTION_LABELS[event.action_type] || "实验操作"} · 对象待核对`;
}


function materialCard(event, index = 0) {
  const mllm = event.provenance?.mllm || {};
  const facts = event.decision?.observed_facts || [];
  const current = productEvidenceText(mllm.current_step || facts[0], "当前动作等待进一步说明");
  const nextStatus = mllm.next_step_status || mllm.next_step_evidence?.status || "unknown";
  const nextLabel = nextStatus === "observed" ? "已观察后续动作" : nextStatus === "inferred" ? "预测下一步" : "后续未知";
  const next = productEvidenceText(mllm.next_step, "没有足够证据支持下一步");
  const observed = facts.length ? facts.map((fact)=>productEvidenceText(fact, "暂无补充说明")) : ["当前档案没有单独记录可直接确认的画面事实。"];
  const cross = event.cross_view_associations?.[0];
  const dualView = eventHasDualViewSupport(event);
  const supportCopy = dualView
    ? "两个视角共同支持这一关键动作"
    : cross
      ? "画面已同步；尚不能确认两个视角共同支持这一动作"
      : "双视角画面已保存，动作关联等待进一步复核";
  const clipSeconds = Math.max(0, (Number(event.end_us) - Number(event.start_us)) / 1_000_000);
  const peakOffset = Math.min(clipSeconds, Math.max(0, (Number(event.peak_timestamp_us ?? event.start_us) - Number(event.start_us)) / 1_000_000));
  const selectionKey = materialSelectionKey(event);
  const selected = state.selectedMaterials.has(selectionKey);
  const uncertainty = nextStatus === "unknown" || /不足|无法|不能|待确认|不确定|未明确/.test(next);
  return `<article class="material-card product-material-card ${selected ? "is-selected" : ""}" data-material-card="${esc(event.event_id)}" data-material-selection-key="${esc(selectionKey)}">
    <header class="material-card-header"><div><span class="material-sequence">关键素材 ${String(index + 1).padStart(2,"0")}</span><h2>${esc(materialOperationTitle(event))}</h2><p>${timecode(Number(event.start_us)/1000)} → ${timecode(Number(event.end_us)/1000)}</p></div><div class="material-card-actions"><button class="material-focus-button" type="button" data-material-focus="${esc(event.event_id)}">${icon("video")}聚焦查看</button><label class="material-select"><input type="checkbox" data-material-select="${esc(event.event_id)}" ${selected ? "checked" : ""}/><span>选择</span></label><span class="material-support-badge ${dualView ? "trusted" : "partial"}">${dualView ? icon("check") : icon("image")}${dualView ? "双视角支持" : "动作待核对"}</span></div></header>
    <figure class="material-visual"><div class="material-frame">${event.aligned_frame_url ? `<img loading="lazy" src="${esc(event.aligned_frame_url)}" alt="第一人称与第三人称关键画面对照"/>` : `<div class="empty-state compact"><strong>关键画面准备中</strong></div>`}</div><figcaption><span>${icon("video")}第一 / 第三人称画面对照</span><span>关键时刻 ${timecode(Number(event.peak_timestamp_us ?? event.start_us)/1000)}</span></figcaption></figure>
    <section class="material-preview-note"><span class="evidence-kind interpreted">步骤理解</span><p>${esc(current)}</p></section>
    <details class="material-product-details"><summary><span>查看详情与播放</span><small>${event.aligned_clip_url ? `视频 ${duration(clipSeconds)}` : "查看分析"}</small>${icon("chevron")}</summary><div class="material-product-content">
      <div class="material-story"><section class="material-story-main evidence-observed"><small><span class="evidence-kind observed">画面确认</span>直接观察事实</small><ul class="fact-list">${observed.slice(0,3).map((fact)=>`<li>${esc(fact)}</li>`).join("")}</ul></section><section class="material-next-step ${uncertainty ? "evidence-uncertain" : "evidence-inferred"}"><small><span class="evidence-kind ${uncertainty ? "uncertain" : "inferred"}">${esc(nextLabel)}</span>${nextStatus === "observed" ? "来自后续画面" : nextStatus === "inferred" ? "模型推测 · 非事实结论" : "当前证据未支持后续判断"}</small><p>${esc(next)}</p></section><p class="material-support-note">${dualView ? icon("check") : icon("image")}<span>${esc(supportCopy)}</span></p></div>
      ${event.aligned_clip_url ? `<details class="material-clip-details"><summary>${icon("video")}<span>播放关键片段</span><small>${duration(clipSeconds)}</small>${icon("chevron")}</summary><div class="material-player-toolbar"><span>${icon("video")}双视角片段</span><small>第一 / 第三人称画面对照</small><div class="material-player-actions"><button type="button" data-review-adjacent="previous" data-review-event="${esc(event.event_id)}" aria-label="上一份素材">上一份</button><label>倍速<select data-video-speed="${esc(event.event_id)}"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label><label class="loop-control"><input type="checkbox" data-video-loop="${esc(event.event_id)}"/>循环</label><button type="button" data-video-fullscreen="${esc(event.event_id)}">全屏</button><button type="button" data-review-adjacent="next" data-review-event="${esc(event.event_id)}" aria-label="下一份素材">下一份</button></div></div><div class="material-video">${videoPreview(event.aligned_clip_url,"关键片段",event.aligned_frame_url || videoPosterUrl(event.aligned_clip_url),{materialId:event.event_id})}<div class="material-time-markers"><button type="button" data-video-seek="${esc(event.event_id)}" data-seconds="0"><i></i><span>片段开始<small>${timecode(Number(event.start_us)/1000)}</small></span></button><button class="peak" type="button" data-video-seek="${esc(event.event_id)}" data-seconds="${peakOffset}"><i></i><span>关键时刻<small>${timecode(Number(event.peak_timestamp_us ?? event.start_us)/1000)}</small></span></button><button type="button" data-video-seek="${esc(event.event_id)}" data-seconds="${clipSeconds}"><i></i><span>片段结束<small>${timecode(Number(event.end_us)/1000)}</small></span></button></div><p class="player-shortcuts">快捷键：空格 播放/暂停 · ←/→ 前后 5 秒 · P/N 上一份/下一份</p></div></details>` : ""}
      <details class="material-evidence-details"><summary><span>${icon("boxes")}查看对象与分析依据</span><small>素材记录 ${esc(event.event_id)}</small>${icon("chevron")}</summary><div class="material-evidence-content"><section><h3>动作相关对象</h3><div class="object-chip-list">${materialObjectItems(event.objects || {})}</div></section>${verificationTrace(event.verification || {})}</div></details>
    </div></details>
  </article>`;
}


function eventHasDualViewSupport(event) {
  return (event.cross_view_associations || []).some((association) =>
    association.both_views_support_action === true && association.consistency === "consistent"
  );
}


function eventHasAlignedDualViewMaterial(event) {
  return event.dual_view_material_ready === true || Boolean(event.aligned_frame_url && event.aligned_clip_url);
}


function ensureMaterialFilters(data) {
  if (state.materialFilters.archive === data.name) return;
  state.materialFilters = {
    archive: data.name,
    group: data.experiment_groups?.[0]?.group_id || "all",
    action: "all",
    object: "all",
    support: "all",
    query: "",
  };
}


function materialSelectionKey(event) {
  return JSON.stringify([event.event_uid || null, String(event.event_id), event.parent_event_id || event.experiment_group?.group_id || null]);
}


function materialSelectionScope(data) {
  return JSON.stringify([data.name, data.staging_run_id || null, data.release_id || null]);
}


function ensureMaterialSelection(data) {
  const source = materialSelectionScope(data);
  if (state.materialSelectionSource === source) return;
  if (state.selectedMaterials.size) toast("素材来源或版本已更新，已清除原来的选择。");
  state.selectedMaterials.clear();
  state.materialSelectionSource = source;
}


function filteredMaterialEvents(data) {
  const filters = state.materialFilters;
  const query = filters.query.trim().toLocaleLowerCase("zh-CN");
  return data.key_events.filter((event) => {
    if (!eventHasAlignedDualViewMaterial(event)) return false;
    if (filters.group !== "all" && event.experiment_group?.group_id !== filters.group) return false;
    if (filters.action !== "all" && event.action_type !== filters.action) return false;
    if (filters.object !== "all" && !eventObjectValues(event).includes(filters.object)) return false;
    if (filters.support === "dual" && !eventHasDualViewSupport(event)) return false;
    if (filters.support === "partial" && eventHasDualViewSupport(event)) return false;
    if (!query || data.material_total_count != null) return true;
    const mllm = event.provenance?.mllm || {};
    const haystack = [
      event.event_id,
      ACTION_LABELS[event.action_type] || event.action_type,
      event.experiment_group?.name,
      mllm.current_step,
      mllm.next_step,
      JSON.stringify(event.objects || {}),
      ...(event.decision?.observed_facts || []),
    ].join(" ").toLocaleLowerCase("zh-CN");
    return haystack.includes(query);
  });
}


function materialResults(data) {
  const events = filteredMaterialEvents(data);
  const formalTotal = Number(data.material_total_count ?? events.length);
  const groupMap = new Map((data.experiment_groups || []).map((group,index) => [group.group_id, { ...group, index }]));
  const grouped = new Map();
  for (const event of events) {
    const groupId = event.experiment_group?.group_id || "ungrouped";
    if (!grouped.has(groupId)) grouped.set(groupId, []);
    grouped.get(groupId).push(event);
  }
  const sections = [...grouped.entries()].sort(([left],[right]) => (groupMap.get(left)?.index ?? 999) - (groupMap.get(right)?.index ?? 999));
  if (!sections.length) return productState("neutral", "search", "没有符合条件的关键素材", "可以放宽实验片段、动作类型、画面支持或关键词筛选。", `<button class="secondary-button" type="button" data-reset-material-filters>重置素材筛选</button>`);
  return `<div class="material-result-summary"><div><strong>已加载 ${number(events.length)} / ${number(formalTotal)} 份关键素材</strong><span>先看概览，需要时展开详情与视频。</span></div><button class="compact-button" type="button" data-select-visible>${icon("check")}选择当前结果</button></div>${sections.map(([groupId,items])=>{ const group = groupMap.get(groupId) || items[0].experiment_group || {}; const index = group.index == null ? "—" : String(group.index + 1).padStart(2,"0"); return `<section class="material-group"><header><span class="material-group-index">${index}</span><div><h2>${esc(group.name || groupId)}</h2><p>${timecode(group.start_ms)} → ${timecode(group.end_ms)} · ${workflowLabel(group)} · ${workflowCompletionLabel(group)}</p></div><span class="badge">已加载 ${number(items.length)} 份</span></header><div class="material-grid">${items.map(materialCard).join("")}</div></section>`; }).join("")}`;
}


function materialSelectionBar(data) {
  const count = state.selectedMaterials.size;
  return `<aside class="material-selection-bar ${count ? "is-visible" : ""}" id="material-selection-bar" aria-live="polite"><div><strong>已选择 <b data-selected-count>${number(count)}</b> 份素材</strong><span>选择仅用于整理，不会修改正式证据档案。</span></div><div><button type="button" data-copy-selected>${icon("copy")}复制编号</button><button type="button" data-export-selected>${icon("file")}导出清单</button><button type="button" data-clear-selected>清除</button></div></aside>`;
}


function retainedMaterialsView(data) {
  const retained = [...(data.preliminary_materials || []), ...(data.quarantined_materials || [])];
  if (!retained.length) return "";
  const priorityCount = retained.filter(item=>item.preview_review?.priority).length;
  const filters = state.retainedMaterialFilters ||= {scope:priorityCount ? "priority" : "all", group:"all", action:"all"};
  const pending = data.partial_delivery?.pending_semantic_results || [];
  const automaticStatus = pending.some(item=>item.status === "disabled")
    ? "本次运行未启用自动语义核验，候选已由系统隔离保存。"
    : pending.length ? "自动语义核验尚未完成，候选已由系统隔离保存。"
    : "以下候选尚未满足自动核验的正式收录条件。";
  const groups = [...new Set(retained.map(item=>item.group_folder).filter(Boolean))];
  const actions = [...new Set(retained.map(item=>item.cv_action_type).filter(Boolean))];
  const visible = retained.filter(item => (filters.scope === "all" || item.preview_review?.priority)
    && (filters.group === "all" || item.group_folder === filters.group)
    && (filters.action === "all" || item.cv_action_type === filters.action));
  const reasons = {short_candidate:"候选持续不足 1 秒",missing_time_bounds:"缺少候选时间边界",missing_cross_role_sources:"缺少第一/第三人称候选来源",incomplete_preview_media:"画面或片段不完整",semantic_review_not_accepted:"语义核验未通过",overlapping_similar_candidate:"与另一同类候选时间重叠"};
  return `<section class="panel retained-workspace" id="retained-workspace"><header class="panel-heading"><div><h2>候选素材 · 未入正式库</h2><p>${automaticStatus}无需人工审批；查阅与筛选不会改变自动判定。</p><p>保留 ${number(retained.length)} 个候选，其中 ${number(priorityCount)} 个优先查阅。</p></div></header><p class="retained-policy">优先查阅要求候选持续至少 1 秒，且有第一、第三人称候选来源；时间重叠的同类对象候选折叠到“全部候选”。这些条件只用于初筛，不确认动作真假；短时动作和单侧画面仍完整保留。</p><div class="product-filter-grid"><label><span>候选范围</span><select data-retained-filter="scope"><option value="priority" ${filters.scope==="priority"?"selected":""}>优先查阅（${priorityCount}）</option><option value="all" ${filters.scope==="all"?"selected":""}>全部候选（${retained.length}）</option></select></label><label><span>实验片段</span><select data-retained-filter="group"><option value="all">全部实验片段</option>${groups.map(group=>`<option value="${esc(group)}" ${filters.group===group?"selected":""}>实验片段 ${esc(group.split("_")[0])}</option>`).join("")}</select></label><label><span>CV 候选类型 · 未确认</span><select data-retained-filter="action"><option value="all">全部候选类型</option>${actions.map(action=>`<option value="${esc(action)}" ${filters.action===action?"selected":""}>${esc(ACTION_LABELS[action]||action)}</option>`).join("")}</select></label></div><p class="retained-count" role="status">当前显示 ${number(visible.length)} / ${number(retained.length)} 个候选</p><div class="material-grid">${visible.map((item,index) => {
    const review = item.preview_review || {};
    const pair = item.view_pairing || {};
    const pairLabel = [pair.first_person_view, pair.third_person_view].filter(Boolean).join(" / ");
    const pairStatus = pair.pair_evidence_status === "context_only_missing_key_time_support" ? "未建立动作机位对应：先展示单侧候选，其他机位仅供参考" : "候选机位对照：同一对象与动作尚未核验";
    const context = item.group_folder ? `${experimentRecordRoute(data)}?group=${encodeURIComponent(item.group_folder)}` : experimentRecordRoute(data);
    return `<article class="experiment-card retained-card" id="candidate-${esc(item.event_id)}"><header title="素材记录：${esc(item.event_id)}"><h3>候选 ${String(index + 1).padStart(2,"0")} · ${esc(ACTION_LABELS[item.cv_action_type] || "动作待确认")}</h3><span class="badge">${automaticReviewLabel(item)}</span></header><p>${item.group_folder?`实验片段 ${esc(item.group_folder.split("_")[0])} · `:""}${timecode(item.timestamp_ms)} · 候选持续 ${review.duration_ms==null?"未知":`${(review.duration_ms/1000).toFixed(2)} 秒`}</p><p>候选对象：${esc((item.cv_objects||[]).map(productObjectLabel).join("、")||"未记录")} · ${number(item.source_views?.length)} 个候选来源机位</p>${review.reason_codes?.length?`<p class="retained-reasons">${esc(review.reason_codes.map(reason=>reasons[reason]||reason).join("；"))}${review.related_event_id?`（${esc(review.related_event_id)}）`:""}</p>`:""}${pairLabel ? `<p class="retained-reasons">${esc(pairStatus)}<br>${esc(pairLabel)}</p>` : ""}${item.frame_url ? `<img loading="lazy" style="width:100%;height:auto" src="${esc(item.frame_url)}" alt="未核验候选画面，检测框不代表动作已确认">` : ""}${item.clip_url ? `<details class="material-clip-details"><summary>${icon("video")}播放候选片段${icon("chevron")}</summary><div class="material-video">${videoPreview(item.clip_url,"候选片段",item.frame_url || videoPosterUrl(item.clip_url))}</div></details>` : ""}${item.context_media?.frame_url ? `<details class="material-clip-details"><summary>查看同时间参考机位（未确认对应）${icon("chevron")}</summary><img loading="lazy" style="width:100%;height:auto" src="${esc(item.context_media.frame_url)}" alt="同时间参考机位，未确认与候选动作对应">${item.context_media.clip_url ? videoPreview(item.context_media.clip_url,"参考机位片段",item.context_media.frame_url) : ""}</details>` : ""}<a class="secondary-button" href="${context}">查看较长实验片段</a></article>`;
  }).join("") || `<p class="empty-state">当前条件没有候选，可切换到全部候选或调整实验片段和候选类型。</p>`}</div></section>`;
}


function bindRetainedMaterialFilters(data) {
  document.querySelectorAll("[data-retained-filter]").forEach(control=>control.addEventListener("change",()=>{
    state.retainedMaterialFilters[control.dataset.retainedFilter] = control.value;
    document.querySelector("#retained-workspace").outerHTML = retainedMaterialsView(data);
    bindRetainedMaterialFilters(data);
    bindVideoPreviews();
  }));
}


function movementScreeningView(data) {
  const review = data.movement_screening;
  if (!review) return "";
  const labels = {supported:"支持相对移动",contradicted:"不支持移动",unverified:"画面不足以判断"};
  return `<section class="panel screening-record" id="screening-record"><header class="panel-heading"><div><h2>素材筛选记录</h2><p>检查候选画面中的相对移动；这些统计不代表已确认的实验操作。</p></div>${review.report_url ? `<a class="secondary-button" href="${esc(review.report_url)}" target="_blank" rel="noopener">打开完整核验记录 ${icon("arrow")}</a>` : ""}</header><div class="screening-counts">${Object.entries(labels).map(([status,label])=>`<article><span>${label}</span><strong>${number(review.counts?.[status] || 0)}</strong><small>${status === "supported" ? "仍需结合对象与操作核对" : "保留记录，不用于确认动作"}</small></article>`).join("")}</div></section>`;
}


function materialsView(data) {
  ensureMaterialSelection(data);
  const screening = data.movement_screening ? `<a class="screening-record-link" href="${experimentRecordRoute(data,"metrics")}">${icon("activity")}查看素材筛选与核验记录 ${icon("arrow")}</a>` : "";
  const retained = retainedMaterialsView(data);
  if (!data.key_events.length && data.material_total_count == null && (retained || screening)) return screening + retained;
  ensureMaterialFilters(data);
  const filters = state.materialFilters;
  const formalEvents = data.key_events.filter(eventHasAlignedDualViewMaterial);
  const quarantinedCount = Number(data.quality_acceptance?.key_materials?.missing_dual_view_material_count || 0);
  const matchingCount = Number(data.material_total_count ?? filteredMaterialEvents(data).length);
  const actionTypes = [...new Set([...Object.keys(ACTION_LABELS), ...formalEvents.map((event)=>event.action_type), filters.action !== "all" ? filters.action : null].filter(Boolean))];
  const objectValues = [...new Set([...formalEvents.flatMap(eventObjectValues), filters.object !== "all" ? filters.object : null].filter(Boolean))].sort();
  return `${screening}<section class="material-workspace" id="material-workspace"><header class="material-toolbar-heading"><div><p class="eyebrow">关键素材</p><h2>按实验查看关键画面与片段</h2><p>先看概览，需要时进入聚焦模式查看画面事实、模型提示和视频。</p></div><span class="badge">符合筛选 ${number(matchingCount)} 份</span></header><div class="evidence-legend" aria-label="证据信息说明"><span><i class="observed"></i><strong>画面确认</strong>可直接观察</span><span><i class="inferred"></i><strong>模型提示</strong>用于辅助理解</span><span><i class="uncertain"></i><strong>证据不足</strong>保留不确定性</span></div>${quarantinedCount ? `<div class="freshness-warning"><strong>${number(quarantinedCount)} 份素材缺少完整双视角画面，暂未展示。</strong></div>` : ""}<div class="material-filter-bar" id="material-filters"><label><span>实验片段</span><select id="material-group-filter"><option value="all" ${filters.group==="all"?"selected":""}>全部实验</option>${(data.experiment_groups||[]).map((group,index)=>`<option value="${esc(group.group_id)}" ${filters.group===group.group_id?"selected":""}>${String(index+1).padStart(2,"0")} · ${esc(group.name)}</option>`).join("")}</select></label><label><span>动作类型</span><select id="material-action-filter"><option value="all">全部动作类型</option>${actionTypes.map((type)=>`<option value="${esc(type)}" ${filters.action===type?"selected":""}>${esc(ACTION_LABELS[type]||type)}</option>`).join("")}</select></label><label><span>相关对象</span><select id="material-object-filter"><option value="all">全部对象</option>${objectValues.map((value)=>`<option value="${esc(value)}" ${filters.object===value?"selected":""}>${esc(productObjectLabel(value))}</option>`).join("")}</select></label><label><span>画面支持</span><select id="material-support-filter"><option value="all">全部关键素材</option><option value="dual" ${filters.support==="dual"?"selected":""}>两个视角相互印证</option><option value="partial" ${filters.support==="partial"?"selected":""}>单个视角更清晰</option></select></label><label class="material-query"><span>搜索关键素材</span><input id="material-query" value="${esc(filters.query)}" placeholder="例如：移液器、开盖、称量纸" /></label></div><p id="material-filter-status" class="analysis-readiness-note" role="status" hidden><span></span> <button type="button" hidden>重试筛选</button></p><div id="material-results">${materialResults(data)}</div>${materialSelectionBar(data)}<div id="material-focus-host"></div></section>${retained}`;
}


function updateMaterialResults(data) {
  const template = document.createElement("template");
  template.innerHTML = materialsView(data);
  for (const selector of ["#material-results", ".material-toolbar-heading .badge", "#material-group-filter", "#material-action-filter", "#material-object-filter"]) {
    const target = document.querySelector(selector);
    const replacement = template.content.querySelector(selector);
    if (!target || !replacement || target === document.activeElement) continue;
    if (selector === "#material-results") pauseMaterialVideos(target);
    target.innerHTML = replacement.innerHTML;
  }
  bindMaterialInteractions(data);
}


function bindMaterialFilters(data) {
  const requestedHash = location.hash;
  const requestId = state.archiveViewRequestId;
  const isCurrent = () => location.hash === requestedHash && state.archiveViewRequestId === requestId;
  let queryTimer, generation = 0, composing = false;
  let loadedQuery = JSON.stringify(state.materialFilters);
  const status = (message, failed = false) => {
    const notice = document.querySelector("#material-filter-status");
    if (notice) {
      notice.hidden = !message;
      notice.querySelector("span").textContent = message;
      notice.querySelector("button").hidden = !failed;
    }
    document.querySelector("#material-results")?.setAttribute("aria-busy", String(Boolean(message) && !failed));
  };
  const rerender = async (cursor = null) => {
    if (!isCurrent() || composing) return;
    clearTimeout(queryTimer);
    const version = ++generation;
    const query = JSON.stringify(state.materialFilters);
    if (query !== loadedQuery) cursor = null;
    const epoch = state.archiveCacheEpoch || 0;
    const latest = () => isCurrent() && generation === version && JSON.stringify(state.materialFilters) === query;
    status("正在筛选关键素材，当前显示上次载入的结果。");
    try {
      const next = data.staging_run_id || archiveProcessStopped(data) ? data : await loadArchiveMaterials(data.name, cursor);
      if (!latest()) return;
      if (epoch !== (state.archiveCacheEpoch || 0) || (next.release_id || null) !== (data.release_id || null)) throw new Error("档案版本已更新，请刷新页面。");
      data = next;
      loadedQuery = query;
      updateMaterialResults(data);
      bindPagination();
      status("");
    } catch (error) {
      if (latest()) status(`筛选未完成，当前保留上次结果。${error.message}`, true);
    }
  };
  for (const [selector, field] of [["#material-group-filter", "group"], ["#material-action-filter", "action"], ["#material-object-filter", "object"], ["#material-support-filter", "support"]]) {
    document.querySelector(selector)?.addEventListener("change", (event) => {
      if (!isCurrent()) return;
      state.materialFilters[field] = event.target.value;
      void rerender();
    });
  }
  const scheduleQuery = (event) => {
    if (!isCurrent()) return;
    state.materialFilters.query = event.target.value;
    clearTimeout(queryTimer);
    generation++;
    if (!composing && !event.isComposing) queryTimer = setTimeout(() => rerender(), 300);
  };
  const input = document.querySelector("#material-query");
  input?.addEventListener("compositionstart", () => {
    composing = true;
    clearTimeout(queryTimer);
    generation++;
  });
  input?.addEventListener("compositionend", (event) => { composing = false; scheduleQuery(event); });
  input?.addEventListener("input", scheduleQuery);
  document.querySelector("#material-filter-status button")?.addEventListener("click", () => { void rerender(); });
  const bindPagination = () => {
    document.querySelector("#material-pagination")?.remove();
    if (!data.material_next_cursor) return;
    document.querySelector("#material-results")?.insertAdjacentHTML("afterend", `<div class="pagination-actions" id="material-pagination"><button class="secondary-button" id="load-more-materials" type="button">加载更多关键素材</button></div>`);
    document.querySelector("#load-more-materials")?.addEventListener("click", async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      try { await rerender(data.material_next_cursor); }
      finally { button.disabled = false; }
    });
  };
  bindPagination();
  bindMaterialInteractions(data);
}


function selectedMaterialEvents(data) {
  if (state.materialSelectionSource !== materialSelectionScope(data)) return [];
  return [...state.selectedMaterials.values()];
}


function updateMaterialSelectionBar() {
  const bar = document.querySelector("#material-selection-bar");
  if (!bar) return;
  const count = state.selectedMaterials.size;
  bar.classList.toggle("is-visible", count > 0);
  const counter = bar.querySelector("[data-selected-count]");
  if (counter) counter.textContent = number(count);
  document.querySelectorAll("[data-material-selection-key]").forEach((card)=>card.classList.toggle("is-selected", state.selectedMaterials.has(card.dataset.materialSelectionKey)));
}


function exportSelectedMaterials(data) {
  const events = selectedMaterialEvents(data);
  if (!events.length) { toast("请先选择需要整理的关键素材。", "error"); return; }
  const payload = {
    schema_version: "visioncortex-user-selection/1",
    artifact_type: "visioncortex_user_selection_manifest",
    evidence_status: "DERIVED_SELECTION_NOT_GROUND_TRUTH",
    source_archive: data.name,
    source_release_id: data.release_id || null,
    source_staging_run_id: data.staging_run_id || null,
    selected_count: events.length,
    created_at: new Date().toISOString(),
    selected_materials: events.map((event)=>({
      event_id: event.event_id,
      event_uid: event.event_uid || null,
      parent_event_id: event.parent_event_id || event.experiment_group?.group_id || null,
      release_id: event.release_id || data.release_id || null,
      material_route: materialFocusRoute(data, event),
      action_type: event.action_type,
      action: ACTION_LABELS[event.action_type] || event.action_type,
      start_us: event.start_us,
      end_us: event.end_us,
      start_timecode: timecode(Number(event.start_us) / 1000),
      end_timecode: timecode(Number(event.end_us) / 1000),
      dual_view_supported: eventHasDualViewSupport(event),
      frame_url: event.aligned_frame_url || null,
      clip_url: event.aligned_clip_url || null,
    })),
  };
  let url;
  let link;
  try {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    url = URL.createObjectURL(blob);
    link = document.createElement("a");
    link.href = url;
    link.download = `VisionCortex-素材清单-${String(data.name).replace(/[^a-zA-Z0-9_-]+/g, "-")}.json`;
    document.body.appendChild(link);
    link.click();
    toast(`已生成 ${events.length} 份素材清单，请查看浏览器下载。`);
  } catch {
    toast("清单下载未能启动，已保留所选素材，请重试。", "error");
  } finally {
    link?.remove();
    if (url) setTimeout(() => URL.revokeObjectURL(url), 60_000);
  }
}


function materialVideo(eventId, control = null) {
  if (control) return control.closest("[data-material-card]")?.querySelector("video[data-material-video]") || null;
  const matches = [...document.querySelectorAll("video[data-material-video]")].filter((video)=>video.dataset.materialVideo === eventId);
  return matches.length === 1 ? matches[0] : null;
}


function pauseMaterialVideos(root) {
  root?.querySelectorAll("video").forEach((video)=>video.pause());
}


function materialFocusMarkup(data, event, events) {
  const index = events.findIndex((item)=>materialSelectionKey(item) === materialSelectionKey(event));
  const facts = event.decision?.observed_facts || [];
  const current = productEvidenceText(event.provenance?.mllm?.current_step || facts[0], "已保存关键实验画面。");
  const nextStep = event.provenance?.mllm || {};
  const next = productEvidenceText(nextStep.next_step, "没有足够证据支持下一步。");
  const clipSeconds = Math.max(0,(Number(event.end_us)-Number(event.start_us))/1_000_000);
  const peakOffset = Math.min(clipSeconds,Math.max(0,(Number(event.peak_timestamp_us??event.start_us)-Number(event.start_us))/1_000_000));
  return `<dialog class="material-focus-dialog" id="material-focus-dialog"><header><div><p class="eyebrow">聚焦查看 · ${number(index+1)} / ${number(events.length)}</p><h2>${esc(ACTION_LABELS[event.action_type]||"关键实验动作")}</h2><p>${esc(productExperimentName(data.name))} · ${timecode(Number(event.start_us)/1000)} → ${timecode(Number(event.end_us)/1000)}</p></div><button class="dialog-close" type="button" data-close-material-focus aria-label="关闭">${icon("x")}</button></header><div class="material-focus-body"><section class="material-focus-player">${event.aligned_clip_url ? videoPreview(event.aligned_clip_url,"关键片段",event.aligned_frame_url || videoPosterUrl(event.aligned_clip_url),{focus:true}) : event.aligned_frame_url ? `<img src="${esc(event.aligned_frame_url)}" alt="双视角关键画面"/>` : productState("neutral","image","画面暂不可用","该素材没有可播放的视频或关键帧。")}${event.aligned_clip_url ? `<div class="focus-time-markers"><button type="button" data-focus-seek="0"><span>片段开始</span><small>${timecode(Number(event.start_us)/1000)}</small></button><button type="button" data-focus-seek="${peakOffset}"><span>关键时刻</span><small>${timecode(Number(event.peak_timestamp_us??event.start_us)/1000)}</small></button><button type="button" data-focus-seek="${clipSeconds}"><span>片段结束</span><small>${timecode(Number(event.end_us)/1000)}</small></button></div>` : ""}</section><aside class="material-focus-insight"><section><small>步骤理解</small><strong>${esc(current)}</strong></section><section><small>画面确认</small><ul>${facts.slice(0,4).map((fact)=>`<li>${esc(productEvidenceText(fact,"暂无补充说明"))}</li>`).join("")||"<li>当前档案没有单独记录可直接确认的画面事实。</li>"}</ul></section><section class="focus-next"><small>${esc(nextStepLabel(nextStep))}${nextStepLabel(nextStep) === "预测后续" ? " · 非事实结论" : ""}</small><p>${esc(next)}</p></section></aside></div><footer><div class="focus-playback-controls"><label>播放速度<select data-focus-speed><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label><label><input type="checkbox" data-focus-loop/>循环播放</label><button type="button" data-focus-fullscreen>${icon("video")}全屏</button></div><div class="focus-navigation"><button type="button" data-focus-nav="previous" ${index<=0?"disabled":""}>${icon("chevron")}上一份</button><button type="button" data-focus-nav="next" ${index>=events.length-1?"disabled":""}>下一份 ${icon("arrow")}</button></div></footer></dialog>`;
}


async function openLinkedMaterial(data, eventId, isCurrent) {
  const query = routeQuery();
  const epoch = state.archiveCacheEpoch || 0;
  const current = () => isCurrent() && (state.archiveCacheEpoch || 0) === epoch;
  try {
    if (query.get("release") && query.get("release") !== data.release_id) throw new Error("该素材链接对应的档案版本已更新，请从素材库重新打开。");
    const matches = (event) => String(event.event_id) === String(eventId)
      && (!query.get("event_uid") || event.event_uid === query.get("event_uid"))
      && (!query.get("parent_event_id") || (event.parent_event_id || event.experiment_group?.group_id) === query.get("parent_event_id"));
    let event = (data.key_events || []).find(matches);
    if (!event && query.get("event_uid")) {
      const uid = encodeURIComponent(query.get("event_uid"));
      const url = data.staging_run_id
        ? `/api/staging-runs/${encodeURIComponent(data.staging_run_id)}/key-events/${uid}`
        : `/api/key-events/${uid}?archive=${encodeURIComponent(data.name)}`;
      const payload = await api(url);
      if (!current()) return;
      event = matches(payload) ? payload : null;
    } else if (!event) {
      const parameters = new URLSearchParams({archive:data.name,q:String(eventId),material_ready:"true",limit:"24"});
      if (query.get("parent_event_id")) parameters.set("parent_event_id",query.get("parent_event_id"));
      const payload = await api(`/api/key-events?${parameters}`);
      if (!current()) return;
      event = (payload.items || []).find(matches);
    }
    if (!event || !eventHasAlignedDualViewMaterial(event)) throw new Error("该正式素材暂不可用，请从素材库重新查找。");
    if ((event.release_id || null) !== (data.release_id || null)) throw new Error("素材所属版本已更新，请重新加载档案。");
    if (current()) openMaterialFocus(data, eventId, event);
  } catch (error) {
    if (current()) toast(error.message, "error");
  }
}


function openMaterialFocus(data, eventId, linkedEvent = null, selectionKey = null) {
  const events = linkedEvent ? [linkedEvent] : filteredMaterialEvents(data);
  const matches = events.filter((item)=>String(item.event_id)===String(eventId) && (!selectionKey || materialSelectionKey(item)===selectionKey));
  if (matches.length > 1) { toast("存在同名素材，请从对应素材卡片打开。", "error"); return; }
  const event = matches[0];
  if (!event) { toast("指定素材不在当前已载入页面，请加载更多或调整筛选。", "error"); return; }
  state.focusedMaterialId = String(event.event_id);
  let host = document.querySelector("#material-focus-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "material-focus-host";
    document.body.append(host);
  }
  pauseMaterialVideos(main);
  pauseMaterialVideos(host);
  host.innerHTML = materialFocusMarkup(data,event,events);
  const dialog = document.querySelector("#material-focus-dialog");
  dialog?.showModal();
  if (dialog) bindVideoPreviews(dialog);
  dialog?.addEventListener("close",()=>{ pauseMaterialVideos(dialog); state.focusedMaterialId=null; });
  dialog?.querySelector("[data-close-material-focus]")?.addEventListener("click",()=>dialog.close());
  dialog?.addEventListener("click",(clickEvent)=>{ if(clickEvent.target===dialog) dialog.close(); });
  dialog?.querySelectorAll("[data-focus-nav]").forEach((button)=>button.addEventListener("click",()=>{
    const currentIndex = events.findIndex((item)=>materialSelectionKey(item)===materialSelectionKey(event));
    const nextIndex = currentIndex + (button.dataset.focusNav === "previous" ? -1 : 1);
    if (events[nextIndex]) openMaterialFocus(data,events[nextIndex].event_id,null,materialSelectionKey(events[nextIndex]));
  }));
  const video = dialog?.querySelector("[data-focus-video]");
  dialog?.querySelector("[data-focus-speed]")?.addEventListener("change",(event)=>{ if(video) video.playbackRate=Number(event.target.value||1); });
  dialog?.querySelector("[data-focus-loop]")?.addEventListener("change",(event)=>{ if(video) video.loop=event.target.checked; });
  dialog?.querySelector("[data-focus-fullscreen]")?.addEventListener("click",async()=>{ try{ if(video?.requestFullscreen) await video.requestFullscreen(); }catch{ toast("当前浏览器无法进入全屏播放。","error"); } });
  dialog?.querySelectorAll("[data-focus-seek]").forEach((button)=>button.addEventListener("click",()=>{ if(video) video.currentTime=Math.min(Number(button.dataset.focusSeek||0),Number.isFinite(video.duration)?video.duration:Number(button.dataset.focusSeek||0)); }));
}


function activateAdjacentMaterial(eventId, direction, sourceCard = null) {
  const cards = [...document.querySelectorAll("[data-material-card]")];
  const matches = cards.filter((card)=>card.dataset.materialCard === eventId);
  const currentIndex = cards.indexOf(sourceCard || (matches.length === 1 ? matches[0] : null));
  if (currentIndex < 0) return;
  const delta = direction === "previous" ? -1 : 1;
  const target = cards[currentIndex + delta];
  if (!target) { toast(direction === "previous" ? "已经是第一份素材。" : "已经是最后一份素材。"); return; }
  document.querySelectorAll(".material-product-details[open]").forEach((details)=>{ details.open = false; });
  const productDetails = target.querySelector(".material-product-details");
  const clipDetails = target.querySelector(".material-clip-details");
  if (productDetails) productDetails.open = true;
  if (clipDetails) clipDetails.open = true;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  setTimeout(()=>target.querySelector("video")?.focus({ preventScroll: true }), 280);
}


function bindMaterialInteractions(data) {
  const eventsByKey = new Map((data.key_events || []).map((event)=>[materialSelectionKey(event), event]));
  document.querySelector("[data-reset-material-filters]")?.addEventListener("click", () => {
    state.materialFilters = { ...state.materialFilters, group: "all", action: "all", object: "all", support: "all", query: "" };
    renderArchive(data.name, "materials", routeParts()[0] === "stage");
  });
  document.querySelectorAll(".material-product-details").forEach((details)=>details.addEventListener("toggle", () => {
    if (!details.open) { pauseMaterialVideos(details); return; }
    document.querySelectorAll(".material-product-details[open]").forEach((other)=>{ if (other !== details) other.open = false; });
  }));
  document.querySelectorAll(".material-clip-details").forEach((details)=>details.addEventListener("toggle",()=>{
    if (!details.open) pauseMaterialVideos(details);
  }));
  document.querySelectorAll("[data-material-select]").forEach((checkbox)=>checkbox.addEventListener("change", () => {
    const key = checkbox.closest("[data-material-selection-key]")?.dataset.materialSelectionKey;
    const event = eventsByKey.get(key);
    if (checkbox.checked && event) state.selectedMaterials.set(key, { ...event, release_id: event.release_id || data.release_id || null });
    else state.selectedMaterials.delete(key);
    updateMaterialSelectionBar();
  }));
  document.querySelectorAll("[data-material-focus]").forEach((button)=>button.addEventListener("click",()=>openMaterialFocus(data,button.dataset.materialFocus,null,button.closest("[data-material-selection-key]")?.dataset.materialSelectionKey)));
  document.querySelector("[data-select-visible]")?.addEventListener("click", () => {
    filteredMaterialEvents(data).forEach((event)=>state.selectedMaterials.set(materialSelectionKey(event), { ...event, release_id: event.release_id || data.release_id || null }));
    document.querySelectorAll("[data-material-select]").forEach((checkbox)=>{ checkbox.checked = true; });
    updateMaterialSelectionBar();
  });
  document.querySelectorAll("[data-video-seek]").forEach((button)=>button.addEventListener("click", () => {
    const video = materialVideo(button.dataset.videoSeek, button);
    if (!video) return;
    video.currentTime = Math.min(Number(button.dataset.seconds || 0), Number.isFinite(video.duration) ? video.duration : Number(button.dataset.seconds || 0));
  }));
  document.querySelectorAll("[data-video-fullscreen]").forEach((button)=>button.addEventListener("click", async () => {
    const video = materialVideo(button.dataset.videoFullscreen, button);
    try { if (video?.requestFullscreen) await video.requestFullscreen(); }
    catch { toast("当前浏览器无法进入全屏播放。", "error"); }
  }));
  document.querySelectorAll("[data-review-adjacent]").forEach((button)=>button.addEventListener("click", ()=>activateAdjacentMaterial(button.dataset.reviewEvent, button.dataset.reviewAdjacent,button.closest("[data-material-card]"))));
  document.querySelectorAll("[data-video-speed]").forEach((select)=>select.addEventListener("change", () => {
    const video = materialVideo(select.dataset.videoSpeed, select);
    if (video) video.playbackRate = Number(select.value || 1);
  }));
  document.querySelectorAll("[data-video-loop]").forEach((checkbox)=>checkbox.addEventListener("change", () => {
    const video = materialVideo(checkbox.dataset.videoLoop, checkbox);
    if (video) video.loop = checkbox.checked;
  }));
  const selectionBar = document.querySelector("#material-selection-bar");
  if (selectionBar && selectionBar.dataset.bound !== "true") {
    selectionBar.dataset.bound = "true";
    selectionBar.querySelector("[data-copy-selected]")?.addEventListener("click", async () => {
      const ids = selectedMaterialEvents(data).map((event)=>String(event.event_id));
      if (!ids.length) { toast("请先选择需要整理的关键素材。", "error"); return; }
      try { await navigator.clipboard.writeText(ids.join("\n")); toast(`已复制 ${ids.length} 个素材编号。`); }
      catch { toast("当前浏览器未开放剪贴板权限。", "error"); }
    });
    selectionBar.querySelector("[data-export-selected]")?.addEventListener("click", ()=>exportSelectedMaterials(data));
    selectionBar.querySelector("[data-clear-selected]")?.addEventListener("click", () => {
      state.selectedMaterials.clear();
      document.querySelectorAll("[data-material-select]").forEach((checkbox)=>{ checkbox.checked = false; });
      updateMaterialSelectionBar();
    });
  }
  updateMaterialSelectionBar();
}


  return Object.freeze({ materialLibraryEntries, automaticReviewLabel, materialFocusRoute, globalMaterialCard, bindLibraryImageFallbacks, renderMaterialsLibrary, materialOperationTitle, materialCard, eventHasDualViewSupport, eventHasAlignedDualViewMaterial, ensureMaterialFilters, materialSelectionKey, materialSelectionScope, ensureMaterialSelection, filteredMaterialEvents, materialResults, materialSelectionBar, retainedMaterialsView, bindRetainedMaterialFilters, movementScreeningView, materialsView, updateMaterialResults, bindMaterialFilters, selectedMaterialEvents, updateMaterialSelectionBar, exportSelectedMaterials, materialVideo, pauseMaterialVideos, materialFocusMarkup, openLinkedMaterial, openMaterialFocus, activateAdjacentMaterial, bindMaterialInteractions });
}
});
