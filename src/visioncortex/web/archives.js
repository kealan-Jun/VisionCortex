"use strict";

// Explicit application ports keep domain modules independently testable.
window.VisionCortexArchives = Object.freeze({
createArchives(context) {
  const { api, archiveProcessStopped, archiveSearchQuery, ensureMaterialFilters, experimentRecords, number, refreshProgressiveSurface, routeParts, state } = context;

function invalidateArchiveCache(name) {
  state.archiveCacheEpoch = (state.archiveCacheEpoch || 0) + 1;
  if (state.archiveView?.name === name && !state.archiveView.staging) state.archiveView.refreshPending = true;
  [...state.archiveCache.keys()].filter((key)=>key === `archive/${name}` || key.startsWith(`${name}:`)).forEach((key)=>state.archiveCache.delete(key));
  [...state.materialCache.keys()].filter((key)=>key.startsWith(`${name}:`)).forEach((key)=>state.materialCache.delete(key));
}


function applyArchiveListingPage(payload, append = false) {
  const previous = new Map(state.archives.map((archive)=>[archive.name, archive]));
  for (const archive of payload.archives || []) {
    // An archive may have cached details even when it is outside the current directory page.
    const known = [previous.get(archive.name), state.archiveCache.get(`archive/${archive.name}`), state.archiveCache.get(`${archive.name}:summary`)].filter(Boolean);
    if (known.some((old)=>old.release_id !== archive.release_id || ["modified_at", "key_event_count"].some((field)=>old[field] !== undefined && old[field] !== archive[field]))) invalidateArchiveCache(archive.name);
  }
  state.archives = append && payload.total_count !== 0 ? [...new Map([...state.archives, ...(payload.archives || [])].map((archive)=>[archive.name, archive])).values()] : payload.archives || [];
  state.archiveNextCursor = payload.next_cursor || null;
  state.archiveTotal = Number(payload.total_count ?? state.archives.length);
  state.archiveTotals = payload.totals || (append ? state.archiveTotals : { experiments: 0, key_events: 0 });
}


async function loadArchiveListing(query = archiveSearchQuery()) {
  const requestId = state.archiveRequestId = (state.archiveRequestId || 0) + 1;
  let payload;
  try {
    payload = await api(`/api/archives?limit=50${query ? `&q=${encodeURIComponent(query)}` : ""}`);
  } catch (error) {
    if (requestId !== state.archiveRequestId || query !== archiveSearchQuery()) return false;
    state.archiveRefreshPending = true;
    throw error;
  }
  if (requestId !== state.archiveRequestId || query !== archiveSearchQuery()) return false;
  applyArchiveListingPage(payload);
  state.archiveListingQuery = query;
  state.archiveRefreshPending = false;
  state.libraryReleaseRefreshPending = false;
  return true;
}


function cachedArchiveDetail(name) {
  return state.archiveCache.get(`archive/${name}`) || null;
}


function libraryRecordKey(record) {
  return record.staging_run_id ? `stage/${record.staging_run_id}` : `archive/${record.name}`;
}


function cachedLibraryDetail(record) {
  return state.archiveCache.get(libraryRecordKey(record)) || null;
}


function libraryRecordQueryKey(record, section) {
  // Retained-run responses are unfiltered; reuse them across local filters.
  return record.staging_run_id ? JSON.stringify(["retained", record.material_revision]) : libraryQueryKey(section);
}


function libraryRequestKey(record, section) {
  return JSON.stringify([section, libraryRecordKey(record), libraryRecordQueryKey(record, section), state.archiveCacheEpoch || 0]);
}


function libraryLoadState(section) {
  const records = experimentRecords();
  const ready = [], failed = [], pending = [];
  for (const record of records) {
    if (cachedLibraryDetail(record)?.libraryKeys?.[section] === libraryRecordQueryKey(record, section)) ready.push(record);
    else if (state.libraryLoadErrors.has(libraryRequestKey(record, section))) failed.push(record);
    else pending.push(record);
  }
  return { records, ready, failed, pending };
}


function libraryFailureNotice(status, section) {
  if (!status.failed.length) return "";
  const label = section === "reports" ? "日报" : "素材";
  return `<div class="freshness-warning" role="alert"><p>${number(status.failed.length)} 个实验的${label}暂未载入，当前结果不完整。</p><button class="secondary-button" type="button" data-retry-library="${section}">重试未载入的${label}</button></div>`;
}


function bindLibraryRetry(section) {
  document.querySelector(`[data-retry-library="${section}"]`)?.addEventListener("click", () => {
    libraryLoadState(section).failed.forEach((record)=>state.libraryLoadErrors.delete(libraryRequestKey(record, section)));
    refreshProgressiveSurface();
  });
}


async function loadLibraryRecord(record, section) {
  if (!record.staging_run_id) return loadLibraryDetail(record.name, section);
  const queryKey = libraryRecordQueryKey(record, section), epoch = state.archiveCacheEpoch || 0;
  const key = libraryRecordKey(record);
  const payload = await api(`/api/staging-runs/${encodeURIComponent(record.staging_run_id)}/archive?section=library-${section}`);
  if (epoch !== (state.archiveCacheEpoch || 0)) throw new Error("档案已更新，请重新载入。");
  const previous = state.archiveCache.get(key) || {};
  const data = {...previous,...payload,libraryKeys:{...(previous.libraryKeys || {}),[section]:queryKey}};
  delete data.fullDetailLoadedAt;
  state.archiveCache.set(key,data);
  return data;
}


function libraryQueryKey(section) {
  return section === "reports" ? "reports" : JSON.stringify([state.globalMaterialFilters, state.search.trim()]);
}


function queueLibraryReleaseRefresh(name, section) {
  invalidateArchiveCache(name);
  state.archiveRefreshPending = true;
  state.libraryReleaseRefreshPending = true;
  // Wait for the directory poll before retrying a conflicting material response.
  state.libraryLoadErrors.add(libraryRequestKey({name}, section));
}


async function loadLibraryDetail(name, section, cursor = null) {
  const queryKey = libraryQueryKey(section);
  const filters = { ...state.globalMaterialFilters };
  const search = state.search.trim();
  const summary = await loadArchive(name);
  const epoch = state.archiveCacheEpoch || 0;
  if (queryKey !== libraryQueryKey(section)) return null;
  let prior = cachedArchiveDetail(name) || {};
  if (prior.release_id !== summary.release_id) prior = {};
  if (cursor && prior.libraryKeys?.[section] !== queryKey) return null;
  let payload;
  if (archiveProcessStopped(summary)) payload = summary;
  else if (section === "reports") {
    payload = { ...await loadArchiveSection(name, "reports") };
    for (const key of ["experiments", "key_events", "preliminary_materials"]) delete payload[key];
  }
  else {
    const parameters = new URLSearchParams({ archive: name, limit: "24", material_ready: "true" });
    if (filters.action !== "all") parameters.set("action_type", filters.action);
    if (["dual", "partial"].includes(filters.support)) parameters.set("cross_view", String(filters.support === "dual"));
    const query = [search, filters.object !== "all" ? filters.object : ""].filter(Boolean).join(" ");
    if (query) parameters.set("q", query);
    if (cursor) parameters.set("cursor", cursor);
    let page;
    try {
      page = await api(`/api/key-events?${parameters}`);
    } catch (error) {
      if (error.status === 409 && queryKey === libraryQueryKey(section) && epoch === (state.archiveCacheEpoch || 0)) queueLibraryReleaseRefresh(name, section);
      throw error;
    }
    if (queryKey !== libraryQueryKey(section) || epoch !== (state.archiveCacheEpoch || 0)) return null;
    if ((page.items || []).some((item)=>(item.release_id || null) !== (summary.release_id || null))) {
      queueLibraryReleaseRefresh(name, section);
      throw new Error("实验档案版本已更新，系统正在重新同步素材。");
    }
    payload = {
      key_events: cursor ? [...(prior.key_events || []), ...(page.items || [])] : page.items || [],
      material_next_cursor: page.next_cursor || null,
      material_total_count: Number(page.total_count || 0),
    };
  }
  if (queryKey !== libraryQueryKey(section) || epoch !== (state.archiveCacheEpoch || 0)) return null;
  prior = cachedArchiveDetail(name) || {};
  if (prior.release_id !== summary.release_id) prior = {};
  const metadata = { ...summary };
  // Empty section placeholders in the summary must not erase another section.
  for (const key of ["experiments", "key_events", "preliminary_materials"]) delete metadata[key];
  const result = { ...prior, ...metadata, ...payload, libraryKeys: { ...prior.libraryKeys, [section]: queryKey } };
  state.archiveCache.set(`archive/${name}`, result);
  return result;
}


function ensureLibraryDetails() {
  if (state.libraryLoadPromise) return state.libraryLoadPromise;
  const section = routeParts()[0] === "reports" ? "reports" : "materials";
  if (section === "materials" && state.libraryReleaseRefreshPending) return Promise.resolve();
  const queryKey = libraryQueryKey(section);
  const epoch = state.archiveCacheEpoch || 0;
  const missing = libraryLoadState(section).pending;
  if (!missing.length) return Promise.resolve();
  state.libraryLoadPromise = (async () => {
    for (let offset = 0; offset < missing.length; offset += 3) {
      if (queryKey !== libraryQueryKey(section) || epoch !== (state.archiveCacheEpoch || 0)) break;
      const batch = missing.slice(offset, offset + 3);
      const requestKeys = batch.map((record)=>libraryRequestKey(record, section));
      const results = await Promise.allSettled(batch.map((archive) => loadLibraryRecord(archive, section)));
      results.forEach((result, index) => {
        if (result.status === "rejected") state.libraryLoadErrors.add(requestKeys[index]);
      });
      refreshProgressiveSurface();
    }
  })().finally(() => {
    state.libraryLoadPromise = null;
    refreshProgressiveSurface();
  });
  return state.libraryLoadPromise;
}


async function loadMoreArchives() {
  if (!state.archiveNextCursor) return;
  const query = state.archiveListingQuery;
  const requestId = state.archiveRequestId;
  const cursor = state.archiveNextCursor;
  if (query !== archiveSearchQuery()) return;
  const payload = await api(`/api/archives?limit=50&cursor=${encodeURIComponent(cursor)}${query ? `&q=${encodeURIComponent(query)}` : ""}`);
  if (requestId !== state.archiveRequestId || query !== archiveSearchQuery() || cursor !== state.archiveNextCursor) return;
  applyArchiveListingPage(payload, true);
}


async function loadArchive(name, staging = false) {
  const epoch = state.archiveCacheEpoch || 0;
  if (staging) {
    const key = `stage/${name}`, cached = state.archiveCache.get(key);
    if (cached?.fullDetailLoadedAt && Date.now()-cached.fullDetailLoadedAt < 2000) return cached;
    state.stagingDetailRequests ||= new Map();
    const requestKey = `${epoch}:${name}`;
    if (state.stagingDetailRequests.has(requestKey)) return state.stagingDetailRequests.get(requestKey);
    const pending = api(`/api/staging-runs/${encodeURIComponent(name)}/archive`).then(data=>{
      data.fullDetailLoadedAt = Date.now();
      // A directory poll may invalidate formal archives while this run loads.
      // Return its response to the route-generation guard, but do not cache it.
      if (epoch === (state.archiveCacheEpoch || 0)) state.archiveCache.set(key,data);
      return data;
    }).finally(()=>state.stagingDetailRequests.delete(requestKey));
    state.stagingDetailRequests.set(requestKey,pending);
    return pending;
  }
  const summaryKey = `${name}:summary`;
  const cachedSummary = state.archiveCache.get(summaryKey);
  if (!cachedSummary || Date.now() - (cachedSummary.loadedAt || 0) > 2000) {
    const summary = await api(`/api/archives/${encodeURIComponent(name)}?section=summary`);
    if (epoch !== (state.archiveCacheEpoch || 0)) throw new Error("档案已更新，请重新载入。");
    const previousRelease = state.archiveCache.get(summaryKey)?.release_id;
    if (previousRelease !== undefined && previousRelease !== summary.release_id) {
      invalidateArchiveCache(name);
    }
    state.archiveCache.set(summaryKey, { experiments: [], key_events: [], preliminary_materials: [], ...summary, loadedAt: Date.now() });
  }
  return state.archiveCache.get(summaryKey);
}


async function loadArchiveSection(name, section, cursor = null) {
  const summary = await loadArchive(name);
  const epoch = state.archiveCacheEpoch || 0;
  const key = `${name}:${summary.release_id || "legacy"}:${section}:${cursor || "first"}`;
  if (!state.archiveCache.has(key)) {
    const url = `/api/archives/${encodeURIComponent(name)}?section=${encodeURIComponent(section)}&limit=24${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`;
    const payload = await api(url);
    if (epoch !== (state.archiveCacheEpoch || 0)) throw new Error("档案已更新，请重新载入。");
    if (payload.release_id !== summary.release_id) {
      invalidateArchiveCache(name);
      throw new Error("实验档案刚刚发布了新版本，请重新打开");
    }
    state.archiveCache.set(key, { ...summary, ...payload });
  }
  return state.archiveCache.get(key);
}


function materialQueryParameters(name, cursor = null) {
  const filters = state.materialFilters;
  const parameters = new URLSearchParams({ archive: name, limit: "24", material_ready: "true" });
  if (filters.group && filters.group !== "all") parameters.set("parent_event_id", filters.group);
  if (filters.action !== "all") parameters.set("action_type", filters.action);
  if (filters.support === "dual") parameters.set("cross_view", "true");
  if (filters.support === "partial") parameters.set("cross_view", "false");
  const query = [filters.query.trim(), filters.object !== "all" ? filters.object : ""].filter(Boolean).join(" ");
  if (query) parameters.set("q", query);
  if (cursor) parameters.set("cursor", cursor);
  return parameters;
}


async function loadArchiveMaterials(name, cursor = null) {
  ensureMaterialFilters({name});
  const parameters = materialQueryParameters(name, cursor);
  const queryKey = materialQueryParameters(name).toString();
  const summary = await loadArchive(name);
  const epoch = state.archiveCacheEpoch || 0;
  const isCurrent = () => state.materialFilters.archive === name && materialQueryParameters(name).toString() === queryKey && (state.archiveCacheEpoch || 0) === epoch;
  if (!isCurrent()) throw new Error("素材筛选或档案已更新，请重新加载。");
  if (archiveProcessStopped(summary)) return summary;
  const groupPayload = await loadArchiveSection(name, "experiments");
  if (!isCurrent()) throw new Error("素材筛选或档案已更新，请重新加载。");
  const baseKey = `${name}:${summary.release_id || "legacy"}:materials:${queryKey}`;
  if (!cursor && state.materialCache.has(baseKey)) return state.materialCache.get(baseKey);
  const previous = cursor ? state.materialCache.get(baseKey) : null;
  if (cursor && previous?.material_next_cursor !== cursor) throw new Error("分页位置已更新，请重新加载素材。");
  const payload = await api(`/api/key-events?${parameters.toString()}`);
  if (!isCurrent()) throw new Error("素材筛选或档案已更新，请重新加载。");
  if ((payload.items || []).some((item) => (item.release_id || null) !== (summary.release_id || null))) {
    invalidateArchiveCache(name);
    throw new Error("素材所属版本已更新，请重新加载档案");
  }
  if (cursor && state.materialCache.get(baseKey) !== previous) return state.materialCache.get(baseKey);
  const result = {
    ...summary,
    experiment_groups: groupPayload.experiment_groups || [],
    key_events: previous ? [...previous.key_events, ...(payload.items || [])] : (payload.items || []),
    material_total_count: Number(payload.total_count || 0),
    material_next_cursor: payload.next_cursor || null,
  };
  state.materialCache.set(baseKey, result);
  return result;
}


async function loadArchiveExperiments(name, cursor = null) {
  const summary = await loadArchive(name);
  const epoch = state.archiveCacheEpoch || 0;
  const baseKey = `${name}:${summary.release_id || "legacy"}:experiments:accumulated`;
  if (!cursor && state.archiveCache.has(baseKey)) return state.archiveCache.get(baseKey);
  const previous = cursor ? state.archiveCache.get(baseKey) : null;
  if (cursor && previous?.next_cursor !== cursor) throw new Error("分页位置已更新，请重新加载实验片段。");
  const payload = await loadArchiveSection(name, "experiments", cursor);
  if (epoch !== (state.archiveCacheEpoch || 0)) throw new Error("档案已更新，请重新载入。");
  if (cursor && state.archiveCache.get(baseKey) !== previous) return state.archiveCache.get(baseKey);
  const result = {
    ...summary,
    ...payload,
    experiments: previous ? [...previous.experiments, ...(payload.experiments || [])] : (payload.experiments || []),
  };
  state.archiveCache.set(baseKey, result);
  return result;
}


async function loadArchiveView(name, tab, cursor = null) {
  if (tab === "speech") return loadArchive(name);
  if (tab === "materials") return loadArchiveMaterials(name, cursor);
  if (tab === "experiments") return loadArchiveExperiments(name, cursor);
  return loadArchiveSection(name, tab === "reports" ? "reports" : "metrics", cursor);
}


  return Object.freeze({ invalidateArchiveCache, applyArchiveListingPage, loadArchiveListing, cachedArchiveDetail, libraryRecordKey, cachedLibraryDetail, libraryRecordQueryKey, libraryRequestKey, libraryLoadState, libraryFailureNotice, bindLibraryRetry, loadLibraryRecord, libraryQueryKey, queueLibraryReleaseRefresh, loadLibraryDetail, ensureLibraryDetails, loadMoreArchives, loadArchive, loadArchiveSection, materialQueryParameters, loadArchiveMaterials, loadArchiveExperiments, loadArchiveView });
}
});
