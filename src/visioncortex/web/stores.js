"use strict";

window.VisionCortexStores = Object.freeze({
createStores() {
const initial = {
  archives: [],
  deviceDayArchives: [],
  deviceDayErrors: [],
  deviceDayQueue: {},
  deviceDayUpdatedAt: 0,
  archiveNextCursor: null,
  archiveTotal: 0,
  archiveTotals: { experiments: 0, key_events: 0 },
  archiveListingQuery: null,
  archiveRequestId: 0,
  archiveCacheEpoch: 0,
  archiveViewRequestId: 0,
  archiveView: null,
  health: null,
  runs: [],
  collections: [],
  nasRecordings: [],
  nasLoading: true,
  nasRequestId: 0,
  nasSyncError: false,
  nasRenderPending: false,
  nasBatches: [],
  nasMonitor: null,
  nasSelection: {},
  nasQuery: "",
  nasCompleteConfirmed: false,
  nasError: "",
  sources: [],
  activeRun: null,
  followRun: null,
  taskDisclosures: new Map(),
  selectedCollectionId: null,
  collectionQuery: "",
  experimentName: "",
  archiveCache: new Map(),
  materialCache: new Map(),
  search: "",
  refreshingTasks: false,
  taskSyncError: false,
  archiveFilters: { status: "all", date: "all", owner: "all", tag: "all", view: "list" },
  materialFilters: { archive: null, group: null, action: "all", object: "all", support: "all", query: "" },
  globalMaterialFilters: { date: "all", action: "all", object: "all", support: "all" },
  globalMaterialLimit: 18,
  libraryLoadPromise: null,
  libraryLoadErrors: new Set(),
  focusedMaterialId: null,
  searchActiveIndex: -1,
  globalMaterialSearch: null,
  selectedMaterials: new Map(),
  materialSelectionSource: null,
  annotationFilters: { priority: "", reviewStatus: "", query: "" },
  activeUploadSessionId: null,
  uploadAbortController: null,
  uploadCancelled: false,
};
const ownership = {"uploads": ["activeUploadSessionId", "collectionQuery", "experimentName", "selectedCollectionId", "sources", "uploadAbortController", "uploadCancelled"], "capture": ["deviceDayArchives", "deviceDayErrors", "deviceDayQueue", "deviceDaySyncError", "deviceDayUpdatedAt", "nasAutomaticProcessing", "nasBatches", "nasCompleteConfirmed", "nasError", "nasLoading", "nasMonitor", "nasQuery", "nasRecordings", "nasRenderPending", "nasRequestId", "nasSelection", "nasSyncError"], "tasks": ["activeRun", "followRun", "refreshingTasks", "runPollRequestId", "runs", "taskDisclosures", "taskSyncError"], "archives": ["annotationFilters", "archiveCache", "archiveCacheEpoch", "archiveFilters", "archiveListingQuery", "archiveNextCursor", "archiveRefreshPending", "archiveRequestId", "archiveTotal", "archiveTotals", "archiveView", "archiveViewRequestId", "archives", "focusedMaterialId", "globalMaterialFilters", "globalMaterialLimit", "globalMaterialSearch", "libraryLoadErrors", "libraryLoadPromise", "libraryReleaseRefreshPending", "librarySyncState", "materialCache", "materialFilters", "materialSelectionSource", "search", "searchActiveIndex", "selectedMaterials"], "shell": ["collections", "experimentMetadataCache", "health", "retainedMaterialFilters", "stagingDetailRequests"]};
const stores = {};
const state = {};
for (const [domain, keys] of Object.entries(ownership)) {
  const store = stores[domain] = {};
  for (const key of keys) {
    store[key] = initial[key];
    Object.defineProperty(state, key, {enumerable: true,
      get: () => store[key], set: value => { store[key] = value; }});
  }
}
// The flat view is the compatibility surface; each value has one owning store.
return {state, stores};
}
});
