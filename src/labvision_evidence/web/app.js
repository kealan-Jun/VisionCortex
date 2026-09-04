const brandLogo = window.VisionCortexBrandLogoDataUrl;
if (brandLogo) {
  const logo = document.querySelector("#product-logo");
  const favicon = document.querySelector("#product-favicon");
  if (logo) logo.src = brandLogo;
  if (favicon) favicon.href = brandLogo;
}

const ICONS = {
  dashboard: '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
  flask: '<svg viewBox="0 0 24 24"><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3M7.5 15h9"/></svg>',
  activity: '<svg viewBox="0 0 24 24"><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  boxes: '<svg viewBox="0 0 24 24"><path d="m12 2 8 4-8 4-8-4 8-4Z"/><path d="m4 10 8 4 8-4M4 14l8 4 8-4M4 18l8 4 8-4"/></svg>',
  file: '<svg viewBox="0 0 24 24"><path d="M6 2h8l4 4v16H6zM14 2v5h5M9 13h6M9 17h6"/></svg>',
  server: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="6" rx="2"/><rect x="3" y="14" width="18" height="6" rx="2"/><path d="M7 7h.01M7 17h.01M17 7h1M17 17h1"/></svg>',
  chevron: '<svg viewBox="0 0 24 24"><path d="m15 18-6-6 6-6"/></svg>',
  search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>',
  refresh: '<svg viewBox="0 0 24 24"><path d="M20 11a8 8 0 1 0-2 5.5M20 4v7h-7"/></svg>',
  upload: '<svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5M4 20h16"/></svg>',
  video: '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="14" height="14" rx="2"/><path d="m17 10 4-2v8l-4-2z"/></svg>',
  clock: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg>',
  gauge: '<svg viewBox="0 0 24 24"><path d="M4 18a9 9 0 1 1 16 0M12 12l4-4M7 18h10"/></svg>',
  brain: '<svg viewBox="0 0 24 24"><path d="M9 4a3 3 0 0 0-5 2 3 3 0 0 0 0 6 3 3 0 0 0 2 5 3 3 0 0 0 6 0V6a3 3 0 0 0-3-2ZM15 4a3 3 0 0 1 5 2 3 3 0 0 1 0 6 3 3 0 0 1-2 5 3 3 0 0 1-6 0V6a3 3 0 0 1 3-2Z"/><path d="M8 9h4M16 9h-4M8 14h4M16 14h-4"/></svg>',
  folder: '<svg viewBox="0 0 24 24"><path d="M3 6h7l2 2h9v11H3z"/></svg>',
  copy: '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/></svg>',
  arrow: '<svg viewBox="0 0 24 24"><path d="M5 12h14M14 7l5 5-5 5"/></svg>',
  image: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 15-5-5L5 20"/></svg>',
  token: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8 9h8M8 13h8M10 17h4"/></svg>',
  target: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>',
};

const state = {
  archives: [],
  health: null,
  runs: [],
  collections: [],
  nasRecordings: [],
  nasLoading: true,
  nasBatches: [],
  nasMonitor: null,
  nasSelection: {},
  nasQuery: "",
  nasCompleteConfirmed: false,
  nasError: "",
  sources: [],
  activeRun: null,
  selectedCollectionId: null,
  collectionQuery: "",
  experimentName: "",
  archiveCache: new Map(),
  search: "",
  refreshingTasks: false,
  archiveFilters: { status: "all", date: "all", owner: "all", tag: "all", view: "list" },
  materialFilters: { archive: null, group: null, action: "all", support: "all", query: "" },
  selectedMaterials: new Set(),
  materialArchive: null,
  annotationFilters: { priority: "", reviewStatus: "", query: "" },
  activeUploadSessionId: null,
  uploadAbortController: null,
  uploadCancelled: false,
};

const main = document.querySelector("#main-content");
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);
const icon = (name) => ICONS[name] || "";
const number = (value) => Number(value || 0).toLocaleString("zh-CN");
const formatDate = (value) => value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—";
const formatBytes = (bytes) => {
  const value = Number(bytes || 0);
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index > 1 ? 2 : 0)} ${units[index]}`;
};
const timecode = (milliseconds) => {
  if (milliseconds === null || milliseconds === undefined) return "—";
  const total = Math.max(0, Number(milliseconds));
  const hours = Math.floor(total / 3600000);
  const minutes = Math.floor((total % 3600000) / 60000);
  const seconds = Math.floor((total % 60000) / 1000);
  const millis = Math.round(total % 1000);
  return `${String(hours).padStart(2,"0")}:${String(minutes).padStart(2,"0")}:${String(seconds).padStart(2,"0")}.${String(millis).padStart(3,"0")}`;
};
const duration = (seconds) => {
  if (seconds === null || seconds === undefined) return "—";
  const value = Math.max(0, Number(seconds));
  if (value >= 3600) return `${Math.floor(value / 3600)}时${Math.floor(value % 3600 / 60)}分${Math.round(value % 60)}秒`;
  if (value >= 60) return `${Math.floor(value / 60)}分${String(Math.round(value % 60)).padStart(2,"0")}秒`;
  return `${value.toFixed(value < 10 ? 2 : 1)}秒`;
};
const percent = (value) => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
const metricStat = (metric, field = "mean", suffix = "") => metric?.[field] == null ? "—" : `${Number(metric[field]).toFixed(1)}${suffix}`;
const latestTelemetry = (run) => run?.observability?.live_telemetry?.latest || {};
const isNasMode = () => state.health?.storage_mode === "nas";
const archiveLabel = () => "实验档案";
const archiveShortLabel = () => "实验";
const EXPERIMENT_META_KEY = "visioncortex-experiment-metadata-v1";

function loadExperimentMetadata() {
  try { return JSON.parse(localStorage.getItem(EXPERIMENT_META_KEY) || "{}"); }
  catch { return {}; }
}

function experimentMetadata(name) {
  return loadExperimentMetadata()[String(name || "")] || {};
}

function saveExperimentMetadata(name, metadata) {
  const records = loadExperimentMetadata();
  records[String(name)] = metadata;
  localStorage.setItem(EXPERIMENT_META_KEY, JSON.stringify(records));
}

const productExperimentName = (value) => {
  const name = String(value || "实验记录");
  const customName = String(experimentMetadata(name).displayName || "").trim();
  if (customName) return customName;
  const batch = name.match(/^Batch-(\d+)-(\d{4})-(\d{2})-(\d{2})(?:_|$)/i);
  if (batch) return `第 ${Number(batch[1])} 批实验 · ${batch[2]}-${batch[3]}-${batch[4]}`;
  const weighing = name.match(/^FlowTest-Weighing-R(\d+)$/i);
  if (weighing) return `称量实验 · 第 ${Number(weighing[1])} 次`;
  const collection = name.match(/^VisionCortex-Collection-(\d{4})(\d{2})(\d{2})-/i);
  if (collection) return `采集实验 · ${collection[1]}-${collection[2]}-${collection[3]}`;
  return name;
};

function pageSkeleton(label = "正在读取实验档案") {
  return `<div class="page-skeleton" aria-busy="true" aria-label="${esc(label)}"><div class="skeleton-hero"><i></i><i></i><i></i></div><div class="skeleton-stat-grid">${Array.from({ length: 4 }, () => `<i></i>`).join("")}</div><div class="skeleton-panel"><i></i><i></i><i></i><i></i></div><span class="sr-only">${esc(label)}</span></div>`;
}

const ACTION_LABELS = {
  hand_object_contact: "手部与物体接触",
  object_movement: "物体移动",
  liquid_movement: "液体移动",
  liquid_transfer: "液体移动",
  pipette_transfer_operation: "移液器源到目标操作（液体不可见）",
  container_state_change: "容器状态变化",
  device_panel_operation: "设备面板操作",
  panel_operation: "设备面板操作",
};
const OBJECT_ROLE_LABELS = {
  actor: "操作者",
  tool: "使用工具",
  source: "来源",
  target: "操作对象",
  destination: "目标位置",
  container: "容器",
  object: "相关对象",
};
const OBJECT_LABELS = {
  gloved_hand: "戴手套的手",
  hand: "手部",
  balance: "电子天平",
  scale: "电子秤",
  spatula: "药匙",
  reagent_bottle: "试剂瓶",
  bottle: "试剂瓶",
  bottle_cap: "瓶盖",
  tube: "离心管",
  tube_rack: "离心管架",
  paper: "称量纸",
  weighing_paper: "称量纸",
  pipette: "移液器",
  beaker: "烧杯",
  container: "容器",
  unknown: "暂未识别",
};

function productObjectLabel(value) {
  const raw = String(value || "unknown");
  const unresolved = /-unresolved$/i.test(raw);
  const token = raw.replace(/-unresolved$/i, "").replace(/-\d+$/i, "").toLocaleLowerCase();
  const label = OBJECT_LABELS[token] || "相关对象";
  return unresolved ? `${label}（具体实例待确认）` : label;
}

function productEvidenceText(value, fallback) {
  let text = String(value || "").trim();
  if (!text || ["未知", "unknown"].includes(text.toLocaleLowerCase())) return fallback;
  text = text.replace(/^(?:未知|unknown)\s*[；;，,:：]\s*/i, "");
  if (!text) return fallback;
  for (const [token,label] of Object.entries(OBJECT_LABELS).sort(([left],[right]) => right.length - left.length)) text = text.replaceAll(token, label);
  return text;
}

function materialObjectItems(objects = {}) {
  const entries = Object.entries(objects);
  if (!entries.length) return `<span class="object-chip muted">暂未识别明确对象</span>`;
  return entries.map(([role,value]) => `<span class="object-chip" title="${esc(role)}: ${esc(value)}"><small>${esc(OBJECT_ROLE_LABELS[role] || "相关对象")}</small>${esc(productObjectLabel(value))}</span>`).join("");
}
const VERIFICATION_STATUS_LABELS = {
  secondary_verification_executed: "已执行二次复核",
  clear_closed_set_evidence: "闭集证据清晰，无需重模型",
  verification_deferred_budget_exhausted: "复核预算已满，保留不确定性",
  verification_recorded: "已有模型复核账本",
  not_available_historical_archive: "历史档案无复核账本",
};
const MODEL_LABELS = {
  closed_set_yolo_tensorrt: "闭集 YOLO TensorRT",
  yolo_world_v2: "YOLO-World v2",
  grounding_dino_base: "Grounding DINO Base",
  sam2_temporal_participant: "SAM2 时序参与对象",
  labpics_pspnet_liquid_semantic: "LabPics PSPNet 液体语义",
};
const STAGE_LABELS = {
  reserving: "准备实验空间",
  original_ingest: "保存实验视频",
  nas_ingest: "读取实验视频",
  queued: "排队等待",
  running: "正在启动",
  preflight: "检查视频",
  alignment: "同步不同视角",
  motion_probe: "定位实验活动",
  candidate_coarse: "识别实验片段",
  candidate_fine: "确认片段边界",
  candidate_audit: "复核实验片段",
  experiment_understanding: "理解实验步骤",
  experiment_clips: "整理实验片段",
  key_materials: "关键素材生成",
  mllm: "关键素材模型理解",
  material_refinement: "复核关键画面",
  semantic_refinement: "复核实验步骤",
  package: "检查结果完整性",
  daily_report: "生成实验报告",
  completed: "分析完成",
  failed: "分析失败",
  interrupted: "服务重启后待续跑",
};

const GUIDED_PIPELINE = [
  {
    id: "originals", number: "01", title: "导入实验视频",
    stages: ["reserving", "original_ingest", "nas_ingest"], completedBy: ["original_ingest"],
    folder: "Original-Experiment-Videos",
    outputs: [["Original-Video-Index.json","原始输入索引"],["nas_ingest.json","NAS 输入清单"]],
    doing: "检查实验视频与时间记录，并保存到本次实验中。",
    outcome: "完整保存实验视频、时间记录与拍摄视角。",
  },
  {
    id: "alignment", number: "02", title: "同步不同视角",
    stages: ["preflight", "alignment"], completedBy: ["alignment"],
    folder: "JSON-Config-Files",
    outputs: [["time_alignment.json","对齐结果"],["aligned_timestamps.csv","对齐时间戳"]],
    doing: "检查视频质量，并把第一人称与第三人称画面对齐到同一时间轴。",
    outcome: "得到可以同步回看的多视角实验视频。",
  },
  {
    id: "discovery", number: "03", title: "识别实验片段",
    stages: ["motion_probe", "candidate_coarse", "candidate_fine", "candidate_audit"], completedBy: ["candidate_audit"],
    folder: "JSON-Config-Files",
    outputs: [["candidate_layer.json","实验候选"],["audit_layer.json","候选审计"]],
    doing: "定位有效实验过程，确认片段开始与结束时间，并检查跨视角连续性。",
    outcome: "保留时间一致、边界清晰的实验片段。",
  },
  {
    id: "clips", number: "04", title: "理解实验步骤",
    stages: ["experiment_understanding", "experiment_clips"], completedBy: ["experiment_clips"],
    folder: "Experiment-Clips",
    outputs: [["experiment_group_understanding.json","步骤理解"]], resultTab: ["experiments","实验片段"],
    doing: "判断独立或连续实验，理解当前步骤和可能的下一步，并整理回看视频。",
    outcome: "得到多视角实验片段与逐步实验说明。",
  },
  {
    id: "materials", number: "05", title: "生成关键素材",
    stages: ["key_materials", "mllm", "material_refinement", "semantic_refinement"], completedBy: ["semantic_refinement"],
    folder: "Key-Materials",
    outputs: [["semantic_key_material_curation.json","关键素材筛选"],["final_key_material_annotation.json","对象框复核"]], resultTab: ["materials","关键素材"],
    doing: "提取关键画面和短片，并结合不同视角理解当前与下一步骤。",
    outcome: "得到按实验整理的关键画面、片段与步骤说明。",
  },
  {
    id: "evidence", number: "06", title: "检查分析结果",
    stages: ["package"], completedBy: ["package"],
    folder: "JSON-Config-Files",
    outputs: [], resultTab: ["metrics","专业报告"],
    doing: "检查片段、关键素材和跨视角信息是否完整，并整理报告数据。",
    outcome: "得到完整性检查结果与可追溯的分析资料。",
  },
  {
    id: "reports", number: "07", title: "生成报告并完成归档",
    stages: ["daily_report", "completed"], completedBy: ["daily_report", "completed"],
    folder: "Lab-Daily-Reports / Professional-PDFs",
    outputs: [], resultTab: ["reports","实验室日报"],
    doing: "汇总实验过程与关键发现，生成实验室日报、专业 PDF 和结构化 JSON 报告。",
    outcome: "实验室日报、专业报告和完整实验档案。",
  },
];

async function api(url, options) {
  const response = await fetch(url, options);
  let payload;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) {
    const detail = payload?.detail;
    const shortage = Number(detail?.missing_bytes || 0);
    const message = typeof detail === "string"
      ? detail
      : `${detail?.message || `${response.status} ${response.statusText}`}${shortage ? `，还差 ${formatBytes(shortage)}` : ""}`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function toast(message, tone = "") {
  const item = document.createElement("div");
  item.className = `toast ${tone}`;
  item.textContent = message;
  document.querySelector("#toast-region").append(item);
  setTimeout(() => item.remove(), 4200);
}

function productState(tone, iconName, title, copy, actions = "") {
  return `<div class="product-state ${esc(tone || "neutral")}" role="${tone === "error" ? "alert" : "status"}"><span class="product-state-icon" aria-hidden="true">${icon(iconName || "file")}</span><div><strong>${esc(title)}</strong><p>${esc(copy)}</p>${actions ? `<div class="product-state-actions">${actions}</div>` : ""}</div></div>`;
}

function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((element) => {
    element.innerHTML = icon(element.dataset.icon);
  });
}

function pageContext(route) {
  if (route === "new") return ["实验工作台", "新建实验"];
  if (route === "tasks") return ["实验工作台", "分析进度"];
  if (route === "materials") return ["实验成果", "关键素材库"];
  if (route === "reports") return ["实验成果", "实验室日报"];
  if (route === "operations") return ["系统", "运行状态"];
  if (route === "annotations") return ["系统", "模型标注"];
  if (route === "archive") return ["实验记录", "实验详情"];
  if (route === "experiments") return ["实验工作台", "实验记录"];
  return ["实验工作台", "概览"];
}

function setChrome(route, pageOverride) {
  const [section, page] = pageContext(route);
  document.querySelector("#section-name").textContent = section;
  document.querySelector("#page-name").textContent = pageOverride || page;
  const active = ["archive", "stage"].includes(route) ? "experiments" : route;
  document.querySelectorAll(".nav-link").forEach((link) => { link.classList.toggle("active", link.dataset.route === active); if (link.dataset.route === active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current"); });
  document.querySelector("#mobile-more-button")?.classList.toggle("active", ["materials","reports","operations"].includes(active));
}

function setPhase(run) {
  const chip = document.querySelector("#phase-chip");
  const stage = run?.state || "idle";
  chip.className = `phase-chip ${stage === "completed" ? "completed" : stage === "failed" ? "failed" : run ? "running" : ""}`;
  chip.querySelector("span").textContent = run ? (STAGE_LABELS[stage] || run.message || stage) : "暂无进行中的任务";
}

function updateServiceChrome() {
  const online = state.health?.status === "ok";
  document.querySelector("#node-dot").classList.toggle("online", online);
  const running = state.runs.filter((run) => !["completed", "failed"].includes(run.state)).length;
  document.querySelector("#service-note").textContent = online
    ? state.health?.run_purpose === "workflow_simulation" ? "测试环境 · 服务正常" : running ? `${running} 个任务正在分析` : "服务运行正常"
    : "分析服务暂不可用";
  const badge = document.querySelector("#running-badge");
  badge.hidden = running === 0;
  badge.textContent = running;
  setPhase(state.activeRun || state.runs.find((run) => !["completed", "failed"].includes(run.state)) || null);
}

async function loadAll() {
  state.nasLoading = true;
  const nasRequest = api("/api/nas-recordings");
  const results = await Promise.allSettled([api("/api/health"), api("/api/archives"), api("/api/runs"), api("/api/collections?limit=200")]);
  if (results[0].status === "fulfilled") state.health = results[0].value;
  if (results[1].status === "fulfilled") state.archives = results[1].value.archives || [];
  if (results[2].status === "fulfilled") state.runs = results[2].value.runs || [];
  if (results[3].status === "fulfilled") state.collections = results[3].value.collections || [];
  updateServiceChrome();
  void nasRequest.then((payload) => {
    state.nasRecordings = payload.recordings || [];
    state.nasBatches = payload.batches || [];
    state.nasMonitor = payload.monitor || null;
    state.nasError = payload.truncated ? "素材较多，当前显示部分结果" : payload.errors?.length ? "部分素材说明无法读取" : "";
  }).catch((error) => {
    state.nasRecordings = [];
    state.nasBatches = [];
    state.nasError = error.message || "无法读取已采集素材";
  }).finally(() => {
    state.nasLoading = false;
    if (routeParts()[0] === "new" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) renderNew();
  });
}

async function refreshTaskSnapshots() {
  if (state.refreshingTasks || document.hidden) return;
  state.refreshingTasks = true;
  try {
    const payload = await api("/api/runs");
    state.runs = payload.runs || [];
    if (state.activeRun?.run_id) {
      state.activeRun = state.runs.find((run) => run.run_id === state.activeRun.run_id) || state.activeRun;
    }
    updateServiceChrome();
    if ((routeParts()[0] || "home") === "tasks") renderTasks();
  } catch {
    // Keep the last durable snapshot visible; the freshness warning explains stale data.
  } finally {
    state.refreshingTasks = false;
  }
}

function archiveProductStatus(archive) {
  if (archive.pipeline_stage === "completed") return { key: "completed", label: "已完成", tone: "success" };
  if (["failed", "interrupted"].includes(archive.pipeline_stage)) return { key: "attention", label: "需要关注", tone: "danger" };
  if (archive.pipeline_stage === "archived") return { key: "archived", label: "已归档", tone: "neutral" };
  return { key: "processing", label: "处理中", tone: "progress" };
}

function archiveWithinDate(archive, range) {
  if (range === "all") return true;
  const modified = new Date(archive.modified_at).getTime();
  if (!Number.isFinite(modified)) return false;
  const today = new Date();
  if (range === "today") return new Date(archive.modified_at).toDateString() === today.toDateString();
  const days = range === "7d" ? 7 : 30;
  return modified >= Date.now() - days * 86400000;
}

function filteredArchives(applyAdvanced = false) {
  const query = state.search.trim().toLocaleLowerCase();
  const filters = state.archiveFilters;
  return state.archives.filter((archive) => {
    const metadata = experimentMetadata(archive.name);
    const haystack = [archive.name, productExperimentName(archive.name), metadata.owner, ...(metadata.tags || [])].join(" ").toLocaleLowerCase();
    if (query && !haystack.includes(query)) return false;
    if (!applyAdvanced) return true;
    if (filters.status !== "all" && archiveProductStatus(archive).key !== filters.status) return false;
    if (!archiveWithinDate(archive, filters.date)) return false;
    if (filters.owner !== "all" && metadata.owner !== filters.owner) return false;
    if (filters.tag !== "all" && !(metadata.tags || []).includes(filters.tag)) return false;
    return true;
  });
}

function statusCard(iconName, label, value, note) {
  return `<article class="status-card"><span>${icon(iconName)}</span><div><small>${esc(label)}</small><strong>${esc(value)}</strong>${note ? `<p>${esc(note)}</p>` : ""}</div></article>`;
}

function archiveRows(archives, target = "experiments", view = "list") {
  if (!archives.length) {
    const searching = Boolean(state.search.trim());
    const title = searching ? "没有找到匹配的实验" : target === "materials" ? "关键素材将在这里汇集" : target === "reports" ? "实验报告将在这里汇集" : "从第一个实验开始";
    const copy = searching ? "试试其他实验名称，或清空搜索查看全部记录。" : "导入实验视频，集中查看片段、关键素材与分析结果。";
    return productState("neutral", searching ? "search" : target === "reports" ? "file" : "folder", title, copy, searching ? `<button class="secondary-button" type="button" data-clear-archive-filters>清空筛选</button>` : `<a class="secondary-button" href="#/new">${icon("plus")}创建实验</a>`);
  }
  return `<div class="archive-list ${view === "cards" ? "archive-card-grid" : ""}">${archives.map((archive) => {
    const metadata = experimentMetadata(archive.name);
    const status = archiveProductStatus(archive);
    return `
    <a class="archive-row ${view === "cards" ? "archive-row-card" : ""}" href="#/archive/${encodeURIComponent(archive.name)}/${target}">
      <span>${icon("check")}</span>
      <span class="archive-identity" title="档案编号：${esc(archive.name)}"><strong>${esc(productExperimentName(archive.name))}</strong><small>${formatDate(archive.modified_at)}</small>${metadata.owner || metadata.tags?.length ? `<em>${metadata.owner ? esc(metadata.owner) : ""}${metadata.owner && metadata.tags?.length ? " · " : ""}${(metadata.tags || []).slice(0,2).map(esc).join(" · ")}</em>` : ""}</span>
      <span class="archive-cell"><small>实验片段</small><strong>${number(archive.experiment_count)} 个</strong></span>
      <span class="archive-cell"><small>关键素材</small><strong>${number(archive.key_event_count)} 个</strong></span>
      <span class="archive-status ${status.tone}">${esc(status.label)}</span>
      <span class="row-link">查看结果 ${icon("arrow")}</span>
    </a>`;
  }).join("")}</div>`;
}

function renderHome() {
  setChrome("home");
  const running = state.runs.filter((run) => !["completed", "failed"].includes(run.state));
  const todayCount = state.archives.filter((archive)=>archiveWithinDate(archive,"today")).length;
  const reportReady = state.archives.filter((archive)=>archive.has_daily_report).length;
  const attention = state.archives.filter((archive)=>archiveProductStatus(archive).key === "attention");
  const health = state.health || {};
  main.innerHTML = `<div class="page workspace-page">
    <header class="page-hero workbench-hero"><div><p class="eyebrow">实验工作台</p><h1>${running.length ? "实验正在有序分析" : attention.length ? "有实验需要处理" : "今天从哪里开始？"}</h1><p>${running.length ? `${running.length} 个任务正在运行，完成后会自动生成素材与报告。` : attention.length ? `${attention.length} 个实验需要检查，其他档案均可正常查看。` : "导入多视角视频，或继续查看最近完成的实验成果。"}</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header>
    <section class="home-focus-grid" aria-label="待办概览">
      <a href="#/experiments"><span>${icon("folder")}</span><small>今日新增</small><strong>${number(todayCount)}</strong><em>查看实验记录 ${icon("arrow")}</em></a>
      <a href="#/tasks" class="${running.length ? "active" : ""}"><span>${icon("activity")}</span><small>正在分析</small><strong>${number(running.length)}</strong><em>${running.length ? "跟进任务进度" : "当前没有运行任务"} ${icon("arrow")}</em></a>
      <a href="#/reports"><span>${icon("file")}</span><small>有报告可查看</small><strong>${number(reportReady)}</strong><em>进入实验室日报 ${icon("arrow")}</em></a>
      <a href="#/experiments" class="${attention.length ? "attention" : ""}"><span>${icon(attention.length ? "x" : "check")}</span><small>需要关注</small><strong>${number(attention.length)}</strong><em>${attention.length ? "检查失败或中断" : "当前状态正常"} ${icon("arrow")}</em></a>
    </section>
    ${running.length ? `<section class="current-work"><span class="current-icon">${icon("activity")}</span><div><small>正在分析</small><h2>${esc(productExperimentName(running[0].experiment_id))}</h2><p>${esc(STAGE_LABELS[running[0].state] || "处理中")}</p></div><a class="secondary-button" href="#/tasks">查看进度 ${icon("arrow")}</a></section>` : ""}
    <div class="workspace-grid">
      <section class="panel recent-panel"><header class="panel-heading"><div><h2>最近完成与更新</h2><p>从上次离开的地方继续</p></div><a href="#/experiments">全部记录 ${icon("arrow")}</a></header>${archiveRows(filteredArchives().slice(0, 6))}</section>
      <aside class="workspace-aside">
        <section class="next-action-panel"><p class="eyebrow">下一步</p><h2>${attention.length ? "检查需要关注的实验" : running.length ? "等待分析完成" : state.archives.length ? "继续查看最近成果" : "创建第一个实验"}</h2><p>${attention.length ? "失败或中断的实验会保留已完成内容，可先查看现有结果。" : "工作台会把最需要处理的内容放在这里。"}</p><a class="secondary-button" href="${attention.length ? "#/experiments" : running.length ? "#/tasks" : state.archives.length ? `#/archive/${encodeURIComponent(state.archives[0].name)}/experiments` : "#/new"}">${attention.length ? "查看异常实验" : running.length ? "查看分析进度" : state.archives.length ? "打开最近实验" : "新建实验"} ${icon("arrow")}</a></section>
        <section class="workspace-status"><header><h2>工作区状态</h2><a href="#/operations" aria-label="查看服务状态">${icon("arrow")}</a></header><div><span>档案存储</span><strong class="${health.archive_available ? "status-ok" : "status-pending"}">${health.archive_available ? "可用" : "待连接"}</strong></div><div><span>智能理解</span><strong class="${health.mllm_enabled && health.ark_key_configured ? "status-ok" : "status-pending"}">${health.mllm_enabled && health.ark_key_configured ? "已启用" : "待启用"}</strong></div></section>
      </aside>
    </div>
    <nav class="workspace-shortcuts" aria-label="实验成果入口"><a href="#/materials"><span>${icon("boxes")}</span><div><strong>关键素材库</strong></div>${icon("arrow")}</a><a href="#/reports"><span>${icon("file")}</span><div><strong>实验室日报</strong></div>${icon("arrow")}</a></nav>
  </div>`;
}

async function rerunBenchmark() {
  const button = document.querySelector("#rerun-benchmark");
  if (button) {
    button.disabled = true;
    button.textContent = "正在准备基准验证…";
  }
  try {
    const run = await api("/api/benchmarks/six-view-three-hour/runs", { method: "POST" });
    state.activeRun = run;
    await loadAll();
    toast(`已复用固定归档：${run.nas_output}`);
    location.hash = "#/tasks";
  } catch (error) {
    toast(error.message, "error");
    if (button) {
      button.disabled = false;
      button.innerHTML = `${icon("activity")}运行基准验证`;
    }
  }
}

function renderExperiments(target = "experiments") {
  setChrome(target);
  const title = target === "materials" ? "关键素材库" : target === "reports" ? "实验室日报" : "实验记录";
  const copy = target === "materials" ? "回看关键帧与操作片段，追溯每一项实验发现。" : target === "reports" ? "汇集实验过程与分析结论，方便回顾和复核。" : "将实验视频、分析过程与结果，整理为可追溯的记录。";
  const filters = state.archiveFilters;
  const metadata = state.archives.map((archive)=>experimentMetadata(archive.name));
  const owners = [...new Set(metadata.map((item)=>item.owner).filter(Boolean))].sort();
  const tags = [...new Set(metadata.flatMap((item)=>item.tags || []))].sort();
  const filtered = filteredArchives(true);
  const activeFilterCount = [filters.status,filters.date,filters.owner,filters.tag].filter((value)=>value !== "all").length + (state.search.trim() ? 1 : 0);
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">${target === "materials" || target === "reports" ? "实验成果" : "实验记录"}</p><h1>${title}</h1><p>${copy}</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header>${target === "experiments" && isNasMode() ? `<section class="panel"><header class="panel-heading"><div><h2>待分析的实验素材</h2><p>从已保存的采集批次中选择素材并开始分析</p></div><a href="#/new">选择素材 ${icon("arrow")}</a></header>${collectionLedger()}</section>` : ""}<section class="panel experiment-records"><header class="panel-heading"><div><h2>${activeFilterCount ? "筛选结果" : "全部记录"}<span class="count-pill">${number(filtered.length)}</span></h2><p>${activeFilterCount ? `已应用 ${activeFilterCount} 项条件` : "按更新时间排序"}</p></div><div class="archive-view-toggle" aria-label="记录视图"><button type="button" data-archive-view="list" class="${filters.view === "list" ? "active" : ""}">列表</button><button type="button" data-archive-view="cards" class="${filters.view === "cards" ? "active" : ""}">卡片</button></div></header><div class="archive-filter-toolbar"><label><span>时间</span><select data-archive-filter="date"><option value="all">全部时间</option><option value="today" ${filters.date === "today" ? "selected" : ""}>今天</option><option value="7d" ${filters.date === "7d" ? "selected" : ""}>最近 7 天</option><option value="30d" ${filters.date === "30d" ? "selected" : ""}>最近 30 天</option></select></label><label><span>状态</span><select data-archive-filter="status"><option value="all">全部状态</option><option value="completed" ${filters.status === "completed" ? "selected" : ""}>已完成</option><option value="processing" ${filters.status === "processing" ? "selected" : ""}>处理中</option><option value="attention" ${filters.status === "attention" ? "selected" : ""}>需要关注</option><option value="archived" ${filters.status === "archived" ? "selected" : ""}>已归档</option></select></label><label><span>负责人</span><select data-archive-filter="owner"><option value="all">全部负责人</option>${owners.map((owner)=>`<option value="${esc(owner)}" ${filters.owner === owner ? "selected" : ""}>${esc(owner)}</option>`).join("")}</select></label><label><span>标签</span><select data-archive-filter="tag"><option value="all">全部标签</option>${tags.map((tag)=>`<option value="${esc(tag)}" ${filters.tag === tag ? "selected" : ""}>${esc(tag)}</option>`).join("")}</select></label><button class="archive-filter-clear" type="button" data-clear-archive-filters ${activeFilterCount ? "" : "disabled"}>清空</button></div>${archiveRows(filtered, target === "materials" ? "materials" : target === "reports" ? "reports" : "experiments", filters.view)}</section></div>`;
  bindArchiveFilters(target);
}

function bindArchiveFilters(target) {
  document.querySelectorAll("[data-archive-filter]").forEach((control)=>control.addEventListener("change", () => {
    state.archiveFilters[control.dataset.archiveFilter] = control.value;
    renderExperiments(target);
  }));
  document.querySelectorAll("[data-archive-view]").forEach((button)=>button.addEventListener("click", () => {
    state.archiveFilters.view = button.dataset.archiveView;
    renderExperiments(target);
  }));
  document.querySelectorAll("[data-clear-archive-filters]").forEach((button)=>button.addEventListener("click", () => {
    state.search = "";
    const search = document.querySelector("#global-search");
    if (search) search.value = "";
    state.archiveFilters = { ...state.archiveFilters, status: "all", date: "all", owner: "all", tag: "all" };
    renderExperiments(target);
  }));
}

function createSource(video = null, csv = null, index = state.sources.length) {
  const stem = video?.name?.replace(/\.[^.]+$/, "") || `view-${String(index + 1).padStart(2,"0")}`;
  const viewId = stem.replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-|-$/g, "").slice(0, 80) || `view-${String(index + 1).padStart(2,"0")}`;
  return { key: crypto.randomUUID(), viewId, role: "", segments: video ? [{ video, csv }] : [] };
}

function normalizePairName(filename) {
  return filename.toLocaleLowerCase().replace(/\.[^.]+$/, "").replace(/[_ -]?(?:rgb录制帧索引|帧时间戳|rgb)$/i, "").replace(/timestamps?|timecodes?|clock|frames?|pts|video|record/g, "").replace(/[^a-z0-9\u4e00-\u9fff]/g, "");
}

function importBatch(fileList) {
  const files = [...(fileList || [])];
  const videos = files.filter((file) => /\.(mp4|mov|m4v|mkv|avi|webm)$/i.test(file.name) || file.type.startsWith("video/"));
  const csvs = files.filter((file) => /\.csv$/i.test(file.name) || file.type === "text/csv");
  if (!videos.length) {
    toast("没有识别到视频文件，请选择 MP4、MOV、MKV、AVI 或 WebM。", "error");
    return;
  }
  state.selectedCollectionId = null;
  const parentKey = (file) => {
    const parts = String(file.webkitRelativePath || "").split("/").filter(Boolean);
    return parts.length >= 3 ? parts.slice(0, -1).join("/") : null;
  };
  const groups = new Map();
  videos.forEach((video, index) => {
    const key = parentKey(video) || `single:${index}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(video);
  });
  const unusedCsvs = [...csvs];
  const pairCsvs = (groupVideos, groupKey) => {
    const sameFolder = unusedCsvs.filter((csv) => groupKey && parentKey(csv) === groupKey);
    const candidates = sameFolder.length ? sameFolder : unusedCsvs;
    return groupVideos.map((video, index) => {
      const normalized = normalizePairName(video.name);
      let csv = candidates.find((item) => {
        const candidate = normalizePairName(item.name);
        return candidate && normalized && (candidate.includes(normalized) || normalized.includes(candidate));
      });
      if (!csv && candidates.length === groupVideos.length) csv = candidates[index];
      if (csv) unusedCsvs.splice(unusedCsvs.indexOf(csv), 1);
      return { video, csv: csv || null };
    });
  };
  state.sources = [...groups.entries()].map(([groupKey, groupVideos], index) => {
    const orderedVideos = [...groupVideos].sort((a, b) => String(a.webkitRelativePath || a.name).localeCompare(String(b.webkitRelativePath || b.name), "zh-CN", { numeric: true }));
    const source = createSource(orderedVideos[0], null, index);
    source.segments = pairCsvs(orderedVideos, groupKey.startsWith("single:") ? null : groupKey);
    if (!groupKey.startsWith("single:")) source.viewId = groupKey.split("/").at(-1).replace(/[^A-Za-z0-9_.-]+/g, "-").slice(0, 80) || source.viewId;
    const previous = state.sources[index];
    if (previous?.role) source.role = previous.role;
    return source;
  });
  renderNew();
  toast(`已生成 ${state.sources.length} 路机位、${videos.length} 个视频分片${csvs.length ? `，识别 ${csvs.length} 个 CSV` : ""}。`);
}

function sourceCard(source, index) {
  const segments = source.segments || [];
  const videoBytes = segments.reduce((total, item) => total + Number(item.video?.size || 0), 0);
  const csvCount = segments.filter((item) => item.csv).length;
  const firstName = segments[0]?.video?.name;
  return `<article class="source-card" data-source="${esc(source.key)}">
    <header class="source-card-head"><span>${icon("video")}</span><div><strong>机位 ${String(index + 1).padStart(2,"0")}${firstName ? ` · ${esc(firstName)}` : ""}</strong><small>${segments.length ? `${segments.length} 个连续分片 · ${formatBytes(videoBytes)} · ${csvCount} 个 CSV` : "等待选择该机位的视频"}</small></div><button class="remove-source" type="button" data-remove="${esc(source.key)}" aria-label="移除机位">${icon("x")}</button></header>
    <div class="source-config-grid">
      <label>机位 ID<input data-view-id="${esc(source.key)}" value="${esc(source.viewId)}" autocomplete="off" /></label>
      <label>拍摄视角<select data-role="${esc(source.key)}"><option value="" ${source.role ? "" : "selected"}>确认拍摄视角</option><option value="first_person" ${source.role === "first_person" ? "selected" : ""}>第一人称 · 手部与精细接触</option><option value="third_person" ${source.role === "third_person" ? "selected" : ""}>第三人称 · 空间与移动轨迹</option></select></label>
    </div>
    <div class="file-grid">
      <label class="file-field ${segments.length ? "has-file" : ""}"><input type="file" multiple accept="video/*,.mp4,.mov,.m4v,.mkv,.avi,.webm" data-video="${esc(source.key)}"/><span class="file-icon">${icon("video")}</span><span><strong>${segments.length ? `${segments.length} 个视频分片` : "选择实验视频或连续分片"}</strong><small>${segments.length ? `${esc(segments[0].video.name)}${segments.length > 1 ? ` … ${esc(segments.at(-1).video.name)}` : ""}` : "按文件名自然顺序组成同一机位时间线"}</small></span>${segments.length ? `<span class="file-check">${icon("check")}</span>` : ""}</label>
      <label class="file-field ${csvCount ? "has-file" : ""}"><input type="file" multiple accept=".csv,text/csv" data-csv="${esc(source.key)}"/><span class="file-icon">${icon("clock")}</span><span><strong>${csvCount ? `${csvCount} 个时间戳 CSV` : "选择对应时间戳 CSV"}</strong><small>${csvCount ? `按名称或顺序对应 ${segments.length} 个分片` : "可选；数量应与视频分片对应"}</small></span>${csvCount ? `<span class="file-check">${icon("check")}</span>` : ""}</label>
    </div>
  </article>`;
}

function roleGuidance() {
  const videos = state.sources.filter((source) => source.segments?.length).length;
  const first = state.sources.filter((source) => source.role === "first_person").length;
  const third = state.sources.filter((source) => source.role === "third_person").length;
  if (videos < 2) return { tone: "warning", text: `已选择 ${videos} 个拍摄视角；请至少添加第一人称和第三人称素材。` };
  if (!first || !third) return { tone: "warning", text: `当前 ${videos} 路视频中，需要至少确认一路第一人称和一路第三人称。` };
  return { tone: "ok", text: `素材已就绪：${first} 个第一人称视角、${third} 个第三人称视角。` };
}


function nasRecordingPicker() {
  const items = state.nasRecordings;
  const rows = items.map((item) => {
    const chosen = Object.hasOwn(state.nasSelection, item.recording_id);
    const role = state.nasSelection[item.recording_id] || "";
    return `<article class="nas-recording-row ${chosen ? "selected" : ""}" data-nas-path="${esc(item.relative_path)}" ${item.relative_path.toLocaleLowerCase().includes(state.nasQuery.toLocaleLowerCase()) ? "" : "hidden"}>
      <label class="nas-recording-choice"><input type="checkbox" data-nas-recording="${esc(item.recording_id)}" aria-label="选择 ${esc(item.relative_path)}" ${chosen ? "checked" : ""} ${item.available ? "" : "disabled"}><span><strong>${esc(item.requires_completion_confirmation ? item.relative_path.split("/").pop() : item.camera_key)}</strong><span>${formatDate(item.recording_start_time)} · ${duration(item.duration_seconds)}</span><span class="nas-file-name">${esc(item.relative_path)} · ${formatBytes(item.size_bytes)}</span></span></label>
      <div class="nas-recording-action"><span class="collection-status ${item.available ? "ready" : "attention"}">${item.available ? "可选择" : "待检查"}</span><select data-nas-role="${esc(item.recording_id)}" aria-label="${esc(item.relative_path)} 的拍摄视角" ${chosen ? "" : "disabled"}><option value="">确认拍摄视角</option><option value="first_person" ${role === "first_person" ? "selected" : ""}>第一人称</option><option value="third_person" ${role === "third_person" ? "selected" : ""}>第三人称</option></select></div>
      ${item.issues.length ? `<p class="nas-recording-issue">${esc(item.issues.join("；"))}</p>` : ""}</article>`;
  }).join("");
  return `<section class="form-section" id="nas-recordings"><div class="section-heading"><span>01</span><div><h2>选择已采集素材 <span class="count-pill">${items.length}</span></h2><p>选择同一次实验的不同视角，无需重复上传视频。</p></div></div>${state.nasError ? `<p class="nas-recording-issue" role="status">${esc(state.nasError)}</p>` : ""}<label class="collection-search">${icon("search")}<input id="nas-search" value="${esc(state.nasQuery)}" placeholder="按实验、文件夹或文件名查找" aria-label="搜索已采集素材"></label><div class="nas-recordings-list">${rows || `<div class="empty-state compact"><strong>${state.nasLoading ? "正在读取已采集素材…" : "还没有采集素材"}</strong></div>`}</div>${items.some((item) => item.requires_completion_confirmation) ? `<label class="nas-completion-confirmation"><input id="nas-complete-confirmed" type="checkbox" ${state.nasCompleteConfirmed ? "checked" : ""}>我确认所选视频已采集完成，不再写入</label>` : ""}<div class="nas-selection-actions"><span id="nas-selection-count">已选择 ${Object.keys(state.nasSelection).length} 个片段</span><button class="secondary-button" type="button" id="save-nas-selection">保存为实验批次</button></div></section>`;
}

function nasBatchPicker() {
  const monitor = state.nasMonitor || {};
  const monitorWatching = monitor.status === "watching";
  const monitorCopy = monitorWatching
    ? `持续监控中 · ${formatDate(monitor.observed_at)}`
    : monitor.status === "retrying"
    ? "NAS 连接中断，正在自动重试"
    : "正在启动采集监控";
  const rows = state.nasBatches.map((batch) => {
    const status = batch.available ? "可分析" : "采集中";
    const issue = batch.issues?.[0] || "";
    return `<article class="collection-card ${batch.available ? "ready" : "recording"}">
      <header><div title="采集编号：${esc(batch.recording_session_id)}"><small>${formatDate(batch.recording_start_time)}</small><strong>多视角采集 · ${number(batch.camera_count)} 个视角</strong></div><span class="collection-status ${batch.available ? "ready" : "recording"}">${status}</span></header>
      <div class="collection-metrics"><span><b>${number(batch.camera_count)}</b> 路相机</span><span><b>${number(batch.recording_count)}</b> 个视频</span><span><b>${duration(batch.duration_seconds)}</b></span><span><b>${formatBytes(batch.size_bytes)}</b></span></div>
      ${issue ? `<small class="collection-issues">${esc(issue)}</small>` : ""}
      <button class="primary-button" type="button" data-run-nas-batch="${esc(batch.batch_id)}" ${batch.available ? "" : "disabled"}>${icon("arrow")}分析此批次</button>
    </article>`;
  }).join("");
  return `<section class="form-section collection-picker" id="nas-batches"><div class="section-heading"><span>01</span><div><h2>选择采集批次</h2><p class="nas-monitor-state ${monitorWatching ? "watching" : "retrying"}"><i></i>${esc(monitorCopy)}</p></div></div><div class="collection-card-list">${rows || `<div class="empty-state compact"><strong>等待相机完成采集</strong></div>`}</div></section>`;
}

function bindNasBatchPicker() {
  document.querySelectorAll("[data-run-nas-batch]").forEach((button) => button.addEventListener("click", async () => {
    const batch = state.nasBatches.find((item) => item.batch_id === button.dataset.runNasBatch);
    if (!batch?.available) return;
    await submitNasBatchRun(batch, button);
  }));
}

function bindNasRecordingPicker() {
  document.querySelector("#nas-search")?.addEventListener("input", (event) => {
    state.nasQuery = event.target.value;
    document.querySelectorAll("[data-nas-path]").forEach((row) => {
      row.hidden = !row.dataset.nasPath.toLocaleLowerCase().includes(state.nasQuery.toLocaleLowerCase());
    });
  });
  document.querySelector("#nas-complete-confirmed")?.addEventListener("change", (event) => {
    state.nasCompleteConfirmed = event.target.checked;
  });
  document.querySelectorAll("[data-nas-recording]").forEach((input) => input.addEventListener("change", () => {
    state.nasCompleteConfirmed = false;
    if (input.checked) state.nasSelection[input.dataset.nasRecording] = "";
    else delete state.nasSelection[input.dataset.nasRecording];
    state.selectedCollectionId = null;
    renderNew();
  }));
  document.querySelectorAll("[data-nas-role]").forEach((input) => input.addEventListener("change", () => {
    state.nasSelection[input.dataset.nasRole] = input.value;
    updateReview();
  }));
  document.querySelector("#save-nas-selection")?.addEventListener("click", async (event) => {
    const name = state.experimentName.trim();
    const recordings = Object.entries(state.nasSelection).map(([recording_id, role]) => ({ recording_id, role }));
    if (!name) { toast("请先填写下方的实验名称", "error"); document.querySelector("#experiment-name").focus(); return; }
    if (recordings.length < 2 || recordings.some((item) => !item.role)) { toast("请选择至少两个视角，并确认各自的拍摄视角", "error"); return; }
    event.currentTarget.disabled = true;
    try {
      const receipt = await api("/api/nas-selections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ experiment_name: name, recordings, recording_complete_confirmed: document.querySelector("#nas-complete-confirmed")?.checked === true }) });
      await loadAll();
      state.selectedCollectionId = receipt.collection_id;
      state.nasSelection = {};
      state.nasCompleteConfirmed = false;
      renderNew();
      toast("实验批次已保存");
    } catch (error) {
      toast(error.message, "error");
      document.querySelector("#save-nas-selection").disabled = false;
    }
  });
}

function selectedCollection() {
  return state.collections.find((item) => item.collection_id === state.selectedCollectionId) || null;
}

function defaultCollectionArchiveName(collection) {
  const date = String(collection?.recording_start_time || "").slice(0, 10).replaceAll("-", "") || "Undated";
  const suffix = String(collection?.experiment_id || "Collection").slice(-8);
  return `VisionCortex-Collection-${date}-${suffix}`;
}

function collectionStatusCopy(collection) {
  const processing = collection.processing?.state;
  if (processing === "archived") return { label: "已处理并留存", tone: "archived", note: `实验档案：${collection.processing.archive_name || "已生成"}` };
  if (["queued", "processing"].includes(processing)) return { label: "处理中", tone: "processing", note: `任务 ${collection.processing.run_id || "已接管"}` };
  if (processing === "failed") return { label: "处理失败", tone: "failed", note: "可在任务进度查看失败原因" };
  if (collection.status === "ready") return { label: "可开始分析", tone: "ready", note: "素材已就绪，视角已确认" };
  if (collection.status === "recording") return { label: "采集中", tone: "recording", note: "等待采集完成" };
  return { label: "需处理", tone: "attention", note: `${number(collection.blocking_issue_count)} 项阻断问题` };
}

function filteredCollections() {
  const query = state.collectionQuery.trim().toLocaleLowerCase();
  const items = query
    ? state.collections.filter((item) => `${item.collection_id} ${item.display_name}`.toLocaleLowerCase().includes(query))
    : state.collections;
  return items.slice(0, 24);
}

function collectionCards() {
  const items = filteredCollections();
  if (!items.length) return `<div class="empty-state compact"><strong>没有匹配的采集批次</strong><p>刷新列表，或尝试其他日期与批次编号。</p></div>`;
  return `<div class="collection-card-list">${items.map((collection) => {
    const status = collectionStatusCopy(collection);
    const views = collection.resolved_view_counts || {};
    const selected = collection.collection_id === state.selectedCollectionId;
    const issues = [...(collection.blocking_issue_codes || []), ...(collection.warning_codes || [])];
    return `<article class="collection-card ${status.tone} ${selected ? "selected" : ""}">
      <header><div><small>${formatDate(collection.recording_start_time)}</small><strong>${esc(collection.display_name || collection.collection_id)}</strong></div><span class="collection-status ${status.tone}">${esc(status.label)}</span></header>
      <p>${esc(status.note)}${collection.approved_override_count ? ` · ${number(collection.approved_override_count)} 个有收据的角色覆盖` : ""}</p>
      <div class="collection-metrics"><span><b>${number(collection.camera_count)}</b> 路</span><span><b>${number(collection.video_segment_count)}</b> 个 MP4</span><span><b>${number(collection.clock_segment_count)}</b> 个 CSV</span><span><b>${number(views.first_person)}</b> 第一 + <b>${number(views.third_person)}</b> 第三</span></div>
      ${issues.length ? `<small class="collection-issues">${esc(issues.join(" · "))}</small>` : ""}
      <button class="${selected ? "secondary-button" : "primary-button"}" type="button" data-select-collection="${esc(collection.collection_id)}" ${collection.ready_to_analyze && !["queued", "processing"].includes(collection.processing?.state) ? "" : "disabled"}>${selected ? `${icon("check")}已选择` : collection.processing?.state === "archived" ? `${icon("check")}重新分析为新归档` : `${icon("server")}选择此批次`}</button>
    </article>`;
  }).join("")}</div>`;
}

function collectionLedger() {
  if (!state.collections.length) return `<div class="empty-state compact"><strong>还没有保存的实验批次</strong><p>选择已采集素材后，即可保存实验批次。</p></div>`;
  const counts = { archived: 0, processing: 0, failed: 0, pending: 0 };
  state.collections.forEach((collection) => {
    const status = collectionStatusCopy(collection);
    if (status.tone === "archived") counts.archived += 1;
    else if (status.tone === "processing") counts.processing += 1;
    else if (status.tone === "failed") counts.failed += 1;
    else counts.pending += 1;
  });
  const rows = state.collections.slice(0, 12).map((collection) => {
    const status = collectionStatusCopy(collection);
    const archiveName = collection.processing?.archive_name;
    const link = status.tone === "archived" && archiveName
      ? `<a href="#/archive/${encodeURIComponent(archiveName)}/experiments">打开正式归档 ${icon("arrow")}</a>`
      : status.tone === "processing" || status.tone === "failed"
        ? `<a href="#/tasks">查看任务证据 ${icon("arrow")}</a>`
        : `<a href="#/new">选择并分析 ${icon("arrow")}</a>`;
    return `<article class="batch-ledger-row"><span class="collection-status ${status.tone}">${esc(status.label)}</span><div><strong>${esc(collection.display_name || collection.collection_id)}</strong><small>${esc(collection.collection_id)} · ${formatDate(collection.recording_start_time)}</small></div><p>${number(collection.camera_count)} 路 · ${number(collection.video_segment_count)} MP4 · ${esc(status.note)}</p>${link}</article>`;
  }).join("");
  return `<div class="batch-ledger-summary"><span><b>${counts.pending}</b> 未处理</span><span><b>${counts.processing}</b> 处理中</span><span><b>${counts.failed}</b> 失败待处理</span><span><b>${counts.archived}</b> 已处理并留存</span></div><div class="batch-ledger">${rows}</div>`;
}

function reviewState() {
  const collection = selectedCollection();
  const title = document.querySelector("#experiment-name")?.value.trim() || state.experimentName.trim();
  if (collection) {
    const views = collection.resolved_view_counts || {};
    return {
      mode: "nas_collection",
      collection,
      videos: Number(collection.camera_count || 0),
      csvs: Number(collection.clock_segment_count || 0),
      first: Number(views.first_person || 0),
      third: Number(views.third_person || 0),
      title,
      ready: Boolean(title && collection.ready_to_analyze),
    };
  }
  if (Object.keys(state.nasSelection).length) {
    const picked = state.nasRecordings.filter((item) => Object.hasOwn(state.nasSelection, item.recording_id));
    return { mode: "nas_draft", title, ready: false, videos: new Set(picked.map((item) => item.camera_key)).size, csvs: picked.length,
      first: new Set(picked.filter((item) => state.nasSelection[item.recording_id] === "first_person").map((item) => item.camera_key)).size,
      third: new Set(picked.filter((item) => state.nasSelection[item.recording_id] === "third_person").map((item) => item.camera_key)).size };
  }
  const videos = state.sources.filter((source) => source.segments?.length).length;
  const videoSegments = state.sources.reduce(
    (total, source) => total + (source.segments?.length || 0),
    0,
  );
  const csvs = state.sources.reduce(
    (total, source) =>
      total + (source.segments || []).filter((segment) => segment.csv).length,
    0,
  );
  const first = state.sources.filter((source) => source.role === "first_person").length;
  const third = state.sources.filter((source) => source.role === "third_person").length;
  const ids = state.sources.map((source) => source.viewId.trim()).filter(Boolean);
  const unique = new Set(ids).size === ids.length;
  const clockMappingsValid = state.sources.every((source) => {
    const mapped = (source.segments || []).filter((segment) => segment.csv).length;
    return mapped === 0 || mapped === source.segments.length;
  });
  return { mode: "upload", videos, videoSegments, csvs, first, third, title, ready: Boolean(title && videos >= 2 && videos === state.sources.length && first && third && state.sources.every((source) => ["first_person", "third_person"].includes(source.role)) && unique && ids.length === state.sources.length && clockMappingsValid) };
}

function renderNew() {
  setChrome("new");
  const guide = roleGuidance();
  const currentTitle = document.querySelector("#experiment-name")?.value;
  if (currentTitle !== undefined) state.experimentName = currentTitle;
  const selected = selectedCollection();
  const nas = isNasMode();
  const directNas = nas && state.health?.collection_ingest?.mode === "directory_metadata";
  const modelEnabled = state.health?.mllm_enabled && state.health?.ark_key_configured;
  main.innerHTML = `<div class="page new-experiment-page">${state.health?.run_purpose === "workflow_simulation" ? `<p class="analysis-readiness-note">当前为测试环境，分析结果将保存到独立测试档案。</p>` : ""}
    <header class="page-hero compact"><div><p class="eyebrow">创建实验</p><h1>新建实验</h1><p>导入不同视角的实验视频，系统会自动整理片段、关键素材和报告。</p></div><a class="secondary-button" href="#/experiments">${icon("chevron")}返回实验记录</a></header>
    <nav class="setup-progress" aria-label="新建实验进度"><a href="#${directNas ? "nas-batches" : nas ? "collection-source" : "recorded-videos"}"><span>01</span><strong>导入视频</strong></a><a href="#experiment-info"><span>02</span><strong>命名实验</strong></a><a href="#analysis-method"><span>03</span><strong>确认分析方式</strong></a><a href="#review-start"><span>04</span><strong>开始处理</strong></a></nav>
    <div class="new-layout"><div class="new-primary">
      ${directNas ? nasBatchPicker() : ""}
      ${nas && (!directNas || state.collections.length) ? `<section class="form-section collection-picker" id="collection-source"><div class="section-heading"><span>01</span><div><h2>选择采集批次</h2><p>查看已保存的实验素材组合。</p></div></div><label class="collection-search">${icon("search")}<input id="collection-search" value="${esc(state.collectionQuery)}" placeholder="按日期或批次编号查找" aria-label="搜索采集批次" /></label><div id="collection-card-results">${collectionCards()}</div>${selected ? `<div class="selected-collection-note"><span>${icon("check")}</span><div><strong>已选择 ${esc(selected.collection_id)}</strong><p>${number(selected.camera_count)} 路视频 · ${number(selected.video_segment_count)} 个片段</p></div><button class="secondary-button" id="clear-collection" type="button">改选批次</button></div>` : ""}</section>` : ""}
      <section class="form-section manual-upload-fallback ${selected ? "is-secondary" : ""}" id="recorded-videos"><div class="section-heading"><span>${nas ? icon("upload") : "01"}</span><div><h2>${nas ? "或上传实验文件" : "导入实验视频"}</h2></div></div>
        <div class="batch-import-panel"><div><span class="upload-symbol">${icon("upload")}</span><strong>选择视频，开始整理实验</strong><p>支持 MP4、MOV、MKV、AVI、WebM 与时间戳 CSV</p></div><div class="batch-actions"><label class="primary-button batch-import-button">${icon("plus")}选择文件<input id="batch-input" aria-label="选择视频与时间戳文件" type="file" multiple accept="video/*,.mp4,.mov,.m4v,.mkv,.avi,.webm,.csv,text/csv" /></label><label class="secondary-button batch-import-button">${icon("folder")}选择文件夹<input id="folder-input" aria-label="选择实验文件夹" type="file" multiple webkitdirectory directory /></label></div></div>
        <div class="role-guidance ${guide.tone}">${esc(guide.text)}</div><div class="source-list">${state.sources.length ? state.sources.map(sourceCard).join("") : ""}</div><button class="secondary-button add-source" id="add-source" type="button">${icon("plus")}添加拍摄视角</button>
      </section>
      <section class="form-section" id="experiment-info"><div class="section-heading"><span>02</span><div><h2>为实验命名</h2><p>使用易于辨认的英文或编号，方便后续查找。</p></div></div><label class="field-label"><span>实验名称</span><input id="experiment-name" value="${esc(state.experimentName)}" maxlength="120" placeholder="例如：Sample-Preparation-01" autocomplete="off" /></label></section>
      <section class="form-section" id="analysis-method"><div class="section-heading"><span>03</span><div><h2>分析方式</h2></div></div><div class="analysis-method-overview"><article class="method-card"><span>${icon("video")}</span><div><small>视频处理</small><strong>片段整理与素材提取</strong><p>对齐不同视角，识别实验片段与关键动作。</p></div></article><article class="method-card"><span>${icon("brain")}</span><div><small>智能理解 · ${modelEnabled ? "已配置" : "待启用"}</small><strong>实验步骤解读</strong><p>${modelEnabled ? "结合实验素材生成步骤解读，并标注不确定项。" : "尚未启用步骤解读。可在服务状态中查看模型配置。"}</p></div></article></div></section>
    </div><aside class="review-card" id="review-start"><div class="section-heading"><span>04</span><div><h2>准备开始</h2></div></div><div class="review-summary" id="review-summary"></div>${state.health?.analysis_ready === false ? `<p class="analysis-readiness-note" role="status">${esc(state.health.analysis_blocker)}。当前可以浏览素材、保存实验批次。</p>` : ""}<div class="upload-progress hidden" id="upload-progress"><div class="progress-copy"><span id="progress-message">准备任务</span><strong id="progress-percent">0%</strong></div><span class="progress-track"><i id="progress-bar" style="width:0%"></i></span><div class="run-stage-list" id="run-stages"></div><button class="secondary-button hidden" id="cancel-upload" type="button">${icon("x")}取消上传</button></div><button class="primary-button" id="start-run" type="button">${icon("arrow")}创建分析任务</button></aside></div>
  </div>`;
  bindNewPage();
  updateReview();
}

function updateReview() {
  const review = reviewState();
  const summary = document.querySelector("#review-summary");
  if (!summary) return;
  summary.innerHTML = review.mode === "nas_draft"
    ? `<div class="review-line"><span>所选相机</span><strong>${review.videos} 路</strong></div><div class="review-line"><span>素材片段</span><strong>${review.csvs} 个</strong></div><div class="review-line"><span>已确认视角</span><strong>${review.first} 第一人称 + ${review.third} 第三人称</strong></div><p>确认名称和视角后，先保存为实验批次。</p>`
    : review.mode === "nas_collection"
    ? `<div class="review-line"><span>采集批次</span><strong>${esc(review.collection.collection_id)}</strong></div><div class="review-line"><span>自动聚合</span><strong>${review.videos} 路 / ${number(review.collection.video_segment_count)} 个 MP4</strong></div><div class="review-line"><span>视角构成</span><strong>${review.first} 第一人称 + ${review.third} 第三人称</strong></div><div class="review-line"><span>源视频复制</span><strong>0 字节</strong></div><div class="review-line"><span>角色解析收据</span><strong>${number(review.collection.approved_override_count)} 个覆盖</strong></div><div class="review-line"><span>自动处理</span><strong>启用</strong></div>`
    : `<div class="review-line"><span>视频素材</span><strong>${review.videos} 个视角 / ${review.videoSegments} 个分片</strong></div><div class="review-line"><span>时间记录</span><strong>${review.csvs} 个</strong></div><div class="review-line"><span>拍摄视角</span><strong>${review.first} 第一人称 + ${review.third} 第三人称</strong></div><div class="review-line"><span>传输保护</span><strong>校验与断点续传</strong></div><div class="review-line"><span>原视频留存</span><strong>${isNasMode() ? "集中存储" : "实验档案"}</strong></div><div class="review-line"><span>自动分析</span><strong>已启用</strong></div>`;
  const button = document.querySelector("#start-run");
  button.disabled = state.health?.analysis_ready === false || !review.ready || Boolean(state.activeRun && !["completed","failed"].includes(state.activeRun.state));
}

function updateSource(key, change) {
  const source = state.sources.find((item) => item.key === key);
  if (!source) return;
  Object.assign(source, change);
}

function bindNewPage() {
  bindNasBatchPicker();
  document.querySelectorAll(".setup-progress a").forEach((link) => link.addEventListener("click", (event) => {
    event.preventDefault();
    document.getElementById(link.getAttribute("href").slice(1))?.scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  const bindCollectionButtons = () => {
    document.querySelectorAll("[data-select-collection]").forEach((button) => button.addEventListener("click", () => {
      const collection = state.collections.find((item) => item.collection_id === button.dataset.selectCollection);
      if (!collection?.ready_to_analyze) return;
      state.selectedCollectionId = collection.collection_id;
      state.experimentName = defaultCollectionArchiveName(collection);
      renderNew();
    }));
  };
  bindCollectionButtons();
  document.querySelector("#collection-search")?.addEventListener("input", (event) => {
    state.collectionQuery = event.target.value;
    const results = document.querySelector("#collection-card-results");
    if (results) {
      results.innerHTML = collectionCards();
      bindCollectionButtons();
    }
  });
  document.querySelector("#clear-collection")?.addEventListener("click", () => {
    state.selectedCollectionId = null;
    state.experimentName = "";
    renderNew();
  });
  document.querySelector("#batch-input").addEventListener("change", (event) => importBatch(event.target.files));
  document.querySelector("#folder-input").addEventListener("change", (event) => importBatch(event.target.files));
  document.querySelector("#experiment-name").addEventListener("input", (event) => { state.experimentName = event.target.value; updateReview(); });
  document.querySelector("#add-source").addEventListener("click", () => { state.selectedCollectionId = null; state.sources.push(createSource()); renderNew(); });
  document.querySelectorAll("[data-remove]").forEach((button) => button.addEventListener("click", () => { state.sources = state.sources.filter((source) => source.key !== button.dataset.remove); renderNew(); }));
  document.querySelectorAll("[data-view-id]").forEach((input) => input.addEventListener("input", () => { updateSource(input.dataset.viewId, { viewId: input.value.replace(/[^A-Za-z0-9_.-]+/g, "-") }); updateReview(); }));
  document.querySelectorAll("[data-role]").forEach((select) => select.addEventListener("change", () => { updateSource(select.dataset.role, { role: select.value }); renderNew(); }));
  document.querySelectorAll("[data-video]").forEach((input) => input.addEventListener("change", () => {
    state.selectedCollectionId = null;
    const source = state.sources.find((item) => item.key === input.dataset.video);
    if (!source) return;
    const videos = [...input.files].sort((a, b) => a.name.localeCompare(b.name, "zh-CN", { numeric: true }));
    const oldCsvs = (source.segments || []).map((item) => item.csv).filter(Boolean);
    source.segments = videos.map((video, index) => ({ video, csv: oldCsvs.length === videos.length ? oldCsvs[index] : null }));
    renderNew();
  }));
  document.querySelectorAll("[data-csv]").forEach((input) => input.addEventListener("change", () => {
    state.selectedCollectionId = null;
    const source = state.sources.find((item) => item.key === input.dataset.csv);
    if (!source) return;
    const csvs = [...input.files].sort((a, b) => a.name.localeCompare(b.name, "zh-CN", { numeric: true }));
    if (csvs.length && csvs.length !== source.segments.length) {
      toast(`该机位有 ${source.segments.length} 个视频分片，请选择相同数量的 CSV，或不选 CSV。`, "error");
      return;
    }
    const unused = [...csvs];
    source.segments = source.segments.map((segment, index) => {
      const normalized = normalizePairName(segment.video.name);
      let csvIndex = unused.findIndex((item) => {
        const candidate = normalizePairName(item.name);
        return candidate && normalized && (candidate.includes(normalized) || normalized.includes(candidate));
      });
      if (csvIndex < 0) csvIndex = 0;
      return { ...segment, csv: unused.splice(csvIndex, 1)[0] || null };
    });
    renderNew();
  }));
  document.querySelector("#start-run").addEventListener("click", submitRun);
  document.querySelector("#cancel-upload")?.addEventListener("click", cancelActiveUpload);
}

const UPLOAD_SESSION_CACHE_KEY = "visioncortex.large-upload-session.v1";

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

function buildUploadPlan(review) {
  const files = [];
  const specs = [];
  let csvIndex = 0;
  let videoIndex = 0;
  state.sources.forEach((source) => {
    const spec = {
      view_id: source.viewId,
      role: source.role,
      calibration_hint_ms: 0,
      segments: [],
    };
    source.segments.forEach((segment) => {
      const currentVideoIndex = videoIndex;
      files.push({
        file_id: `video-${currentVideoIndex}`,
        kind: "video",
        file_index: currentVideoIndex,
        name: segment.video.name,
        size: segment.video.size,
        last_modified: segment.video.lastModified,
        browser_file: segment.video,
      });
      const mapping = { video_index: currentVideoIndex };
      videoIndex += 1;
      if (segment.csv) {
        const currentCsvIndex = csvIndex;
        files.push({
          file_id: `timestamp-csv-${currentCsvIndex}`,
          kind: "timestamp_csv",
          file_index: currentCsvIndex,
          name: segment.csv.name,
          size: segment.csv.size,
          last_modified: segment.csv.lastModified,
          browser_file: segment.csv,
        });
        mapping.csv_index = currentCsvIndex;
        csvIndex += 1;
      }
      spec.segments.push(mapping);
    });
    if (spec.segments.length === 1) {
      const [single] = spec.segments;
      delete spec.segments;
      spec.video_index = single.video_index;
      if (single.csv_index !== undefined) spec.csv_index = single.csv_index;
    }
    specs.push(spec);
  });
  const signature = JSON.stringify({
    experiment_name: review.title,
    view_specs: specs,
    files: files.map(({ browser_file, ...metadata }) => metadata),
  });
  return { experiment_name: review.title, view_specs: specs, files, signature };
}

function cachedUploadSession(signature) {
  try {
    const cached = JSON.parse(localStorage.getItem(UPLOAD_SESSION_CACHE_KEY) || "null");
    return cached?.signature === signature ? cached : null;
  } catch {
    return null;
  }
}

function rememberUploadSession(signature, sessionId) {
  try {
    localStorage.setItem(UPLOAD_SESSION_CACHE_KEY, JSON.stringify({ signature, session_id: sessionId }));
  } catch {
    // Upload still works when browser storage is disabled; only page-reload resume is lost.
  }
}

function forgetUploadSession() {
  try { localStorage.removeItem(UPLOAD_SESSION_CACHE_KEY); } catch { /* no-op */ }
}

async function createOrResumeUploadSession(plan) {
  const cached = cachedUploadSession(plan.signature);
  if (cached?.session_id) {
    try {
      const existing = await api(`/api/upload-sessions/${encodeURIComponent(cached.session_id)}`);
      if (existing.status === "open") return existing;
      if (["finalized", "released"].includes(existing.status) && existing.run_id) return existing;
    } catch (error) {
      if (![404, 410].includes(Number(error.status))) throw error;
      forgetUploadSession();
    }
  }
  const created = await api("/api/upload-sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      experiment_name: plan.experiment_name,
      view_specs: plan.view_specs,
      files: plan.files.map(({ browser_file, ...metadata }) => metadata),
    }),
  });
  rememberUploadSession(plan.signature, created.session_id);
  return created;
}

async function validateBrowserPlan(plan) {
  const decoder = new TextDecoder("utf-8", { fatal: false });
  for (const descriptor of plan.files) {
    if (!Number.isFinite(descriptor.size) || descriptor.size <= 0) throw new Error(`${descriptor.name} 是空文件，不能上传。`);
    const head = new Uint8Array(await descriptor.browser_file.slice(0, 4096).arrayBuffer());
    if (descriptor.kind === "video") {
      const ascii = (start, end) => String.fromCharCode(...head.slice(start, end));
      const isoAtom = head.length >= 12 ? ascii(4, 8) : "";
      const mp4 = ["ftyp", "moov", "wide", "free", "mdat"].includes(isoAtom);
      const ebml = head.length >= 4 && head[0] === 0x1a && head[1] === 0x45 && head[2] === 0xdf && head[3] === 0xa3;
      const avi = head.length >= 12 && ascii(0, 4) === "RIFF" && ascii(8, 12) === "AVI ";
      if (!(mp4 || ebml || avi)) throw new Error(`${descriptor.name} 的文件头不是支持的视频容器，请检查文件是否损坏或扩展名是否错误。`);
    } else {
      const text = decoder.decode(head).replace(/^\uFEFF/, "");
      const firstLine = text.split(/\r?\n/, 1)[0] || "";
      if (!/[;,\t]/.test(firstLine) && !firstLine.includes(",")) throw new Error(`${descriptor.name} 缺少可识别的 CSV 表头。`);
    }
  }
}

function sha256Fallback(buffer) {
  const bytes = new Uint8Array(buffer);
  const bitLength = bytes.length * 8;
  const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x100000000), false);
  view.setUint32(paddedLength - 4, bitLength >>> 0, false);
  const constants = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
  ];
  const h = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
  const w = new Uint32Array(64);
  const rotate = (value, bits) => (value >>> bits) | (value << (32 - bits));
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let index = 0; index < 16; index += 1) w[index] = view.getUint32(offset + index * 4, false);
    for (let index = 16; index < 64; index += 1) {
      const s0 = rotate(w[index - 15], 7) ^ rotate(w[index - 15], 18) ^ (w[index - 15] >>> 3);
      const s1 = rotate(w[index - 2], 17) ^ rotate(w[index - 2], 19) ^ (w[index - 2] >>> 10);
      w[index] = (w[index - 16] + s0 + w[index - 7] + s1) >>> 0;
    }
    let [a,b,c,d,e,f,g,hh] = h;
    for (let index = 0; index < 64; index += 1) {
      const s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const choice = (e & f) ^ (~e & g);
      const temp1 = (hh + s1 + choice + constants[index] + w[index]) >>> 0;
      const s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + majority) >>> 0;
      hh = g; g = f; f = e; e = (d + temp1) >>> 0; d = c; c = b; b = a; a = (temp1 + temp2) >>> 0;
    }
    [a,b,c,d,e,f,g,hh].forEach((value, index) => { h[index] = (h[index] + value) >>> 0; });
  }
  return h.map((value) => value.toString(16).padStart(8, "0")).join("");
}

async function chunkSha256(buffer) {
  if (!globalThis.crypto?.subtle) return sha256Fallback(buffer);
  const hash = await globalThis.crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(hash), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sendUploadChunk(sessionId, fileId, offset, buffer, signal = null) {
  const headers = {
    "Content-Type": "application/octet-stream",
    "Upload-Offset": String(offset),
  };
  headers["X-Chunk-SHA256"] = await chunkSha256(buffer);
  const response = await fetch(
    `/api/upload-sessions/${encodeURIComponent(sessionId)}/files/${encodeURIComponent(fileId)}`,
    { method: "PATCH", headers, body: buffer, signal },
  );
  let payload;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message || `上传失败：HTTP ${response.status}`);
  }
  return payload;
}

async function uploadPlanFiles(plan, initialSession, onProgress) {
  const session = initialSession;
  const totalBytes = plan.files.reduce((total, item) => total + item.size, 0);
  const uploadedById = new Map(session.files.map((item) => [item.file_id, Number(item.uploaded_bytes || 0)]));
  const reportProgress = (message) => {
    const uploaded = [...uploadedById.values()].reduce((total, value) => total + value, 0);
    onProgress(totalBytes ? uploaded / totalBytes : 1, message);
  };

  const uploadOne = async (descriptor) => {
    let offset = Number(uploadedById.get(descriptor.file_id) || 0);
    if (offset > descriptor.size) throw new Error(`${descriptor.name} 的服务器续传位置无效`);
    if (offset > 0) {
      const probeStart = Math.max(0, offset - 64 * 1024);
      const probe = await descriptor.browser_file.slice(probeStart, offset).arrayBuffer();
      try {
        const verified = await sendUploadChunk(session.session_id, descriptor.file_id, probeStart, probe, state.uploadAbortController?.signal);
        offset = Number(verified.uploaded_bytes);
        uploadedById.set(descriptor.file_id, offset);
      } catch (error) {
        throw new Error(`${descriptor.name} 与上次上传的文件不一致，不能从旧断点拼接。${error.message}`);
      }
    }
    let failures = 0;
    while (offset < descriptor.size) {
      const chunkSize = Math.max(1, Number(session.chunk_size_bytes || 16 * 1024 * 1024));
      const end = Math.min(offset + chunkSize, descriptor.size);
      try {
        const buffer = await descriptor.browser_file.slice(offset, end).arrayBuffer();
        if (state.uploadCancelled) throw new DOMException("上传已取消", "AbortError");
        const result = await sendUploadChunk(session.session_id, descriptor.file_id, offset, buffer, state.uploadAbortController?.signal);
        offset = Number(result.uploaded_bytes);
        uploadedById.set(descriptor.file_id, offset);
        failures = 0;
        reportProgress(`正在断点续传：${descriptor.name}（${formatBytes(offset)} / ${formatBytes(descriptor.size)}）`);
      } catch (error) {
        failures += 1;
        if (failures > 8) throw new Error(`${descriptor.name} 多次重试仍失败；再次点击可从断点继续。${error.message}`);
        reportProgress(`网络中断，正在恢复 ${descriptor.name}（第 ${failures} 次重试）`);
        await delay(Math.min(8000, 750 * 2 ** (failures - 1)));
        if (state.uploadCancelled || error.name === "AbortError") throw error;
        const refreshed = await api(`/api/upload-sessions/${encodeURIComponent(session.session_id)}`);
        const remote = refreshed.files.find((item) => item.file_id === descriptor.file_id);
        offset = Number(remote?.uploaded_bytes || 0);
        uploadedById.set(descriptor.file_id, offset);
      }
    }
  };
  let cursor = 0;
  const worker = async () => {
    while (cursor < plan.files.length) {
      const descriptor = plan.files[cursor];
      cursor += 1;
      await uploadOne(descriptor);
    }
  };
  const parallelFiles = Math.max(1, Math.min(Number(session.parallel_files || 2), plan.files.length));
  await Promise.all(Array.from({ length: parallelFiles }, worker));
  return api(`/api/upload-sessions/${encodeURIComponent(session.session_id)}`);
}

function renderStages(activeStage, progressValue) {
  const stages = ["capacity_reservation","source_transfer","original_ingest","nas_ingest","input_seal","input_preflight","queued","preflight","alignment","motion_probe","candidate_coarse","candidate_fine","candidate_audit","experiment_understanding","experiment_clips","key_materials","mllm","package","daily_report"];
  const activeIndex = stages.indexOf(activeStage);
  const element = document.querySelector("#run-stages");
  if (!element) return;
  element.innerHTML = stages.map((stage, index) => `<div class="${activeStage === "completed" || index < activeIndex ? "done" : index === activeIndex ? "active" : ""}"><i></i><span>${esc(STAGE_LABELS[stage])}</span></div>`).join("");
  setProgress(progressValue, STAGE_LABELS[activeStage] || activeStage);
}

function setProgress(value, message) {
  const percent = Math.max(0, Math.min(100, Math.round(Number(value || 0) * 100)));
  const bar = document.querySelector("#progress-bar");
  if (!bar) return;
  bar.style.width = `${percent}%`;
  document.querySelector("#progress-percent").textContent = `${percent}%`;
  document.querySelector("#progress-message").textContent = message;
}

async function cancelActiveUpload() {
  const sessionId = state.activeUploadSessionId;
  if (!sessionId) return;
  state.uploadCancelled = true;
  state.uploadAbortController?.abort();
  setProgress(0, "正在取消上传并释放预留空间");
  try {
    const result = await api(`/api/upload-sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    forgetUploadSession();
    state.activeUploadSessionId = null;
    document.querySelector("#cancel-upload")?.classList.add("hidden");
    document.querySelector("#start-run").disabled = false;
    setProgress(0, result.archive_cleanup === "removed" ? "上传已取消，暂存文件和空间预留已释放" : "上传已取消，空间预留已释放；残留暂存目录等待服务清理");
    toast("上传已取消，未创建分析任务。", "");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function submitRun() {
  const review = reviewState();
  if (!review.ready) { toast(review.mode === "nas_collection" ? "该批次尚未通过封口与视角质量门，或缺少归档名称。" : "请先填写实验名称，选择至少两路视频，并确认第一/第三人称。", "error"); return; }
  if (review.mode === "nas_collection") {
    await submitCollectionRun(review);
    return;
  }
  const plan = buildUploadPlan(review);
  document.querySelector("#upload-progress").classList.remove("hidden");
  document.querySelector("#start-run").disabled = true;
  state.uploadCancelled = false;
  state.uploadAbortController = new AbortController();
  renderStages("capacity_reservation", 0);
  setProgress(0, "正在本地检查文件头、CSV 表头和真实大小");
  try {
    await validateBrowserPlan(plan);
    setProgress(.002, "本地轻量检查通过，正在按真实大小预留归档空间");
    let session = await createOrResumeUploadSession(plan);
    state.activeUploadSessionId = session.session_id;
    document.querySelector("#cancel-upload")?.classList.remove("hidden");
    if (["finalized", "released"].includes(session.status) && session.run_id) {
      const archiveName = session.archive_name || review.title;
      state.activeRun = { run_id: session.run_id, state: "queued", progress: .15, experiment_id: archiveName };
      setPhase(state.activeRun);
      forgetUploadSession();
      state.activeUploadSessionId = null;
      document.querySelector("#cancel-upload")?.classList.add("hidden");
      await pollRun(session.run_id, archiveName);
      return;
    }
    const capacity = session.capacity;
    if (capacity) {
      setProgress(.005, `空间已预留 ${formatBytes(capacity.reserved_bytes)}；开始传输 ${formatBytes(session.expected_source_bytes)}`);
    }
    renderStages("source_transfer", .005);
    session = await uploadPlanFiles(plan, session, (value, message) => setProgress(.005 + value * .13, message));
    renderStages("input_seal", .14);
    setProgress(.14, "所有分块已校验到达，正在生成输入封条；大文件无需再次全盘读取");
    renderStages("input_preflight", .145);
    setProgress(.145, "服务器正在探测媒体可读性、分片顺序与跨视角时钟覆盖");
    const created = await api(`/api/upload-sessions/${encodeURIComponent(session.session_id)}/finalize`, { method: "POST" });
    const archiveName = new URL(created.archive_url, location.origin).searchParams.get("archive") || review.title;
    state.activeRun = { run_id: created.run_id, state: "queued", progress: .15, experiment_id: archiveName };
    forgetUploadSession();
    state.activeUploadSessionId = null;
    state.uploadAbortController = null;
    document.querySelector("#cancel-upload")?.classList.add("hidden");
    renderStages("queued", .15);
    setPhase(state.activeRun);
    await pollRun(created.run_id, archiveName);
  } catch (error) {
    if (state.uploadCancelled || error.name === "AbortError") return;
    toast(error.message, "error");
    setProgress(1, `失败：${error.message}`);
    document.querySelector("#start-run").disabled = false;
  }
}

async function submitCollectionRun(review) {
  document.querySelector("#upload-progress").classList.remove("hidden");
  document.querySelector("#start-run").disabled = true;
  setProgress(.01, "正在锁定采集批次并生成零复制输入清单");
  try {
    const created = await api(`/api/collections/${encodeURIComponent(review.collection.collection_id)}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiment_name: review.title }),
    });
    state.activeRun = {
      run_id: created.run_id,
      state: "queued",
      progress: .02,
      experiment_id: created.archive_name,
      source_collection_id: review.collection.collection_id,
    };
    setPhase(state.activeRun);
    setProgress(.02, "已开始分析，正在检查视频与时间记录");
    await pollRun(created.run_id, created.archive_name);
  } catch (error) {
    toast(error.message, "error");
    setProgress(1, `失败：${error.message}`);
    document.querySelector("#start-run")?.removeAttribute("disabled");
  }
}

async function submitNasBatchRun(batch, button) {
  document.querySelector("#upload-progress")?.classList.remove("hidden");
  button.disabled = true;
  setProgress(.01, "正在锁定采集批次并创建分析任务");
  try {
    const name = state.experimentName.trim();
    const created = await api(
      `/api/nas-batches/${encodeURIComponent(batch.batch_id)}/runs`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(name ? { experiment_name: name } : {}),
      },
    );
    state.activeRun = {
      run_id: created.run_id,
      state: "queued",
      progress: .02,
      experiment_id: created.archive_name,
      source_collection_id: created.collection_id,
    };
    setPhase(state.activeRun);
    setProgress(.02, "批次已接管，正在校验视频与时间戳");
    await pollRun(created.run_id, created.archive_name);
  } catch (error) {
    toast(error.message, "error");
    setProgress(1, `失败：${error.message}`);
    button.disabled = false;
  }
}

async function pollRun(runId, archiveName) {
  let consecutiveReadFailures = 0;
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    let run;
    try {
      run = await api(`/api/runs/${encodeURIComponent(runId)}`);
      consecutiveReadFailures = 0;
    } catch (error) {
      consecutiveReadFailures += 1;
      const retrySeconds = Math.min(10, 2 ** Math.min(3, consecutiveReadFailures));
      const progress = .15 + Number(state.activeRun?.progress || 0) * .85;
      setProgress(
        progress,
        `服务暂时繁忙，任务仍在后台运行；${retrySeconds} 秒后继续读取进度`,
      );
      await delay(retrySeconds * 1000);
      continue;
    }
    state.activeRun = run;
    setPhase(run);
    renderStages(run.state, .15 + Number(run.progress || 0) * .85);
    if (run.state === "completed") {
      setProgress(1, "分析完成，全部实验成果已保存");
      toast("实验分析完成，结果已保存。", "");
      await loadAll();
      if (routeParts()[0] === "new") location.hash = `#/archive/${encodeURIComponent(archiveName)}/experiments`;
      return;
    }
    if (run.state === "failed") {
      toast(run.error || "分析失败", "error");
      setProgress(1, `失败：${run.error || "请检查服务日志"}`);
      document.querySelector("#start-run")?.removeAttribute("disabled");
      return;
    }
  }
}

function renderTasks() {
  setChrome("tasks");
  const terminal = new Set(["completed", "failed", "interrupted"]);
  const runs = [...state.runs].sort((a, b) => Number(terminal.has(a.state)) - Number(terminal.has(b.state)) || String(b.updated_at || b.observability?.status?.updated_at || "").localeCompare(String(a.updated_at || a.observability?.status?.updated_at || "")));
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">实验分析</p><h1>分析进度</h1><p>查看实验处理状态；技术参数与阶段文件默认收起。</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header><section class="status-grid">${statusCard("activity","全部任务",number(runs.length),"")}${statusCard("gauge","正在分析",number(runs.filter((run)=>!["completed","failed","interrupted"].includes(run.state)).length),"")}${statusCard("check","已完成",number(runs.filter((run)=>run.state==="completed").length),"")}${statusCard("file","需要关注",number(runs.filter((run)=>["failed","interrupted"].includes(run.state)).length),"")}</section>${runs.length ? runs.map((run)=>runObservabilityCard(run,false)).join("") : `<section class="panel"><div class="empty-state"><span class="empty-illustration" aria-hidden="true">${icon("activity")}</span><strong>暂无分析任务</strong><a class="secondary-button" href="#/new">${icon("plus")}新建实验</a></div></section>`}</div>`;
  bindArchiveActions();
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
  if (!["completed","failed","interrupted"].includes(run.state) && updated && !Number.isNaN(updated.getTime())) {
    value += Math.max(0, (Date.now() - updated.getTime()) / 1000);
  }
  return value;
}

function guidedStageState(run, definition, receipts, index) {
  const status = run.observability?.status || {};
  const current = status.stage || run.state;
  const failed = status.failed_stage || (run.state === "failed" ? current : null);
  const completedReceipt = definition.completedBy.map((stage)=>receipts.get(stage)).find(Boolean);
  if (run.state === "completed" || completedReceipt) return { state: "done", receipt: completedReceipt };
  if (["failed","interrupted"].includes(run.state) && definition.stages.includes(failed)) return { state: run.state, receipt: null };
  if (definition.stages.includes(current)) return { state: "active", receipt: null };
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
  const snapshot = run.observability || {};
  const receipts = new Map((snapshot.stage_receipts || []).map((item)=>[item.stage,item]));
  const rows = GUIDED_PIPELINE.map((definition,index)=>({ definition, ...guidedStageState(run,definition,receipts,index) }));
  const active = rows.find((item)=>["active","failed","interrupted"].includes(item.state)) || rows.find((item)=>item.state === "waiting") || rows.at(-1);
  const activeIndex = rows.indexOf(active);
  const next = rows.slice(activeIndex+1).find((item)=>item.state === "waiting");
  const status = snapshot.status || {};
  const stateLabel = { active:"正在进行", done:"已归档", "done-unreceipted":"已通过", waiting:"等待中", failed:"本环节失败", interrupted:"等待续跑" };
  const cards = rows.map(({definition,state:stageState,receipt})=>{
    const deliveredReceipts = definition.stages.map((stage)=>receipts.get(stage)).filter(Boolean);
    const artifacts = deliveredReceipts.flatMap((item)=>item.artifacts || []).filter((item)=>item.available);
    const filesByName = new Map(artifacts.filter((item)=>item.kind === "file" && item.relative_path).map((item)=>[item.name,item]));
    const featured = (definition.outputs || []).map(([name,label])=>({ item: filesByName.get(name), label })).filter(({item})=>item);
    const latestReceipt = receipt || deliveredReceipts.at(-1);
    const receiptUrl = latestReceipt ? stageArtifactUrl(run, latestReceipt.receipt) : "";
    const resultLink = definition.resultTab && deliveredReceipts.length
      ? `<a href="${esc(stageResultRoute(run,definition.resultTab[0]))}">${icon("arrow")}${esc(definition.resultTab[1])}</a>` : "";
    const artifactLinks = deliveredReceipts.length ? `<div class="journey-artifacts">${resultLink}${featured.map(({item,label})=>`<a target="_blank" href="${esc(stageArtifactUrl(run,item.relative_path))}" title="${esc(item.relative_path)}">${icon("file")}${esc(label)}</a>`).join("")}${receiptUrl ? `<a class="receipt-link" target="_blank" href="${esc(receiptUrl)}">${icon("check")}完整清单</a>` : ""}</div>` : "";
    const detail = stageState === "done"
      ? `${receipt?.stage_duration_seconds != null ? `耗时 ${duration(receipt.stage_duration_seconds)}` : "产出已保存"}`
      : stageState === "active" ? definition.doing
      : stageState === "failed" ? `失败位置：${STAGE_LABELS[status.failed_stage] || status.failed_stage || run.error || "本环节"}`
      : stageState === "interrupted" ? "已完成产出保留，可从持久账本续跑。"
      : stageState === "done-unreceipted" ? "后续阶段已开始；该历史运行没有独立阶段回执。"
      : "等待前序环节完成";
    return `<article class="journey-step ${stageState}"><span class="journey-number">${definition.number}</span><div class="journey-copy"><header><strong>${esc(definition.title)}</strong><span>${esc(stateLabel[stageState])}</span></header><p>${esc(detail)}</p>${artifactLinks}</div></article>`;
  }).join("");
  return `<section class="current-guide ${active.state}"><div><span class="guide-kicker">${esc(stateLabel[active.state])} · 第 ${esc(active.definition.number)} 步，共 7 步</span><h3>${esc(active.definition.title)}</h3><p>${esc(status.message || active.definition.doing)}</p></div><aside><small>${next ? "接下来" : "最终结果"}</small><strong>${esc(next?.definition.title || "实验成果已完整保存")}</strong></aside></section><div class="task-progress-line"><span>${["failed", "interrupted"].includes(run.state) ? "处理已停止" : `已完成 ${Math.round(Number(run.progress || 0) * 100)}%`}</span><span>已用 ${duration(elapsedForRun(run, status))}</span></div><details class="technical-observability" ${outputsOpen ? "open" : ""}><summary>查看各环节与生成文件</summary><div class="pipeline-journey">${cards}</div></details>`;
}

function runObservabilityCard(run, outputsOpen = false) {
  const snapshot = run.observability || {};
  const status = snapshot.status || {};
  const live = latestTelemetry(run);
  const gpu = live.gpu || {};
  const network = live.host_network || {};
  const processIo = live.pipeline_process_tree_io || {};
  const views = normalizedViews(status);
  const metrics = snapshot.metrics || {};
  const tokens = metrics.tokens?.run_total || {};
  const freshness = newestFreshness(snapshot);
  const current = status.stage || run.state;
  const archivePath = (run.state === "completed" ? run.nas_output : run.nas_staging) || run.nas_output || "运行目录待登记";
  const gpuCompute = gpu["utilization.gpu"] ?? gpu.utilization_percent ?? "—";
  const nvdec = gpu["utilization.decoder"] ?? gpu.decoder_percent ?? "—";
  const memory = gpu["memory.used"] ?? gpu.memory_used_mib ?? "—";
  const freshnessNote = freshness.stale && !["completed","failed","interrupted"].includes(run.state)
    ? `<div class="freshness-warning">状态数据已 ${duration(freshness.age)} 未更新：NAS 遥测可能延迟，不能据此判定任务停止；页面仍会继续刷新。</div>` : "";
  const previewAvailable = (snapshot.stage_receipts || []).length > 0;
  const resultRoute = run.nas_staging && run.state !== "completed"
    ? `stage/${encodeURIComponent(run.run_id)}` : `archive/${encodeURIComponent(run.experiment_id || "")}`;
  const archiveLink = run.state === "completed" || previewAvailable
    ? `<a class="secondary-button" href="#/${resultRoute}/experiments">${run.state === "completed" ? "查看实验结果" : "预览已完成内容"}</a>` : "";
  return `<section class="panel live-run-card"><header class="panel-heading"><div title="实验编号：${esc(run.experiment_id || run.run_id)}"><p class="panel-kicker">实验分析</p><h2>${esc(productExperimentName(run.experiment_id || run.run_id))}</h2></div><div class="run-heading-actions">${archiveLink}<span class="queue-status ${esc(run.state)}"><i></i>${esc(STAGE_LABELS[current] || current)}</span></div></header>${freshnessNote}${guidedPipelineView(run,outputsOpen)}<details class="technical-observability"><summary>技术信息与运行参数</summary><div class="technical-actions"><p>${esc(run.run_id)} · ${esc(archivePath)}</p><button class="secondary-button" type="button" data-copy-path="${esc(archivePath)}">${icon("copy")}复制文件路径</button></div><div class="live-metric-grid"><article><small>总体进度</small><strong>${Math.round(Number(run.progress ?? status.progress ?? 0)*100)}%</strong><span>${duration(elapsedForRun(run,status))} 已用</span></article><article><small>GPU / NVDEC（瞬时）</small><strong>${gpuCompute}% / ${nvdec}%</strong><span>单点样本 · ${memory} MiB 显存</span></article><article><small>CPU / 内存（瞬时）</small><strong>${live.cpu_percent ?? "—"}% / ${live.memory_percent ?? "—"}%</strong><span>单点主机采样</span></article><article><small>NAS/网络读取（瞬时）</small><strong>${network.received_mib_per_second ?? "—"} MiB/s</strong><span>进程树读 ${processIo.read_mib_per_second ?? "—"} MiB/s</span></article><article><small>模型 Token</small><strong>${number(tokens.total_tokens)}</strong><span>${number(tokens.input_tokens)} 输入 + ${number(tokens.output_tokens)} 输出</span></article><article><small>最近更新</small><strong>${freshness.value ? formatDate(freshness.value) : "待采样"}</strong><span>${freshness.age == null ? "正式运行记录" : `${duration(freshness.age)} 前`}</span></article></div>${views.length ? `<div class="view-runtime-grid">${views.map(([viewId,item])=>{ const done=item.completed_units ?? item.completed_work_units; const total=item.total_units ?? item.total_work_units; return `<article><span class="view-state-dot ${item.state === "completed" ? "completed" : ""}"></span><div><strong>${esc(viewId)}</strong><small>${esc(item.role || "待识别角色")} · ${esc(item.decode_backend || "待分配解码")} · ${esc(item.state || "waiting")} · ${total ? `${number(done)}/${number(total)} 单元` : `${number(item.segment_count)} 分片`}</small></div></article>`; }).join("")}</div>` : `<div class="empty-state compact-empty">等待逐视角运行记录。</div>`}</details></section>`;
}

function renderOperations() {
  setChrome("operations");
  const health = state.health || {};
  const modelStatus = !health.ark_key_configured ? "待配置密钥" : health.mllm_enabled ? "已启用" : "密钥已配置 · 待启用";
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">系统</p><h1>运行状态</h1><p>确认分析与存储功能是否可用。</p></div></header><section class="health-grid"><article class="health-item"><span>${icon("server")}</span><div><small>分析服务</small><strong>${health.status === "ok" ? "运行正常" : "暂不可用"}</strong></div></article><article class="health-item"><span>${icon("folder")}</span><div><small>NAS 存储</small><strong>${health.nas_available && health.archive_available ? "连接正常" : "待连接"}</strong></div></article><article class="health-item"><span>${icon("brain")}</span><div><small>智能理解</small><strong>${modelStatus}</strong></div></article><article class="health-item"><span>${icon("video")}</span><div><small>视频处理</small><strong>准备就绪</strong></div></article></section><section class="panel"><header class="panel-heading"><div><h2>存储位置</h2></div><span class="badge">${health.nas_available && health.archive_available ? "NAS 已连接" : "需要检查"}</span></header><div class="capability-list"><div><span>${icon("folder")}</span><p><strong>实验产出</strong><small>NAS / VisionCortexExperimentArchive</small></p></div><div><span>${icon("server")}</span><p><strong>处理缓存</strong><small>NAS / VisionCortexExperimentCache</small></p></div></div></section>${isNasMode() ? `<details class="panel technical-observability"><summary>管理员工具</summary><div class="maintenance-action"><p>仅在系统排障时运行环境检查。</p><button class="secondary-button" id="rerun-benchmark" type="button">${icon("activity")}运行环境检查</button></div></details>` : ""}</div>`;
  document.querySelector("#rerun-benchmark")?.addEventListener("click", rerunBenchmark);
}

async function loadArchive(name, staging = false) {
  const key = `${staging ? "stage" : "archive"}/${name}`;
  const cached = state.archiveCache.get(key);
  if (cached?.observability?.status?.stage === "completed" && !staging) return cached;
  const data = await api(staging ? `/api/staging-runs/${encodeURIComponent(name)}/archive` : `/api/archives/${encodeURIComponent(name)}`);
  state.archiveCache.set(key, data);
  return data;
}

function resultHeader(data, tab) {
  const metrics = data.metrics || {};
  const currentRun = metrics.preprocessing_display?.current_run || {};
  const analysisDuration = currentRun.total_seconds ?? metrics.total_duration_seconds;
  const materialCount = data.key_events.length || data.preliminary_materials?.length || 0;
  const reportCount = ["daily_report_pdf", "daily_report_json"].filter((key)=>data.links?.[key]).length;
  const metadata = experimentMetadata(data.name);
  const tags = Array.isArray(metadata.tags) ? metadata.tags : [];
  const metadataChips = [metadata.owner ? `负责人 · ${metadata.owner}` : "", ...tags].filter(Boolean);
  return `<div class="result-header"><header class="page-hero compact experiment-hero"><div title="档案编号：${esc(data.name)}"><p class="eyebrow">实验详情</p><h1>${esc(productExperimentName(data.name))}</h1><p>查看实验过程、关键素材与已生成的报告。</p>${metadataChips.length ? `<div class="experiment-meta-chips">${metadataChips.map((item)=>`<span>${esc(item)}</span>`).join("")}</div>` : ""}</div><div class="hero-actions"><button class="secondary-button" type="button" data-edit-experiment>${icon("file")}编辑信息</button><button class="primary-button" type="button" data-open-folder="${esc(data.name)}">${icon("folder")}打开实验文件</button></div></header><details class="technical-observability archive-file-details"><summary>实验文件位置</summary><div class="archive-path-row"><p class="archive-path">${esc(data.path)}</p><button class="secondary-button" type="button" data-copy-path="${esc(data.path)}">${icon("copy")}复制路径</button></div></details><section class="status-grid result-status-grid">${statusCard("flask","实验片段",number(data.experiments.length),"")}${statusCard("image",data.preliminary_materials?.length ? "初步素材" : "关键素材",number(materialCount),"")}${statusCard("clock","分析用时",duration(analysisDuration),"")}${statusCard("file","专业报告",number(reportCount),"PDF / JSON")}</section><div class="result-nav-shell"><strong class="result-nav-name" title="${esc(data.name)}">${esc(productExperimentName(data.name))}</strong><nav class="result-tabs" aria-label="实验结果导航"><a class="result-tab ${tab==="experiments"?"active":""}" href="#/archive/${encodeURIComponent(data.name)}/experiments">${icon("video")}<span>实验片段和步骤理解</span></a><a class="result-tab ${tab==="materials"?"active":""}" href="#/archive/${encodeURIComponent(data.name)}/materials">${icon("boxes")}<span>关键素材</span></a><a class="result-tab ${tab==="reports"?"active":""}" href="#/archive/${encodeURIComponent(data.name)}/reports">${icon("file")}<span>实验室日报</span></a><a class="result-tab ${tab==="metrics"?"active":""}" href="#/archive/${encodeURIComponent(data.name)}/metrics">${icon("folder")}<span>专业 PDF / JSON 报告</span></a></nav><button class="result-nav-folder" type="button" data-open-folder="${esc(data.name)}" aria-label="打开实验文件">${icon("folder")}</button></div><dialog class="experiment-meta-dialog" id="experiment-meta-dialog"><form id="experiment-meta-form"><header><div><p class="eyebrow">工作区信息</p><h2>完善实验信息</h2><p>便于日常查找与协作，不会改写正式证据档案。</p></div><button class="dialog-close" type="button" data-close-experiment-meta aria-label="关闭">${icon("x")}</button></header><div class="experiment-meta-fields"><label><span>显示名称</span><input name="display_name" value="${esc(metadata.displayName || "")}" placeholder="例如：电子天平称量验证" /></label><label><span>负责人</span><input name="owner" value="${esc(metadata.owner || "")}" placeholder="姓名或团队" /></label><label class="wide"><span>标签</span><input name="tags" value="${esc(tags.join("、"))}" placeholder="例如：称量、质控、第二轮" /></label><label class="wide"><span>实验备注</span><textarea name="note" rows="4" placeholder="记录目的、批次说明或后续事项">${esc(metadata.note || "")}</textarea></label></div>${metadata.note ? `<p class="experiment-note-preview"><strong>当前备注</strong>${esc(metadata.note)}</p>` : ""}<footer><span>原始档案编号始终保留：${esc(data.name)}</span><div><button class="secondary-button" type="button" data-close-experiment-meta>取消</button><button class="primary-button" type="submit">保存信息</button></div></footer></form></dialog></div>`;
}

function experimentCard(experiment, preliminary = false) {
  const stepCount = experiment.steps?.length || 0;
  const videoTile = (label, url, className = "") => `<article class="experiment-media ${className}"><div class="video-frame">${url ? `<video controls preload="metadata" src="${esc(url)}"></video>` : `<div class="empty-state">${esc(label)}尚未生成</div>`}</div><p class="video-caption">${icon("video")}${esc(label)}</p></article>`;
  const videos = `${videoTile("同步双视角", experiment.aligned_video_url, "experiment-media--aligned")}${videoTile("第一人称", experiment.first_person_video_url)}${videoTile("第三人称", experiment.third_person_video_url)}`;
  return `<article class="experiment-card experiment-card--understanding"><header class="experiment-card-header"><div class="experiment-title-copy"><span class="experiment-label">实验片段</span><h2>${esc(experiment.name)}</h2><div class="experiment-meta"><span>${icon("clock")}${timecode(experiment.start_ms)} → ${timecode(experiment.end_ms)}</span><span>${icon("activity")}${number(stepCount)} 个步骤</span></div></div><span class="badge">${experiment.continuity_type === "continuous" ? "连续实验" : "独立实验"}</span></header><div class="video-understanding"><div class="experiment-media-set">${videos}</div><div class="understanding-panel"><div class="understanding-item is-summary"><span class="understanding-index">01</span><div><strong>${preliminary ? "初步理解 · 待复核" : "实验整体理解"}</strong><p>${esc(experiment.summary || "暂无模型摘要")}</p></div></div><div class="understanding-item"><span class="understanding-index">02</span><div><strong>步骤粒度</strong><p>${number(stepCount)} 个带时间边界的细粒度步骤；每步明确当前正在做什么和可支持的下一步。</p></div></div>${experiment.uncertainties?.length ? `<div class="understanding-item is-warning"><span class="understanding-index">!</span><div><strong>不确定项</strong><p>${esc(experiment.uncertainties.join("；"))}</p></div></div>` : ""}</div></div><details class="experiment-steps" ${stepCount <= 10 ? "open" : ""}><summary class="experiment-steps-toggle"><span class="experiment-steps-title"><i>${icon("activity")}</i><span><strong>细粒度步骤</strong><small>按时间顺序查看当前动作与下一步建议</small></span></span><span class="experiment-step-count">${number(stepCount)} 个 ${icon("chevron")}</span></summary><div class="step-list">${(experiment.steps || []).map((step,index)=>`<article class="step-card"><span class="step-number"><b>${String(step.step_index ?? index+1).padStart(2,"0")}</b><small>${timecode(step.start_global_ms)}<br/>${timecode(step.end_global_ms)}</small></span><div class="step-body"><strong>当前：${esc(step.current_step || step.observed_action || "未描述")}</strong><p class="step-next"><b>下一步：</b>${esc(step.next_step || "没有足够证据支持下一步")}</p><span class="step-meta">对象：${esc((step.objects || []).join("、") || "未明确")} · 置信度：${step.confidence ?? "—"}</span></div></article>`).join("")}</div></details></article>`;
}

function verificationTrace(verification) {
  const confidence = verification.confidence || {};
  const timing = verification.timing || {};
  const models = verification.models || [];
  const reasons = verification.uncertainty_reasons || [];
  const views = verification.views || [];
  const status = VERIFICATION_STATUS_LABELS[verification.status] || verification.status || "未记录";
  return `<details class="verification-trace"><summary>查看分析依据与复核信息</summary><div class="verification-grid"><div><small>结论</small><strong>${esc(status)}</strong></div><div><small>置信度范围</small><strong>${confidence.minimum == null ? "—" : `${percent(confidence.minimum)}–${percent(confidence.maximum)}`}</strong></div><div><small>视觉分析</small><strong>${duration(timing.inference_seconds)}</strong></div><div><small>准备模型</small><strong>${duration(timing.model_load_seconds)}</strong></div></div><div class="model-chip-list">${models.map((model)=>`<span>${esc(MODEL_LABELS[model] || model)}</span>`).join("") || "<span>无模型记录</span>"}</div>${views.length ? `<table class="metric-table verification-view-table"><thead><tr><th>视角</th><th>参与对象</th><th>最低置信度</th><th>复核状态</th></tr></thead><tbody>${views.map((view)=>`<tr><td>${esc(view.role_label || view.view_id)}</td><td>${esc((view.rendered_classes || []).join("、") || "无")}</td><td>${percent(view.minimum_rendered_confidence)}</td><td>${esc(VERIFICATION_STATUS_LABELS[view.status] || view.status)}</td></tr>`).join("")}</tbody></table>` : ""}${reasons.length ? `<p class="verification-warning"><strong>需要注意：</strong>${esc(reasons.join("；"))}</p>` : `<p class="verification-pass">当前没有需要特别说明的不确定项。</p>`}</details>`;
}

function materialCard(event, index = 0) {
  const mllm = event.provenance?.mllm || {};
  const facts = event.decision?.observed_facts || [];
  const current = productEvidenceText(mllm.current_step || facts[0], "当前动作等待进一步说明");
  const next = productEvidenceText(mllm.next_step || event.decision?.supported_inferences?.[0], "当前画面不足以判断下一步");
  const observed = facts.length ? facts.map((fact)=>productEvidenceText(fact, "暂无补充说明")) : ["当前档案没有单独记录可直接确认的画面事实。"];
  const cross = event.cross_view_associations?.[0];
  const dualView = eventHasDualViewSupport(event);
  const supportCopy = dualView
    ? "两个视角共同支持这一关键动作"
    : cross
      ? "画面已同步；这一动作在其中一个视角中更清晰"
      : "双视角画面已保存，动作关联等待进一步复核";
  const clipSeconds = Math.max(0, (Number(event.end_us) - Number(event.start_us)) / 1_000_000);
  const peakOffset = Math.min(clipSeconds, Math.max(0, (Number(event.peak_timestamp_us ?? event.start_us) - Number(event.start_us)) / 1_000_000));
  const selected = state.selectedMaterials.has(String(event.event_id));
  const uncertainty = /不足|无法|不能|待确认|不确定|未明确/.test(next);
  return `<article class="material-card product-material-card ${selected ? "is-selected" : ""}" data-material-card="${esc(event.event_id)}">
    <header class="material-card-header"><div><span class="material-sequence">关键素材 ${String(index + 1).padStart(2,"0")}</span><h2>${esc(ACTION_LABELS[event.action_type] || "关键实验动作")}</h2><p>${timecode(Number(event.start_us)/1000)} → ${timecode(Number(event.end_us)/1000)}</p></div><div class="material-card-actions"><label class="material-select"><input type="checkbox" data-material-select="${esc(event.event_id)}" ${selected ? "checked" : ""}/><span>选择</span></label><span class="material-support-badge ${dualView ? "trusted" : "partial"}">${dualView ? icon("check") : icon("image")}${dualView ? "双视角印证" : "主要视角清晰"}</span></div></header>
    <figure class="material-visual"><div class="material-frame">${event.aligned_frame_url ? `<img loading="lazy" src="${esc(event.aligned_frame_url)}" alt="第一人称与第三人称同步关键画面"/>` : `<div class="empty-state compact"><strong>关键画面准备中</strong></div>`}</div><figcaption><span>${icon("video")}第一 / 第三人称同步画面</span><span>关键时刻 ${timecode(Number(event.peak_timestamp_us ?? event.start_us)/1000)}</span></figcaption></figure>
    <section class="material-preview-note"><span class="evidence-kind interpreted">步骤理解</span><p>${esc(current)}</p></section>
    <details class="material-product-details"><summary><span>查看详情与播放</span><small>${event.aligned_clip_url ? `视频 ${duration(clipSeconds)}` : "查看分析"}</small>${icon("chevron")}</summary><div class="material-product-content">
      <div class="material-story"><section class="material-story-main evidence-observed"><small><span class="evidence-kind observed">画面确认</span>直接观察事实</small><ul class="fact-list">${observed.slice(0,3).map((fact)=>`<li>${esc(fact)}</li>`).join("")}</ul></section><section class="material-next-step ${uncertainty ? "evidence-uncertain" : "evidence-inferred"}"><small><span class="evidence-kind ${uncertainty ? "uncertain" : "inferred"}">${uncertainty ? "证据不足" : "模型提示"}</span>下一步推测 · 非事实结论</small><p>${esc(next)}</p></section><p class="material-support-note">${dualView ? icon("check") : icon("image")}<span>${esc(supportCopy)}</span></p></div>
      ${event.aligned_clip_url ? `<details class="material-clip-details"><summary>${icon("video")}<span>播放关键片段</span><small>${duration(clipSeconds)}</small>${icon("chevron")}</summary><div class="material-player-toolbar"><span>${icon("video")}同步双视角</span><small>当前档案提供第一 / 第三人称同步合成画面</small><div class="material-player-actions"><button type="button" data-review-adjacent="previous" data-review-event="${esc(event.event_id)}" aria-label="上一份素材">上一份</button><label>倍速<select data-video-speed="${esc(event.event_id)}"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label><label class="loop-control"><input type="checkbox" data-video-loop="${esc(event.event_id)}"/>循环</label><button type="button" data-video-fullscreen="${esc(event.event_id)}">全屏</button><button type="button" data-review-adjacent="next" data-review-event="${esc(event.event_id)}" aria-label="下一份素材">下一份</button></div></div><div class="material-video"><video controls preload="metadata" tabindex="0" data-material-video="${esc(event.event_id)}" src="${esc(event.aligned_clip_url)}"></video><div class="material-time-markers"><button type="button" data-video-seek="${esc(event.event_id)}" data-seconds="0"><i></i><span>片段开始<small>${timecode(Number(event.start_us)/1000)}</small></span></button><button class="peak" type="button" data-video-seek="${esc(event.event_id)}" data-seconds="${peakOffset}"><i></i><span>关键时刻<small>${timecode(Number(event.peak_timestamp_us ?? event.start_us)/1000)}</small></span></button><button type="button" data-video-seek="${esc(event.event_id)}" data-seconds="${clipSeconds}"><i></i><span>片段结束<small>${timecode(Number(event.end_us)/1000)}</small></span></button></div><p class="player-shortcuts">快捷键：空格 播放/暂停 · ←/→ 前后 5 秒 · P/N 上一份/下一份</p></div></details>` : ""}
      <details class="material-evidence-details"><summary><span>${icon("boxes")}查看对象与分析依据</span><small>素材记录 ${esc(event.event_id)}</small>${icon("chevron")}</summary><div class="material-evidence-content"><section><h3>动作相关对象</h3><div class="object-chip-list">${materialObjectItems(event.objects || {})}</div></section>${verificationTrace(event.verification || {})}</div></details>
    </div></details>
  </article>`;
}

function eventHasDualViewSupport(event) {
  return (event.cross_view_associations || []).some((association) => association.both_views_support_action === true);
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
    support: "all",
    query: "",
  };
  state.materialArchive = data.name;
  state.selectedMaterials = new Set();
}

function filteredMaterialEvents(data) {
  const filters = state.materialFilters;
  const query = filters.query.trim().toLocaleLowerCase("zh-CN");
  return data.key_events.filter((event) => {
    if (!eventHasAlignedDualViewMaterial(event)) return false;
    if (filters.group !== "all" && event.experiment_group?.group_id !== filters.group) return false;
    if (filters.action !== "all" && event.action_type !== filters.action) return false;
    if (filters.support === "dual" && !eventHasDualViewSupport(event)) return false;
    if (filters.support === "partial" && eventHasDualViewSupport(event)) return false;
    if (!query) return true;
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
  const formalTotal = data.key_events.filter(eventHasAlignedDualViewMaterial).length;
  const groupMap = new Map((data.experiment_groups || []).map((group,index) => [group.group_id, { ...group, index }]));
  const grouped = new Map();
  for (const event of events) {
    const groupId = event.experiment_group?.group_id || "ungrouped";
    if (!grouped.has(groupId)) grouped.set(groupId, []);
    grouped.get(groupId).push(event);
  }
  const sections = [...grouped.entries()].sort(([left],[right]) => (groupMap.get(left)?.index ?? 999) - (groupMap.get(right)?.index ?? 999));
  if (!sections.length) return productState("neutral", "search", "没有符合条件的关键素材", "可以放宽实验片段、动作类型、画面支持或关键词筛选。", `<button class="secondary-button" type="button" data-reset-material-filters>重置素材筛选</button>`);
  return `<div class="material-result-summary"><div><strong>显示 ${number(events.length)} / ${number(formalTotal)} 份关键素材</strong><span>先看概览，需要时展开详情与视频。</span></div><button class="compact-button" type="button" data-select-visible>${icon("check")}选择当前结果</button></div>${sections.map(([groupId,items])=>{ const group = groupMap.get(groupId) || items[0].experiment_group || {}; const index = group.index == null ? "—" : String(group.index + 1).padStart(2,"0"); return `<section class="material-group"><header><span class="material-group-index">${index}</span><div><h2>${esc(group.name || groupId)}</h2><p>${timecode(group.start_ms)} → ${timecode(group.end_ms)} · ${group.continuity_type === "continuous" ? "连续实验" : "独立实验"}</p></div><span class="badge">${number(items.length)} 份素材</span></header><div class="material-grid">${items.map(materialCard).join("")}</div></section>`; }).join("")}`;
}

function materialSelectionBar(data) {
  const count = state.selectedMaterials.size;
  return `<aside class="material-selection-bar ${count ? "is-visible" : ""}" id="material-selection-bar" aria-live="polite"><div><strong>已选择 <b data-selected-count>${number(count)}</b> 份素材</strong><span>选择仅用于整理，不会修改正式证据档案。</span></div><div><button type="button" data-copy-selected>${icon("copy")}复制编号</button><button type="button" data-export-selected>${icon("file")}导出清单</button><button type="button" data-clear-selected>清除</button></div></aside>`;
}

function materialsView(data) {
  if (!data.key_events.length && data.preliminary_materials?.length) {
    return `<section class="panel"><header class="panel-heading"><div><h2>已生成的关键素材 · ${number(data.preliminary_materials.length)}</h2><p>步骤理解与动作复核尚未完成，当前内容为初步素材。</p></div></header><div class="material-grid">${data.preliminary_materials.map((item,index) => `<article class="experiment-card"><header title="素材记录：${esc(item.event_id)}"><h3>初步关键素材 ${String(index + 1).padStart(2,"0")}</h3><span class="badge">待复核 · ${timecode(item.timestamp_ms)}</span></header>${item.frame_url ? `<img loading="lazy" style="width:100%;height:auto" src="${esc(item.frame_url)}" alt="双视角关键画面">` : ""}${item.clip_url ? `<details class="material-clip-details"><summary>${icon("video")}播放关键片段${icon("chevron")}</summary><div class="material-video"><video controls preload="none" src="${esc(item.clip_url)}"></video></div></details>` : ""}</article>`).join("")}</div></section>`;
  }
  ensureMaterialFilters(data);
  const filters = state.materialFilters;
  const formalEvents = data.key_events.filter(eventHasAlignedDualViewMaterial);
  const quarantinedCount = data.key_events.length - formalEvents.length;
  const actionTypes = [...new Set(formalEvents.map((event)=>event.action_type).filter(Boolean))];
  const actionCounts = Object.fromEntries(actionTypes.map((type)=>[type,formalEvents.filter((event)=>event.action_type===type).length]));
  return `<section class="material-workspace"><header class="material-toolbar-heading"><div><p class="eyebrow">关键素材</p><h2>按实验查看关键画面与片段</h2><p>先看概览，需要复核时再展开画面事实、模型提示和视频。</p></div><span class="badge">${number(formalEvents.length)} 份关键素材</span></header><div class="evidence-legend" aria-label="证据信息说明"><span><i class="observed"></i><strong>画面确认</strong>可直接观察</span><span><i class="inferred"></i><strong>模型提示</strong>用于辅助理解</span><span><i class="uncertain"></i><strong>证据不足</strong>保留不确定性</span></div>${quarantinedCount ? `<div class="freshness-warning"><strong>${number(quarantinedCount)} 份素材缺少完整双视角画面，暂未展示。</strong></div>` : ""}<div class="material-filter-bar"><label><span>实验片段</span><select id="material-group-filter"><option value="all" ${filters.group==="all"?"selected":""}>全部实验（${number(formalEvents.length)}）</option>${(data.experiment_groups||[]).map((group,index)=>`<option value="${esc(group.group_id)}" ${filters.group===group.group_id?"selected":""}>${String(index+1).padStart(2,"0")} · ${esc(group.name)}（${number(group.key_event_count)}）</option>`).join("")}</select></label><label><span>动作类型</span><select id="material-action-filter"><option value="all">全部动作类型</option>${actionTypes.map((type)=>`<option value="${esc(type)}" ${filters.action===type?"selected":""}>${esc(ACTION_LABELS[type]||type)}（${number(actionCounts[type])}）</option>`).join("")}</select></label><label><span>画面支持</span><select id="material-support-filter"><option value="all">全部关键素材</option><option value="dual" ${filters.support==="dual"?"selected":""}>两个视角相互印证</option><option value="partial" ${filters.support==="partial"?"selected":""}>单个视角更清晰</option></select></label><label class="material-query"><span>搜索关键素材</span><input id="material-query" value="${esc(filters.query)}" placeholder="例如：移液器、开盖、称量纸" /></label></div><div id="material-results">${materialResults(data)}</div>${materialSelectionBar(data)}</section>`;
}

function bindMaterialFilters(data) {
  const rerender = () => { document.querySelector("#material-results").innerHTML = materialResults(data); bindMaterialInteractions(data); };
  document.querySelector("#material-group-filter")?.addEventListener("change", (event)=>{ state.materialFilters.group=event.target.value; rerender(); });
  document.querySelector("#material-action-filter")?.addEventListener("change", (event)=>{ state.materialFilters.action=event.target.value; rerender(); });
  document.querySelector("#material-support-filter")?.addEventListener("change", (event)=>{ state.materialFilters.support=event.target.value; rerender(); });
  document.querySelector("#material-query")?.addEventListener("input", (event)=>{ state.materialFilters.query=event.target.value; rerender(); });
  bindMaterialInteractions(data);
}

function selectedMaterialEvents(data) {
  return (data.key_events || []).filter((event)=>state.selectedMaterials.has(String(event.event_id)));
}

function updateMaterialSelectionBar() {
  const bar = document.querySelector("#material-selection-bar");
  if (!bar) return;
  const count = state.selectedMaterials.size;
  bar.classList.toggle("is-visible", count > 0);
  const counter = bar.querySelector("[data-selected-count]");
  if (counter) counter.textContent = number(count);
  document.querySelectorAll("[data-material-card]").forEach((card)=>card.classList.toggle("is-selected", state.selectedMaterials.has(card.dataset.materialCard)));
}

function exportSelectedMaterials(data) {
  const events = selectedMaterialEvents(data);
  if (!events.length) { toast("请先选择需要整理的关键素材。", "error"); return; }
  const payload = {
    artifact_type: "visioncortex_user_selection_manifest",
    evidence_status: "DERIVED_SELECTION_NOT_GROUND_TRUTH",
    source_archive: data.name,
    created_at: new Date().toISOString(),
    selected_materials: events.map((event)=>({
      event_id: event.event_id,
      action: ACTION_LABELS[event.action_type] || event.action_type,
      start_timecode: timecode(Number(event.start_us) / 1000),
      end_timecode: timecode(Number(event.end_us) / 1000),
      dual_view_supported: eventHasDualViewSupport(event),
      frame_url: event.aligned_frame_url || null,
      clip_url: event.aligned_clip_url || null,
    })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `VisionCortex-素材清单-${String(data.name).replace(/[^a-zA-Z0-9_-]+/g, "-")}.json`;
  link.click();
  URL.revokeObjectURL(url);
  toast(`已导出 ${events.length} 份素材清单。`);
}

function materialVideo(eventId) {
  return [...document.querySelectorAll("video[data-material-video]")].find((video)=>video.dataset.materialVideo === eventId);
}

function activateAdjacentMaterial(eventId, direction) {
  const cards = [...document.querySelectorAll("[data-material-card]")];
  const currentIndex = cards.findIndex((card)=>card.dataset.materialCard === eventId);
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
  document.querySelector("[data-reset-material-filters]")?.addEventListener("click", () => {
    state.materialFilters = { ...state.materialFilters, group: "all", action: "all", support: "all", query: "" };
    renderArchive(data.name, "materials", routeParts()[0] === "stage");
  });
  document.querySelectorAll(".material-product-details").forEach((details)=>details.addEventListener("toggle", () => {
    if (!details.open) return;
    document.querySelectorAll(".material-product-details[open]").forEach((other)=>{ if (other !== details) other.open = false; });
  }));
  document.querySelectorAll("[data-material-select]").forEach((checkbox)=>checkbox.addEventListener("change", () => {
    const id = checkbox.dataset.materialSelect;
    if (checkbox.checked) state.selectedMaterials.add(id); else state.selectedMaterials.delete(id);
    updateMaterialSelectionBar();
  }));
  document.querySelector("[data-select-visible]")?.addEventListener("click", () => {
    filteredMaterialEvents(data).forEach((event)=>state.selectedMaterials.add(String(event.event_id)));
    document.querySelectorAll("[data-material-select]").forEach((checkbox)=>{ checkbox.checked = true; });
    updateMaterialSelectionBar();
  });
  document.querySelectorAll("[data-video-seek]").forEach((button)=>button.addEventListener("click", () => {
    const video = materialVideo(button.dataset.videoSeek);
    if (!video) return;
    video.currentTime = Math.min(Number(button.dataset.seconds || 0), Number.isFinite(video.duration) ? video.duration : Number(button.dataset.seconds || 0));
  }));
  document.querySelectorAll("[data-video-fullscreen]").forEach((button)=>button.addEventListener("click", async () => {
    const video = materialVideo(button.dataset.videoFullscreen);
    try { if (video?.requestFullscreen) await video.requestFullscreen(); }
    catch { toast("当前浏览器无法进入全屏播放。", "error"); }
  }));
  document.querySelectorAll("[data-review-adjacent]").forEach((button)=>button.addEventListener("click", ()=>activateAdjacentMaterial(button.dataset.reviewEvent, button.dataset.reviewAdjacent)));
  document.querySelectorAll("[data-video-speed]").forEach((select)=>select.addEventListener("change", () => {
    const video = materialVideo(select.dataset.videoSpeed);
    if (video) video.playbackRate = Number(select.value || 1);
  }));
  document.querySelectorAll("[data-video-loop]").forEach((checkbox)=>checkbox.addEventListener("change", () => {
    const video = materialVideo(checkbox.dataset.videoLoop);
    if (video) video.loop = checkbox.checked;
  }));
  const selectionBar = document.querySelector("#material-selection-bar");
  if (selectionBar && selectionBar.dataset.bound !== "true") {
    selectionBar.dataset.bound = "true";
    selectionBar.querySelector("[data-copy-selected]")?.addEventListener("click", async () => {
      const ids = [...state.selectedMaterials];
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

function dailyReportTimelineVisual(timeline, actions) {
  if (!timeline.length) return productState("neutral", "clock", "暂无可展示的实验时间线", "报告保留可用摘要；时间边界完成后会自动显示在这里。");
  const start = Math.min(...timeline.map((group)=>Number(group.start_global_ms || 0)));
  const end = Math.max(...timeline.map((group)=>Number(group.end_global_ms || group.start_global_ms || 0)));
  const span = Math.max(1, end - start);
  const actionMax = Math.max(1, ...actions.map((item)=>Number(item.event_count || 0)));
  const segments = timeline.map((group,index) => {
    const left = Math.max(0, Math.min(100, (Number(group.start_global_ms || 0) - start) / span * 100));
    const width = Math.max(3, Math.min(100 - left, (Number(group.end_global_ms || group.start_global_ms || 0) - Number(group.start_global_ms || 0)) / span * 100));
    return `<span class="report-timeline-segment" style="left:${left.toFixed(2)}%;width:${width.toFixed(2)}%" title="${esc(group.experiment_name)}"><b>${String(index + 1).padStart(2,"0")}</b><small>${number(group.steps?.length)} 步</small></span>`;
  }).join("");
  return `<div class="report-visual-summary"><section class="report-timeline-visual"><header><div><strong>实验过程分布</strong><small>按统一实验相对时间展示</small></div><span>${timecode(start)} → ${timecode(end)}</span></header><div class="report-timeline-track">${segments}</div><div class="report-timeline-labels"><span>${timecode(start)}</span><span>${timecode(start + span / 2)}</span><span>${timecode(end)}</span></div></section><section class="action-distribution"><header><strong>关键动作分布</strong><small>按已归档关键素材统计</small></header><div>${actions.map((item)=>`<span><label><b>${esc(item.action_label)}</b><em>${number(item.event_count)}</em></label><i><u style="width:${(Number(item.event_count || 0) / actionMax * 100).toFixed(2)}%"></u></i></span>`).join("")}</div></section></div>`;
}

function dailyReportView(data) {
  const report = data.daily_report || {};
  const overview = report.overview || {};
  const alignment = report.alignment_summary || {};
  const actions = report.action_summary || [];
  const timeline = report.experiment_timeline || [];
  if (!report.report_id) return `<section class="panel">${productState("progress", "clock", "实验日报正在准备", "完成前序分析与验收后，日报会自动显示在这里。", `<a class="secondary-button" href="#/tasks">查看分析进度</a>`)}</section>`;
  return `<section class="daily-report-cover"><div title="实验编号：${esc(report.experiment_id)}"><p class="eyebrow">实验室日报</p><h2>${esc(report.report_date)} · 实验摘要</h2><p>${esc(productExperimentName(report.experiment_id))}</p></div><span class="badge">${!overview.evidence_package_eval_passed ? "内容生成中" : data.quality_acceptance?.passed === true ? "内容已生成" : "等待复核"}</span></section>
  <section class="status-grid">${statusCard("video","拍摄视角",number(overview.input_view_count),`${number(overview.first_person_views)} 第一人称 + ${number(overview.third_person_views)} 第三人称`)}${statusCard("flask","实验片段",number(overview.experiment_group_count),"独立 / 连续实验")}${statusCard("image","关键素材",number(overview.key_event_count),"")}${statusCard("activity","状态变化",number(overview.physical_change_count),"")}</section>
  <section class="panel"><header class="panel-heading"><div><h2>实验时间线与步骤理解</h2><p>时间均为统一实验相对时间；每项保留证据事件与跨视角素材链接。</p></div></header>${dailyReportTimelineVisual(timeline,actions)}<div class="daily-timeline">${timeline.map((group,index)=>`<article><span class="daily-index">${String(index+1).padStart(2,"0")}</span><div><header><h3>${esc(group.experiment_name)}</h3><span class="badge">${group.continuity_type === "continuous" ? "连续实验" : "独立实验"}</span></header><p class="timecode">${esc(group.start_timecode)} → ${esc(group.end_timecode)}</p><p>${esc(group.overall_summary || "无模型摘要")}</p><details><summary>查看 ${number(group.steps?.length)} 个细粒度步骤</summary><div class="step-list">${(group.steps||[]).map((step)=>`<article class="step-card"><span class="step-number">步骤 ${esc(step.step_index)}<small>${esc(step.start_timecode)}<br/>${esc(step.end_timecode)}</small></span><div class="step-body"><strong>当前：${esc(step.current_step || "未说明")}</strong><p class="step-next"><b>下一步：</b>${esc(step.next_step || "证据不足")}</p><span class="step-meta">对象：${esc((step.objects||[]).join("、") || "未明确")} · ${number(step.supporting_views?.length)} 路证据</span></div></article>`).join("")}</div></details></div></article>`).join("")}</div></section>
  <section class="panel"><header class="panel-heading"><div><h2>报告状态</h2><p>用于判断当前报告是否适合继续复核或导出。</p></div></header><table class="metric-table report-summary-table"><tbody><tr><td>多视角同步</td><td>${number(alignment.aligned)}/${number(alignment.view_count)} 路画面已对齐</td></tr><tr><td>需要留意</td><td>${number(report.uncertainties?.length)} 组不确定项，${number(report.contradictions?.length)} 项视角差异</td></tr><tr><td>人工复核</td><td>${esc(report.human_review?.status === "approved" ? "已完成" : "待复核")}</td></tr></tbody></table></section>`;
}

function professionalReportsView(data) {
  const report = data.daily_report || {};
  const quality = data.quality_acceptance || {};
  const reportFiles = [
    { key: "daily_report_pdf", format: "PDF", title: "专业实验报告", copy: "适合审阅、打印和对外分享的正式版报告。" },
    { key: "daily_report_json", format: "JSON", title: "结构化实验报告", copy: "包含实验摘要、步骤、关键素材和复核状态。" },
  ].filter((item)=>data.links?.[item.key]);
  const extraFiles = [
    ["daily_report_html", "网页版报告"],
    ["daily_report_markdown", "Markdown 报告"],
  ].filter(([key])=>data.links?.[key]);
  const status = quality.passed === true ? "报告已生成" : report.report_id ? "报告已生成 · 等待复核" : "报告生成中";
  return `<section class="professional-report-hero"><div><span class="report-hero-icon">${icon("file")}</span><p class="eyebrow">专业报告</p><h2>导出完整实验成果</h2><p>PDF 用于阅读与分享，JSON 用于结构化存档和后续系统对接。</p></div><span class="badge">${status}</span></section><section class="report-file-grid">${reportFiles.map((item)=>`<a class="report-file-card" target="_blank" href="${esc(data.links[item.key])}"><span class="file-format">${item.format}</span><div><strong>${item.title}</strong><p>${item.copy}</p><small>${icon("arrow")}打开报告</small></div></a>`).join("") || productState("progress", "clock", "专业报告正在生成", "分析完成后，PDF 与 JSON 报告会显示在这里。", `<a class="secondary-button" href="#/tasks">查看分析进度</a>`)}</section>${extraFiles.length ? `<section class="panel report-extra-files"><header class="panel-heading"><div><h2>其他阅读格式</h2><p>按需要选择网页版或可编辑文本格式。</p></div></header><div class="result-links">${extraFiles.map(([key,label])=>`<a target="_blank" href="${esc(data.links[key])}">${icon("file")}${label}</a>`).join("")}</div></section>` : ""}<details class="advanced-report-details"><summary><span>${icon("gauge")}</span><div><strong>技术附录与校验信息</strong><small>耗时、模型用量、质量检查和资源参数</small></div>${icon("chevron")}</summary><div class="advanced-report-content">${metricsView(data)}</div></details>`;
}

function keyMaterialEvaluationSummary(data) {
  const quality = data.quality_acceptance || {};
  const materials = quality.key_materials || {};
  const recallEval = data.key_material_recall_eval || {};
  const recallAtHalf = (recallEval.threshold_results || []).find(
    (item)=>Number(item.temporal_iou_threshold) === 0.5
  );
  const precisionInterval = recallAtHalf?.precision_confidence_interval;
  const recallInterval = recallAtHalf?.recall_confidence_interval;
  const intervalLabel = precisionInterval && recallInterval
    ? "；95%区间 P " + percent(precisionInterval.lower) + "–"
      + percent(precisionInterval.upper) + "，R "
      + percent(recallInterval.lower) + "–" + percent(recallInterval.upper)
    : "";
  const recallLabel = recallEval.evaluated && recallAtHalf
    ? percent(recallAtHalf.precision) + " / " + percent(recallAtHalf.recall)
      + "（tIoU 0.5，对象约束；标注覆盖 "
      + percent(recallEval.annotation_coverage?.coverage_ratio) + intervalLabel
      + (recallEval.small_sample_warning ? "；小样本" : "") + "）"
    : "未评估（没有适用于本数据集的逐事件人工真值）";
  const unobservedActions = (materials.unobserved_action_types || []).map(
    (item)=>ACTION_LABELS[item] || item
  );
  const missingActions = (materials.missing_action_types || []).map(
    (item)=>ACTION_LABELS[item] || item
  );
  const categoryCoverageLabel = materials.category_coverage_is_acceptance_gate
    ? (missingActions.length
      ? "未通过：缺少 " + missingActions.join("、")
      : "固定基准要求的动作类别均已覆盖")
    : (unobservedActions.length
      ? "自然未观察到 " + unobservedActions.join("、") + "；不作为失败"
      : "本实验观察到全部动作类别");
  return {
    recallLabel,
    categoryCoverageLabel,
    displayNote: (
      quality.display_note
      || "边界基线只参与评估，不参与推理；没有基线时不猜测准确率。"
    ) + " 关键素材：" + recallLabel + "；动作覆盖：" + categoryCoverageLabel + "。",
  };
}

function metricsView(data) {
  const metrics = data.metrics || {};
  const tokens = metrics.tokens || {};
  const stages = metrics.stage_durations || [];
  const webIngest = metrics.web_ingest || metrics.nas_index_ingest;
  const endToEnd = metrics.web_end_to_end || metrics.fixed_benchmark_end_to_end;
  const retention = webIngest?.retention_mode === "nas_only" ? "上传并保存至 NAS" : "上传并按存储配置留存";
  const extraRows = `${webIngest?.duration_seconds != null ? `<tr><td>input_ingest</td><td>${metrics.web_ingest ? retention : "NAS 索引输入准备/复用"}</td><td>${duration(webIngest.duration_seconds)}</td></tr>` : ""}${endToEnd?.total_duration_seconds != null ? `<tr><td>end_to_end</td><td>请求进入至最终 NAS 归档完成</td><td>${duration(endToEnd.total_duration_seconds)}</td></tr>` : ""}`;
  const quality = data.quality_acceptance || {};
  const boundary = quality.experiment_boundaries || {};
  const materials = quality.key_materials || {};
  const materialEvaluation = keyMaterialEvaluationSummary(data);
  const verification = data.key_material_verification || {};
  const summaries = data.observability?.telemetry_summary?.stage_summaries || {};
  const performance = metrics.preprocessing_display || {};
  const fullColdStart = performance.full_cold_start || {};
  const currentRun = performance.current_run || {};
  const statusLabel = quality.status === "evidence_package_passed_no_boundary_ground_truth"
    ? "证据包结构与媒体已验收；没有人工边界真值"
    : quality.status || "未生成质量账本";
  const boundaryAccuracy = boundary.evaluated
    ? `${percent(boundary.precision)} / ${percent(boundary.recall)}`
    : "未评估（历史档案无人工边界基线）";
  const boundaryContinuity = boundary.evaluated
    ? `${percent(boundary.boundary_pass_rate)} / ${percent(boundary.continuity_accuracy)}`
    : "未评估；不使用结构验收冒充准确率";
  const performanceSummary = fullColdStart.measured
    ? `历史完整冷启动预处理 ${duration(fullColdStart.seconds)}；本次归档运行 ${duration(currentRun.total_seconds)}，其中预处理 ${duration(currentRun.preprocessing_seconds)}。`
    : `本次运行预处理 ${duration(currentRun.preprocessing_seconds ?? metrics.display_preprocessing_seconds)}；流水线 ${duration(currentRun.total_seconds ?? metrics.total_duration_seconds)}。`;
  const reuseNote = currentRun.reused_validated_cv_ledgers
    ? "本次归档运行复用了已验收 CV 检测账本，因此不能当作冷启动速度。"
    : "本次运行未标记为复用已验收 CV 检测账本。";
  const links = {
    experiment_understanding: "实验级步骤理解 JSON",
    key_material_understanding: "关键素材模型理解 JSON",
    key_material_category_index: "关键素材实验/五类目录索引 JSON",
    metrics: "耗时与 Token JSON",
    acceptance: "验收汇总 JSON",
    quality_acceptance: "自动质量验收 JSON",
    evidence_package_eval: "证据包结构/媒体验收 JSON",
    key_material_recall_eval: "关键素材 Precision / Recall 评估 JSON",
    final_key_material_annotation: "关键素材对象框与二次复核 JSON",
  };
  const verificationSection = `<section class="panel"><header class="panel-heading"><div><h2>关键素材模型质量账本</h2><p>逐事件显示实际执行的视觉模型、对象框置信度、二次复核耗时和明确保留的不确定性；没有人工真值时不展示伪造的准确率。</p></div></header><table class="metric-table"><tbody><tr><td>复核账本</td><td>${verification.available ? "可用" : "历史档案未生成"}</td></tr><tr><td>事件 / 渲染视角</td><td>${number(verification.event_count)} / ${number(verification.rendered_view_count)}</td></tr><tr><td>仅渲染动作参与对象</td><td>${number(materials.participant_only_annotation_pass_count)} / ${number(materials.event_count)}（${materials.participant_only_annotation_gate_passed ? "门禁通过" : "未通过或无账本"}）</td></tr><tr><td>动作参与对象可见性</td><td>${number(materials.action_participant_visibility_pass_count)} / ${number(materials.event_count)}（${materials.action_participant_visibility_gate_passed ? "门禁通过" : "未通过或无账本"}）</td></tr><tr><td>决策状态</td><td>${esc(Object.entries(verification.decision_status_counts || {}).map(([key,value])=>`${key}: ${value}`).join("；") || "无")}</td></tr><tr><td>实际模型执行</td><td>${esc(Object.entries(verification.model_execution_counts || {}).map(([key,value])=>`${MODEL_LABELS[key] || key}: ${value}`).join("；") || "无")}</td></tr><tr><td>二次复核推理耗时</td><td>YOLO-World ${duration(verification.timing?.open_vocabulary_inference_seconds)}；Grounding DINO ${duration(verification.timing?.grounding_dino_inference_seconds)}；总墙钟 ${duration(verification.timing?.wall_seconds)}</td></tr><tr><td>保留不确定性的事件</td><td>${number(verification.uncertain_event_count)}</td></tr><tr><td>预算使用</td><td>${number(verification.budget?.admitted_event_count)} 事件 / ${number(verification.budget?.admitted_view_count)} 视角；延后 ${number(verification.budget?.deferred_count)}</td></tr></tbody></table></section>`;
  return `<section class="panel"><header class="panel-heading"><div><h2>耗时口径</h2><p>${performanceSummary} ${reuseNote}</p></div></header><div class="performance-compare"><article><small>历史完整冷启动预处理</small><strong>${duration(fullColdStart.seconds)}</strong><span>${esc(fullColdStart.includes || "预检 + 对齐 + 全量粗扫 + 有界精扫 + 边界审计；不含模型理解")}</span></article><article><small>当前归档运行总耗时</small><strong>${duration(currentRun.total_seconds ?? metrics.total_duration_seconds)}</strong><span>${currentRun.reused_validated_cv_ledgers ? "复用已验收 CV 账本，重新生成理解/媒体/证据包" : "以本次运行账本为准"}</span></article><article><small>当前运行预处理</small><strong>${duration(currentRun.preprocessing_seconds)}</strong><span>${currentRun.reused_validated_cv_ledgers ? "不是冷启动基准" : "当前运行实际值"}</span></article></div><table class="metric-table"><thead><tr><th>阶段</th><th>说明</th><th>耗时</th></tr></thead><tbody>${extraRows}${stages.map((stage)=>`<tr><td>${esc(stage.stage)}</td><td>${esc(STAGE_LABELS[stage.stage] || stage.stage)}</td><td>${duration(stage.duration_seconds)}</td></tr>`).join("")}</tbody></table></section><section class="panel"><header class="panel-heading"><div><h2>Token 用量</h2><p>CV、FFmpeg 与 TensorRT 不消耗模型 Token；断点复用的模型结果不重复计入本次实际消耗。</p></div></header><table class="metric-table"><thead><tr><th>阶段</th><th>执行 / 复用</th><th>输入 Token</th><th>输出 Token</th><th>总 Token</th></tr></thead><tbody><tr><td>实验片段步骤理解</td><td>${number(tokens.experiment_groups?.executed_call_count ?? tokens.experiment_groups?.call_count)} / ${number(tokens.experiment_groups?.reused_call_count)}</td><td>${number(tokens.experiment_groups?.input_tokens)}</td><td>${number(tokens.experiment_groups?.output_tokens)}</td><td>${number(tokens.experiment_groups?.total_tokens)}</td></tr><tr><td>关键素材理解</td><td>${number(tokens.key_materials?.executed_call_count ?? tokens.key_materials?.call_count)} / ${number(tokens.key_materials?.reused_call_count)}</td><td>${number(tokens.key_materials?.input_tokens)}</td><td>${number(tokens.key_materials?.output_tokens)}</td><td>${number(tokens.key_materials?.total_tokens)}</td></tr><tr><td><strong>全任务</strong></td><td>—</td><td><strong>${number(tokens.run_total?.input_tokens)}</strong></td><td><strong>${number(tokens.run_total?.output_tokens)}</strong></td><td><strong>${number(tokens.run_total?.total_tokens)}</strong></td></tr></tbody></table></section><section class="panel"><header class="panel-heading"><div><h2>质量验收</h2><p>${esc(materialEvaluation.displayNote)}</p></div></header><table class="metric-table"><tbody><tr><td>总体状态</td><td>${esc(statusLabel)}</td></tr><tr><td>实验检出 Precision / Recall</td><td>${boundaryAccuracy}</td></tr><tr><td>关键素材 Precision / Recall</td><td>${esc(materialEvaluation.recallLabel)}</td></tr><tr><td>边界通过率 / 连续性准确率</td><td>${boundaryContinuity}</td></tr><tr><td>证据包结构与媒体</td><td>${boundary.evidence_package_eval_passed || materials.evidence_package_eval_passed ? "通过自动验收" : "未通过或无验收记录"}</td></tr><tr><td>关键素材类别覆盖</td><td>${esc(materialEvaluation.categoryCoverageLabel)}</td></tr><tr><td>第一/第三人称成套素材</td><td>${number(materials.dual_view_material_count)} / ${number(materials.event_count)}（${percent(materials.dual_view_material_rate)}）</td></tr><tr><td>双侧共同佐证动作</td><td>${number(materials.cross_view_supported_count)} / ${number(materials.event_count)}（${percent(materials.cross_view_supported_rate)}）</td></tr><tr><td>双视角关联可审计</td><td>${number(materials.cross_view_or_explicit_uncertainty_count)} / ${number(materials.event_count)}</td></tr></tbody></table></section>${verificationSection}<section class="panel"><header class="panel-heading"><div><h2>资源遥测（按阶段）</h2><p>来源：resource_telemetry.json；主机网络包含其他流量，进程树 I/O 单独列出。</p></div></header><table class="metric-table"><thead><tr><th>阶段</th><th>GPU mean/max</th><th>NVDEC mean/max</th><th>CPU mean/max</th><th>网络接收 mean/max</th></tr></thead><tbody>${Object.entries(summaries).map(([stage,item])=>`<tr><td>${esc(STAGE_LABELS[stage]||stage)}</td><td>${metricStat(item.gpu_compute_percent)}/${metricStat(item.gpu_compute_percent,"max","%")}</td><td>${metricStat(item.nvdec_percent)}/${metricStat(item.nvdec_percent,"max","%")}</td><td>${metricStat(item.cpu_percent)}/${metricStat(item.cpu_percent,"max","%")}</td><td>${metricStat(item.host_network_receive_mib_s,"mean"," MiB/s")}/${metricStat(item.host_network_receive_mib_s,"max"," MiB/s")}</td></tr>`).join("")}</tbody></table></section><section class="panel"><header class="panel-heading"><div><h2>模型理解与验收文件</h2><p>这些链接直接指向当前${archiveShortLabel()}档案中的正式文件；历史档案没有的文件不会显示为可点击链接。</p></div></header><div class="result-links" style="padding:20px">${Object.entries(links).map(([key,label])=>data.links[key]?`<a target="_blank" href="${esc(data.links[key])}">${icon("file")}${label}</a>`:"").join("")}</div></section>`;
}

async function renderArchive(name, tab = "experiments", staging = false) {
  const requestedHash = location.hash;
  setChrome("archive", productExperimentName(name));
  main.innerHTML = pageSkeleton(`正在读取${archiveLabel()}`);
  try {
    const data = await loadArchive(name, staging);
    if (location.hash !== requestedHash) return;
    const stage = data.observability?.status?.stage;
    const pending = staging || (stage && stage !== "completed");
    const understandingPending = pending && !(data.observability?.stage_receipts || []).some((item) => item.stage === "semantic_refinement" && item.status === "completed");
    const notice = pending ? `<p class="analysis-readiness-note" role="status">${["failed", "interrupted"].includes(stage) ? "处理已停止，已完成的阶段产出保留供查看。" : "分析进行中，已完成的阶段产出可查看，后续结果将继续更新。"} <a href="#/tasks">查看进度</a></p>` : "";
    main.innerHTML = `<div class="page">${notice}${resultHeader(data,tab)}${tab === "materials" ? materialsView(data) : tab === "reports" ? dailyReportView(data) : tab === "metrics" ? professionalReportsView(data) : `<section class="step-list experiment-list">${data.experiments.map((item) => experimentCard(item, understandingPending)).join("") || productState("progress", "clock", "实验片段正在生成", "分析完成后会在这里显示同步视频与步骤理解。", `<a class="secondary-button" href="#/tasks">查看分析进度</a>`)}</section>`}</div>`;
    if (staging) {
      document.querySelectorAll(".result-tab").forEach((link) => {
        link.hash = `/stage/${encodeURIComponent(name)}/${link.hash.split("/").at(-1)}`;
      });
      document.querySelectorAll("[data-open-folder]").forEach((button)=>button.remove());
    }
    bindArchiveActions(data);
    if (tab === "materials") bindMaterialFilters(data);
    requestAnimationFrame(updateResultNavDensity);
  } catch (error) {
    if (location.hash !== requestedHash) return;
    main.innerHTML = productState("error", "x", "无法读取该实验档案", error.message, `<button class="primary-button" type="button" data-retry-archive>重新加载</button><a class="secondary-button" href="#/experiments">返回实验记录</a>`);
    document.querySelector("[data-retry-archive]")?.addEventListener("click", ()=>renderArchive(name,tab,staging));
  }
}

function annotationDecisionOptions(selected = "") {
  return [
    ["", "请选择判断"],
    ["confirmed_false_negative", "确认漏检"],
    ["false_positive", "确认误检"],
    ["class_error", "类别错误"],
    ["correct_detection", "检测正确"],
    ["not_actionable", "无需标注"],
    ["needs_review", "需要复审"],
  ].map(([value,label])=>`<option value="${value}" ${selected === value ? "selected" : ""}>${label}</option>`).join("");
}

function annotationCard(item) {
  const decision = item.decision || {};
  const xyxy = decision.xyxy || [];
  return `<article class="annotation-card">
    <div class="annotation-image"><img loading="lazy" src="/api/annotation-workspace/items/${encodeURIComponent(item.item_id)}/image" alt="${esc(item.event_id)} ${esc(item.class_name)} badcase"/>${item.image_available ? "" : `<span>图片不可用</span>`}<b>${esc(item.priority)}</b></div>
    <div class="annotation-copy"><header><div><small>${esc(item.event_id)} · ${esc(item.role)}</small><h3>${esc(item.class_name || "未指定类别")}</h3></div><span class="badge ${item.effective_review_status === "reviewed" ? "trusted" : "uncertain"}">${item.effective_review_status === "reviewed" ? "已审核" : "待审核"}</span></header><p>${esc(item.action_type)} · ${esc(item.issue_type)}</p>
      <form class="annotation-form" data-annotation-form="${esc(item.item_id)}"><label>人工判断<select name="decision" required>${annotationDecisionOptions(decision.decision || "")}</select></label><label>正确类别<input name="corrected_class" value="${esc(decision.corrected_class || item.class_name || "")}" placeholder="例如 weighing_paper"/></label><label>人工框 xyxy<input name="xyxy" value="${esc(xyxy.join(", "))}" placeholder="x1, y1, x2, y2"/></label><label>审核人<input name="reviewer" value="${esc(decision.reviewer || "")}" placeholder="姓名或工号" required/></label><label class="annotation-notes">备注<textarea name="notes" rows="2" placeholder="遮挡、可见性或判断依据">${esc(decision.notes || "")}</textarea></label><button class="primary-button" type="submit">${icon("check")}保存审核</button></form>
    </div>
  </article>`;
}

async function renderAnnotations() {
  setChrome("annotations");
  const filters = state.annotationFilters;
  const query = new URLSearchParams({ limit: "48" });
  if (filters.priority) query.set("priority", filters.priority);
  if (filters.reviewStatus) query.set("review_status", filters.reviewStatus);
  if (filters.query) query.set("q", filters.query);
  main.innerHTML = pageSkeleton("正在读取标注队列");
  try {
    const data = await api(`/api/annotation-workspace?${query}`);
    const summary = data.summary || {};
    main.innerHTML = `<div class="page annotation-page"><header class="page-hero compact"><div><p class="eyebrow">MODEL QUALITY WORKBENCH</p><h1>YOLO badcase 标注工作台</h1><p>把模型漏检、误检与类别混淆转成可复核训练数据。CV 推理结果与人工真值严格分离，只有人工实际提交的框才进入导出。</p></div><div class="hero-actions"><a class="secondary-button" target="_blank" href="/api/annotation-workspace/export">${icon("file")}导出已审核真值</a></div></header>
      <section class="status-grid">${statusCard("target","badcase 总数",number(summary.total),"来自已冻结审计队列")}${statusCard("check","已审核",number(summary.reviewed),"决策已保存，可追溯")}${statusCard("clock","待审核",number(summary.pending),"按 P0/P1/P2 排序处理")}${statusCard("file","P0 高优先级",number(summary.priority_counts?.P0),"确认漏检和高风险问题")}</section>
      <section class="panel"><header class="panel-heading"><div><h2>审核队列</h2><p>${esc(data.queue_path || "尚未发现 YOLO-Annotation-Queue.json")}</p></div></header><div class="annotation-toolbar"><label>优先级<select id="annotation-priority"><option value="">全部</option><option value="P0" ${filters.priority==="P0"?"selected":""}>P0</option><option value="P1" ${filters.priority==="P1"?"selected":""}>P1</option><option value="P2" ${filters.priority==="P2"?"selected":""}>P2</option></select></label><label>状态<select id="annotation-status"><option value="">全部</option><option value="pending" ${filters.reviewStatus==="pending"?"selected":""}>待审核</option><option value="reviewed" ${filters.reviewStatus==="reviewed"?"selected":""}>已审核</option></select></label><label class="annotation-query">搜索<input id="annotation-query" value="${esc(filters.query)}" placeholder="事件、类别、动作或问题"/></label><button class="secondary-button" id="annotation-filter" type="button">${icon("search")}筛选</button></div>${data.available ? `<div class="annotation-grid">${data.items.map(annotationCard).join("")}</div>` : `<div class="empty-state"><strong>未找到 badcase 队列</strong><p>先运行 YOLO badcase 审计，或在配置中指定 yolo_annotation_queue_path。</p></div>`}</section></div>`;
    bindAnnotationActions();
  } catch (error) {
    main.innerHTML = `<div class="empty-state"><strong>无法读取标注工作台</strong><p>${esc(error.message)}</p></div>`;
  }
}

function bindAnnotationActions() {
  document.querySelector("#annotation-filter")?.addEventListener("click", () => {
    state.annotationFilters = { priority: document.querySelector("#annotation-priority").value, reviewStatus: document.querySelector("#annotation-status").value, query: document.querySelector("#annotation-query").value.trim() };
    renderAnnotations();
  });
  document.querySelectorAll("[data-annotation-form]").forEach((form)=>form.addEventListener("submit", async (event)=>{
    event.preventDefault();
    const values = new FormData(form);
    const rawBox = String(values.get("xyxy") || "").trim();
    const xyxy = rawBox ? rawBox.split(",").map((value)=>Number(value.trim())) : null;
    if (xyxy && (xyxy.length !== 4 || xyxy.some((value)=>!Number.isFinite(value)))) { toast("人工框需要填写 4 个逗号分隔数字。", "error"); return; }
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    try {
      await api(`/api/annotation-workspace/items/${encodeURIComponent(form.dataset.annotationForm)}/decision`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision: values.get("decision"), corrected_class: values.get("corrected_class"), xyxy, reviewer: values.get("reviewer"), notes: values.get("notes") }) });
      toast("人工审核已保存到可追溯账本。");
      await renderAnnotations();
    } catch (error) { toast(error.message, "error"); button.disabled = false; }
  }));
}

function bindArchiveActions(data) {
  document.querySelectorAll("[data-copy-path]").forEach((button) => button.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(button.dataset.copyPath); toast("归档路径已复制。" ); }
    catch { toast(`请手工复制：${button.dataset.copyPath}`, "error"); }
  }));
  document.querySelectorAll("[data-open-folder]").forEach((button) => button.addEventListener("click", async () => {
    try { const result = await api(`/api/archives/${encodeURIComponent(button.dataset.openFolder)}/open`, { method: "POST" }); toast(`已打开：${result.path}`); }
    catch (error) { toast(error.message, "error"); }
  }));
  const dialog = document.querySelector("#experiment-meta-dialog");
  document.querySelector("[data-edit-experiment]")?.addEventListener("click", ()=>dialog?.showModal());
  document.querySelectorAll("[data-close-experiment-meta]").forEach((button)=>button.addEventListener("click", ()=>dialog?.close()));
  document.querySelector("#experiment-meta-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const values = new FormData(event.currentTarget);
    const tags = String(values.get("tags") || "").split(/[、,，]/).map((tag)=>tag.trim()).filter(Boolean).slice(0,8);
    try {
      saveExperimentMetadata(data.name, {
        displayName: String(values.get("display_name") || "").trim(),
        owner: String(values.get("owner") || "").trim(),
        tags,
        note: String(values.get("note") || "").trim(),
      });
      dialog?.close();
      toast("实验信息已保存到当前工作区，正式证据档案未改变。");
      renderArchive(data.name, routeParts()[2] || "experiments", routeParts()[0] === "stage");
    } catch { toast("当前浏览器无法保存工作区信息。", "error"); }
  });
}

function updateResultNavDensity() {
  const shell = document.querySelector(".result-nav-shell");
  const hero = document.querySelector(".experiment-hero");
  if (!shell || !hero) return;
  const compact = hero.getBoundingClientRect().bottom < 88 && window.innerWidth > 560;
  shell.classList.toggle("is-compact", compact);
  if (compact) {
    const content = document.querySelector(".product-content")?.getBoundingClientRect();
    if (content) {
      shell.style.left = `${Math.max(0, content.left)}px`;
      shell.style.right = `${Math.max(0, window.innerWidth - content.right)}px`;
    }
  } else {
    shell.style.removeProperty("left");
    shell.style.removeProperty("right");
  }
}

function handleMaterialReviewShortcut(event) {
  if (routeParts()[2] !== "materials" || event.metaKey || event.ctrlKey || event.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement?.tagName)) return;
  const video = document.querySelector(".material-clip-details[open] video[data-material-video]");
  if (!video) return;
  const key = event.key.toLocaleLowerCase();
  if (key === " ") {
    event.preventDefault();
    if (video.paused) video.play().catch(()=>{}); else video.pause();
  } else if (key === "arrowleft" || key === "arrowright") {
    event.preventDefault();
    const offset = key === "arrowleft" ? -5 : 5;
    video.currentTime = Math.max(0, Math.min(Number.isFinite(video.duration) ? video.duration : Infinity, video.currentTime + offset));
  } else if (key === "p" || key === "n") {
    event.preventDefault();
    activateAdjacentMaterial(video.dataset.materialVideo, key === "p" ? "previous" : "next");
  }
}

function routeParts() {
  return location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
}

async function router() {
  const parts = routeParts();
  const route = parts[0] || "home";
  if (route === "archive" && parts[1]) return renderArchive(parts[1], parts[2] || "experiments");
  if (route === "stage" && parts[1]) return renderArchive(parts[1], parts[2] || "experiments", true);
  if (route === "new") return renderNew();
  if (route === "tasks") return renderTasks();
  if (route === "operations") return renderOperations();
  if (route === "annotations") return renderAnnotations();
  if (["experiments","materials","reports"].includes(route)) return renderExperiments(route);
  renderHome();
}

document.querySelector("#nav-toggle").addEventListener("click", () => {
  const shell = document.querySelector("#vc-shell");
  const collapsed = shell.dataset.collapsed !== "true";
  shell.dataset.collapsed = String(collapsed);
  document.querySelector("#nav-toggle").setAttribute("aria-label", collapsed ? "展开全局导航" : "收起全局导航");
});
document.querySelector("#global-search").addEventListener("input", (event) => {
  state.search = event.target.value;
  const route = routeParts()[0] || "home";
  if (["home","experiments","materials","reports"].includes(route)) router();
});
document.querySelector("#refresh-button").addEventListener("click", async () => {
  state.archiveCache.clear();
  await loadAll();
  await router();
  toast("实验档案与任务状态已刷新。" );
});
const mobileMoreDialog = document.querySelector("#mobile-more-dialog");
document.querySelector("#mobile-more-button")?.addEventListener("click", ()=>mobileMoreDialog?.showModal());
document.querySelectorAll("[data-close-mobile-more]").forEach((button)=>button.addEventListener("click", ()=>mobileMoreDialog?.close()));
mobileMoreDialog?.querySelectorAll("a").forEach((link)=>link.addEventListener("click", ()=>mobileMoreDialog.close()));
window.addEventListener("hashchange", router);
window.addEventListener("scroll", updateResultNavDensity, { passive: true });
window.addEventListener("resize", updateResultNavDensity, { passive: true });
window.addEventListener("keydown", handleMaterialReviewShortcut);
window.setInterval(refreshTaskSnapshots, 4000);

hydrateIcons();
const legacyArchive = new URLSearchParams(location.search).get("archive");
if (legacyArchive && !location.hash) location.hash = `#/archive/${encodeURIComponent(legacyArchive)}/experiments`;
loadAll().then(router).catch((error) => {
  main.innerHTML = productState("error", "server", "无法连接分析服务", error.message, `<button class="primary-button" type="button" data-retry-service>重新连接</button>`);
  document.querySelector("[data-retry-service]")?.addEventListener("click", ()=>location.reload());
});

let refreshingNasRecordings = false;
window.setInterval(async () => {
  if (refreshingNasRecordings || routeParts()[0] !== "new" || state.health?.collection_ingest?.mode !== "directory_metadata") return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  refreshingNasRecordings = true;
  try {
    const payload = await api("/api/nas-recordings");
    state.nasRecordings = payload.recordings || [];
    state.nasBatches = payload.batches || [];
    state.nasMonitor = payload.monitor || null;
    state.nasError = payload.truncated ? "素材较多，当前显示部分结果" : payload.errors?.length ? "部分素材说明无法读取" : "";
    const batchPicker = document.querySelector("#nas-batches");
    if (batchPicker) { batchPicker.outerHTML = nasBatchPicker(); bindNasBatchPicker(); }
    const picker = document.querySelector("#nas-recordings");
    if (picker) { picker.outerHTML = nasRecordingPicker(); bindNasRecordingPicker(); }
  } catch (error) {
    state.nasError = error.message;
  } finally { refreshingNasRecordings = false; }
}, 30000);
