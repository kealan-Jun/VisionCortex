const brandLogo = window.VisionCortexBrandLogoDataUrl;
if (brandLogo) {
  const logo = document.querySelector("#product-logo");
  const favicon = document.querySelector("#product-favicon");
  if (logo) logo.src = brandLogo;
  if (favicon) favicon.href = brandLogo;
}

function mllmConnectionLabel(health) {
  if (!(health.mllm_key_configured ?? health.ark_key_configured)) return "待配置密钥";
  if (!health.mllm_enabled) return "密钥已配置 · 待启用";
  return health.mllm_connection?.status === "verified" ? "连接已验证" : "已配置 · 调用待验证";
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
  const rounded = Math.round(value);
  if (rounded >= 3600) return `${Math.floor(rounded / 3600)}时${Math.floor(rounded % 3600 / 60)}分${rounded % 60}秒`;
  if (rounded >= 60) return `${Math.floor(rounded / 60)}分${String(rounded % 60).padStart(2,"0")}秒`;
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
  try {
    const raw = localStorage.getItem(EXPERIMENT_META_KEY) || "{}";
    if (state.experimentMetadataCache?.raw === raw) return state.experimentMetadataCache.records;
    const parsed = JSON.parse(raw);
    const records = !parsed || typeof parsed !== "object" || Array.isArray(parsed) ? {} : Object.fromEntries(Object.entries(parsed).filter(([,item])=>item && typeof item === "object" && !Array.isArray(item)).map(([name,item])=>[name, {
      ...item,
      displayName: typeof item.displayName === "string" ? item.displayName : "",
      owner: typeof item.owner === "string" ? item.owner : "",
      tags: Array.isArray(item.tags) ? item.tags.filter((tag)=>typeof tag === "string") : [],
      note: typeof item.note === "string" ? item.note : "",
    }]));
    state.experimentMetadataCache = {raw, records};
    return records;
  }
  catch { return {}; }
}

function experimentMetadata(name) {
  const records = loadExperimentMetadata();
  return Object.hasOwn(records, String(name || "")) ? records[String(name || "")] : {};
}

function saveExperimentMetadata(name, metadata) {
  const records = { ...loadExperimentMetadata() };
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
  reagent_bottle_open: "开口试剂瓶",
  sample_bottle: "样品瓶",
  sample_bottle_blue: "蓝盖样品瓶",
  bottle: "试剂瓶",
  bottle_cap: "瓶盖",
  tube: "离心管",
  tube_cap: "管盖",
  tube_rack: "离心管架",
  paper: "称量纸",
  weighing_paper: "称量纸",
  pipette: "移液器",
  spearhead: "移液枪头",
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
  capacity_reservation: "检查可用空间",
  source_transfer: "传入实验视频",
  input_seal: "核对输入文件",
  reserving: "准备实验空间",
  original_ingest: "保存实验视频",
  nas_ingest: "读取实验视频",
  input_preflight: "检查实验输入",
  queued: "排队等待",
  running: "正在启动",
  preflight: "检查视频",
  alignment: "同步不同视角",
  speech: "实验录音转写",
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
  quality_attention: "保存阶段成果与缺口",
  daily_report: "生成实验报告",
  finalizing: "封存溯源并正式发布",
  completed: "分析完成",
  partial: "分析结束 · 证据不足",
  failed: "分析失败",
  interrupted: "服务重启后待续跑",
};

const GUIDED_PIPELINE = [
  {
    id: "originals", number: "01", title: "导入实验视频",
    stages: ["reserving", "original_ingest", "nas_ingest", "input_preflight"], completedBy: ["original_ingest"],
    folder: "Original-Experiment-Videos",
    outputs: [["Original-Video-Index.json","原始输入索引"],["nas_ingest.json","NAS 输入清单"]],
    doing: "检查实验视频与时间记录，并保存到本次实验中。",
    outcome: "完整保存实验视频、时间记录与拍摄视角。",
  },
  {
    id: "alignment", number: "02", title: "同步不同视角",
    stages: ["preflight", "alignment", "speech"], completedBy: ["alignment"],
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
    stages: ["daily_report", "finalizing", "completed"], completedBy: ["daily_report", "finalizing", "completed"],
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
  const effectiveTone = tone || "success";
  item.className = `toast ${effectiveTone}`;
  item.setAttribute("role", effectiveTone === "error" ? "alert" : "status");
  item.innerHTML = `<span aria-hidden="true">${icon(effectiveTone === "error" ? "x" : effectiveTone === "loading" ? "refresh" : "check")}</span><p>${esc(message)}</p>`;
  document.querySelector("#toast-region").append(item);
  setTimeout(() => { item.classList.add("is-leaving"); setTimeout(()=>item.remove(),180); }, 4200);
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
  if (route === "tasks") return ["实验工作台", "任务进度"];
  if (route === "materials") return ["实验成果", "关键素材库"];
  if (route === "reports") return ["实验成果", "实验室日报"];
  if (route === "operations") return ["系统", "运行状态"];
  if (route === "ai-settings") return ["系统", "AI 服务设置"];
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
  chip.className = `phase-chip ${stage === "partial" ? "partial" : stage === "completed" ? "completed" : ["failed", "interrupted"].includes(stage) ? "failed" : run ? "running" : ""}`;
  chip.querySelector("span").textContent = run?.parent_run_id ? (stage === "completed" ? "阶段结果已更新" : ["failed","interrupted"].includes(stage) ? "阶段更新未完成" : "正在处理所选阶段") : run ? (STAGE_LABELS[stage] || run.message || stage) : "暂无进行中的任务";
  const follow = document.querySelector("#follow-stage-results");
  if (follow) {
    follow.hidden = !run || ["completed","partial","failed","interrupted"].includes(stage);
    follow.textContent = state.followRun?.enabled ? "自动跟随中 · 暂停" : "跟随分析进度";
    follow.ariaPressed = String(Boolean(state.followRun?.enabled));
    follow.title = "环节产出保存后自动打开对应页面；手动切换页面会暂停跟随。";
  }
}

function updateServiceChrome() {
  document.querySelectorAll("[data-ai-settings-link]").forEach(link => { link.hidden = !state.health?.ai_settings_enabled; });
  const online = state.health?.status === "ok";
  document.querySelector("#node-dot").classList.toggle("online", online);
  const running = state.runs.filter((run) => !["completed", "partial", "failed", "interrupted"].includes(run.state)).length;
  document.querySelector("#service-note").textContent = online
    ? state.health?.run_purpose === "workflow_simulation" ? "测试环境 · 服务正常" : running ? `${running} 个任务正在分析` : "服务运行正常"
    : "分析服务暂不可用";
  const badge = document.querySelector("#running-badge");
  badge.hidden = running === 0;
  badge.textContent = running;
  setPhase(state.activeRun || state.runs.find((run) => !["completed", "partial", "failed", "interrupted"].includes(run.state)) || null);
}

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

async function loadNasRecordings() {
  const requestId = state.nasRequestId = (state.nasRequestId || 0) + 1;
  state.nasLoading = true;
  try {
    const payload = await api("/api/nas-recordings");
    if (requestId !== state.nasRequestId) return null;
    state.nasRecordings = payload.recordings || [];
    state.nasBatches = payload.batches || [];
    state.nasMonitor = payload.monitor || null;
    state.nasSyncError = false;
    state.nasError = payload.truncated ? "素材较多，当前显示部分结果" : payload.errors?.length ? "部分素材说明无法读取" : "";
  } catch (error) {
    if (requestId !== state.nasRequestId) return null;
    // Keep the last list and selection readable during a temporary connection failure.
    state.nasSyncError = true;
    state.nasError = error.message || "无法读取已采集素材";
  } finally {
    if (requestId === state.nasRequestId) {
      state.nasLoading = false;
      state.nasRenderPending = true;
    }
  }
  return requestId;
}

async function loadAll() {
  const runRequestId = state.runPollRequestId || 0;
  const nasRequest = loadNasRecordings();
  const archiveQuery = archiveSearchQuery();
  const results = await Promise.allSettled([api("/api/health"), loadArchiveListing(archiveQuery), api("/api/runs"), api("/api/collections?limit=200")]);
  if (results[0].status === "fulfilled") state.health = results[0].value;
  if (runRequestId === (state.runPollRequestId || 0)) {
    state.taskSyncError = results[2].status === "rejected";
    if (!state.taskSyncError) state.runs = results[2].value.runs || [];
  }
  if (results[3].status === "fulfilled") state.collections = results[3].value.collections || [];
  updateServiceChrome();
  void nasRequest.then((requestId) => {
    if (requestId !== state.nasRequestId) return;
    if (routeParts()[0] === "new") refreshNasPickers();
  });
  // Each page decides whether it needs the directory; direct archive links remain readable.
  return results[1].status === "fulfilled" && results[1].value === true;
}

async function refreshTaskSnapshots() {
  if (state.refreshingTasks || document.hidden) return;
  state.refreshingTasks = true;
  const runRequestId = state.runPollRequestId || 0;
  try {
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
    updateServiceChrome();
    if (state.followRun) followStageResults(state.runs.find(run=>run.run_id === state.followRun.runId));
    const route = routeParts()[0] || "home";
    if (route === "tasks") renderTasks();
    const current = JSON.stringify(experimentRecords().map((record)=>[libraryRecordKey(record), record.material_revision, record.key_event_count]));
    refreshLibraryOverview(archivesUpdated || current !== previous);
  } catch {
    if (runRequestId !== (state.runPollRequestId || 0)) return;
    state.taskSyncError = true;
    if (routeParts()[0] === "tasks") renderTasks();
    refreshLibraryOverview();
  } finally {
    state.refreshingTasks = false;
  }
}

function refreshLibraryOverview(recordsChanged = false) {
  if (state.archiveView?.refreshPending) void refreshOpenArchive();
  const syncState = JSON.stringify([Boolean(state.taskSyncError), Boolean(state.archiveRefreshPending)]);
  const syncChanged = syncState !== (state.librarySyncState || "[false,false]");
  state.librarySyncState = syncState;
  // Search spans all archives, independently of the current page's directory query.
  if (state.globalMaterialSearch && (recordsChanged || state.globalMaterialSearch.key !== globalMaterialSearchKey(state.search))) {
    const panel = document.querySelector("#global-search-results");
    if (panel && !panel.hidden) renderGlobalSearchResults(state.search);
  }
  if (!recordsChanged && !syncChanged) return;
  if (recordsChanged) state.libraryLoadErrors.clear();
  const route = routeParts()[0] || "home";
  if (["home", "experiments", "materials", "reports"].includes(route) && state.archiveListingQuery !== archiveSearchQuery()) return;
  if (route === "materials") renderMaterialsLibrary(true);
  if (route === "reports") renderReportsLibrary(true);
  if (route === "home") renderHome();
  if (route === "experiments") { syncArchiveFiltersFromRoute(); renderExperiments(); }
}

function archiveSyncNotice() {
  if (state.taskSyncError) return '<div class="freshness-warning" role="alert" aria-label="成果同步状态">任务状态同步中断，当前列表与数量可能尚未包含最新成果；连接恢复后会自动同步。</div>';
  if (state.archiveRefreshPending) return '<div class="freshness-warning" role="status" aria-label="成果同步状态">实验档案目录仍在同步，当前列表与数量尚未确认包含最新成果；系统会自动重试。</div>';
  return "";
}

function archiveProductStatus(archive) {
  if (archive.pipeline_stage === "partial") return { key: "attention", label: "分析结束 · 证据不足", tone: "neutral" };
  if (archive.staging_run_id && ["failed", "interrupted"].includes(archive.pipeline_stage)) return { key: "attention", label: "阶段产出 · 待补全", tone: "danger" };
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

function routeQuery() {
  const raw = location.hash.replace(/^#\/?/, "");
  const marker = raw.indexOf("?");
  return new URLSearchParams(marker >= 0 ? raw.slice(marker + 1) : "");
}

function archiveLibraryHash(target = "experiments") {
  if (target !== "experiments") return `#/${target}`;
  const filters = state.archiveFilters;
  const query = new URLSearchParams();
  if (filters.status !== "all") query.set("status", filters.status);
  if (filters.date !== "all") query.set("date", filters.date);
  if (filters.owner !== "all") query.set("owner", filters.owner);
  if (filters.tag !== "all") query.set("tag", filters.tag);
  if (filters.view !== "list") query.set("view", filters.view);
  const suffix = query.toString();
  return `#/${target}${suffix ? `?${suffix}` : ""}`;
}

function syncArchiveFiltersFromRoute() {
  const query = routeQuery();
  const parts = routeParts();
  const validStatus = new Set(["all", "completed", "processing", "attention", "archived"]);
  const validDate = new Set(["all", "today", "7d", "30d"]);
  const validView = new Set(["list", "cards"]);
  const status = parts[1] === "attention" ? "attention" : query.get("status") || "all";
  state.archiveFilters = {
    ...state.archiveFilters,
    status: validStatus.has(status) ? status : "all",
    date: validDate.has(query.get("date")) ? query.get("date") : "all",
    owner: query.get("owner") || "all",
    tag: query.get("tag") || "all",
    view: validView.has(query.get("view")) ? query.get("view") : "list",
  };
}

function updateArchiveFilterRoute() {
  const next = archiveLibraryHash("experiments");
  if (location.hash === next) void router();
  else location.hash = next;
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

function eventObjectValues(event) {
  return Object.values(event?.objects || event?.cv_objects || {}).flatMap((value) => Array.isArray(value) ? value : [value]).filter(Boolean);
}

function archiveDurationLabel(detail) {
  const current = detail?.metrics?.preprocessing_display?.current_run || {};
  return duration(current.total_seconds ?? detail?.metrics?.total_duration_seconds);
}

function refreshProgressiveSurface() {
  const route = routeParts()[0] || "home";
  if (route === "materials") renderMaterialsLibrary(true);
  if (route === "reports") renderReportsLibrary(true);
  if (!document.querySelector("#global-search-results")?.hidden) renderGlobalSearchResults(state.search);
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

function archiveSearchQuery() {
  if (!["home", "experiments"].includes(routeParts()[0] || "home")) return "";
  const query = state.search.trim();
  const normalized = query.toLocaleLowerCase();
  // Names and tags saved in this browser are not part of the server's archive-name index.
  const matchesLocalMetadata = query && Object.values(loadExperimentMetadata()).some((metadata)=>
    [metadata.displayName, metadata.owner, ...(metadata.tags || [])].join(" ").toLocaleLowerCase().includes(normalized));
  return matchesLocalMetadata ? "" : query;
}

function experimentRecords() {
  const retained = state.runs.filter((run) => run.state !== "completed" && run.nas_staging
    && run.result_available !== false && run.observability?.stage_receipts?.length
    && !state.archives.some((archive) => archive.path === run.nas_staging)).map((run) => ({
      name: run.experiment_id || run.run_id,
      staging_run_id: run.run_id,
      pipeline_stage: run.state,
      modified_at: run.observability.status?.updated_at || run.updated_at,
      experiment_count: run.observability.retained_experiment_count || 0,
      has_experiment_videos: run.observability.stage_receipts.some((receipt) =>
        receipt.stage === "experiment_clips" && receipt.status === "completed"
        && receipt.artifacts?.some((artifact) => artifact.available
          && artifact.relative_path === "Experiment-Clips")),
      key_event_count: 0,
      candidate_material_count: Number(run.observability.partial_delivery?.quarantined_event_count || 0),
      material_revision: JSON.stringify([run.state, run.observability.partial_delivery?.created_at,
        run.observability.stage_receipts]),
      has_daily_report: false,
      formal_accuracy_claim_allowed: false,
    }));
  return [...state.archives, ...retained].sort((left, right) =>
    new Date(right.modified_at || 0) - new Date(left.modified_at || 0));
}

function experimentRecordRoute(record, target = "experiments") {
  return record.staging_run_id ? `#/stage/${encodeURIComponent(record.staging_run_id)}/${target}`
    : `#/archive/${encodeURIComponent(record.name)}/${target}`;
}

function filteredArchives(applyAdvanced = false) {
  const query = state.search.trim().toLocaleLowerCase();
  const filters = state.archiveFilters;
  return experimentRecords().filter((archive) => {
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

function bindArchivePagination(target = "experiments") {
  if (!state.archiveNextCursor) return;
  document.querySelector("#main-content .page")?.insertAdjacentHTML("beforeend", `<div class="pagination-actions"><button class="secondary-button" id="load-more-archives" type="button">加载更多实验档案（已载入 ${number(state.archives.length)} / ${number(state.archiveTotal)}）</button></div>`);
  document.querySelector("#load-more-archives")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const requestedHash = location.hash;
    const requestId = state.archiveRequestId;
    const isCurrent = () => location.hash === requestedHash && state.archiveRequestId === requestId;
    button.disabled = true;
    try { await loadMoreArchives(); if (isCurrent()) renderExperiments(target); }
    catch (error) { if (isCurrent()) toast(error.message, "error"); }
    finally { button.disabled = false; }
  });
}

function statusCard(iconName, label, value, note) {
  return `<article class="status-card"><span>${icon(iconName)}</span><div><small>${esc(label)}</small><strong>${esc(value)}</strong>${note ? `<p>${esc(note)}</p>` : ""}</div></article>`;
}

function archiveRows(archives, target = "experiments", view = "list", hasActiveFilters = Boolean(state.search.trim())) {
  if (!archives.length) {
    const pending = Boolean(state.archiveNextCursor);
    const title = pending ? "已载入的记录中暂无匹配项" : hasActiveFilters ? "没有符合条件的实验" : target === "materials" ? "关键素材将在这里汇集" : target === "reports" ? "实验报告将在这里汇集" : "从第一个实验开始";
    const copy = pending ? "仍有档案未载入，当前空结果不代表没有匹配实验。请继续加载后查找。" : hasActiveFilters ? "可以调整筛选条件，或清空筛选查看全部记录。" : "导入实验视频，集中查看片段、关键素材与分析结果。";
    return productState("neutral", hasActiveFilters || pending ? "search" : target === "reports" ? "file" : "folder", title, copy, hasActiveFilters ? `<button class="secondary-button" type="button" data-clear-archive-filters>清空筛选</button>` : pending ? `<a class="secondary-button" href="#/experiments">查看实验记录</a>` : `<a class="secondary-button" href="#/new">${icon("plus")}创建实验</a>`);
  }
  return `<div class="archive-list ${view === "cards" ? "archive-card-grid" : ""}">${archives.map((archive) => {
    const metadata = experimentMetadata(archive.name);
    const status = archiveProductStatus(archive);
    const videosAvailable = target === "experiments" && Number(archive.experiment_count) > 0
      && (archive.has_experiment_videos || !archive.staging_run_id);
    return `
    <a class="archive-row ${view === "cards" ? "archive-row-card" : ""}" href="${experimentRecordRoute(archive, target)}">
      <span class="archive-leading-icon ${status.tone}">${icon(status.tone === "danger" ? "x" : status.tone === "progress" ? "activity" : "check")}</span>
      <span class="archive-identity" title="档案编号：${esc(archive.name)}"><strong>${esc(productExperimentName(archive.name))}</strong><small>${formatDate(archive.modified_at)}</small>${videosAvailable ? `<small class="archive-video-availability">${number(archive.experiment_count)} 组视频片段可查看</small>` : ""}${metadata.owner || metadata.tags?.length ? `<em>${metadata.owner ? esc(metadata.owner) : ""}${metadata.owner && metadata.tags?.length ? " · " : ""}${(metadata.tags || []).slice(0,2).map(esc).join(" · ")}</em>` : ""}</span>
      <span class="archive-cell"><small>实验片段</small><strong>${number(archive.experiment_count)} 个</strong></span>
      <span class="archive-cell"><small>正式素材 / 候选素材</small><strong>${number(archive.key_event_count)} / ${number(archive.candidate_material_count)} 个</strong></span>
      <span class="archive-status ${status.tone}">${esc(status.label)}</span>
      <span class="row-link">${videosAvailable ? "查看视频" : "查看结果"} ${icon("arrow")}</span>
    </a>`;
  }).join("")}</div>`;
}

function renderHome() {
  setChrome("home");
  const running = state.runs.filter((run) => !["completed", "partial", "failed", "interrupted"].includes(run.state));
  const todayCount = experimentRecords().filter((archive)=>archiveWithinDate(archive,"today")).length;
  const reportReady = state.archives.filter((archive)=>archive.has_daily_report).length;
  const attention = experimentRecords().filter((archive)=>archiveProductStatus(archive).key === "attention");
  const attentionTarget = attention.length === 1 ? experimentRecordRoute(attention[0]) : attention.length ? "#/experiments?status=attention" : "#/experiments";
  const health = state.health || {};
  main.innerHTML = `<div class="page workspace-page">
    <header class="page-hero workbench-hero"><div><p class="eyebrow">实验工作台</p><h1>实验运行概览</h1><p>${running.length ? `${running.length} 个实验正在分析，完成后将自动生成关键素材和报告。` : attention.length ? `${attention.length} 个实验需要关注，其余档案可正常查看。` : "集中查看实验状态、分析进度与最近生成的成果。"}</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header>${archiveSyncNotice()}
    <section class="home-focus-grid" aria-label="待办概览">
      <a href="#/experiments"><span>${icon("folder")}</span><small>今日新增</small><strong>${number(todayCount)}</strong><em>查看实验记录 ${icon("arrow")}</em></a>
      <a href="#/tasks" class="${running.length ? "active" : ""}"><span>${icon("activity")}</span><small>正在分析</small><strong>${number(running.length)}</strong><em>${running.length ? "查看任务进度" : "当前没有运行任务"} ${icon("arrow")}</em></a>
      <a href="#/reports"><span>${icon("file")}</span><small>有报告可查看</small><strong>${number(reportReady)}</strong><em>进入实验室日报 ${icon("arrow")}</em></a>
      <a href="${attentionTarget}" class="${attention.length ? "attention" : ""}"><span>${icon(attention.length ? "x" : "check")}</span><small>需要关注</small><strong>${number(attention.length)}</strong><em>${attention.length === 1 ? "直接打开异常实验" : attention.length ? "查看异常实验列表" : "当前状态正常"} ${icon("arrow")}</em></a>
    </section>
    <section class="home-launchpad" aria-labelledby="home-launchpad-title" aria-describedby="home-launchpad-description">
      <span class="home-launchpad-icon">${icon("upload")}</span>
      <div><p class="eyebrow">快速开始</p><h2 id="home-launchpad-title">把多视角视频拖到这里</h2><p id="home-launchpad-description">也可以选择视频文件；随后确认视角与实验信息，再创建分析任务。</p></div>
      <div class="home-launchpad-flow" aria-label="分析流程"><span><b>1</b>选择视频</span><span><b>2</b>确认视角</span><span><b>3</b>创建任务</span></div>
      <label class="primary-button batch-import-button home-upload-button">${icon("upload")}选择视频<input id="home-upload-input" aria-label="从首页选择视频文件" type="file" multiple accept="video/*,.mp4,.mov,.m4v,.mkv,.avi,.webm,.csv,text/csv" /></label>
    </section>
    ${running.length ? `<section class="current-work"><span class="current-icon">${icon("activity")}</span><div><small>正在分析</small><h2>${esc(productExperimentName(running[0].experiment_id))}</h2><p>${esc(STAGE_LABELS[running[0].state] || "处理中")}</p></div><a class="secondary-button" href="#/tasks">查看任务进度 ${icon("arrow")}</a></section>` : ""}
    <div class="workspace-grid">
      <section class="panel recent-panel"><header class="panel-heading"><div><h2>最近完成与更新</h2><p>从上次离开的地方继续</p></div><a href="#/experiments">全部记录 ${icon("arrow")}</a></header>${archiveRows(filteredArchives().slice(0, 6))}</section>
      <aside class="workspace-aside">
        <section class="next-action-panel"><p class="eyebrow">下一步</p><h2>${attention.length ? "检查需要关注的实验" : running.length ? "等待分析完成" : state.archives.length ? "继续查看最近成果" : "创建第一个实验"}</h2><p>${attention.length ? "失败或中断的实验会保留已完成内容，可先查看现有结果。" : "工作台会把最需要处理的内容放在这里。"}</p><a class="secondary-button" href="${attention.length ? attentionTarget : running.length ? "#/tasks" : state.archives.length ? `#/archive/${encodeURIComponent(state.archives[0].name)}/experiments` : "#/new"}">${attention.length === 1 ? "打开异常实验" : attention.length ? "查看异常实验列表" : running.length ? "查看任务进度" : state.archives.length ? "打开最近实验" : "新建实验"} ${icon("arrow")}</a></section>
        <section class="workspace-status"><header><h2>工作区状态</h2><a href="#/operations" aria-label="查看服务状态">${icon("arrow")}</a></header><div><span>档案存储</span><strong class="${health.archive_available ? "status-ok" : "status-pending"}">${health.archive_available ? "可用" : "待连接"}</strong></div><div><span>智能理解</span><strong class="${health.mllm_enabled && health.ark_key_configured ? "status-ok" : "status-pending"}">${mllmConnectionLabel(health)}</strong></div></section>
      </aside>
    </div>
    <nav class="workspace-shortcuts" aria-label="实验成果入口"><a href="#/materials"><span>${icon("boxes")}</span><div><strong>关键素材库</strong></div>${icon("arrow")}</a><a href="#/reports"><span>${icon("file")}</span><div><strong>实验室日报</strong></div>${icon("arrow")}</a></nav>
  </div>`;
  bindHomeLaunchpad();
  bindArchiveFilters();
}

function bindHomeLaunchpad() {
  const launchpad = document.querySelector(".home-launchpad");
  const input = document.querySelector("#home-upload-input");
  if (!launchpad || !input) return;
  const importFromHome = (fileList) => {
    const files = [...(fileList || [])];
    const hasVideo = files.some((file) => /\.(mp4|mov|m4v|mkv|avi|webm)$/i.test(file.name) || file.type.startsWith("video/"));
    if (!hasVideo) {
      toast("没有识别到视频文件，请选择 MP4、MOV、MKV、AVI 或 WebM。", "error");
      return;
    }
    location.hash = "#/new";
    importBatch(files);
  };
  input.addEventListener("change", (event) => importFromHome(event.target.files));
  launchpad.addEventListener("dragenter", (event) => {
    event.preventDefault();
    launchpad.classList.add("is-dragging");
  });
  launchpad.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    launchpad.classList.add("is-dragging");
  });
  launchpad.addEventListener("dragleave", (event) => {
    if (!launchpad.contains(event.relatedTarget)) launchpad.classList.remove("is-dragging");
  });
  launchpad.addEventListener("drop", (event) => {
    event.preventDefault();
    launchpad.classList.remove("is-dragging");
    importFromHome(event.dataTransfer.files);
  });
}

async function rerunBenchmark() {
  const button = document.querySelector("#rerun-benchmark");
  if (button?.disabled) return;
  if (button) {
    button.disabled = true;
    button.textContent = "正在准备基准验证…";
  }
  try {
    const run = await api("/api/benchmarks/six-view-three-hour/runs", { method: "POST" });
    showAcceptedRun(run);
    toast("基准任务已创建，可在任务进度中查看。");
  } catch (error) {
    toast(error.message, "error");
    if (button) {
      button.disabled = false;
      button.innerHTML = `${icon("activity")}运行基准验证`;
    }
  }
}

function renderExperiments(target = "experiments") {
  if (target === "materials") return renderMaterialsLibrary();
  if (target === "reports") return renderReportsLibrary();
  setChrome("experiments");
  const filters = state.archiveFilters;
  const metadata = experimentRecords().map((archive)=>experimentMetadata(archive.name));
  const owners = [...new Set([...metadata.map((item)=>item.owner), filters.owner !== "all" ? filters.owner : ""].filter(Boolean))].sort();
  const tags = [...new Set([...metadata.flatMap((item)=>item.tags || []), filters.tag !== "all" ? filters.tag : ""].filter(Boolean))].sort();
  const filtered = filteredArchives(true);
  const activeFilterCount = [filters.status,filters.date,filters.owner,filters.tag].filter((value)=>value !== "all").length + (state.search.trim() ? 1 : 0);
  const quickFilter = filters.status === "all" && filters.date === "all" ? "all" : filters.status === "completed" && filters.date === "all" ? "completed" : filters.status === "attention" && filters.date === "all" ? "attention" : filters.status === "all" && filters.date === "today" ? "today" : "advanced";
  const advancedOpen = [filters.owner,filters.tag].some((value)=>value !== "all") || ["7d","30d"].includes(filters.date) || ["processing","archived"].includes(filters.status);
  const pendingMaterials = isNasMode() ? `<a class="pending-materials-strip" href="#/new"><span>${icon("plus")}</span><div><strong>有新素材？开始一次实验分析</strong><small>导入多视角视频后，系统会自动整理步骤、素材与报告。</small></div>${icon("arrow")}</a>` : "";
  main.innerHTML = `<div class="page"><header class="page-hero compact library-hero"><div><p class="eyebrow">实验记录</p><h1>实验记录</h1><p>集中查看实验视频、分析过程与可追溯结果。</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header>${archiveSyncNotice()}${pendingMaterials}<section class="panel experiment-records"><header class="panel-heading"><div><h2>${activeFilterCount ? "筛选结果" : state.archiveNextCursor ? "已载入记录" : "全部记录"}<span class="count-pill">${number(filtered.length)}</span></h2><p>${activeFilterCount ? `已应用 ${activeFilterCount} 项条件` : "按更新时间排序"}${state.archiveNextCursor ? ` · 当前仅匹配已载入的记录（正式档案 ${number(state.archives.length)} / ${number(state.archiveTotal)}），加载更多可继续查找` : ""}</p></div><div class="archive-view-toggle" aria-label="记录视图"><button type="button" data-archive-view="list" class="${filters.view === "list" ? "active" : ""}">列表</button><button type="button" data-archive-view="cards" class="${filters.view === "cards" ? "active" : ""}">卡片</button></div></header><nav class="archive-quick-filters" aria-label="快速筛选"><button type="button" data-archive-quick="all" class="${quickFilter === "all" ? "active" : ""}">全部</button><button type="button" data-archive-quick="today" class="${quickFilter === "today" ? "active" : ""}">今天</button><button type="button" data-archive-quick="completed" class="${quickFilter === "completed" ? "active" : ""}">已完成</button><button type="button" data-archive-quick="attention" class="${quickFilter === "attention" ? "active" : ""}">需要关注</button></nav><details class="archive-advanced-filters" ${advancedOpen || quickFilter === "advanced" ? "open" : ""}><summary>更多筛选${advancedOpen ? " · 已应用" : ""}</summary><div class="archive-filter-toolbar"><label><span>更新时间</span><select data-archive-filter="date"><option value="all">全部时间</option><option value="today" ${filters.date === "today" ? "selected" : ""}>今天</option><option value="7d" ${filters.date === "7d" ? "selected" : ""}>最近 7 天</option><option value="30d" ${filters.date === "30d" ? "selected" : ""}>最近 30 天</option></select></label><label><span>状态</span><select data-archive-filter="status"><option value="all">全部状态</option><option value="completed" ${filters.status === "completed" ? "selected" : ""}>已完成</option><option value="processing" ${filters.status === "processing" ? "selected" : ""}>处理中</option><option value="attention" ${filters.status === "attention" ? "selected" : ""}>需要关注</option><option value="archived" ${filters.status === "archived" ? "selected" : ""}>已归档</option></select></label><label><span>负责人</span><select data-archive-filter="owner"><option value="all">全部负责人</option>${owners.map((owner)=>`<option value="${esc(owner)}" ${filters.owner === owner ? "selected" : ""}>${esc(owner)}</option>`).join("")}</select></label><label><span>标签</span><select data-archive-filter="tag"><option value="all">全部标签</option>${tags.map((tag)=>`<option value="${esc(tag)}" ${filters.tag === tag ? "selected" : ""}>${esc(tag)}</option>`).join("")}</select></label><button class="archive-filter-clear" type="button" data-clear-archive-filters ${activeFilterCount ? "" : "disabled"}>清空</button></div></details>${archiveRows(filtered, "experiments", filters.view, activeFilterCount > 0)}</section></div>`;
  bindArchiveFilters();
  bindArchivePagination("experiments");
}

function bindArchiveFilters() {
  document.querySelectorAll("[data-archive-filter]").forEach((control)=>control.addEventListener("change", () => {
    state.archiveFilters[control.dataset.archiveFilter] = control.value;
    updateArchiveFilterRoute();
  }));
  document.querySelectorAll("[data-archive-view]").forEach((button)=>button.addEventListener("click", () => {
    state.archiveFilters.view = button.dataset.archiveView;
    updateArchiveFilterRoute();
    toast(`已切换为${button.dataset.archiveView === "cards" ? "卡片" : "列表"}视图。`, "success");
  }));
  document.querySelectorAll("[data-archive-quick]").forEach((button)=>button.addEventListener("click", () => {
    const value = button.dataset.archiveQuick;
    state.archiveFilters.status = value === "completed" || value === "attention" ? value : "all";
    state.archiveFilters.date = value === "today" ? "today" : "all";
    updateArchiveFilterRoute();
  }));
  document.querySelectorAll("[data-clear-archive-filters]").forEach((button)=>button.addEventListener("click", () => {
    state.search = "";
    const search = document.querySelector("#global-search");
    if (search) search.value = "";
    state.archiveFilters = { ...state.archiveFilters, status: "all", date: "all", owner: "all", tag: "all" };
    updateArchiveFilterRoute();
  }));
}

function libraryCardSkeleton(count = 6) {
  return `<div class="library-card-grid library-skeleton-grid" aria-busy="true">${Array.from({length: count},()=>`<article class="library-card-skeleton"><i></i><span></span><span></span><span></span></article>`).join("")}</div>`;
}

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
  return `<a class="global-material-card ${preliminary ? "is-preliminary" : ""}" href="${target}"><figure>${imageUrl ? `<img loading="lazy" src="${esc(imageUrl)}" alt="${esc(productExperimentName(archive.name))}的关键画面"/>` : `<span class="material-image-fallback">${icon("image")}<small>画面暂不可用</small></span>`}<em>${staged ? "尚未通过整体验收" : preliminary ? automaticReviewLabel(event) : dual ? "双视角印证" : "主要视角清晰"}</em></figure><div class="global-material-copy"><small>${esc(productExperimentName(archive.name))} · ${formatDate(archive.modified_at)}</small><h2>${esc(preliminary ? `${staged ? "阶段素材" : "候选"} · ${ACTION_LABELS[event.cv_action_type || event.action_type] || "动作待确认"}` : ACTION_LABELS[event.action_type] || "关键实验动作")}</h2><p>${esc(current)}</p><span>查看素材 ${icon("arrow")}</span></div></a>`;
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

function reportLibraryCard(archive, detail) {
  const report = detail.daily_report || {};
  const timeline = report.experiment_timeline || [];
  const summary = timeline[0]?.overall_summary || report.summary || (archive.staging_run_id ? "已保存的操作与证据已整理为阶段日报，完整质量检查尚未通过。" : "实验过程与关键发现已整理，可进入日报查看完整内容。");
  const concernCount = Number(report.uncertainties?.length || 0) + Number(report.contradictions?.length || 0);
  const formats = [["daily_report_pdf","PDF"],["daily_report_json","JSON"],["daily_report_html","HTML"],["daily_report_markdown","Markdown"],["partial_pdf","PDF"],["partial_json","JSON"],["partial_daily_report","HTML"]].filter(([key])=>detail.links?.[key]).map(([,label])=>label);
  return `<a class="report-library-card" href="${experimentRecordRoute(archive,"reports")}"><header><span class="report-cover-mark">${icon("file")}</span><div><small>${esc(report.report_date || formatDate(archive.modified_at))}</small><h2>${esc(productExperimentName(archive.name))}</h2></div><em class="${concernCount ? "has-concern" : ""}">${archive.staging_run_id ? "阶段版 · 待核对" : concernCount ? `${number(concernCount)} 项需留意` : "报告已生成"}</em></header><p>${esc(summary)}</p><footer><span>${formats.map((format)=>`<b>${format}</b>`).join("")}</span><strong>查看日报 ${icon("arrow")}</strong></footer></a>`;
}

function renderReportsLibrary(progressive = false) {
  setChrome("reports");
  void ensureLibraryDetails();
  const query = state.search.trim().toLocaleLowerCase("zh-CN");
  const status = libraryLoadState("reports");
  const loadedArchives = status.ready;
  const reportEntries = loadedArchives.map((archive)=>[archive,cachedLibraryDetail(archive)]).filter(([,detail])=>detail?.daily_report?.report_id || detail?.links?.partial_daily_report);
  const filtered = reportEntries.filter(([archive,detail])=>!query || [productExperimentName(archive.name),detail.daily_report?.report_date,detail.daily_report?.experiment_timeline?.[0]?.overall_summary].join(" ").toLocaleLowerCase("zh-CN").includes(query));
  const attention = status.records.filter(archive=>archive.staging_run_id || archiveProductStatus(archive).key==="attention");
  const unavailable = status.ready.filter(archive=>{ const detail=cachedLibraryDetail(archive); return !detail?.daily_report?.report_id && !detail?.links?.partial_daily_report; });
  const loading = status.pending.length > 0;
  const latest = reportEntries[0];
  main.innerHTML = `<div class="page library-page"><header class="page-hero compact library-hero"><div><p class="eyebrow">实验室日报</p><h1>实验室日报</h1><p>快速了解每次实验的过程、结论与需要留意的内容。</p></div>${latest ? `<div class="hero-actions"><a class="primary-button" href="${experimentRecordRoute(latest[0],"reports")}">${icon("file")}打开最新日报</a></div>` : ""}</header>${archiveSyncNotice()}<section class="library-summary-strip report-summary-strip"><span><small>已载入日报</small><strong>${number(reportEntries.length)}</strong></span><span><small>已载入 PDF</small><strong>${number(reportEntries.filter(([,detail])=>detail.links?.daily_report_pdf || detail.links?.partial_pdf).length)}</strong></span><span><small>需要关注</small><strong>${number(attention.length)}</strong></span><span><small>已载入实验</small><strong>${number(loadedArchives.length)} / ${number(status.records.length)}</strong></span></section>${libraryFailureNotice(status,"reports")}${unavailable.length ? `<a class="library-attention-strip" href="#/experiments?status=attention"><span>${icon("activity")}</span><div><strong>${number(unavailable.length)} 个实验尚未生成日报</strong><small>失败或中断的实验会保留已完成内容，不会显示为完整报告。</small></div>${icon("arrow")}</a>` : ""}<div class="library-results-heading" aria-live="polite"><div><strong>${query ? "搜索结果" : "最近日报"}</strong><span>${number(filtered.length)} 份</span></div>${loading ? `<small><i></i>正在继续载入报告摘要…</small>` : ""}</div>${filtered.length ? `<section class="report-library-grid">${filtered.map(([archive,detail])=>reportLibraryCard(archive,detail)).join("")}</section>` : loading ? libraryCardSkeleton(progressive ? 2 : 4) : status.failed.length ? "" : productState("neutral","file",query ? "没有找到匹配的实验日报" : "尚未生成实验日报",query ? "可以尝试其他实验名称或关键词。" : "完成实验分析后，日报会自动出现在这里。",query ? `<button class="secondary-button" type="button" data-clear-library-search>清空搜索</button>` : `<a class="primary-button" href="#/new">新建实验</a>`)}</div>`;
  document.querySelector("[data-clear-library-search]")?.addEventListener("click",()=>{ state.search=""; const input=document.querySelector("#global-search"); if(input) input.value=""; renderReportsLibrary(true); });
  bindLibraryRetry("reports");
  bindArchivePagination("reports");
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
    ${segments.length ? `<details class="speech-upload"><summary>附带录音（可选；启用转写后自动读取视频音轨）</summary>${segments.map((segment, ordinal)=>`<div class="source-config-grid"><label>第 ${ordinal + 1} 段 · ${esc(segment.video.name)}<input type="file" accept="audio/*,.opus,.wav,.mp3,.m4a,.ogg,.flac" data-audio="${esc(source.key)}" data-segment="${ordinal}" /><small>${segment.audio ? esc(segment.audio.name) : "未附带录音"}</small></label><label>录音开始相对视频的偏移（秒）<input type="number" step="0.001" data-audio-offset="${esc(source.key)}" data-segment="${ordinal}" value="${segment.audioOffsetSeconds ?? ""}" placeholder="未知留空；同时开始填 0" /><small>正数表示录音晚于视频开始。</small></label></div>`).join("")}</details>` : ""}
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
  return `<section class="form-section" id="nas-recordings"><div class="section-heading"><span>01</span><div><h2>选择已采集素材 <span class="count-pill">${items.length}</span></h2><p>选择同一次实验的不同视角，无需重复上传视频。</p></div></div>${state.nasError ? `<p class="nas-recording-issue" role="status">${esc(state.nasError)}</p>` : ""}<label class="collection-search">${icon("search")}<input id="nas-search" value="${esc(state.nasQuery)}" placeholder="按实验、文件夹或文件名查找" aria-label="搜索已采集素材"></label><div class="nas-recordings-list">${rows || `<div class="empty-state compact"><strong>${state.nasLoading ? "正在读取已采集素材…" : state.nasSyncError ? "暂时无法读取采集素材" : "还没有采集素材"}</strong></div>`}</div>${items.some((item) => item.requires_completion_confirmation) ? `<label class="nas-completion-confirmation"><input id="nas-complete-confirmed" type="checkbox" ${state.nasCompleteConfirmed ? "checked" : ""}>我确认所选视频已采集完成，不再写入</label>` : ""}<div class="nas-selection-actions"><span id="nas-selection-count">已选择 ${Object.keys(state.nasSelection).length} 个片段</span><button class="secondary-button" type="button" id="save-nas-selection">保存为实验批次</button></div></section>`;
}

function captureQualityCopy(reason) {
  const labels = {
    "rgb coverage below 98 percent": "视频时间覆盖率低于 98%",
    "rgb contains a gap over 500 ms": "视频存在超过 0.5 秒的帧间隔",
    "depth coverage below 98 percent": "深度数据时间覆盖率低于 98%",
    "depth contains a gap over 500 ms": "深度数据存在超过 0.5 秒的帧间隔",
    "RGB/depth duration drift exceeds 500 ms": "视频与深度数据的时长差超过 0.5 秒",
  };
  return String(reason || "").split(";").map(part=>labels[part.trim()] || part.trim()).join("；");
}

function nasBatchPicker() {
  const monitor = state.nasMonitor || {};
  const monitorWatching = !state.nasSyncError && monitor.status === "watching";
  const monitorCopy = state.nasSyncError ? "采集素材列表同步中断，当前保留上次结果；连接恢复后会自动更新。" : monitorWatching
    ? `持续监控中 · ${formatDate(monitor.observed_at)}`
    : monitor.status === "retrying"
    ? "NAS 连接中断，正在自动重试"
    : "正在启动采集监控";
  const rows = state.nasBatches.map((batch) => {
    const issues = batch.issues || [];
    const writing = issues.some(issue=>/正在写入|尚未写入|尚未关闭|等待采集完成|等待文件写入|尚未就绪/.test(issue));
    const status = batch.available ? "可分析" : writing ? "等待采集完成" : "需要处理";
    return `<article class="collection-card ${batch.available ? "ready" : "blocked"}">
      <header><div title="采集编号：${esc(batch.recording_session_id)}"><small>${formatDate(batch.recording_start_time)}</small><strong>多视角采集 · ${number(batch.camera_count)} 个视角</strong></div><span class="collection-status ${batch.available ? "ready" : "blocked"}">${status}</span></header>
      <div class="collection-metrics"><span><b>${number(batch.camera_count)}</b> 路相机</span><span><b>${number(batch.recording_count)}</b> 个视频</span><span><b>${duration(batch.duration_seconds)}</b></span><span><b>${formatBytes(batch.size_bytes)}</b></span></div>
      ${issues.length ? `<ul class="collection-blockers">${issues.map(issue=>`<li>${esc(issue)}</li>`).join("")}</ul>` : ""}
      ${!batch.available ? `<p class="batch-next-action">${issues.some(issue=>/视角/.test(issue)) ? '请在下方“选择视频”中指定第一／第三人称视角。' : ""}${issues.some(issue=>/不完整/.test(issue)) ? "采集质量未通过，请查看相机记录；等待分析不会修复缺失的数据。" : ""}</p>` : ""}
      ${(batch.issue_details || []).length ? `<details class="batch-capture-details"><summary>查看各相机原因</summary>${batch.issue_details.map(item=>`<section><strong>${esc(item.camera_key)}</strong><p>${esc((item.issues || []).join("；"))}</p>${item.capture_quality?.reason ? `<p>采集程序记录：${esc(captureQualityCopy(item.capture_quality.reason))}</p>` : ""}${item.capture_quality?.rgb_coverage_ratio != null ? `<p>采集程序报告的视频覆盖率：${(Number(item.capture_quality.rgb_coverage_ratio)*100).toFixed(1)}%${item.capture_quality.rgb_max_frame_gap_us != null ? ` · 最大帧间隔 ${(Number(item.capture_quality.rgb_max_frame_gap_us)/1000000).toFixed(2)} 秒` : ""}</p>` : ""}</section>`).join("")}</details>` : ""}
      <button class="primary-button" type="button" data-run-nas-batch="${esc(batch.batch_id)}" ${batch.available ? "" : "disabled"}>${icon("arrow")}分析此批次</button>
    </article>`;
  }).join("");
  return `<section class="form-section collection-picker" id="nas-batches"><div class="section-heading"><span>01</span><div><h2>选择采集批次</h2><p>优先直接分析原视频目录中的已有视频：建立输入与时间戳索引，无需重新上传或复制原片。</p><p class="nas-monitor-state ${monitorWatching ? "watching" : "retrying"}"><i></i>${esc(monitorCopy)}</p></div></div><div class="collection-card-list">${rows || `<div class="empty-state compact"><strong>${state.nasSyncError ? "暂时无法读取采集批次" : state.nasLoading ? "正在读取采集批次…" : "等待相机完成采集"}</strong></div>`}</div></section>`;
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
  if (processing === "partial") return { label: "分析结束 · 阶段成果已保存", tone: "attention", note: "证据缺口已列入阶段报告" };
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
  const nas = state.health?.collection_ingest?.enabled !== false && (isNasMode() || state.health?.collection_ingest?.mode === "directory_metadata");
  const directNas = nas && state.health?.collection_ingest?.mode === "directory_metadata";
  const modelEnabled = state.health?.mllm_enabled && state.health?.ark_key_configured;
  main.innerHTML = `<div class="page new-experiment-page">${state.health?.run_purpose === "workflow_simulation" ? `<p class="analysis-readiness-note">当前为测试环境，分析结果将保存到独立测试档案。</p>` : ""}
    <header class="page-hero compact"><div><p class="eyebrow">创建实验</p><h1>新建实验</h1><p>按实际机位数量添加视频，系统会自动整理片段、关键素材和报告。至少需要一路第一人称和一路第三人称。</p></div><a class="secondary-button" href="#/experiments">${icon("chevron")}返回实验记录</a></header>
    <nav class="setup-progress" aria-label="新建实验进度"><a href="#${directNas ? "nas-batches" : nas ? "collection-source" : "recorded-videos"}"><span>01</span><strong>导入视频</strong></a><a href="#experiment-info"><span>02</span><strong>命名实验</strong></a><a href="#analysis-method"><span>03</span><strong>确认分析方式</strong></a><a href="#review-start"><span>04</span><strong>开始处理</strong></a></nav>
    <div class="new-layout"><div class="new-primary">
      ${directNas ? nasBatchPicker() + `<details class="panel" ${state.nasBatches.length ? "" : "open"}><summary>按文件选择实验视角</summary>${nasRecordingPicker()}</details>` : ""}
      ${nas && (!directNas || state.collections.length) ? `<section class="form-section collection-picker" id="collection-source"><div class="section-heading"><span>01</span><div><h2>选择采集批次</h2><p>视频已在 NAS 时优先选择已有批次：建立输入与时间戳索引，直接读取原片分析，无需重新上传或复制。</p></div></div><label class="collection-search">${icon("search")}<input id="collection-search" value="${esc(state.collectionQuery)}" placeholder="按日期或批次编号查找" aria-label="搜索采集批次" /></label><div id="collection-card-results">${collectionCards()}</div>${selected ? `<div class="selected-collection-note"><span>${icon("check")}</span><div><strong>已选择 ${esc(selected.collection_id)}</strong><p>${number(selected.camera_count)} 路视频 · ${number(selected.video_segment_count)} 个片段</p></div><button class="secondary-button" id="clear-collection" type="button">改选批次</button></div>` : ""}</section>` : ""}
      <section class="form-section manual-upload-fallback ${selected ? "is-secondary" : ""}" id="recorded-videos"><div class="section-heading"><span>${nas ? icon("upload") : "01"}</span><div><h2>${nas ? "或上传实验文件" : "导入实验视频"}</h2>${nas ? "<p>视频位于已配置的原视频目录时，优先使用上方索引入口；其他文件可在这里上传。</p>" : ""}</div></div>
        <div class="batch-import-panel"><div><span class="upload-symbol">${icon("upload")}</span><strong>选择视频，开始整理实验</strong><p>支持 MP4、MOV、MKV、AVI、WebM 与时间戳 CSV</p></div><div class="batch-actions"><label class="primary-button batch-import-button">${icon("plus")}选择文件<input id="batch-input" aria-label="选择视频与时间戳文件" type="file" multiple accept="video/*,.mp4,.mov,.m4v,.mkv,.avi,.webm,.csv,text/csv" /></label><label class="secondary-button batch-import-button">${icon("folder")}选择文件夹<input id="folder-input" aria-label="选择实验文件夹" type="file" multiple webkitdirectory directory /></label></div></div>
        <div class="role-guidance ${guide.tone}">${esc(guide.text)}</div><div class="source-list">${state.sources.length ? state.sources.map(sourceCard).join("") : ""}</div><button class="secondary-button add-source" id="add-source" type="button">${icon("plus")}添加拍摄视角</button>
      </section>
      <section class="form-section" id="experiment-info"><div class="section-heading"><span>02</span><div><h2>为实验命名</h2><p>使用易于辨认的英文或编号，方便后续查找。</p></div></div><label class="field-label"><span>实验名称</span><input id="experiment-name" value="${esc(state.experimentName)}" maxlength="120" placeholder="例如：Sample-Preparation-01" autocomplete="off" /></label></section>
      <section class="form-section" id="analysis-method"><div class="section-heading"><span>03</span><div><h2>分析方式</h2></div></div><div class="analysis-method-overview"><article class="method-card"><span>${icon("video")}</span><div><small>视频处理</small><strong>片段整理与素材提取</strong><p>对齐不同视角，识别实验片段与关键动作。</p></div></article><article class="method-card"><span>${icon("brain")}</span><div><small>智能理解 · ${modelEnabled ? "已配置" : "待启用"}</small><strong>实验步骤解读</strong><p>${modelEnabled ? "结合实验素材生成步骤解读，并标注不确定项。" : "尚未启用步骤解读。可在服务状态中查看模型配置。"}</p></div></article></div><p class="analysis-readiness-note">${state.health?.speech_recognition?.enabled ? "录音转写已启用：自动关联配套录音，字幕和可搜索文字保存到本实验。" : "本服务尚未启用录音转写；附带录音会随实验输入保存。"}</p></section>
    </div><aside class="review-card" id="review-start"><div class="section-heading"><span>04</span><div><h2>准备开始</h2></div></div><div class="review-summary" id="review-summary"></div>${state.health?.analysis_ready === false ? `<p class="analysis-readiness-note" role="status">${esc(state.health.analysis_blocker)}。当前可以浏览素材、保存实验批次。</p>` : ""}<div class="upload-progress hidden" id="upload-progress"><div class="progress-copy"><span id="progress-message">准备任务</span><strong id="progress-percent">0%</strong></div><span class="progress-track"><i id="progress-bar" style="width:0%"></i></span><div class="run-stage-list" id="run-stages"></div><button class="secondary-button hidden" id="cancel-upload" type="button">${icon("x")}取消上传</button></div><button class="primary-button" id="start-run" type="button">${icon("arrow")}创建分析任务</button></aside></div>
  </div>`;
  state.nasRenderPending = false;
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
  button.disabled = state.health?.analysis_ready === false || !review.ready || Boolean(state.activeRun && !["completed","partial","failed","interrupted"].includes(state.activeRun.state));
}

function updateSource(key, change) {
  const source = state.sources.find((item) => item.key === key);
  if (!source) return;
  Object.assign(source, change);
}

function bindNewPage() {
  bindNasBatchPicker();
  bindNasRecordingPicker();
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
  document.querySelectorAll("[data-audio]").forEach(input => input.addEventListener("change", () => {
    const source = state.sources.find(item => item.key === input.dataset.audio);
    const segment = source?.segments[Number(input.dataset.segment)];
    if (segment) { segment.audio = input.files[0] || null; renderNew(); }
  }));
  document.querySelectorAll("[data-audio-offset]").forEach(input => input.addEventListener("input", () => {
    const source = state.sources.find(item => item.key === input.dataset.audioOffset);
    const segment = source?.segments[Number(input.dataset.segment)];
    if (segment) segment.audioOffsetSeconds = input.value === "" ? null : Number(input.value);
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
  let audioIndex = 0;
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
      if (segment.audio) {
        files.push({file_id: `audio-${audioIndex}`, kind: "audio", file_index: audioIndex,
          name: segment.audio.name, size: segment.audio.size, last_modified: segment.audio.lastModified,
          browser_file: segment.audio});
        mapping.audio_index = audioIndex++;
        mapping.audio_offset_ms = segment.audioOffsetSeconds == null ? null : segment.audioOffsetSeconds * 1000;
      }
      spec.segments.push(mapping);
    });
    if (spec.segments.length === 1) {
      const [single] = spec.segments;
      delete spec.segments;
      spec.video_index = single.video_index;
      if (single.csv_index !== undefined) spec.csv_index = single.csv_index;
      if (single.audio_index !== undefined) { spec.audio_index = single.audio_index; spec.audio_offset_ms = single.audio_offset_ms; }
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
  const stages = ["preflight","alignment","speech","motion_probe","candidate_coarse","candidate_fine","candidate_audit","experiment_understanding","experiment_clips","key_materials","mllm","material_refinement","semantic_refinement","package","daily_report","finalizing"];
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
  if (outcome?.status === "skipped") return {state:"skipped",label:"已跳过",symbol:"−",reason:outcome.reason};
  if (outcome?.status === "failed") return {state:"failed",label:"未完成",symbol:"!",reason:outcome.reason};
  if (["failed","interrupted"].includes(run.state) && status.failed_stage === stage) return {state:"failed",label:"已停止",symbol:"!"};
  if (activeStage === stage) return {state:"active",label:stage === "queued" ? "排队中" : "处理中",symbol:"↻"};
  if (outcome?.status === "completed") return {state:"done",label:"已完成",symbol:"✓"};
  if ((status.completed_stages || []).includes(stage)) return {state:"passed",label:"已通过",symbol:"✓"};
  return {state:"waiting",label:["completed","partial","failed","interrupted"].includes(run.state) ? "未执行" : "待处理",symbol:"·"};
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
      beginStageFollow(session.run_id);
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
    beginStageFollow(created.run_id);
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
    beginStageFollow(created.run_id);
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
    beginStageFollow(created.run_id);
    await pollRun(created.run_id, created.archive_name);
  } catch (error) {
    toast(error.message, "error");
    setProgress(1, `失败：${error.message}`);
    button.disabled = false;
  }
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

function beginStageFollow(runId) {
  state.followRun = {runId, enabled: true, seen: new Set(), expectedHash: null};
}

function followStageResults(run) {
  const follow = state.followRun;
  if (!run || !follow?.enabled || follow.runId !== run.run_id) return;
  const receipts = (run.observability?.stage_receipts || []).filter(item=>item.status === "completed")
    .sort((a,b)=>String(a.completed_at || "").localeCompare(String(b.completed_at || "")));
  const fresh = receipts.filter(item=>!follow.seen.has(`${item.stage}:${item.completed_at}`))
    .sort((a,b)=>String(a.completed_at || "").localeCompare(String(b.completed_at || "")));
  const finished = run.state === "completed" && !follow.finished;
  const latest = fresh.at(-1) || (finished ? receipts.at(-1) : null);
  if (!latest) return;
  if (state.archiveView && archiveViewBusy()) return;
  for (const item of receipts) follow.seen.add(`${item.stage}:${item.completed_at}`);
  if (finished) follow.finished = true;
  const tabs = {speech:"speech", experiment_understanding:"experiments", experiment_clips:"experiments", key_materials:"materials", mllm:"materials", material_refinement:"materials", semantic_refinement:"experiments", package:"metrics", daily_report:"reports", finalizing:"reports"};
  const target = tabs[latest.stage] ? stageResultRoute(run, tabs[latest.stage]) : "#/tasks";
  toast(`${STAGE_LABELS[latest.stage] || latest.stage}已完成，产出已保存。`, "");
  if (location.hash === target) { void router(); return; }
  follow.expectedHash = target;
  location.hash = target;
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
  setChrome("tasks");
  const terminal = new Set(["completed", "partial", "failed", "interrupted"]);
  const runs = [...state.runs].sort((a, b) => Number(terminal.has(a.state)) - Number(terminal.has(b.state)) || String(b.updated_at || b.observability?.status?.updated_at || "").localeCompare(String(a.updated_at || a.observability?.status?.updated_at || "")));
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">实验分析</p><h1>任务进度</h1><p>查看实验处理状态和当前所在环节。</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header>${state.taskSyncError ? `<div class="freshness-warning" role="alert">任务状态同步中断，当前显示最近收到的数据；连接恢复后会自动更新。</div>` : state.archiveRefreshPending ? `<div class="freshness-warning" role="status">任务状态已更新，实验档案目录仍在同步；系统会自动重试。</div>` : ""}<section class="status-grid">${statusCard("activity","全部任务",number(runs.length),"")}${statusCard("gauge","正在分析",number(runs.filter((run)=>!["completed","partial","failed","interrupted"].includes(run.state)).length),"")}${statusCard("check","已归档",number(runs.filter((run)=>run.state==="completed"&&!run.parent_run_id).length),"")}${statusCard("file","需要关注",number(runs.filter((run)=>["partial","failed","interrupted"].includes(run.state)).length),"")}</section>${runs.length ? runs.map((run)=>runObservabilityCard(run,false)).join("") : `<section class="panel"><div class="empty-state"><span class="empty-illustration" aria-hidden="true">${icon("activity")}</span><strong>暂无分析任务</strong><a class="secondary-button" href="#/new">${icon("plus")}新建实验</a></div></section>`}</div>`;
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
    dialog.innerHTML=`<header><div><p class="eyebrow">继续处理已保存的实验</p><h2>恢复与补全</h2></div><button class="dialog-close" type="button" data-recovery-close aria-label="关闭恢复方案">${icon("x")}</button></header><p>已保存 ${number(plan.retained_stage_count)} 个阶段回执。原任务模型：${esc(plan.provider||"未记录")} · ${esc(plan.model||"未记录")}。</p>${plan.model_ready?"":`<p class="recovery-model-notice">${esc(plan.model_check_message)} <a href="#/ai-settings" data-recovery-settings>前往 AI 服务设置</a></p>`}<div class="recovery-options"><button type="button" data-recovery-action="operations" ${plan.actions.operations?'':'disabled'}><strong>重新整理操作步骤</strong><span>使用已审核事件和保存画面，整理具体操作、减少重复记录。可能产生模型费用。</span></button><button type="button" data-recovery-action="reports" ${plan.actions.reports?'':'disabled'}><strong>更新已有结果的报告</strong><span>不调用模型；质量未通过时更新阶段报告。</span></button><button type="button" data-recovery-action="retry" ${plan.actions.retry?'':'disabled'}><strong>复跑完整流程</strong><span>使用原输入，重新检查并补全各环节；校验通过的缓存继续使用。</span></button></div><details><summary>复用范围、历史产出与费用说明</summary><p>${esc(plan.cache_policy)}</p><p>${esc(plan.retry_effect)}</p><p>${esc(plan.model_cost)}</p><p>${esc(plan.quality_policy)}</p><p class="recovery-path">产出路径：${esc(plan.output_path)}</p></details><p role="status" data-recovery-status>请选择本次需要处理的范围。</p>`;
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
  const url=scope==='retry'?`${base}/retry?revision=${encodeURIComponent(plan.revision)}`:`${base}/refresh/${scope}${scope==='operations'?`?revision=${encodeURIComponent(plan.group_revision)}`:''}`;
  const result=await api(url,{method:'POST'});
  state.archiveCache.clear();showAcceptedRun(result);
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

function guidedStageState(run, definition, receipts, index) {
  const status = run.observability?.status || {};
  const current = status.stage || run.state;
  if (run.state === "partial" && definition.stages.includes("package")) return { state: "partial", receipt: null };
  const failed = status.failed_stage || (run.state === "failed"
    ? run.observability?.metrics?.nas_index_ingest?.failure_stage || current : null);
  if (["failed","interrupted"].includes(run.state) && definition.stages.includes(failed)) return { state: run.state, receipt: null };
  const completedReceipt = definition.completedBy.map((stage)=>receipts.get(stage)).find(item=>item?.status === "completed");
  if (!["completed","partial","failed","interrupted"].includes(run.state) && definition.stages.includes(current)) return { state: "active", receipt: completedReceipt };
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
  const snapshot = run.observability || {};
  const receipts = new Map((snapshot.stage_receipts || []).map((item)=>[item.stage,item]));
  const rows = GUIDED_PIPELINE.map((definition,index)=>({ definition, ...guidedStageState(run,definition,receipts,index) }));
  const active = rows.find((item)=>["active","partial","failed","interrupted"].includes(item.state)) || rows.find((item)=>item.state === "waiting") || rows.at(-1);
  const activeIndex = rows.indexOf(active);
  const next = rows.slice(activeIndex+1).find((item)=>item.state === "waiting");
  const status = snapshot.status || {};
  const stateLabel = { active:"正在进行", done:"已完成", "done-unreceipted":"已通过", waiting:"等待中", partial:"证据不足", withheld:"未发布", failed:"本环节失败", interrupted:"等待续跑" };
  const cards = rows.map(({definition,state:stageState,receipt})=>{
    const deliveredReceipts = definition.stages.map((stage)=>receipts.get(stage)).filter(Boolean);
    const artifacts = deliveredReceipts.flatMap((item)=>item.artifacts || []).filter((item)=>item.available);
    const filesByName = new Map(artifacts.filter((item)=>item.kind === "file" && item.relative_path).map((item)=>[item.name,item]));
    const featured = (definition.outputs || []).map(([name,label])=>({ item: filesByName.get(name), label })).filter(({item})=>item);
    const latestReceipt = receipt || deliveredReceipts.at(-1);
    const receiptUrl = latestReceipt ? stageArtifactUrl(run, latestReceipt.receipt) : "";
    const resultLink = definition.resultTab && deliveredReceipts.some(item=>item.status === "completed")
      ? `<a href="${esc(stageResultRoute(run,definition.resultTab[0]))}">${icon("arrow")}${esc(definition.resultTab[1])}</a>` : "";
    const artifactLinks = deliveredReceipts.length ? `<div class="journey-artifacts">${resultLink}${featured.map(({item,label})=>`<a target="_blank" href="${esc(stageArtifactUrl(run,item.relative_path))}">${icon("file")}${esc(label)}</a>`).join("")}${receiptUrl ? `<a class="receipt-link" target="_blank" href="${esc(receiptUrl)}">${icon("check")}完整清单</a>` : ""}</div>` : "";
    const detail = stageState === "done"
      ? `${receipt?.stage_duration_seconds != null ? `耗时 ${duration(receipt.stage_duration_seconds)}` : "产出已保存"}`
      : stageState === "withheld" ? "完整质量检查通过后生成；当前阶段成果可正常查看。"
      : stageState === "partial" ? "分析已结束，阶段成果和具体证据缺口已保存。"
      : stageState === "active" ? definition.doing
      : stageState === "failed" ? `停止位置：${STAGE_LABELS[status.failed_stage] || "当前分析环节"}`
      : stageState === "interrupted" ? "已完成内容会保留，可以稍后重新分析。"
      : stageState === "done-unreceipted" ? "该环节已通过，后续分析已经开始。"
      : "等待前序环节完成";
    const componentNotes = deliveredReceipts.filter(item=>["skipped","failed"].includes(item.status)).map(item=>`<p class="component-outcome">${esc(STAGE_LABELS[item.stage] || item.stage)} · ${item.status === "skipped" ? "已跳过" : "未完成"}：${esc(item.reason || "请查看该环节记录")}</p>`).join("");
    return `<article class="journey-step ${stageState}"><span class="journey-number">${definition.number}</span><div class="journey-copy"><header><strong>${esc(definition.title)}</strong><span>${esc(stateLabel[stageState])}</span></header><p>${esc(detail)}</p>${componentNotes}${artifactLinks}</div></article>`;
  }).join("");
  return `<section class="current-guide ${active.state}"><div><span class="guide-kicker">${esc(stateLabel[active.state])} · 第 ${esc(active.definition.number)} 步，共 7 步</span><h3>${esc(active.definition.title)}</h3><p>${esc(productRunMessage(run, active.definition.doing))}</p></div><aside><small>${run.state === "partial" ? "本次结果" : next ? "接下来" : "最终结果"}</small><strong>${esc(run.state === "partial" ? "阶段成果已保存，完整结论未发布" : next?.definition.title || "实验成果已完整保存")}</strong></aside></section><div class="task-progress-line"><span>${["failed", "interrupted"].includes(run.state) ? "处理已停止" : `已完成 ${Math.round(Number(run.progress || 0) * 100)}%`}</span><span>已用 ${duration(elapsedForRun(run, status))}</span></div><details class="technical-observability" ${outputsOpen ? "open" : ""}><summary>查看各环节与生成文件</summary><div class="pipeline-journey">${cards}</div></details>`;
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
  const captureWarnings = (snapshot.capture_quality?.records || []).flatMap(item=>(item.warnings || []).map(message=>`${item.view_id}：${message}`));
  const captureNotice = captureWarnings.length ? `<details class="analysis-readiness-note" open><summary>采集质量预检发现问题（仅限抽样）</summary>${captureWarnings.map(message=>`<p>${esc(message)}</p>`).join("")}</details>` : "";
  const retryButton = ["partial", "failed", "interrupted"].includes(run.state) && run.nas_staging && run.retry_available !== false
    ? `<button class="secondary-button" data-retry-run="${esc(run.run_id)}" type="button">${icon("refresh")}复跑并补全</button>` : "";
  const previewAvailable = (snapshot.stage_receipts || []).some((receipt)=>
    ["experiment_understanding", "experiment_clips", "key_materials", "mllm", "material_refinement", "semantic_refinement", "package", "daily_report"].includes(receipt.stage));
  const resultRoute = run.nas_staging && run.state !== "completed"
    ? `stage/${encodeURIComponent(run.run_id)}` : `archive/${encodeURIComponent(run.experiment_id || "")}`;
  const archiveLink = run.state === "completed" || previewAvailable
    ? `<a class="secondary-button" href="#/${resultRoute}/experiments">${run.state === "completed" ? "查看实验结果" : "预览已完成内容"}</a>` : "";
  return `<section class="panel live-run-card"><header class="panel-heading"><div><p class="panel-kicker">实验分析</p><h2>${esc(productExperimentName(run.experiment_id || run.run_id))}</h2></div><div class="run-heading-actions">${archiveLink}${retryButton}<span class="queue-status ${esc(run.state)}"><i></i>${esc(run.state === "failed" && partial ? "阶段产出已保存 · 待补全" : STAGE_LABELS[current] || (run.state === "failed" ? "处理失败" : "处理中"))}</span></div></header>${freshnessNote}${captureNotice}${guidedPipelineView(run,outputsOpen)}</section>`;
}

function productRunMessage(run, fallback) {
  if (run?.state === "partial") return "分析结束，已完成成果和证据缺口均已保存，可查看阶段报告。";
  if (["failed", "interrupted"].includes(run?.state)) return friendlyFailureReason(run);
  const raw = String(run?.observability?.status?.message || "").trim();
  if (!raw || raw.length > 180 || /(?:runtimeerror|traceback|exception|[a-z]:\\|\/(?:mnt|home|tmp|var)\/)/i.test(raw)) return fallback;
  return raw;
}

function renderOperations() {
  setChrome("operations");
  const health = state.health || {};
  const modelStatus = mllmConnectionLabel(health);
  const storageReady = Boolean(health.archive_available);
  const storageLabel = isNasMode() ? "NAS 存储" : "本地存储";
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">系统</p><h1>运行状态</h1><p>确认分析与存储功能是否可用。</p></div></header><section class="health-grid"><article class="health-item"><span>${icon("server")}</span><div><small>分析服务</small><strong>${health.status === "ok" ? "运行正常" : "暂不可用"}</strong></div></article><article class="health-item"><span>${icon("folder")}</span><div><small>${storageLabel}</small><strong>${storageReady ? "目录可用" : "需要检查"}</strong></div></article><article class="health-item"><span>${icon("brain")}</span><div><small>智能理解</small><strong>${modelStatus}</strong></div></article><article class="health-item"><span>${icon("video")}</span><div><small>视频处理</small><strong>准备就绪</strong></div></article></section><section class="panel"><header class="panel-heading"><div><h2>存储位置</h2></div><span class="badge">${storageReady ? "目录可用" : "需要检查"}</span></header><div class="capability-list"><div><span>${icon("folder")}</span><p><strong>实验产出</strong><small>${esc(health.archive_root || "未配置")}</small></p></div><div><span>${icon("server")}</span><p><strong>处理缓存</strong><small>${esc(health.fixed_benchmark?.local_cache_root || "未配置")}</small></p></div></div></section>${isNasMode() && health.fixed_benchmark?.enabled !== false ? `<details class="panel technical-observability"><summary>管理员工具</summary><div class="maintenance-action"><p>仅在系统排障时运行环境检查。</p><button class="secondary-button" id="rerun-benchmark" type="button">${icon("activity")}运行环境检查</button></div></details>` : ""}</div>`;
  document.querySelector("#rerun-benchmark")?.addEventListener("click", rerunBenchmark);
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

function archiveProcessStopped(data) {
  return ["partial","failed","interrupted"].includes(data?.observability?.status?.stage);
}

function friendlyFailureReason(data) {
  const status = data?.observability?.status || {};
  const delivery = data?.partial_delivery || data?.observability?.partial_delivery;
  if (status.stage === "partial" || data?.state === "partial") {
    const gaps = (delivery?.quality_gaps || []).map(item => item.message).filter(Boolean);
    return `分析已结束，阶段成果已保存。${gaps.length ? gaps.join("；") : "部分证据未满足完整质量要求，具体缺口已列入阶段报告。"}`;
  }
  if (data?.result_review?.result_updated) {
    const review=data.result_review;
    return `原分析未通过完整质量检查，之后步骤结果已有更新。${review.latest_check_current ? (review.step_consistency_passed ? "最新步骤文字与引用动作的检查已通过。" : "最新步骤检查仍有问题需要处理。") : "最新结果尚未重新检查。"}实验完整性和跨视角证据仍需分别核验。`;
  }
  const raw = String(status.error || status.message || data?.error || "");
  if (/WinError\s*206|ENAMETOOLONG|文件名或扩展名太长|路径过长|filename.{0,20}too long|path.{0,20}too long/i.test(raw)) return "Windows 文件路径过长，当前环节未能继续。已保存的阶段成果仍可查看；请更新应用的短路径缓存修复后重试。";
  const failedStage = status.failed_stage || data?.observability?.metrics?.nas_index_ingest?.failure_stage;
  if (["nas_ingest", "input_preflight"].includes(failedStage)) {
    return "实验输入检查未通过，分析尚未开始。请检查所选视频、时间记录与拍摄视角，再重试该任务。";
  }
  const quality = data?.quality_acceptance || {};
  if (quality.passed === false || /quality acceptance failed/i.test(raw)) {
    const reasons = [];
    if (quality.segmentation_integrity?.canonical_pair_coverage_passed === false) reasons.push("有实验片段缺少第一与第三人称共同支持的动作证据");
    if (quality.step_action_consistency?.passed === false) reasons.push("步骤文字与最终动作证据不一致");
    return `本次结果未通过自动质量检查${reasons.length ? `：${reasons.join("；")}` : ""}。已完成片段和素材已保留，请查看质量检查记录。`;
  }
  const partial = data?.partial_delivery || data?.observability?.partial_delivery;
  if (partial?.status === "awaiting_semantic_recovery") {
    const pending = partial.pending_semantic_results || [];
    if (pending.length && pending.every((item)=>item.status === "disabled")) return "该任务运行时未启用智能理解，因此未生成完整分析与报告。已完成片段和候选素材已保留；可在新建实验中使用当前 AI 配置重新分析原视频。";
    if (pending.some((item)=>item.status === "skipped_missing_api_key")) return "该任务运行时缺少可用的 AI 密钥，智能理解未完成。已完成内容已保留；请确认当前 AI 配置后，在新建实验中重新分析原视频。";
    return "该任务的智能理解尚未完成，已完成片段和候选素材已保留。请查看任务记录中的具体原因，确认 AI 配置后重新分析；此状态不代表当前账户不可用。";
  }
  if (/decode|frame|sam2/i.test(raw)) return "关键素材精细处理时，系统无法读取其中一段视频的指定画面。";
  if (/cuda|gpu|memory|out of memory/i.test(raw)) return "视频分析所需的计算资源暂时不足，任务已安全停止。";
  if (/network|timeout|connection/i.test(raw)) return "分析过程中连接暂时中断，已完成的内容仍然保留。";
  return "本次分析没有完整结束，系统已保留停止前完成的内容。";
}

function detailAnchorNav(tab) {
  if (tab === "speech") return "";
  const items = tab === "materials" ? [["material-workspace","素材概览"],["material-filters","筛选素材"],["material-results","关键素材"]]
    : tab === "reports" ? [["daily-report-cover","日报摘要"],["daily-timeline-section","实验时间线"],["daily-status-section","报告状态"]]
      : tab === "metrics" ? [["professional-report-hero","报告概览"],["professional-files","报告文件"],["professional-appendix","技术附录"]]
        : [["archive-overview","结果概览"],["experiment-results","实验片段"]];
  return `<nav class="detail-anchor-nav" aria-label="本页内容">${items.map(([target,label],index)=>`<button type="button" data-detail-section="${target}" class="${index===0?"active":""}">${label}</button>`).join("")}</nav>`;
}

function componentResultCards(data, name, staging) {
  const labels = {completed:"已完成", partial:"部分完成", insufficient:"证据不足", not_generated:"未生成", not_available:"暂无产出", failed:"待处理", running:"处理中", pending:"待检查", disabled:"未启用"};
  return `<section class="component-results" aria-label="分项分析结果">${(data.observability?.components || []).map(item=>`<article><span>${esc(item.label)}</span><strong data-component-state="${esc(item.state)}">${esc(labels[item.state] || item.state)}</strong><p>${esc(item.detail)}</p>${["completed","partial"].includes(item.state)?`<a href="#/${staging?"stage":"archive"}/${encodeURIComponent(name)}/${item.tab}">查看成果</a>`:""}</article>`).join("")}</section>`;
}

function experimentAttentionPanel(data, tab = "experiments") {
  const status = data.observability?.status || {};
  const receipts = (data.observability?.stage_receipts || []).filter((item)=>item.status==="completed");
  const lastReceipt = [...receipts].sort((a,b)=>(Date.parse(a.completed_at)||0)-(Date.parse(b.completed_at)||0)).at(-1);
  const preliminaryCount = (data.preliminary_materials?.length || 0) + (data.quarantined_materials?.length || 0);
  const materialCount = Number(data.counts?.key_events ?? data.key_events?.length ?? 0);
  const available = [data.experiments?.length ? `${number(data.counts?.experiments ?? data.experiments.length)} 个实验片段` : "", materialCount ? `${number(materialCount)} 份关键素材` : "", preliminaryCount ? `${number(preliminaryCount)} 份候选素材` : "", ...(data.observability?.components || []).filter(item=>["speech","understanding"].includes(item.key)&&item.state==="completed").map(item=>item.label)].filter(Boolean).join("、") || "任务记录与停止位置";
  const attentionTitle = data.result_review?.result_updated ? "阶段结果已更新，完整验收仍待补全" : status.stage === "partial" ? "分析结束，阶段成果已保存" : (data.observability?.components || []).some(item=>item.state==="completed") ? "部分成果已完成，仍有环节需要处理" : "本次分析在完成前停止";
  const hasVideo = (data.experiments || []).some(item => item.aligned_video_url || item.first_person_video_url || item.third_person_video_url);
  const videoAction = !hasVideo ? "" : tab === "experiments"
    ? '<button class="primary-button" type="button" data-detail-section="experiment-results">查看已生成视频</button>'
    : `<a class="primary-button" href="${experimentRecordRoute(data)}">查看已生成视频</a>`;
  const attentionArchives = experimentRecords().filter((archive)=>archiveProductStatus(archive).key==="attention");
  const position = attentionArchives.findIndex((archive)=>archive.name===data.name);
  const previous = attentionArchives[position-1];
  const next = attentionArchives[position+1];
  return `<section class="experiment-attention-panel is-compact" role="alert"><div class="attention-icon">${icon("activity")}</div><div class="attention-main"><h2>${attentionTitle}</h2><p>${esc(friendlyFailureReason(data))}</p><p class="attention-report-state">完整报告尚未生成，已保存的视频和阶段结果可查看。</p><div class="attention-actions">${videoAction}${data.links?.partial_report?`<a class="secondary-button" href="${esc(data.links.partial_report)}" target="_blank" rel="noopener">查看阶段结果报告</a>`:""}${rerunArchiveAction(data)}<a class="text-button" href="#/tasks">查看任务记录</a></div><details class="attention-run-details"><summary>停止记录与重新分析</summary><div class="attention-facts"><span><small>停止环节</small><strong>${esc(STAGE_LABELS[status.failed_stage||status.stage]||"分析处理中")}</strong></span><span><small>最后完成</small><strong>${esc(STAGE_LABELS[lastReceipt?.stage]||"已保存输入")}</strong></span><span><small>当前可查看</small><strong>${esc(available)}</strong></span></div><div class="attention-actions">${data.links?.partial_json ? `<a class="secondary-button" href="${esc(data.links.partial_json)}" download="VisionCortex-阶段分析.json">导出阶段 JSON</a>` : ""}<a class="text-button" href="#/experiments?status=attention">全部异常实验 ${icon("arrow")}</a></div></details></div></section>`;

}

function rerunArchiveAction(data) {
  if (data.retry_available === false) return '<a class="secondary-button" href="#/new">选择原输入重新分析</a>';
  return `<button class="primary-button" type="button" data-rerun-archive="${esc(data.name)}">${icon("refresh")}恢复与补全</button>`;
}

function resultHeader(data, tab, staging = false) {
  const metrics = data.metrics || {};
  const currentRun = metrics.preprocessing_display?.current_run || {};
  const status = data.observability?.status || {};
  const running = status.stage && !["completed", "partial", "failed", "interrupted"].includes(status.stage);
  const analysisDuration = running ? status.elapsed_seconds : currentRun.total_seconds ?? metrics.total_duration_seconds;
  const formalMaterialCount = Number(data.counts?.key_events ?? data.key_events?.length ?? 0);
  const preliminaryMaterialCount = (data.preliminary_materials?.length || 0) + (data.quarantined_materials?.length || 0);
  const materialCount = formalMaterialCount || preliminaryMaterialCount;
  const reportCount = [data.links?.daily_report_pdf || data.links?.partial_pdf, data.links?.daily_report_json || data.links?.partial_json].filter(Boolean).length;
  const metadata = experimentMetadata(data.name);
  const tags = Array.isArray(metadata.tags) ? metadata.tags : [];
  const metadataChips = [metadata.owner ? `负责人 · ${metadata.owner}` : "", ...tags].filter(Boolean);
  const stopped = archiveProcessStopped(data);
  const backTarget = staging ? "#/tasks" : tab === "materials" ? "#/materials" : ["reports","metrics"].includes(tab) ? "#/reports" : archiveLibraryHash("experiments");
  const backLabel = staging ? "返回任务进度" : tab === "materials" ? "返回关键素材库" : ["reports","metrics"].includes(tab) ? "返回实验室日报" : "返回实验记录";
  const nextAction = tab === "experiments" ? [experimentRecordRoute(data,"materials"),"image","查看关键素材"] : tab === "materials" ? [experimentRecordRoute(data,"reports"),"file","查看实验日报"] : tab === "reports" ? [experimentRecordRoute(data,"metrics"),"activity","查看分析记录"] : null;
  const heroActions = stopped ? "" : `<div class="hero-actions"><button class="secondary-button" type="button" data-edit-experiment>${icon("file")}编辑信息</button>${nextAction?`<a class="primary-button" href="${nextAction[0]}">${icon(nextAction[1])}${nextAction[2]}</a>`:""}</div>`;
  return `<div class="result-header" id="archive-overview"><a class="result-back-link" href="${backTarget}">${icon("chevron")}${backLabel}</a><header class="page-hero compact experiment-hero ${stopped?"is-attention":""}"><div title="档案编号：${esc(data.name)}"><p class="eyebrow">${stopped?"阶段性实验结果":"实验详情"}</p><h1>${esc(productExperimentName(data.name))}</h1><p>${stopped?"本页展示已保存的阶段成果及证据限制。":"查看实验过程、关键素材与已生成的报告。"}</p>${metadataChips.length ? `<div class="experiment-meta-chips">${metadataChips.map((item)=>`<span>${esc(item)}</span>`).join("")}</div>` : ""}</div>${heroActions}</header><div class="result-facts" aria-label="实验概览"><span><b>${number(data.counts?.experiments ?? data.experiments.length)}</b> 个片段</span><span><b>${number(materialCount)}</b> 份${formalMaterialCount?"素材":"候选素材"}</span><span>用时 <b>${duration(analysisDuration)}</b></span></div><div class="result-nav-shell"><strong class="result-nav-name" title="${esc(data.name)}">${esc(productExperimentName(data.name))}</strong><nav class="result-tabs" aria-label="实验结果导航"><a class="result-tab ${tab==="experiments"?"active":""}" ${tab==="experiments"?'aria-current="page"':""} href="${experimentRecordRoute(data)}">${icon("video")}<span>视频与操作步骤</span></a><a class="result-tab ${tab==="speech"?"active":""}" ${tab==="speech"?'aria-current="page"':""} href="#/archive/${encodeURIComponent(data.name)}/speech">${icon("file")}<span>录音与转写</span></a><a class="result-tab ${tab==="materials"?"active":""}" ${tab==="materials"?'aria-current="page"':""} href="#/archive/${encodeURIComponent(data.name)}/materials">${icon("boxes")}<span>关键素材</span></a><a class="result-tab ${tab==="reports"?"active":""}" ${tab==="reports"?'aria-current="page"':""} href="#/archive/${encodeURIComponent(data.name)}/reports">${icon("file")}<span>日报与报告</span></a><a class="result-tab ${tab==="metrics"?"active":""}" ${tab==="metrics"?'aria-current="page"':""} href="#/archive/${encodeURIComponent(data.name)}/metrics">${icon("folder")}<span>分析记录</span></a></nav><button class="result-nav-folder" type="button" data-open-folder="${esc(data.name)}" aria-label="打开实验文件">${icon("folder")}</button></div><dialog class="experiment-meta-dialog" id="experiment-meta-dialog"><form id="experiment-meta-form"><header><div><p class="eyebrow">实验信息</p><h2>完善实验信息</h2><p>便于日常查找与协作，不会改写正式证据档案。</p></div><button class="dialog-close" type="button" data-close-experiment-meta aria-label="关闭">${icon("x")}</button></header><div class="experiment-meta-fields"><label><span>显示名称</span><input name="display_name" value="${esc(metadata.displayName || "")}" placeholder="例如：电子天平称量验证" /></label><label><span>负责人</span><input name="owner" value="${esc(metadata.owner || "")}" placeholder="姓名或团队" /></label><label class="wide"><span>标签</span><input name="tags" value="${esc(tags.join("、"))}" placeholder="例如：称量、质控、第二轮" /></label><label class="wide"><span>实验备注</span><textarea name="note" rows="4" placeholder="记录目的、批次说明或后续事项">${esc(metadata.note || "")}</textarea></label></div>${metadata.note ? `<p class="experiment-note-preview"><strong>当前备注</strong>${esc(metadata.note)}</p>` : ""}<footer><span>实验原始编号会继续保留</span><div><button class="secondary-button" type="button" data-close-experiment-meta>取消</button><button class="primary-button" type="submit">保存信息</button></div></footer></form></dialog></div>`;
}

function experimentExecutiveSummary(data) {
  const stage = data.observability?.status?.stage;
  if (stage && !["completed", "partial", "failed", "interrupted"].includes(stage)) {
    return `<section class="experiment-executive-summary is-partial"><header><div><p class="eyebrow">阶段预览</p><h2>本次分析仍在进行</h2></div><span class="evidence-kind uncertain">自动核验中</span></header><p>下方可预览已保存的片段与素材，复跑时也会保留之前的阶段内容。当前步骤与完整报告仍在更新，以本轮完成并通过验收后的结果为准。</p><a href="#/tasks">查看本次任务进度 ${icon("arrow")}</a></section>`;
  }
  const report = data.daily_report || {};
  const overview = report.overview || {};
  const stopped = archiveProcessStopped(data);
  const steps = (data.experiments || []).flatMap(item => item.steps || []);
  const summary = steps.length ? `已整理 ${number(steps.length)} 条操作记录。按片段和时间阅读操作过程、前后变化，并回看对应视频。` : "步骤尚未生成，可先查看已保存的视频。";
  const concernCount = Number(report.uncertainties?.length||0)+Number(report.contradictions?.length||0);
  const viewCount = overview.input_view_count ?? data.input_view_count ?? data.observability?.status?.input_view_count ?? 0;
  return `<section class="experiment-executive-summary ${stopped?"is-partial":""}"><header><div><p class="eyebrow">${stopped?"阶段性结果":"实验结论总览"}</p><h2>${stopped?"已完成内容概览":"先看结论，再进入证据"}</h2></div><span class="evidence-kind ${stopped?"uncertain":"interpreted"}">${stopped?"非完整结论":"模型归纳"}</span></header><div class="executive-summary-grid"><article><small>${stopped?"当前可查看":"本次完成"}</small><strong>${number(overview.experiment_group_count??data.counts?.experiments??data.experiments?.length)} 个实验片段 · ${number(overview.key_event_count??data.counts?.key_events??data.key_events?.length)} 份关键素材</strong><p>${number(viewCount)} 路画面参与分析。</p></article><article class="executive-finding"><small>${stopped?"已生成的步骤理解":"关键发现"}</small><strong>${esc(summary)}</strong></article><article class="${stopped||concernCount?"has-concern":""}"><small>${stopped?"完整性状态":"需要留意"}</small><strong>${stopped?(stage==="partial"?"分析已结束，部分证据不足":"本次处理未完整结束"):concernCount?`${number(concernCount)} 项需留意`:report.report_id?"暂无报告级视角差异":"请在日报中查看完整不确定性"}</strong><p>${stopped?"以下理解仅来自停止前已完成的阶段。":"仍应以实际素材为准。"}</p></article><article><small>建议下一步</small><strong>${stopped?"查看已有视频及阶段报告中的缺口。":concernCount?"优先查看存在不确定性的素材。":"继续查看关键素材。"}</strong><a href="${stopped?"#/tasks":`#/archive/${encodeURIComponent(data.name)}/materials`}">${stopped?"查看任务记录":"查看相关素材"} ${icon("arrow")}</a></article></div></section>`;
}

function videoPosterUrl(url) {
  if (!url) return "";
  const parsed = new URL(url, location.href);
  if (parsed.origin !== location.origin || !["/api/archive-file", "/api/staging-file"].includes(parsed.pathname)) return "";
  parsed.searchParams.set("poster", "true");
  return `${parsed.pathname}?${parsed.searchParams}`;
}

function videoPreview(url, label, poster = videoPosterUrl(url), identity = {}) {
  const binding = identity.focus ? "data-focus-video" : identity.materialId ? `data-material-video="${esc(identity.materialId)}"` : "";
  return `<div class="preview-player"><video controls playsinline preload="none" tabindex="0" ${binding} aria-label="${esc(label)}" ${poster ? `poster="${esc(poster)}"` : ""} src="${esc(url)}"></video><button class="preview-play" type="button" aria-label="播放${esc(label)}">▶ <span>播放${esc(label)}</span></button><p class="preview-status" role="status" hidden></p><a class="preview-open" href="${esc(url)}" target="_blank" rel="noopener" hidden>单独打开视频</a></div>`;
}

function bindVideoPreviews(root = main) {
  root.querySelectorAll(".preview-player").forEach((player) => {
    if (player.dataset.bound) return;
    player.dataset.bound = "true";
    const video = player.querySelector("video"), button = player.querySelector(".preview-play");
    const status = player.querySelector(".preview-status"), link = player.querySelector(".preview-open");
    const failed = () => {
      status.textContent = "视频未能加载，请重试或单独打开。";
      status.hidden = false; link.hidden = false; button.hidden = false;
      button.querySelector("span").textContent = "重新播放";
    };
    button.addEventListener("click", async () => {
      button.hidden = true; link.hidden = true;
      status.textContent = "正在加载视频…"; status.hidden = false;
      if (video.error) video.load();
      try { await video.play(); } catch { failed(); }
    });
    video.addEventListener("playing", () => { button.hidden = true; status.hidden = true; link.hidden = true; });
    video.addEventListener("error", failed);
    video.addEventListener("ended", () => { button.hidden = false; });
  });
}

function evidenceSentences(value) {
  const result = [];
  let part = "", depth = 0;
  for (const character of String(value || "")) {
    part += character;
    if ("（([【".includes(character)) depth++;
    if ("）)]】".includes(character)) depth = Math.max(0,depth-1);
    if (!depth && "。；;\n".includes(character)) { if (part.trim()) result.push(part.trim()); part = ""; }
  }
  if (part.trim()) result.push(part.trim());
  return result;
}

function readableEvidenceParts(value) {
  const narrative = [], technical = [];
  // Split at sentence boundaries only: keep negations and entire clauses intact.
  for (const raw of evidenceSentences(value)) {
    if (/\bCV\b|CV标签|CV 标签|CV 收据|\bPTS\b|clip_timeline|clip_late|pair_count|pair_evidence_status|same_action_pair_verified|轨迹\s*ID|before_after_container_state|候选\s*[a-z]+(?:_[a-z]+)+/.test(raw)) {
      technical.push(raw.trim());
      continue;
    }
    let sentence = raw.trim();
    for (const [key,label] of Object.entries({...OBJECT_LABELS, ...ACTION_LABELS}).sort((a,b)=>b[0].length-a[0].length)) sentence = sentence.replaceAll(key,label);
    narrative.push(sentence);
  }
  return {narrative,technical};
}

function evidenceParagraphs(items) {
  return `<ul class="evidence-prose">${items.map(item=>`<li>${esc(item)}</li>`).join("")}</ul>`;
}

function experimentStepDescription(step) {
  const current = readableEvidenceParts(step.current_step || step.observed_action);
  const timingNote = step.time_scope?.complete_operation_boundaries_proven === false ? '<p class="muted">该时间用于定位已记录画面，完整操作的起止位置尚未核对。</p>' : "";
  const physical = readableEvidenceParts(step.physical_change);
  const next = readableEvidenceParts(step.next_step);
  const technical = [...current.technical,...physical.technical,...next.technical];
  return `<div class="step-description">${timingNote}<section><h4>操作过程</h4>${current.narrative.length ? evidenceParagraphs(current.narrative) : "<p>当前记录没有可独立陈述的动作描述，需结合下方分析依据核对。</p>"}</section>${physical.narrative.length ? `<section><h4>${step.observed_result ? "当前结果" : "前后变化"}</h4>${evidenceParagraphs(physical.narrative)}</section>` : ""}<section class="step-next"><h4>${esc(nextStepLabel(step))}${nextStepLabel(step)==="预测后续" ? " · 推测，画面尚未确认" : ""}</h4>${next.narrative.length ? evidenceParagraphs(next.narrative) : "<p>没有足够证据支持下一步。</p>"}</section>${technical.length ? `<p class="step-technical-notice">另有 ${number(technical.length)} 条检测或时间校验限制，可能影响上述判断。</p><details class="step-evidence"><summary>检测与时间校验记录（${number(technical.length)} 条）</summary>${evidenceParagraphs(technical)}</details>` : ""}</div>`;
}

function experimentLimitations(items) {
  const groups = {"动作与状态":[],"对象与手部":[],"视角与时间":[]}, technical = [];
  for (const item of items) {
    const parts = readableEvidenceParts(item);
    technical.push(...parts.technical);
    for (const sentence of parts.narrative) {
      const category = /视角|人称|同步|时序|先后|采样|关键帧|模糊/.test(sentence) ? "视角与时间" : /手|对象|身份|实例|标签|成分|归属|容器/.test(sentence) ? "对象与手部" : "动作与状态";
      groups[category].push(sentence);
    }
  }
  return `${Object.entries(groups).filter(([,values])=>values.length).map(([label,values])=>`<section class="limitation-group"><h4>${label}</h4>${evidenceParagraphs(values)}</section>`).join("")}${technical.length ? `<p>检测标签、机位对应或时间校验另有 ${number(technical.length)} 条限制，相关动作仍需结合依据核对。</p><details class="understanding-disclosure"><summary>检测与时间校验记录（${number(technical.length)} 条）</summary>${evidenceParagraphs(technical)}</details>` : ""}`;
}

function experimentObjects(steps) {
  return [...new Set(steps.flatMap(step=>step.objects || []).filter(value=>!/^(gloved_hand|hand)(-|$)/.test(value)).map(productObjectLabel).filter(label=>!["相关对象","暂未识别"].includes(label)))];
}

function experimentStepTitle(step) {
  if (typeof step.operation_title === "string" && step.operation_title.trim()) return step.operation_title.trim();
  const subject = experimentObjects([step])[0];
  const verb = {object_movement:"移动",hand_object_contact:"接触",device_panel_operation:"操作",panel_operation:"操作"}[step.action_type];
  return subject && verb ? `${verb}${subject}` : ACTION_LABELS[step.action_type] || "查看操作过程";
}

function readerClock(ms) {
  return timecode(ms).split(".")[0];
}

function readerStepSections(experiment) {
  const steps = experiment.steps || [];
  const valid = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const sections = (experiment.workflow_units || []).filter(unit =>
    valid(unit.start_ms) && valid(unit.end_ms) && Number(unit.end_ms) > Number(unit.start_ms)
  ).map((unit,index) => ({key:String(index), name:unit.name || `实验单元 ${index+1}`,
    start:Number(unit.start_ms), end:Number(unit.end_ms), indices:[]}));
  const remaining = {key:"unassigned", name:sections.length ? "单元间及待归属记录" : "操作记录", indices:[]};
  steps.forEach((step,index) => {
    const matches = valid(step.start_global_ms) && valid(step.end_global_ms) && Number(step.end_global_ms) > Number(step.start_global_ms)
      ? sections.filter(section => Number(step.start_global_ms) >= section.start && Number(step.end_global_ms) <= section.end) : [];
    (matches.length === 1 ? matches[0] : remaining).indices.push(index);
  });
  return [...sections, ...(remaining.indices.length ? [remaining] : [])];
}

function readerPlaybackMatches(intervals, timestamp) {
  if (!Number.isFinite(timestamp)) return [];
  return intervals.flatMap((interval,index) => {
    const {start,end} = interval;
    return Number.isFinite(start) && Number.isFinite(end) && end > start && timestamp >= start && timestamp < end ? [index] : [];
  });
}

function readerSeekVideo(video, seconds, startPlayback, onSeek) {
  if (!video || !Number.isFinite(seconds)) return () => {};
  // The play control may reload failed media. Seek after that reset, not before.
  if (startPlayback) startPlayback();
  else if (video.error) video.load();
  const apply = () => {
    video.currentTime = Math.max(0,Math.min(seconds,Number.isFinite(video.duration)?video.duration:seconds));
    onSeek();
  };
  if (video.readyState) apply();
  else {
    video.addEventListener("loadedmetadata",apply,{once:true});
    if (!startPlayback) video.load();
  }
  return () => video.removeEventListener("loadedmetadata",apply);
}

function stageDeliveryView(data) {
  const receipts = (data.observability?.stage_receipts || []).filter(item=>item.receipt_url);
  if (!receipts.length) return "";
  const ordered = [...receipts].sort((a,b)=>(Date.parse(a.completed_at)||0)-(Date.parse(b.completed_at)||0));
  return `<section class="stage-delivery" aria-label="已保存的阶段成果"><header><div><strong>阶段成果</strong><p>各环节独立留存，后续处理不会清除已保存记录。</p></div><label>选择环节<select data-stage-version>${ordered.map((item,index)=>`<option value="${index}" ${index===ordered.length-1?"selected":""}>${esc(STAGE_LABELS[item.stage]||item.stage)} · ${item.status==="completed"?"已保存":item.status==="skipped"?"已跳过":"未完成"}</option>`).join("")}</select></label></header>${ordered.map((item,index)=>`<div data-stage-version-panel="${index}" ${index===ordered.length-1?"":"hidden"}><p>${esc(item.reason||"本阶段文件已保存，可在后续分析进行时查看。")}</p><nav aria-label="阶段文件">${item.version_url?`<a class="secondary-button" href="${esc(item.version_url)}" target="_blank" rel="noopener">当时的结果与溯源</a>`:""}<a class="text-button" href="${esc(item.receipt_url)}" target="_blank" rel="noopener">完成记录</a>${(item.artifacts||[]).filter(x=>x.url).slice(0,3).map(x=>`<a class="text-button" href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.name)}</a>`).join("")}</nav>${!item.version_url?'<small>历史阶段仅有完成记录，未保存独立版本快照。</small>':""}</div>`).join("")}</section>`;
}

function experimentResultTools(data, staging = false) {
  const links = data.links || {}, path = data.path || data.network_path || "";
  const report = links.partial_pdf || links.daily_report_pdf || links.partial_daily_report || links.partial_report || links.daily_report_json;
  const exported = links.partial_json || links.daily_report_json;
  const source = staging ? data.staging_run_id : data.name;
  return `<section class="experiment-result-tools" aria-label="成果与保存位置"><div><strong>${staging?"已保存的阶段成果":"实验成果"}</strong>${path?`<p class="output-location"><span>实际保存目录</span><code>${esc(path)}</code><button type="button" class="text-button" data-copy-path="${esc(path)}">复制路径</button></p>`:"<small>保存位置未提供</small>"}</div><nav aria-label="成果操作">${report?`<a class="secondary-button" href="${esc(report)}" target="_blank" rel="noopener">${links.partial_report?"查看阶段报告":"查看报告"}</a>`:'<span class="result-pending">报告尚未生成</span>'}${exported?`<a class="secondary-button" href="${esc(exported)}" download="VisionCortex-${staging?"阶段结果":"实验报告"}.json">导出 JSON</a>`:""}${path&&source?`<button type="button" class="primary-button" ${staging?"data-open-staging-folder":"data-open-folder"}="${esc(source)}">${icon("folder")}打开产出文件夹</button>`:""}</nav></section>`;
}

function experimentGroupBrowser(data, preliminary = false, selectedFolder = "") {
  const groups = [...(data.experiments || [])].sort((a,b)=>Number(Boolean(a.activity_assessment?.is_auxiliary))-Number(Boolean(b.activity_assessment?.is_auxiliary)));
  if (!groups.length) {
    const stopped = archiveProcessStopped(data);
    return `<section id="experiment-results">${productState(stopped?"error":"progress",stopped?"x":"clock",stopped?"本次处理未生成实验片段":"实验片段正在生成",stopped?"可以在任务记录中查看停止位置，并按原设置重新分析。":"视频片段保存后即可播放，后续模型分析会继续。开启顶部的跟随分析进度可自动查看。",'<a class="secondary-button" href="#/tasks">查看任务进度</a>')}</section>`;
  }
  const matched = groups.findIndex(group=>group.folder===selectedFolder||(group.source_archive_folders||[]).includes(selectedFolder));
  const selected = Math.max(0,matched);
  const route = group=>`${experimentRecordRoute(data)}?group=${encodeURIComponent(group.folder)}`;
  return `<section class="experiment-browser" id="experiment-results"><div class="experiment-switcher"><label><span>选择视频片段</span><select data-experiment-group>${groups.map((group,index)=>`<option title="${timecode(group.start_ms)} — ${timecode(group.end_ms)}" value="${esc(route(group))}" ${index===selected?"selected":""}>${group.activity_assessment?.is_auxiliary?"辅助活动":"片段"} ${index+1} · ${readerClock(group.start_ms)} — ${readerClock(group.end_ms)} · ${number(group.steps?.length)} 条记录</option>`).join("")}</select></label><nav aria-label="切换实验片段">${selected>0?`<a class="secondary-button" href="${esc(route(groups[selected-1]))}">上一个片段</a>`:'<button class="secondary-button" disabled>上一个片段</button>'}<span>${selected+1} / ${groups.length}</span>${selected<groups.length-1?`<a class="secondary-button" href="${esc(route(groups[selected+1]))}">下一个片段</a>`:'<button class="secondary-button" disabled>下一个片段</button>'}</nav></div>${selectedFolder&&matched<0?'<p class="analysis-readiness-note" role="status">指定片段尚未载入，当前显示已载入的第一段。</p>':""}${data.next_cursor?'<p class="analysis-readiness-note">这里列出已载入的片段，可在下方加载更多。</p>':""}${experimentCard(groups[selected],preliminary)}</section>`;
}

function workflowCompletionLabel(experiment) {
  if (experiment.activity_assessment?.is_auxiliary) return "辅助活动 · 不计入实验";
  return {ongoing_at_recording_end:"录像结束，实验待续",unresolved:"实验结束位置待核实",
    observed_complete:"已观察到实验结束",unreviewed:"实验边界尚未复核"}[experiment.completion_status] || "实验边界尚未复核";
}

function workflowLabel(experiment) {
  if (experiment.activity_assessment?.is_auxiliary) return experiment.activity_assessment.label || "辅助活动";
  return experiment.workflow_kind === "continuous_workflow" ? "连续实验链"
    : experiment.workflow_kind === "independent_experiment" ? "独立实验" : "实验操作片段";
}

function experimentCard(experiment, preliminary = false) {
  const steps = experiment.steps || [], units = experiment.workflow_units || [];
  const sections = readerStepSections(experiment), objects = experimentObjects(steps);
  const title = experiment.activity_assessment?.is_auxiliary ? `${experiment.activity_assessment.label}记录` : experiment.workflow_kind === "continuous_workflow" ? experiment.name || "连续实验链" : objects.length ? `${objects.slice(0,3).join("、")}操作记录` : "实验操作记录";
  const uncertainties = experiment.uncertainties || [];
  const directoryOpen = steps.length <= 6;
  const videoTile = (label, url, className = "") => `<article class="experiment-media ${className}"><p class="video-caption"><span>${icon("video")}${esc(label)}</span><small>${className.includes("experiment-primary-video") ? "同步回看" : "独立播放"}</small></p><div class="video-frame">${url ? videoPreview(url, label) : `<div class="empty-state">${esc(label)}尚未生成</div>`}</div></article>`;
  const individual = `${videoTile("第一人称", experiment.first_person_video_url)}${videoTile("第三人称", experiment.third_person_video_url)}`;
  const videos = experiment.aligned_video_url ? `${videoTile("同步双视角", experiment.aligned_video_url, "experiment-media--aligned experiment-primary-video")}${individual}` : individual;
  const hasVideo = experiment.aligned_video_url || experiment.first_person_video_url || experiment.third_person_video_url;
  const coverage=experiment.operation_coverage||{}, reviewGaps=coverage.review_intervals||[];
  const stepButton = (index) => {
    const step = steps[index], observation = readableEvidenceParts(step.current_step || step.observed_action).narrative[0] || "";
    return `<button type="button" data-experiment-step="${index}" aria-pressed="${index===0}" title="${esc(observation)}"><span>${String(index+1).padStart(2,"0")}</span><strong>${esc(experimentStepTitle(step))}</strong><small title="原片时间 ${timecode(step.start_global_ms)}">${readerClock(step.start_global_ms)}</small>${observation?`<em>${esc(observation)}</em>`:""}</button>`;
  };
  return `<article id="experiment-${esc(experiment.folder)}" class="experiment-card experiment-card--understanding experiment-reader" data-reader-start="${Number(experiment.start_ms)}">
    <header class="experiment-card-header"><div class="experiment-title-copy"><span class="experiment-label" title="${timecode(experiment.start_ms)} — ${timecode(experiment.end_ms)}">${workflowLabel(experiment)} · ${readerClock(experiment.start_ms)} — ${readerClock(experiment.end_ms)}</span><h2>${esc(title)}</h2><p class="experiment-reader-caption">${number(steps.length)} 条操作记录 · 选择步骤定位画面，或跟随视频阅读</p></div><span class="badge">${workflowCompletionLabel(experiment)}</span></header>
    ${experiment.activity_assessment?.is_auxiliary?`<p class="activity-classification-note">${esc(experiment.activity_assessment.reason)} · ${experiment.activity_assessment.source==="user_review"?"用户复核":"模型分类"}，原视频与操作依据已保留。</p>`:""}
    ${units.length > 1 ? `<nav class="workflow-unit-nav" aria-label="连续实验中的单元">${units.map(unit=>`<button type="button" class="compact-button" data-step-seek="${Math.max(0,(unit.start_ms-experiment.start_ms)/1000)}">${esc(unit.name)}<small title="${timecode(unit.start_ms)} — ${timecode(unit.end_ms)}">${readerClock(unit.start_ms)} — ${readerClock(unit.end_ms)}</small></button>`).join("")}</nav>` : ""}
    <div class="experiment-reader-layout"><div class="experiment-reader-overview"><div class="experiment-reader-media"><div class="experiment-media-set">${videos}</div><p class="reader-playback-status" data-reader-playing role="status">${hasVideo?"选择步骤可定位画面":"本段尚无可播放视频"}</p><p class="experiment-reader-source">说明由模型根据画面整理。同步视频提供对照，不代表每一步都获得两个视角共同支持。${(experiment.view_timeline||[]).some(row=>!row.third_person_view)?"移动途中或尚未确认机位的部分保留第一人称，右侧标明无对应机位。":""}</p></div>
    <aside class="experiment-reader-companion" aria-label="操作目录与步骤说明"><header class="reader-controls"><div class="reader-controls-heading"><h3>操作与步骤</h3><button type="button" class="text-button" data-reader-focus aria-expanded="false">扩大阅读区</button></div>
    <div class="reader-unit-picker"><label ${sections.length<2?"hidden":""}>实验单元<select data-reader-unit aria-label="按实验单元查看步骤"><option value="all">全部记录（${steps.length}）</option>${sections.map(section=>`<option value="${section.key}">${esc(section.name)}（${section.indices.length}）</option>`).join("")}</select></label><button type="button" class="secondary-button" data-toggle-directory aria-expanded="${directoryOpen}">${directoryOpen?"收起目录":"查看目录"}</button></div>
    <div class="reader-step-controls"><button type="button" class="text-button" data-reader-prev disabled>上一步</button><span data-reading-counter>正在阅读 ${steps.length?1:0} / ${steps.length}</span><button type="button" class="text-button" data-reader-next ${steps.length<2?"disabled":""}>下一步</button></div><label class="reader-follow"><input type="checkbox" data-reader-follow checked ${hasVideo?"":"disabled"}>跟随视频切换步骤</label></header>
    <section class="experiment-reader-index" data-reader-directory ${directoryOpen?"":"hidden"}><nav class="experiment-step-nav" aria-label="实验步骤">${sections.map(section=>`<section class="reader-directory-group" data-reader-group="${section.key}"><h4>${sections.length===1?"本段操作记录":esc(section.name)}<small>${section.indices.length} 条</small></h4>${section.indices.map(stepButton).join("") || '<p>此单元尚无完整归属的步骤记录。</p>'}</section>`).join("")}</nav></section>
    <div class="experiment-reader-story" role="region" aria-label="步骤说明"><div class="experiment-steps-toolbar"><h3>具体操作说明</h3>${steps.length?'<button type="button" class="text-button" data-expand-steps aria-expanded="false">展开全部步骤</button>':""}</div>${steps.map((step,index)=>`<details class="experiment-step-record" data-experiment-step-panel="${index}" data-step-start="${step.start_global_ms==null?"":Number(step.start_global_ms)}" data-step-end="${step.end_global_ms==null?"":Number(step.end_global_ms)}" ${index?"hidden":"open"}><summary><span class="experiment-step-number">${String(index+1).padStart(2,"0")}</span><span class="experiment-step-heading"><strong>${esc(experimentStepTitle(step))}</strong><small title="原片时间 ${timecode(step.start_global_ms)} — ${timecode(step.end_global_ms)}">原片 ${readerClock(step.start_global_ms)} — ${readerClock(step.end_global_ms)}</small></span><span class="experiment-step-affordance">查看说明 ${icon("chevron")}</span></summary><div class="experiment-step-full">${hasVideo ? `<div class="experiment-step-play"><button type="button" class="text-button" data-step-seek="${Math.max(0,(Number(step.start_global_ms)-Number(experiment.start_ms))/1000)}">${icon("video")}从这一步播放</button></div>` : ""}${experimentStepDescription(step)}<details class="step-evidence"><summary>查看此步骤的原始分析记录</summary><p>原片定位区间：${timecode(step.start_global_ms)} — ${timecode(step.end_global_ms)}</p><p>对象记录：${esc((step.objects || []).join("、"))} · 模型置信度：${esc(step.confidence ?? "未记录")}（非实测准确率）</p>${evidenceParagraphs(evidenceSentences(step.current_step || step.observed_action))}${evidenceParagraphs(evidenceSentences(step.physical_change))}${evidenceParagraphs(evidenceSentences(step.next_step))}${(step.source_operation_records || []).map(record=>`<p>原事件 ${esc(record.event_id)} 的整段审阅描述：</p>${evidenceParagraphs(evidenceSentences(record.current_step))}`).join("")}${step.source_next_step ? `<p>原后续判断（尚未定位）：${esc(step.source_next_step.text || "未知")}</p>` : ""}</details></div></details>`).join("") || '<p>步骤仍待整理，已保存的视频可以播放。</p>'}</div></aside></div></div>
    <dialog class="reader-dialog" aria-label="步骤阅读"><form method="dialog"><button class="dialog-close" aria-label="关闭阅读区">${icon("x")}</button></form></dialog>
    <div class="reader-review-entry"><span>需要核对录像范围或分析依据？</span><button type="button" class="text-button" data-reader-evidence>查看本段分析记录 ${icon("arrow")}</button></div><dialog class="reader-evidence-dialog" aria-label="本段分析记录"><form method="dialog"><button class="dialog-close" aria-label="关闭分析记录">${icon("x")}</button></form><h2>本段分析记录</h2><section class="reader-coverage-details">
    <div class="reader-result-state" aria-label="本段结果状态"><span><small>录像范围</small><strong>${experiment.completion_status==="ongoing_at_recording_end"?"已到录像末尾，实验待续":workflowCompletionLabel(experiment)}</strong></span><span><small>步骤理解</small><strong>${experiment.boundary_extension_requires_step_review?"新增画面的步骤待核验":steps.length?`${number(steps.length)} 条已有记录`:"尚未生成"}</strong></span></div>
    ${reviewGaps.length?`<details class="operation-coverage"><summary>有 ${number(reviewGaps.length)} 段时间尚无操作记录，可回看核对</summary><p>这些区间可能包含操作、等待或遮挡，不能仅凭无记录判断没有实验。${experiment.operation_review_accepted?'已有证据已按操作整理。':''}</p><div class="operation-gap-list">${reviewGaps.map(g=>`<button class="compact-button" type="button" data-review-gap="${Number(g.start_ms)}">${readerClock(g.start_ms)} — ${readerClock(g.end_ms)} ${icon("video")}</button>`).join('')}</div></details>`:''}
    ${experiment.completion_status === "ongoing_at_recording_end" || experiment.boundary_extension_requires_step_review ? `<details class="experiment-reader-notices"><summary>查看本段范围与核验说明</summary>${experiment.completion_status === "ongoing_at_recording_end" ? '<p class="analysis-readiness-note">本段保留到现有录像末尾，实验员仍在操作。后续录像尚未纳入，不能据此认定实验完成。</p>' : ""}${experiment.boundary_extension_requires_step_review ? '<p class="analysis-readiness-note">后续视频已保留，新增画面的步骤核验仍未完成。下方操作目录仅列出已有审核记录。</p>' : ""}</details>` : ""}
    </section>
    ${uncertainties.length ? `<details class="experiment-reader-limits"><summary>阅读时需要留意的细节</summary><p>以下限制会影响对应动作的判断。</p>${experimentLimitations(uncertainties)}</details>` : ""}
    <details class="experiment-reader-audit"><summary>查看片段分析原始记录</summary><p>原模型命名：${esc(experiment.name)}</p>${evidenceParagraphs(evidenceSentences(experiment.summary || "暂无模型摘要"))}</details></dialog>
  </article>`;
}

function bindExperimentReaders(root = main) {
  root.querySelectorAll("[data-experiment-group]").forEach(select=>select.addEventListener("change",()=>{ location.hash=select.value; }));
  root.querySelectorAll("[data-reader-evidence]").forEach(button=>button.addEventListener("click",()=>button.closest(".experiment-reader").querySelector(".reader-evidence-dialog").showModal()));
  root.querySelectorAll(".experiment-reader").forEach(card=>{
    if (card.dataset.readerBound) return;
    card.dataset.readerBound="true";
    const records=[...card.querySelectorAll("[data-experiment-step-panel]")], buttons=[...card.querySelectorAll("[data-experiment-step]")];
    const companion=card.querySelector(".experiment-reader-companion"), story=card.querySelector(".experiment-reader-story");
    const directory=card.querySelector("[data-reader-directory]"), toggle=card.querySelector("[data-toggle-directory]"), unit=card.querySelector("[data-reader-unit]");
    const follow=card.querySelector("[data-reader-follow]"), expand=card.querySelector("[data-expand-steps]");
    const videos=[...card.querySelectorAll(".preview-player video")];
    const primary=card.querySelector(".experiment-primary-video video") || videos[0];
    const intervals=records.map(record=>({start:record.dataset.stepStart===""?NaN:Number(record.dataset.stepStart),end:record.dataset.stepEnd===""?NaN:Number(record.dataset.stepEnd)}));
    let selectedStep=0, activeVideo=primary, cancelSeek=()=>{};
    const showDirectory=value=>{directory.hidden=!value;toggle.setAttribute("aria-expanded",String(value));toggle.textContent=value?"收起目录":"查看目录";};
    const filterDirectory=()=>card.querySelectorAll("[data-reader-group]").forEach(group=>{group.hidden=unit.value!=="all"&&group.dataset.readerGroup!==unit.value;});
    const setExpanded=value=>{
      card.dataset.stepsExpanded=String(value);
      records.forEach((record,index)=>{record.hidden=!value&&index!==selectedStep;record.open=value||index===selectedStep;});
      if(expand){expand.textContent=value?"收起全部步骤":"展开全部步骤";expand.setAttribute("aria-expanded",String(value));}
    };
    const selectStep=(index,manual=false)=>{
      if(!records[index])return;
      selectedStep=index;if(manual)follow.checked=false;
      setExpanded(false);
      buttons.forEach(button=>button.setAttribute("aria-pressed",String(Number(button.dataset.experimentStep)===index)));
      card.querySelector("[data-reading-counter]").textContent=`正在阅读 ${index+1} / ${records.length}`;
      card.querySelector("[data-reader-prev]").disabled=index===0;
      card.querySelector("[data-reader-next]").disabled=index===records.length-1;
      if(unit.value!=="all") {unit.value=buttons.find(button=>Number(button.dataset.experimentStep)===index)?.closest("[data-reader-group]")?.dataset.readerGroup || "all";filterDirectory();}
    };
    const revealStory=()=>{
      const heading=companion.querySelector(".reader-controls");
      const top=story.getBoundingClientRect().top-companion.getBoundingClientRect().top+companion.scrollTop-heading.getBoundingClientRect().height;
      companion.scrollTo({top:Math.max(0,top),behavior:"auto"});
    };
    const updatePlayback=()=>{
      if(!activeVideo)return;
      const globalTime=Number(card.dataset.readerStart)+activeVideo.currentTime*1000;
      const matches=readerPlaybackMatches(intervals,globalTime);
      buttons.forEach(button=>button.classList.toggle("is-playing",!activeVideo.paused&&matches.includes(Number(button.dataset.experimentStep))));
      const label=activeVideo.getAttribute("aria-label") || "视频";
      const suffix=matches.length?`对应步骤 ${matches.map(index=>index+1).join("、")}`:"这段画面暂无对应步骤记录";
      card.querySelector("[data-reader-playing]").textContent=`${label} · ${activeVideo.readyState<2?"正在定位画面":activeVideo.paused?"当前画面":"正在播放"} · 原片 ${readerClock(globalTime)} · ${suffix}`;
      if(follow.checked&&matches.length&&!matches.includes(selectedStep)&&card.dataset.stepsExpanded!=="true") {selectStep(matches[0]);if(directory.hidden)companion.scrollTo({top:0,behavior:"auto"});}
    };
    const seek=(seconds,play=false)=>{
      const video=activeVideo || primary;
      if(!video||!Number.isFinite(seconds))return;
      cancelSeek();
      if(play)follow.checked=true;
      cancelSeek=readerSeekVideo(video,seconds,play?()=>video.closest(".preview-player").querySelector(".preview-play").click():null,updatePlayback);
    };
    toggle.addEventListener("click",()=>showDirectory(directory.hidden));
    card.querySelectorAll("[data-review-gap]").forEach(button=>button.addEventListener("click",()=>{
      follow.checked=false;
      seek((Number(button.dataset.reviewGap)-Number(card.dataset.readerStart))/1000,true);
      primary?.scrollIntoView({block:"center",behavior:"smooth"});
    }));
    unit.addEventListener("change",()=>{filterDirectory();showDirectory(true);follow.checked=false;});
    buttons.forEach(button=>button.addEventListener("click",()=>{
      const index=Number(button.dataset.experimentStep);selectStep(index,true);showDirectory(false);
      seek((intervals[index].start-Number(card.dataset.readerStart))/1000);revealStory();
      records[index].querySelector("summary").focus({preventScroll:true});
    }));
    for(const [selector,delta] of [["[data-reader-prev]",-1],["[data-reader-next]",1]])card.querySelector(selector).addEventListener("click",()=>{selectStep(selectedStep+delta,true);showDirectory(false);revealStory();});
    expand?.addEventListener("click",()=>{follow.checked=false;setExpanded(card.dataset.stepsExpanded!=="true");});
    follow.addEventListener("change",()=>{if(follow.checked){setExpanded(false);showDirectory(false);updatePlayback();}});
    records.forEach((record,index)=>record.addEventListener("toggle",()=>{
      if(record.open&&!record.hidden&&card.dataset.stepsExpanded!=="true"&&index!==selectedStep)selectStep(index,true);
    }));
    card.querySelectorAll("[data-step-seek]").forEach(button=>button.addEventListener("click",()=>{
      const index=button.closest("[data-experiment-step-panel]")?.dataset.experimentStepPanel;
      if(index!==undefined)selectStep(Number(index));
      seek(Number(button.dataset.stepSeek),true);
    }));
    videos.forEach(video=>{
      video.addEventListener("play",()=>{activeVideo=video;videos.forEach(other=>{if(other!==video)other.pause();});updatePlayback();});
      for(const event of ["timeupdate","seeked","pause","ended"])video.addEventListener(event,()=>{if(video===activeVideo)updatePlayback();});
    });
    const dialog=card.querySelector(".reader-dialog"), focus=card.querySelector("[data-reader-focus]");
    const home=companion.parentElement;
    dialog.querySelector(".dialog-close").addEventListener("click",event=>{event.preventDefault();dialog.close();});
    dialog.addEventListener("keydown",event=>{if(event.key==="Escape"){event.preventDefault();dialog.close();}});
    focus.addEventListener("click",()=>{
      if(dialog.open){dialog.close();return;}
      showDirectory(false);dialog.append(companion);focus.textContent="返回视频";focus.setAttribute("aria-expanded","true");dialog.showModal();companion.scrollTop=0;
    });
    dialog.addEventListener("close",()=>{home.append(companion);focus.textContent="扩大阅读区";focus.setAttribute("aria-expanded","false");focus.focus({preventScroll:true});});
  });
}

function nextStepLabel(step) {
  const status = step?.next_step_status || step?.next_step_evidence?.status || "unknown";
  return status === "observed" ? "已观察后续" : status === "inferred" ? "预测后续" : "后续未知";
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
  const nextStatus = mllm.next_step_status || mllm.next_step_evidence?.status || "unknown";
  const nextLabel = nextStatus === "observed" ? "已观察后续动作" : nextStatus === "inferred" ? "预测下一步" : "后续未知";
  const next = productEvidenceText(mllm.next_step, "没有足够证据支持下一步");
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
  const selectionKey = materialSelectionKey(event);
  const selected = state.selectedMaterials.has(selectionKey);
  const uncertainty = nextStatus === "unknown" || /不足|无法|不能|待确认|不确定|未明确/.test(next);
  return `<article class="material-card product-material-card ${selected ? "is-selected" : ""}" data-material-card="${esc(event.event_id)}" data-material-selection-key="${esc(selectionKey)}">
    <header class="material-card-header"><div><span class="material-sequence">关键素材 ${String(index + 1).padStart(2,"0")}</span><h2>${esc(ACTION_LABELS[event.action_type] || "关键实验动作")}</h2><p>${timecode(Number(event.start_us)/1000)} → ${timecode(Number(event.end_us)/1000)}</p></div><div class="material-card-actions"><button class="material-focus-button" type="button" data-material-focus="${esc(event.event_id)}">${icon("video")}聚焦查看</button><label class="material-select"><input type="checkbox" data-material-select="${esc(event.event_id)}" ${selected ? "checked" : ""}/><span>选择</span></label><span class="material-support-badge ${dualView ? "trusted" : "partial"}">${dualView ? icon("check") : icon("image")}${dualView ? "双视角印证" : "主要视角清晰"}</span></div></header>
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

function incompleteReportView(data, title) {
  const groups = data.experiments || [];
  return `<section class="incomplete-report"><h2>${esc(title)}</h2><p>${esc(friendlyFailureReason(data))}</p><p id="daily-status-section"><strong>当前为阶段记录，完整日报与正式 PDF / JSON 尚未生成。</strong>以下条目用于查找已有内容，不确认实验动作。</p><div class="attention-actions">${data.links?.partial_report ? `<a class="primary-button" href="${esc(data.links.partial_report)}" target="_blank" rel="noopener">打开阶段结果报告（HTML）</a>` : ""}<a class="secondary-button" href="${experimentRecordRoute(data)}">查看已保存片段</a><a class="secondary-button" href="#/tasks">查看停止记录</a></div>${groups.length ? `<ol class="partial-report-timeline" id="daily-timeline-section">${groups.map((group,index)=>`<li><a href="${experimentRecordRoute(data)}?group=${encodeURIComponent(group.folder || "")}">片段 ${index+1} · ${timecode(group.start_ms)} → ${timecode(group.end_ms)}</a><span>${number(group.steps?.length)} 个模型步骤 · ${number(group.uncertainties?.length)} 项证据限制</span></li>`).join("")}</ol>` : ""}</section>`;
}

function stageDailyReportView(data) {
  const groups = data.experiments || [];
  return `<section class="stage-daily-reader" id="daily-report-cover"><header><p class="eyebrow">本次实验记录</p><h2>已记录的实验过程</h2><p>以下内容来自已保存的步骤说明。实验是否结束和整体验收状态分别保留；未记录的操作仍需核对。</p></header><div class="stage-daily-groups" id="daily-timeline-section">${groups.map((group,index)=>`<article><header><span class="daily-index">${String(index+1).padStart(2,"0")}</span><div><h3>${esc(group.name)}</h3><p>${readerClock(group.start_ms)} — ${readerClock(group.end_ms)} · ${workflowCompletionLabel(group)}</p></div><a class="secondary-button" href="${experimentRecordRoute(data)}?group=${encodeURIComponent(group.folder)}">回看视频 ${icon("video")}</a></header><ol>${(group.steps||[]).map(step=>`<li><time>${readerClock(step.start_global_ms)}</time><strong>${esc(experimentStepTitle(step))}</strong></li>`).join("")}</ol></article>`).join("")}</div></section>`;
}

function dailyReportView(data) {
  const report = data.daily_report || {};
  const overview = report.overview || {};
  const alignment = report.alignment_summary || {};
  const actions = report.action_summary || [];
  const timeline = report.experiment_timeline || [];
  const quality = report.quality_acceptance || {};
  const qualityPassed = quality.passed === true || (quality.passed == null && overview.evidence_package_eval_passed);
  const evidenceLevelLabel = quality.evidence_level === "dataset_measured"
    ? "本实验质量已测量"
    : quality.evidence_level === "partially_measured"
      ? "部分质量已测量"
      : quality.evidence_level == null && overview.evidence_package_eval_passed
        ? "历史证据包已验收 · 质量层级未记录"
      : "结构已验收 · 本实验准确率未测量";
  if (!report.report_id && data.links?.partial_daily_report) return stageDailyReportView(data);
  if (!report.report_id) return `<section class="panel" id="daily-report-cover">${archiveProcessStopped(data) ? incompleteReportView(data,"本次处理未生成实验日报") : productState("progress", "clock", "实验日报正在准备", "完成前序分析后，日报会自动显示在这里。", `<a class="secondary-button" href="#/tasks">查看任务进度</a>`)}</section>`;
  return `<section class="daily-report-cover" id="daily-report-cover"><div title="实验编号：${esc(report.experiment_id)}"><p class="eyebrow">实验室日报</p><h2>${esc(report.report_date)} · 实验摘要</h2><p>${esc(productExperimentName(report.experiment_id))}</p></div><span class="badge">${overview.evidence_package_eval_passed && qualityPassed ? esc(evidenceLevelLabel) : "内容整理中"}</span></section>
  <section class="status-grid">${statusCard("video","拍摄视角",number(overview.input_view_count),`${number(overview.first_person_views)} 第一人称 + ${number(overview.third_person_views)} 第三人称`)}${statusCard("flask","实验片段",number(overview.experiment_group_count),"独立 / 连续实验")}${statusCard("image","关键素材",number(overview.key_event_count),"")}${statusCard("activity","状态变化",number(overview.physical_change_count),"")}</section>
  <section class="panel" id="daily-timeline-section"><header class="panel-heading"><div><h2>实验时间线与步骤理解</h2><p>时间均为统一实验相对时间；每项保留证据事件与跨视角素材链接。</p></div></header>${dailyReportTimelineVisual(timeline,actions)}<div class="daily-timeline">${timeline.map((group,index)=>`<article><span class="daily-index">${String(index+1).padStart(2,"0")}</span><div><header><h3>${esc(group.experiment_name)}</h3><span class="badge">${workflowLabel(group)} · ${workflowCompletionLabel(group)}</span></header><p class="timecode">${esc(group.start_timecode)} → ${esc(group.end_timecode)}</p><p>${esc(group.overall_summary || "无模型摘要")}</p><details><summary>查看 ${number(group.steps?.length)} 个细粒度步骤</summary><div class="step-list">${(group.steps||[]).map((step)=>`<article class="step-card"><span class="step-number">步骤 ${esc(step.step_index)}<small>${esc(step.start_timecode)}<br/>${esc(step.end_timecode)}</small></span><div class="step-body"><strong>当前：${esc(step.current_step || "未说明")}</strong><p class="step-next"><b>${esc(nextStepLabel(step))}：</b>${esc(step.next_step || "证据不足")}</p><span class="step-meta">对象：${esc((step.objects||[]).join("、") || "未明确")} · ${number(step.supporting_views?.length)} 路证据</span></div></article>`).join("")}</div></details></div></article>`).join("")}</div></section>
  <section class="panel" id="daily-status-section"><header class="panel-heading"><div><h2>报告状态</h2><p>用于判断当前报告的信息是否完整。</p></div></header><table class="metric-table report-summary-table"><tbody><tr><td>多视角同步</td><td>${number(alignment.aligned)}/${number(alignment.view_count)} 路画面已对齐</td></tr><tr><td>需要留意</td><td>${number(report.uncertainties?.length)} 组不确定项，${number(report.contradictions?.length)} 项视角差异</td></tr><tr><td>内容完整性</td><td>${overview.evidence_package_eval_passed ? "实验过程、素材与报告已完整关联" : "部分关联信息仍在整理"}</td></tr></tbody></table></section>`;
}

function professionalReportsView(data) {
  const report = data.daily_report || {};
  const quality = data.quality_acceptance || {};
  const partial = !report.report_id;
  const reportFiles = [
    {key:partial ? "partial_pdf" : "daily_report_pdf", format:"PDF", title:partial ? "实验报告 · 阶段版" : "专业实验报告", copy:"按实验过程整理操作与证据，适合阅读和打印。"},
    {key:partial ? "partial_json" : "daily_report_json", format:"JSON", title:"可追溯实验数据", copy:"保留步骤、事件引用、时间、用量与控制文件版本。"},
    {key:partial ? "partial_daily_report" : "daily_report_html", format:"HTML", title:"实验室日报", copy:"在独立页面阅读本次已记录的实验过程。"},
  ].filter(item=>data.links?.[item.key]);
  const extraFiles = [
    ["project_annotation_comparison", "项目标注与微调对照"],
    ["daily_report_html", "网页版报告"],
    ["daily_report_markdown", "Markdown 报告"],
  ].filter(([key])=>data.links?.[key]);
  const status = partial && reportFiles.length ? "阶段版 · 质量待核对" : quality.passed === true || report.report_id ? "报告已生成" : archiveProcessStopped(data) ? "本次未生成" : "报告生成中";
  const emptyReports = archiveProcessStopped(data) ? incompleteReportView(data,"本次处理未生成专业报告") : productState("progress", "clock", "专业报告正在生成", "分析完成后，PDF 与 JSON 报告会显示在这里。", `<a class="secondary-button" href="#/tasks">查看任务进度</a>`);
  return `<section class="professional-report-hero" id="professional-report-hero"><div><span class="report-hero-icon">${icon("file")}</span><p class="eyebrow">专业报告</p><h2>日报与报告</h2><p>PDF 用于阅读与分享，JSON 用于结构化存档和后续系统对接。</p></div><span class="badge">${status}</span></section><section class="report-file-grid" id="professional-files">${reportFiles.map((item)=>`<a class="report-file-card" target="_blank" href="${esc(data.links[item.key])}"><span class="file-format">${item.format}</span><div><strong>${item.title}</strong><p>${item.copy}</p><small>${icon("arrow")}打开报告</small></div></a>`).join("") || emptyReports}</section>${extraFiles.length ? `<section class="panel report-extra-files"><header class="panel-heading"><div><h2>附加结果与阅读格式</h2><p>查看项目核对结果，或选择其他报告格式。</p></div></header><div class="result-links">${extraFiles.map(([key,label])=>`<a target="_blank" href="${esc(data.links[key])}">${icon("file")}${label}</a>`).join("")}</div></section>` : ""}`;
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

function metricTokenValue(bucket, field) {
  const value = bucket?.[field];
  if (value == null) return bucket?.executed_call_count === 0 ? "0" : "未知";
  return number(value);
}

function metricsView(data) {
  const metrics = data.metrics || {};
  const stages = metrics.stage_durations || [];
  const webIngest = metrics.web_ingest || metrics.nas_index_ingest;
  const endToEnd = metrics.web_end_to_end || metrics.fixed_benchmark_end_to_end;
  const retention = webIngest?.retention_mode === "nas_only" ? "上传并保存至 NAS" : "上传并按存储配置留存";
  const extraRows = `${webIngest?.duration_seconds != null ? `<tr><td>input_ingest</td><td>${metrics.web_ingest ? retention : "NAS 索引输入准备/复用"}</td><td>${duration(webIngest.duration_seconds)}</td></tr>` : ""}${endToEnd?.total_duration_seconds != null ? `<tr><td>end_to_end</td><td>${archiveProcessStopped(data) ? "首次提交至本次停止（含等待与复跑）" : "首次提交至当前归档阶段（含等待与复跑）"}</td><td>${duration(endToEnd.total_duration_seconds)}</td></tr>` : ""}`;
  const quality = data.quality_acceptance || {};
  const boundary = quality.experiment_boundaries || {};
  const materials = quality.key_materials || {};
  const materialEvaluation = keyMaterialEvaluationSummary(data);
  const verification = data.key_material_verification || {};
  const performance = metrics.preprocessing_display || {};
  const fullColdStart = performance.full_cold_start || {};
  const currentRun = performance.current_run || {};
  const statusLabel = quality.evidence_level === "dataset_measured"
    ? "结构验收与本实验质量测量均已通过"
    : quality.evidence_level === "partially_measured"
      ? "结构已验收；仅部分质量指标具备独立真值"
      : quality.passed
        ? "结构与媒体已验收；本实验准确率未逐任务测量"
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
    key_material_category_index: "关键素材实验/六类目录索引 JSON",
    metrics: "耗时与 Token JSON",
    acceptance: "验收汇总 JSON",
    quality_acceptance: "自动质量验收 JSON",
    evidence_package_eval: "证据包结构/媒体验收 JSON",
    key_material_recall_eval: "关键素材 Precision / Recall 评估 JSON",
    final_key_material_annotation: "关键素材对象框与二次复核 JSON",
    run_provenance: "代码、配置、模型与证据溯源 JSON",
  };
  const verificationSection = `<section class="panel"><header class="panel-heading"><div><h2>关键素材模型质量账本</h2><p>逐事件显示实际执行的视觉模型、对象框置信度、二次复核耗时和明确保留的不确定性；没有人工真值时不展示伪造的准确率。</p></div></header><table class="metric-table"><tbody><tr><td>复核账本</td><td>${verification.available ? "可用" : "历史档案未生成"}</td></tr><tr><td>事件 / 渲染视角</td><td>${number(verification.event_count)} / ${number(verification.rendered_view_count)}</td></tr><tr><td>仅渲染动作参与对象</td><td>${number(materials.participant_only_annotation_pass_count)} / ${number(materials.event_count)}（${materials.participant_only_annotation_gate_passed ? "门禁通过" : "未通过或无账本"}）</td></tr><tr><td>动作参与对象可见性</td><td>${number(materials.action_participant_visibility_pass_count)} / ${number(materials.event_count)}（${materials.action_participant_visibility_gate_passed ? "门禁通过" : "未通过或无账本"}）</td></tr><tr><td>决策状态</td><td>${esc(Object.entries(verification.decision_status_counts || {}).map(([key,value])=>`${key}: ${value}`).join("；") || "无")}</td></tr><tr><td>实际模型执行</td><td>${esc(Object.entries(verification.model_execution_counts || {}).map(([key,value])=>`${MODEL_LABELS[key] || key}: ${value}`).join("；") || "无")}</td></tr><tr><td>二次复核推理耗时</td><td>YOLO-World ${duration(verification.timing?.open_vocabulary_inference_seconds)}；Grounding DINO ${duration(verification.timing?.grounding_dino_inference_seconds)}；总墙钟 ${duration(verification.timing?.wall_seconds)}</td></tr><tr><td>保留不确定性的事件</td><td>${number(verification.uncertain_event_count)}</td></tr><tr><td>预算使用</td><td>${number(verification.budget?.admitted_event_count)} 事件 / ${number(verification.budget?.admitted_view_count)} 视角；延后 ${number(verification.budget?.deferred_count)}</td></tr></tbody></table></section>`;
  return `<section class="panel"><header class="panel-heading"><div><h2>耗时口径</h2><p>${performanceSummary} ${reuseNote}</p></div></header><div class="performance-compare"><article><small>历史完整冷启动预处理</small><strong>${duration(fullColdStart.seconds)}</strong><span>${esc(fullColdStart.includes || "预检 + 对齐 + 全量粗扫 + 有界精扫 + 边界审计；不含模型理解")}</span></article><article><small>当前归档运行总耗时</small><strong>${duration(currentRun.total_seconds ?? metrics.total_duration_seconds)}</strong><span>${currentRun.reused_validated_cv_ledgers ? "复用已验收 CV 账本，重新生成理解/媒体/证据包" : "以本次运行账本为准"}</span></article><article><small>当前运行预处理</small><strong>${duration(currentRun.preprocessing_seconds)}</strong><span>${currentRun.reused_validated_cv_ledgers ? "不是冷启动基准" : "当前运行实际值"}</span></article></div><table class="metric-table"><thead><tr><th>阶段</th><th>说明</th><th>耗时</th></tr></thead><tbody>${extraRows}${stages.map((stage)=>`<tr><td>${esc(stage.stage)}</td><td>${esc(STAGE_LABELS[stage.stage] || stage.stage)}</td><td>${duration(stage.duration_seconds)}</td></tr>`).join("")}</tbody></table></section><section class="panel"><header class="panel-heading"><div><h2>质量验收</h2><p>${esc(materialEvaluation.displayNote)}</p></div></header><table class="metric-table"><tbody><tr><td>总体状态</td><td>${esc(statusLabel)}</td></tr><tr><td>实验检出 Precision / Recall</td><td>${boundaryAccuracy}</td></tr><tr><td>关键素材 Precision / Recall</td><td>${esc(materialEvaluation.recallLabel)}</td></tr><tr><td>边界通过率 / 连续性准确率</td><td>${boundaryContinuity}</td></tr><tr><td>证据包结构与媒体</td><td>${boundary.evidence_package_eval_passed || materials.evidence_package_eval_passed ? "通过自动验收" : "未通过或无验收记录"}</td></tr><tr><td>关键素材类别覆盖</td><td>${esc(materialEvaluation.categoryCoverageLabel)}</td></tr><tr><td>第一/第三人称成套素材</td><td>${number(materials.dual_view_material_count)} / ${number(materials.event_count)}（${percent(materials.dual_view_material_rate)}）</td></tr><tr><td>双侧共同佐证动作</td><td>${number(materials.cross_view_supported_count)} / ${number(materials.event_count)}（${percent(materials.cross_view_supported_rate)}）</td></tr><tr><td>双视角关联可审计</td><td>${number(materials.cross_view_or_explicit_uncertainty_count)} / ${number(materials.event_count)}</td></tr></tbody></table></section>${verificationSection}<section class="panel"><header class="panel-heading"><div><h2>模型理解与验收文件</h2><p>这些链接指向当前实验已保存的文件；验收状态以本页质量检查记录为准。缺失文件不会显示为可点击链接。</p></div></header><div class="result-links" style="padding:20px">${Object.entries(links).map(([key,label])=>data.links[key]?`<a target="_blank" href="${esc(data.links[key])}">${icon("file")}${label}</a>`:"").join("")}</div></section>`;
}

function archiveViewBusy() {
  const active = document.activeElement;
  const editing = main.contains(active) && (["INPUT", "TEXTAREA", "SELECT"].includes(active?.tagName) || active?.isContentEditable);
  return Boolean(document.querySelector("dialog[open]")) || editing
    || [...main.querySelectorAll("video, audio")].some((media)=>!media.paused && !media.ended);
}

function archiveRefreshNotice(message) {
  const notice = document.querySelector("#archive-sync-notice");
  if (notice) { notice.hidden = false; notice.textContent = message; }
}

function refreshOpenArchive() {
  const view = state.archiveView;
  if (!view?.refreshPending || view.loading || view.hash !== location.hash) return;
  const busy = archiveViewBusy();
  archiveRefreshNotice(busy ? "实验结果已更新，将在当前播放或编辑结束后自动同步。" : "正在同步最新实验结果，当前显示上次载入的内容。");
  if (!busy) return renderArchive(view.name, view.tab, view.staging, true);
}

function stageSnapshotVersion(snapshot) {
  return JSON.stringify([snapshot?.status?.stage, (snapshot?.stage_receipts || []).map(item=>[item.stage,item.status,item.completed_at])]);
}

function resultWorkspaceContent(data, tab, staging, notice, experimentResults) {
  const stopped = archiveProcessStopped(data);
  const label = data.quality_acceptance?.passed === false ? "处理已结束 · 结果待核对" : "处理已停止 · 已完成内容已保留";
  const status = stopped ? `<div class="result-state-line" role="status"><span>${icon("activity")}<strong>${label}</strong><small>可查看已有视频与报告，完整质量检查尚未通过。</small></span><a href="${experimentRecordRoute(data,"metrics")}">查看分析记录 ${icon("arrow")}</a></div>` : notice;
  const insights = tab === "metrics" ? (globalThis.VisionCortexRunInsights?.render(data,{compact:false,labels:STAGE_LABELS,href:experimentRecordRoute(data,"metrics")}) || "") : "";
  const content = tab === "speech" ? '<section id="experiment-speech" class="panel speech-results"></section>'
    : tab === "materials" ? materialsView(data)
    : tab === "reports" ? `${professionalReportsView(data)}<div id="report-refresh-controls"></div>${dailyReportView(data)}`
    : tab === "metrics" ? `<div class="analysis-record-workspace">${insights}${resultReviewPanel(data)}${stopped ? experimentAttentionPanel(data,tab) : ""}${componentResultCards(data,data.staging_run_id || data.name,staging)}${movementScreeningView(data)}${metricsView(data)}</div>`
    : `${experimentResults}${experimentResultTools(data,staging)}`;
  return `<div class="page result-workspace result-workspace--${esc(tab)}"><p id="archive-sync-notice" class="analysis-readiness-note" role="status" hidden></p>${resultHeader(data,tab,staging)}${tab === "metrics" ? "" : status}${stageDeliveryView(data)}${content}</div>`;
}

async function renderArchive(name, tab = "experiments", staging = false, background = false) {
  const requestedHash = location.hash;
  const requestId = state.archiveViewRequestId = (state.archiveViewRequestId || 0) + 1;
  const view = {name, tab, staging, hash:requestedHash, loading:true, refreshPending:false};
  state.archiveView = view;
  const isCurrent = () => location.hash === requestedHash && state.archiveViewRequestId === requestId;
  setChrome("archive", productExperimentName(name));
  if (!background) main.innerHTML = pageSkeleton(`正在读取${archiveLabel()}`);
  try {
    const data = staging ? await loadArchive(name, true) : await loadArchiveView(name, tab);
    if (!isCurrent()) return;
    // Playback or editing may have started while the background request was in flight.
    if (background && archiveViewBusy()) { view.refreshPending = true; return; }
    const status = data.observability?.status || {};
    view.stageVersion = stageSnapshotVersion(data.observability);
    const stage = status.stage;
    const pending = stage ? stage !== "completed" : staging;
    const stopped = archiveProcessStopped(data);
    const currentStarted = Date.parse(status.updated_at) - Number(status.elapsed_seconds || 0) * 1000;
    const understandingPending = pending && !(data.observability?.stage_receipts || []).some((item) => item.stage === "semantic_refinement" && item.status === "completed" && (stopped || Date.parse(item.completed_at) >= currentStarted - 1000));
    const notice = pending && !stopped ? `<p class="analysis-readiness-note" role="status">分析进行中，已完成的阶段产出可查看，后续结果将继续更新。 <a href="#/tasks">查看任务进度</a></p>` : "";
    if (tab === "materials" && routeQuery().get("candidate")) state.retainedMaterialFilters = {scope:"all", group:"all", action:"all"};
    const experimentResults = tab === "experiments" ? experimentGroupBrowser(data,understandingPending,routeQuery().get("group")||"") : "";
    main.innerHTML = resultWorkspaceContent(data, tab, staging, notice, experimentResults);
    view.refreshPending = false;
    if (staging && stopped) {
      const controls = document.createElement("div");
      controls.className = "speech-downloads";
      const choices = [["reports", "刷新报告"]];
      if (!(data.experiments || []).length && (data.observability?.components || []).some(item => item.key === "understanding" && ["completed","partial","failed"].includes(item.state))) choices.unshift(["understanding", "刷新录音理解"]);
      for (const [scope, label] of choices) {
        const button = document.createElement("button");
        button.type = "button"; button.className = "secondary-button"; button.textContent = label;
        button.dataset.refreshStage = scope;
        button.addEventListener("click", async () => {
          controls.querySelectorAll("button").forEach(item => { item.disabled = true; });
          try {
            const result = await api(`/api/runs/${encodeURIComponent(name)}/refresh/${scope}`, {method:"POST"});
            toast("已提交所选阶段；完成后刷新此页查看成果。");
            const link = document.createElement("a"); link.href = "#/tasks"; link.textContent = "查看阶段任务";
            controls.append(link); button.textContent = `${label}已排队`;
            button.dataset.refreshRun = result.run_id;
          } catch (error) { toast(error.message, "error"); controls.querySelectorAll("button").forEach(item => { item.disabled = false; }); }
        });
        controls.append(button);
      }
      main.querySelector(tab === "reports" ? "#report-refresh-controls" : ".component-results")?.append(controls);
    }
    if (tab === "speech") await window.VisionCortexSpeech.render({
      container: document.querySelector("#experiment-speech"), name, staging,
      release: data.release_id, experiments: data.experiments || [], api, esc, toast, isCurrent,
    });
    if (!isCurrent()) return;
    if (staging) {
      const archivePrefix = `#/archive/${encodeURIComponent(data.name)}/`;
      main.querySelectorAll("a[href]").forEach((link) => {
        if (link.hash.startsWith(archivePrefix)) link.hash = `#/stage/${encodeURIComponent(name)}/${link.hash.slice(archivePrefix.length)}`;
      });
      document.querySelectorAll("[data-open-folder]").forEach((button)=>button.remove());
    }
    if (tab === "experiments") window.VisionCortexSpeech?.attach({main, data, name, staging});
    bindArchiveActions(data);
    globalThis.VisionCortexRunInsights?.bind(main,data,{labels:STAGE_LABELS});
    bindVideoPreviews();
    bindExperimentReaders();
    bindDetailAnchors();
    bindRerunActions(data);
    if (tab === "experiments" && data.next_cursor && !staging) {
      document.querySelector("#experiment-results")?.insertAdjacentHTML("afterend", `<div class="pagination-actions"><button class="secondary-button" id="load-more-experiments" type="button">加载更多实验片段</button></div>`);
      document.querySelector("#load-more-experiments")?.addEventListener("click", async (event) => {
        const button = event.currentTarget;
        button.disabled = true;
        try { await loadArchiveExperiments(name, data.next_cursor); if (isCurrent()) await renderArchive(name, "experiments"); }
        catch (error) { if (isCurrent()) toast(error.message, "error"); }
        finally { button.disabled = false; }
      });
    }
    if (tab === "materials") {
      bindMaterialFilters(data);
      bindRetainedMaterialFilters(data);
      const focus = routeQuery().get("focus");
      const candidate = routeQuery().get("candidate");
      if (candidate && !background) requestAnimationFrame(()=>{ if (isCurrent()) document.getElementById(`candidate-${candidate}`)?.scrollIntoView({block:"start"}); });
      if (focus && !background) requestAnimationFrame(()=>{ if (isCurrent()) void openLinkedMaterial(data,focus,isCurrent); });
    }
    requestAnimationFrame(() => {
      if (!isCurrent()) return;
      updateResultNavDensity();
      const group = routeQuery().get("group");
      if (tab === "experiments" && group && !background) document.getElementById(`experiment-${group}`)?.scrollIntoView({block:"start"});
    });
  } catch (error) {
    if (!isCurrent()) return;
    if (background) {
      view.refreshPending = true;
      archiveRefreshNotice("最新实验结果暂时无法同步，当前保留上次载入的内容；系统会自动重试。");
      return;
    }
    main.innerHTML = productState("error", "x", "无法读取该实验档案", error.message, `<button class="primary-button" type="button" data-retry-archive>重新加载</button><a class="secondary-button" href="#/experiments">返回实验记录</a>`);
    document.querySelector("[data-retry-archive]")?.addEventListener("click", ()=>renderArchive(name,tab,staging));
  } finally {
    view.loading = false;
  }
}

function bindDetailAnchors() {
  const buttons = [...document.querySelectorAll("[data-detail-section]")];
  buttons.forEach((button)=>button.addEventListener("click",()=>{
    const target = document.getElementById(button.dataset.detailSection);
    if (!target) { toast("该部分内容尚未生成。", "error"); return; }
    buttons.forEach((item)=>item.classList.toggle("active",item===button));
    target.scrollIntoView({behavior:"smooth",block:"start"});
  }));
}

async function rerunArchive(data, button) {
  if (data.staging_run_id) return retryRetainedRun(data.staging_run_id, button);
  if (button.disabled) return;
  const originalLabel = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `${icon("refresh")}正在创建分析任务…`;
  try {
    const previousRun = state.runs.find((run)=>run.experiment_id===data.name && run.source_collection_id);
    let created;
    if (previousRun?.source_collection_id) {
      created = await api(`/api/collections/${encodeURIComponent(previousRun.source_collection_id)}/runs`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({experiment_name:data.name})});
    } else {
      const manifestResponse = await fetch(`/api/archive-file?archive=${encodeURIComponent(data.name)}&path=${encodeURIComponent("JSON-Config-Files/original_upload_manifest.json")}`);
      if (!manifestResponse.ok) throw new Error("该历史实验没有可复用的原始输入清单，请从“新建实验”重新选择素材。");
      const uploadRecord = await manifestResponse.json();
      if (!uploadRecord?.manifest?.views?.length) throw new Error("原始输入清单不完整，请从“新建实验”重新选择素材。");
      created = await api("/api/runs/from-paths", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(uploadRecord.manifest)});
    }
    state.archiveCache.clear();
    showAcceptedRun(created, data.name);
    toast("已按原设置创建新的分析任务。", "success");
  } catch (error) {
    button.disabled = false;
    button.innerHTML = originalLabel;
    toast(error.message, "error");
  }
}

function bindRerunActions(data) {
  document.querySelectorAll("[data-rerun-archive]").forEach((button)=>button.addEventListener("click",()=>rerunArchive(data,button)));
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

function resultReviewPanel(data) {
  const review=data.result_review;
  if(!review?.available) return "";
  const canRefresh=Boolean(data.staging_run_id && archiveProcessStopped(data));
  const pending=review.findings||[];
  const names={unrecorded_interval:"尚无操作记录的区间",unfinished_boundary:"实验结束尚未确认",overlapping_operations:"操作时间重叠",continuity_unverified:"相邻实验承接待核对",references:"操作证据引用缺口"};
  const counts=Object.entries(names).map(([kind,label])=>({label,count:pending.filter(f=>f.kind===kind).length})).filter(x=>x.count);
  return `<section class="result-review-panel"><header><div><p class="eyebrow">最新结果检查</p><h2>${review.latest_check_current?(review.step_consistency_passed?"步骤引用与文字检查通过":"步骤记录仍需核对"):"当前结果尚未检查"}</h2><p>${review.checked_at?`检查时间 ${esc(new Date(review.checked_at).toLocaleString())}`:"更新结果后可重新检查"}${review.result_updated || review.original_quality_passed!==true?" · 完整实验验收尚未重跑":""}</p></div>${canRefresh?'<button class="secondary-button" type="button" data-result-check>检查最新结果</button>':''}</header><div class="result-check-status"><span><small>结果版本</small><strong>${esc(review.revision.slice(0,8))}</strong></span><span><small>原完整质量检查</small><strong>${review.original_quality_passed===true?"已通过":review.original_quality_passed===false?"未通过（历史记录）":"未记录"}</strong></span><span><small>全部操作是否已识别</small><strong>仍需视频核验</strong></span></div><details><summary>完整性检查 · ${number(pending.length)} 项待核对 ${counts.map(x=>`${x.label} ${x.count}`).join(' · ')}</summary><p>无记录可能是等待、遮挡或漏识别；重叠操作可能同时发生，时间相邻不能证明属于同一实验。补充观察不会自动变成已确认动作，也不会改变原验收结果。</p>${canRefresh && review.windows?.length?`<div class="gap-review-controls"><label>选择缺口分析区间<select data-gap-window aria-label="选择缺口分析区间">${review.windows.map(w=>`<option value="${esc(w.window_id)}">${esc(w.group_id)} · ${readerClock(w.start_ms)} — ${readerClock(w.end_ms)}</option>`).join('')}</select></label><button class="primary-button" type="button" data-gap-analyze>补充分析选定区间</button></div><p>每次最多 30 秒、8 张同步画面，使用原任务模型，可能产生模型费用。当前最多列出 256 个待分析窗口，未分析的其余时间继续保留为缺口。</p>`:''}${(review.gap_observations||[]).map(g=>`<section class="gap-observations"><h3>${esc(g.group_id)} · ${readerClock(g.start_ms)} — ${readerClock(g.end_ms)}</h3><p>补充观察 · ${number(g.sample_count)} 张采样画面 · 尚未纳入已审核操作</p>${g.status==='supplementary_observations'?(g.observations||[]).map(o=>`<details class="gap-observation"><summary>${esc(o.title)}</summary>${evidenceParagraphs(evidenceSentences(o.description))}<p class="muted">引用画面：${esc((o.frame_ids||[]).join("、"))}</p></details>`).join('')||'<p>采样画面不足以给出具体操作观察。</p>':'<p>本次补充分析未通过检查，原记录已保留。</p>'}${g.limitation?`<p class="muted">${esc(g.limitation)}</p>`:''}</section>`).join('')}</details></section>`;
}

async function submitResultReview(data, scope, target, button) {
  if(button.disabled) return;
  button.disabled=true;
  try {
    const params=new URLSearchParams({revision:data.result_review.revision});
    if(target) params.set("target",target);
    const result=await api(`/api/runs/${encodeURIComponent(data.staging_run_id)}/refresh/${scope}?${params}`,{method:"POST"});
    invalidateArchiveCache(data.name);showAcceptedRun(result,data.name);
  } catch(error) {toast(error.message,"error");button.disabled=false;}
}

function bindArchiveActions(data) {
  document.querySelectorAll("[data-result-check]").forEach(button=>button.addEventListener("click",()=>submitResultReview(data,"result_check",null,button)));
  document.querySelectorAll("[data-gap-analyze]").forEach(button=>button.addEventListener("click",()=>submitResultReview(data,"gap_review",document.querySelector("[data-gap-window]")?.value,button)));
  document.querySelectorAll("[data-copy-path]").forEach((button) => button.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(button.dataset.copyPath); toast("归档路径已复制。" ); }
    catch { toast(`请手工复制：${button.dataset.copyPath}`, "error"); }
  }));
  document.querySelectorAll("[data-open-folder]").forEach((button) => button.addEventListener("click", async () => {
    try { await api(`/api/archives/${encodeURIComponent(button.dataset.openFolder)}/open`, { method: "POST" }); toast("已在系统文件管理器中打开实验文件。"); }
    catch { toast("无法打开实验文件，请检查系统文件访问权限。", "error"); }
  }));
  document.querySelectorAll("[data-open-staging-folder]").forEach(button=>button.addEventListener("click",async()=>{
    button.disabled=true;
    try { await api(`/api/staging-runs/${encodeURIComponent(button.dataset.openStagingFolder)}/open`,{method:"POST"});toast("已请求打开产出文件夹，请在运行此应用的电脑上查看。"); }
    catch { toast("无法打开文件夹，可复制保存路径后在文件管理器中打开。","error"); }
    finally { button.disabled=false; }
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
  if (routeParts()[2] !== "materials" || event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement?.tagName)) return;
  if (document.activeElement?.isContentEditable) return;
  const dialog = document.querySelector("dialog[open]");
  if (dialog && !dialog.matches(".material-focus-dialog")) return;
  const video = dialog ? dialog.querySelector("[data-focus-video]")
    : document.querySelector(".material-product-details[open] .material-clip-details[open] video[data-material-video]");
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
    const direction = key === "p" ? "previous" : "next";
    if (dialog) {
      const button = dialog.querySelector(`[data-focus-nav="${direction}"]`);
      if (button && !button.disabled) button.click();
    } else activateAdjacentMaterial(video.dataset.materialVideo, direction, video.closest("[data-material-card]"));
  }
}

function globalMaterialSearchKey(query) {
  return JSON.stringify([query.trim(), state.archiveCacheEpoch || 0]);
}

async function loadGlobalMaterialSearch(query) {
  if (!query.trim()) return;
  const key = globalMaterialSearchKey(query);
  if (state.globalMaterialSearch?.key === key) return;
  const request = {key,status:"loading",items:[],total:0};
  state.globalMaterialSearch = request;
  const current = () => state.globalMaterialSearch === request && globalMaterialSearchKey(state.search) === key;
  try {
    const parameters = new URLSearchParams({q:query.trim(),material_ready:"true",limit:"5"});
    const payload = await api(`/api/key-events?${parameters}`);
    if (!current()) return;
    request.items = (payload.items || []).filter(eventHasAlignedDualViewMaterial);
    request.total = Number(payload.total_count || 0);
    request.status = "ready";
  } catch {
    if (current()) request.status = "error";
  } finally {
    const panel = document.querySelector("#global-search-results");
    if (current() && panel && !panel.hidden) renderGlobalSearchResults(state.search);
  }
}

function globalSearchGroups(query) {
  const normalized = query.trim().toLocaleLowerCase("zh-CN");
  if (!normalized) return [];
  const includes = (...values) => values.filter(Boolean).join(" ").toLocaleLowerCase("zh-CN").includes(normalized);
  const records = experimentRecords();
  const experimentResults = records.filter((archive)=>{
    const metadata = experimentMetadata(archive.name);
    return includes(productExperimentName(archive.name),archive.name,metadata.owner,...(metadata.tags||[]));
  }).slice(0,5).map((archive)=>({href:experimentRecordRoute(archive),title:productExperimentName(archive.name),meta:`实验 · ${archiveProductStatus(archive).label}`}));
  const stepResults = [];
  const search = state.globalMaterialSearch;
  const materialResults = search?.key === globalMaterialSearchKey(query) && search.status === "ready"
    ? search.items.map((event)=>({href:materialFocusRoute({name:event.archive_name},event),title:ACTION_LABELS[event.action_type]||"关键实验动作",meta:`正式素材 · ${productExperimentName(event.archive_name)}`})) : [];
  const reportResults = [];
  records.forEach((archive)=>{
    const detail = cachedLibraryDetail(archive);
    if (!detail) return;
    for (const experiment of detail.experiments||[]) {
      const matchingStep = (experiment.steps||[]).find((step)=>includes(step.current_step,step.observed_action,step.next_step,...(step.objects||[])));
      if (matchingStep && stepResults.length<5) stepResults.push({href:experimentRecordRoute(archive),title:matchingStep.current_step||matchingStep.observed_action||"实验步骤",meta:`${archive.staging_run_id ? "阶段步骤" : "步骤"} · ${productExperimentName(archive.name)}`});
    }
    const report = detail.daily_report||{};
    if (!archive.staging_run_id && report.report_id && reportResults.length<5 && includes(productExperimentName(archive.name),report.report_date,report.experiment_timeline?.[0]?.overall_summary)) reportResults.push({href:experimentRecordRoute(archive,"reports"),title:productExperimentName(archive.name),meta:`实验室日报 · ${report.report_date||"已生成"}`});
  });
  return [["实验",experimentResults],["步骤",stepResults],["关键素材",materialResults],["报告",reportResults]].filter(([,items])=>items.length);
}

function closeGlobalSearch() {
  const panel = document.querySelector("#global-search-results");
  const input = document.querySelector("#global-search");
  if (panel) panel.hidden = true;
  if (input) input.setAttribute("aria-expanded","false");
  state.searchActiveIndex = -1;
}

function renderGlobalSearchResults(query) {
  const panel = document.querySelector("#global-search-results");
  const input = document.querySelector("#global-search");
  if (!panel || !input) return;
  if (!query.trim()) { closeGlobalSearch(); return; }
  void loadGlobalMaterialSearch(query);
  const groups = globalSearchGroups(query);
  const search = state.globalMaterialSearch;
  const loading = search?.status === "loading";
  const failed = search?.status === "error";
  panel.innerHTML = groups.map(([label,items])=>`<section><h2>${label}<span>${number(items.length)}</span></h2>${items.map((item)=>`<a class="global-search-result" role="option" href="${item.href}"><span>${esc(item.title)}</span><small>${esc(item.meta)}</small>${icon("arrow")}</a>`).join("")}</section>`).join("")
    + (loading ? `<p class="global-search-loading"><i></i>正在搜索全部档案中的正式素材…</p>` : failed ? `<div class="global-search-empty" role="alert">正式素材搜索暂不可用，结果可能不完整。<button class="compact-button" type="button" data-retry-global-material-search>重新搜索</button></div>` : `<p class="global-search-loading">全部档案共找到 ${number(search?.total)} 份正式素材，当前显示 ${number(search?.items.length)} 份。</p>`)
    + (!groups.length && !loading && !failed ? `<div class="global-search-empty">没有找到匹配内容<small>可尝试实验名称、动作或对象关键词</small></div>` : "")
    + `<p class="global-search-loading">实验、步骤和报告仅搜索已载入内容。</p>`;
  panel.hidden = false;
  input.setAttribute("aria-expanded","true");
  state.searchActiveIndex = -1;
  panel.querySelectorAll("a").forEach((link)=>link.addEventListener("click",closeGlobalSearch));
  panel.querySelector("[data-retry-global-material-search]")?.addEventListener("click",()=>{
    state.globalMaterialSearch = null;
    input.focus();
    renderGlobalSearchResults(query);
  });
}

function handleGlobalSearchKey(event) {
  const panel = document.querySelector("#global-search-results");
  if (!panel || panel.hidden) return;
  const links = [...panel.querySelectorAll(".global-search-result")];
  if (event.key === "Escape") { closeGlobalSearch(); event.currentTarget.blur(); return; }
  if (!links.length) return;
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const delta = event.key === "ArrowDown" ? 1 : -1;
    state.searchActiveIndex = (state.searchActiveIndex + delta + links.length) % links.length;
    links.forEach((link,index)=>link.classList.toggle("active",index===state.searchActiveIndex));
    links[state.searchActiveIndex].scrollIntoView({block:"nearest"});
  } else if (event.key === "Enter") {
    event.preventDefault();
    (links[state.searchActiveIndex]||links[0]).click();
  }
}

function routeParts() {
  return location.hash.replace(/^#\/?/, "").split("?",1)[0].split("/").filter(Boolean).map(decodeURIComponent);
}

async function router() {
  const parts = routeParts();
  const route = parts[0] || "home";
  if (["home", "experiments", "materials", "reports"].includes(route) && state.archiveListingQuery !== archiveSearchQuery()) {
    const requestedHash = location.hash;
    setChrome(route);
    main.innerHTML = pageSkeleton("正在读取实验目录");
    try {
      const loaded = await loadArchiveListing();
      if (!loaded || requestedHash !== location.hash) return;
    } catch {
      if (requestedHash !== location.hash) return;
      main.innerHTML = productState("error", "server", "实验目录暂时无法载入", "目录读取失败，暂时无法确认素材与报告数量；系统会自动重试。", '<button class="secondary-button" type="button" data-retry-library-list>重试载入目录</button>');
      document.querySelector("[data-retry-library-list]")?.addEventListener("click",()=>router());
      return;
    }
  }
  if (route === "archive" && parts[1]) return renderArchive(parts[1], parts[2] || "experiments");
  if (route === "stage" && parts[1]) return renderArchive(parts[1], parts[2] || "experiments", true);
  if (route === "new") return renderNew();
  if (route === "tasks") return renderTasks();
  if (route === "operations") return renderOperations();
  if (route === "ai-settings") return window.VisionCortexAISettings.render({main, esc, icon, setChrome, api, state});
  if (route === "annotations") return renderAnnotations();
  if (route === "experiments") { syncArchiveFiltersFromRoute(); return renderExperiments(); }
  if (route === "materials") return renderMaterialsLibrary();
  if (route === "reports") return renderReportsLibrary();
  renderHome();
}

document.querySelector("#nav-toggle").addEventListener("click", () => {
  const shell = document.querySelector("#vc-shell");
  const collapsed = shell.dataset.collapsed !== "true";
  shell.dataset.collapsed = String(collapsed);
  document.querySelector("#nav-toggle").setAttribute("aria-label", collapsed ? "展开全局导航" : "收起全局导航");
});
let archiveSearchTimer;
const globalSearchInput = document.querySelector("#global-search");
globalSearchInput.addEventListener("input", (event) => {
  state.search = event.target.value;
  clearTimeout(archiveSearchTimer);
  archiveSearchTimer=setTimeout(async ()=>{
    const requestedHash = location.hash;
    const query = state.search;
    const route = routeParts()[0] || "home";
    if (!["home","experiments","materials","reports"].includes(route)) {
      if (document.activeElement === globalSearchInput) renderGlobalSearchResults(query);
      return;
    }
    try {
      if (["home","experiments"].includes(route) && state.archiveListingQuery !== archiveSearchQuery()) {
        if (!await loadArchiveListing()) return;
      }
      if (location.hash !== requestedHash || state.search !== query) return;
      await router();
      if (location.hash === requestedHash && state.search === query && document.activeElement === globalSearchInput) renderGlobalSearchResults(query);
    } catch (error) {
      if (location.hash === requestedHash && state.search === query) toast(error.message,"error");
    }
  },300);
});
globalSearchInput.addEventListener("focus",()=>renderGlobalSearchResults(state.search));
globalSearchInput.addEventListener("keydown",handleGlobalSearchKey);
document.addEventListener("pointerdown",(event)=>{ if(!event.target.closest(".topbar-search-shell")) closeGlobalSearch(); });
document.querySelector("#refresh-button").addEventListener("click", async () => {
  const button = document.querySelector("#refresh-button");
  button.disabled = true;
  button.classList.add("is-busy");
  state.archiveCacheEpoch = (state.archiveCacheEpoch || 0) + 1;
  state.archiveCache.clear();
  state.materialCache.clear();
  state.libraryLoadErrors.clear();
  try {
    await loadAll();
    await router();
    const unavailable = [state.archiveRefreshPending && "实验目录", state.taskSyncError && "任务状态"].filter(Boolean);
    if (unavailable.length) toast(`${unavailable.join("与")}暂时无法同步，当前数据可能不完整；系统会自动重试。`, "error");
    else toast("实验档案与任务状态已刷新。", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.classList.remove("is-busy");
  }
});
const mobileMoreDialog = document.querySelector("#mobile-more-dialog");
document.querySelector("#mobile-more-button")?.addEventListener("click", ()=>mobileMoreDialog?.showModal());
document.querySelectorAll("[data-close-mobile-more]").forEach((button)=>button.addEventListener("click", ()=>mobileMoreDialog?.close()));
mobileMoreDialog?.querySelectorAll("a").forEach((link)=>link.addEventListener("click", ()=>mobileMoreDialog.close()));
function routeFromNavigation() {
  const follow = state.followRun;
  if (follow?.enabled) {
    if (location.hash !== follow.expectedHash) follow.enabled = false;
    follow.expectedHash = null;
  }
  window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  return router();
}

document.querySelector("#follow-stage-results")?.addEventListener("click", () => {
  if (state.followRun?.enabled) state.followRun.enabled = false;
  else {
    const run = state.activeRun || state.runs.find(item=>!["completed","partial","failed","interrupted"].includes(item.state));
    if (!run) return;
    beginStageFollow(run.run_id);
    followStageResults(run);
  }
  updateServiceChrome();
});
window.addEventListener("hashchange", routeFromNavigation);
window.addEventListener("scroll", updateResultNavDensity, { passive: true });
window.addEventListener("resize", updateResultNavDensity, { passive: true });
window.addEventListener("keydown", handleMaterialReviewShortcut);
window.setInterval(refreshTaskSnapshots, 4000);

hydrateIcons();
const legacyArchive = new URLSearchParams(location.search).get("archive");
if (legacyArchive && !location.hash) location.hash = `#/archive/${encodeURIComponent(legacyArchive)}/experiments`;
loadAll().then(routeFromNavigation).catch(() => {
  main.innerHTML = productState("error", "server", "实验目录暂时无法载入", "目录读取失败，暂时无法确认素材与报告数量；系统会自动重试。", `<button class="primary-button" type="button" data-retry-service>重新连接</button>`);
  document.querySelector("[data-retry-service]")?.addEventListener("click", ()=>location.reload());
});

let refreshingNasRecordings = false;
function refreshNasPickers() {
  if (!state.nasRenderPending || state.nasLoading || routeParts()[0] !== "new") return;
  let deferred = false;
  for (const [selector, markup, bind] of [["#nas-batches", nasBatchPicker, bindNasBatchPicker], ["#nas-recordings", nasRecordingPicker, bindNasRecordingPicker]]) {
    const picker = document.querySelector(selector);
    if (!picker) continue;
    // Protect the control being used, while allowing the rest of the form to stay live.
    if (picker.contains(document.activeElement)) { deferred = true; continue; }
    picker.outerHTML = markup();
    bind();
  }
  state.nasRenderPending = deferred;
}

function flushNasPickersAfterFocus(event) {
  if (state.nasRenderPending && event.target.closest("#nas-batches, #nas-recordings")) setTimeout(refreshNasPickers, 0);
}

async function refreshNasRecordings() {
  if (refreshingNasRecordings || state.nasLoading || document.hidden || routeParts()[0] !== "new" || state.health?.collection_ingest?.mode !== "directory_metadata") return;
  refreshingNasRecordings = true;
  try {
    const requestId = await loadNasRecordings();
    if (requestId === state.nasRequestId) refreshNasPickers();
  } finally { refreshingNasRecordings = false; }
}
window.setInterval(refreshNasRecordings, 30000);
document.addEventListener("focusout", flushNasPickersAfterFocus);

// Browsing a retained stage changes only this view, never the running job.
document.addEventListener("change", event => {
  if (!event.target.matches("[data-stage-version]")) return;
  const section=event.target.closest(".stage-delivery");
  section.querySelectorAll("[data-stage-version-panel]").forEach(panel=>{panel.hidden=panel.dataset.stageVersionPanel!==event.target.value;});
});
