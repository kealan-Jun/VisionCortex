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
  sources: [],
  activeRun: null,
  selectedCollectionId: null,
  collectionQuery: "",
  experimentName: "",
  archiveCache: new Map(),
  search: "",
  refreshingTasks: false,
  materialFilters: { archive: null, group: null, action: "all", support: "all", query: "" },
  annotationFilters: { priority: "", reviewStatus: "", query: "" },
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
const archiveLabel = () => state.health?.archive_label || (isNasMode() ? "NAS 正式归档" : "实验归档");
const archiveShortLabel = () => isNasMode() ? "NAS" : "本地";

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
const STAGE_LABELS = {
  reserving: "锁定固定基准归档",
  original_ingest: "原视频安全留存",
  nas_ingest: "读取索引并准备六路输入",
  queued: "排队等待",
  running: "正在启动",
  preflight: "视频探测与预检",
  alignment: "多路时间戳对齐",
  motion_probe: "低成本运动探针",
  candidate_coarse: "低成本粗扫",
  candidate_fine: "疑似区间精扫",
  candidate_audit: "有界片段审计",
  experiment_understanding: "实验步骤级理解",
  experiment_clips: "实验片段归档",
  key_materials: "关键素材生成",
  mllm: "关键素材模型理解",
  package: "证据包与指标归档",
  daily_report: "实验室日报生成与校验",
  completed: "分析完成",
  failed: "分析失败",
  interrupted: "服务重启后待续跑",
};

const GUIDED_PIPELINE = [
  {
    id: "originals", number: "01", title: "原视频安全留存",
    stages: ["reserving", "original_ingest", "nas_ingest"], completedBy: ["original_ingest"],
    folder: "Original-Experiment-Videos",
    doing: "校验上传文件与时钟 CSV，并把原始输入登记到本次实验档案。",
    outcome: "可追溯的原视频、CSV、来源角色和文件校验信息。",
  },
  {
    id: "alignment", number: "02", title: "预检与多路时间对齐",
    stages: ["preflight", "alignment"], completedBy: ["alignment"],
    folder: "JSON-Config-Files",
    doing: "检查视频可读性、模型和存储，再用 CSV 最近邻与视觉锚点统一六路时间轴。",
    outcome: "每一路的时间变换、置信度与对齐时间戳账本。",
  },
  {
    id: "discovery", number: "03", title: "发现真实有界实验",
    stages: ["motion_probe", "candidate_coarse", "candidate_fine", "candidate_audit"], completedBy: ["candidate_audit"],
    folder: "JSON-Config-Files",
    doing: "运动探针缩小范围，YOLO 粗扫和精扫收紧边界，再审计跨视角连续性。",
    outcome: "仅保留第一/第三人称时间一致、边界可信的真实实验候选。",
  },
  {
    id: "clips", number: "04", title: "实验命名、理解与片段归档",
    stages: ["experiment_understanding", "experiment_clips"], completedBy: ["experiment_clips"],
    folder: "Experiment-Clips",
    doing: "判断独立实验或连续实验，细分当前/下一步骤，并按模型实验名生成三份视频。",
    outcome: "每个实验的第一人称、第三人称、并排 MP4 及对应步骤 JSON。",
  },
  {
    id: "materials", number: "05", title: "关键素材与细粒度步骤理解",
    stages: ["key_materials", "mllm"], completedBy: ["mllm"],
    folder: "Key-Materials",
    doing: "提取分层动作证据的对齐关键帧、片段和时间戳，并理解当前与下一步骤。",
    outcome: "跨视角关键帧/关键片段，以及统一事件 JSON 和模型证据。",
  },
  {
    id: "evidence", number: "06", title: "证据包与质量验收",
    stages: ["package"], completedBy: ["package"],
    folder: "JSON-Config-Files",
    doing: "汇总边界、跨视角支持、动作覆盖、耗时与 Token，并执行自动验收。",
    outcome: "可审计的证据包、质量结论、阶段耗时与输入/输出 Token 账本。",
  },
  {
    id: "reports", number: "07", title: "日报、PDF 与正式归档",
    stages: ["daily_report", "completed"], completedBy: ["daily_report", "completed"],
    folder: "Lab-Daily-Reports / Professional-PDFs",
    doing: "基于已验收证据填充固定日报模板，生成并校验 PDF，最后提升为正式档案。",
    outcome: "实验室日报、专业 PDF 和完整 NAS 正式归档。",
  },
];

async function api(url, options) {
  const response = await fetch(url, options);
  let payload;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.message || `${response.status} ${response.statusText}`;
    throw new Error(message);
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

function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((element) => {
    element.innerHTML = icon(element.dataset.icon);
  });
}

function pageContext(route) {
  if (route === "new") return ["核心工作", "新建实验"];
  if (route === "tasks") return ["核心工作", "任务进度"];
  if (route === "materials") return ["实验产出", "关键素材库"];
  if (route === "reports") return ["实验产出", "实验室日报"];
  if (route === "operations") return ["系统管理", "服务健康"];
  if (route === "annotations") return ["系统管理", "YOLO 标注"];
  if (route === "archive") return ["实验结果", "实验详情"];
  if (route === "experiments") return ["核心工作", "实验记录"];
  return ["核心工作", "总览"];
}

function setChrome(route, pageOverride) {
  const [section, page] = pageContext(route);
  document.querySelector("#section-name").textContent = section;
  document.querySelector("#page-name").textContent = pageOverride || page;
  const active = route === "archive" ? "experiments" : route;
  document.querySelectorAll(".nav-link").forEach((link) => link.classList.toggle("active", link.dataset.route === active));
}

function setPhase(run) {
  const chip = document.querySelector("#phase-chip");
  const stage = run?.state || "idle";
  chip.className = `phase-chip ${stage === "completed" ? "completed" : stage === "failed" ? "failed" : run ? "running" : ""}`;
  chip.querySelector("span").textContent = run ? (STAGE_LABELS[stage] || run.message || stage) : "等待新任务";
}

function updateServiceChrome() {
  const online = state.health?.status === "ok";
  document.querySelector("#node-dot").classList.toggle("online", online);
  const running = state.runs.filter((run) => !["completed", "failed"].includes(run.state)).length;
  document.querySelector("#service-note").textContent = online
    ? running ? `${running} 个任务正在分析` : `${archiveLabel()}与分析服务可用`
    : "分析服务暂不可用";
  const badge = document.querySelector("#running-badge");
  badge.hidden = running === 0;
  badge.textContent = running;
  setPhase(state.activeRun || state.runs.find((run) => !["completed", "failed"].includes(run.state)) || null);
}

async function loadAll() {
  const results = await Promise.allSettled([api("/api/health"), api("/api/archives"), api("/api/runs"), api("/api/collections?limit=200")]);
  if (results[0].status === "fulfilled") state.health = results[0].value;
  if (results[1].status === "fulfilled") state.archives = results[1].value.archives || [];
  if (results[2].status === "fulfilled") state.runs = results[2].value.runs || [];
  if (results[3].status === "fulfilled") state.collections = results[3].value.collections || [];
  updateServiceChrome();
}

async function refreshTaskSnapshots() {
  if (state.refreshingTasks || (routeParts()[0] || "home") !== "tasks") return;
  state.refreshingTasks = true;
  try {
    const payload = await api("/api/runs");
    state.runs = payload.runs || [];
    updateServiceChrome();
    renderTasks();
  } catch {
    // Keep the last durable snapshot visible; the freshness warning explains stale data.
  } finally {
    state.refreshingTasks = false;
  }
}

function filteredArchives() {
  const query = state.search.trim().toLocaleLowerCase();
  return query ? state.archives.filter((archive) => archive.name.toLocaleLowerCase().includes(query)) : state.archives;
}

function statusCard(iconName, label, value, note) {
  return `<article class="status-card"><span>${icon(iconName)}</span><div><small>${esc(label)}</small><strong>${esc(value)}</strong><p>${esc(note)}</p></div></article>`;
}

function archiveRows(archives, target = "experiments") {
  if (!archives.length) return `<div class="empty-state"><strong>还没有实验档案</strong><p>点击“新建实验”上传第一批多视角视频；原视频会先留存在${archiveLabel()}。</p><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div>`;
  return `<div class="archive-list">${archives.map((archive) => `
    <a class="archive-row" href="#/archive/${encodeURIComponent(archive.name)}/${target}">
      <span>${icon("check")}</span>
      <span class="archive-identity"><strong>${esc(archive.name)}</strong><small>${formatDate(archive.modified_at)}</small></span>
      <span class="archive-cell"><small>有界实验</small><strong>${number(archive.experiment_count)} 个</strong></span>
      <span class="archive-cell"><small>关键事件</small><strong>${number(archive.key_event_count)} 个</strong></span>
      <span class="row-link">查看结果 ${icon("arrow")}</span>
    </a>`).join("")}</div>`;
}

function renderHome() {
  setChrome("home");
  const archives = filteredArchives();
  const running = state.runs.filter((run) => !["completed", "failed"].includes(run.state)).length;
  const experimentTotal = state.archives.reduce((sum, archive) => sum + Number(archive.experiment_count || 0), 0);
  const keyTotal = state.archives.reduce((sum, archive) => sum + Number(archive.key_event_count || 0), 0);
  const recent = archives.slice(0, 6);
  const archiveName = archiveLabel();
  main.innerHTML = `<div class="page">
    <header class="page-hero"><div><p class="eyebrow">LABORATORY SITUATIONAL AWARENESS</p><h1>多视角湿实验工作台</h1><p>上传任意实际路数的连续长视频，自动完成原视频留存、时间轴对齐、有界实验筛选、跨视角关键素材和细粒度步骤理解。系统容量已按至少 6 路、每路 3 小时设计。</p></div><div class="hero-actions"><button class="primary-button" id="rerun-benchmark" type="button">${icon("activity")}重跑六路 3 小时基准</button><a class="secondary-button" href="#/new">${icon("upload")}新建其他实验</a></div></header>
    <section class="current-work"><span class="current-icon">${icon(running ? "activity" : "folder")}</span><div><small>${running ? "当前分析" : `最近${archiveName}`}</small><h2>${esc(running ? state.runs.find((run) => !["completed","failed"].includes(run.state))?.experiment_id : state.archives[0]?.name || "等待创建第一个实验")}</h2><p>${running ? `产出会在每个阶段完成时持续写入${archiveName}` : state.archives[0] ? `所有结果均从${archiveName}读取` : "选择多路视频即可开始"}</p></div>${state.archives[0] ? `<a class="secondary-button" href="#/archive/${encodeURIComponent(state.archives[0].name)}/experiments">打开</a>` : ""}</section>
    <section class="status-grid">
      ${statusCard("folder", `${archiveShortLabel()}实验档案`, number(state.archives.length), archiveName)}
      ${statusCard("flask", "有界实验", number(experimentTotal), "独立或连续实验组")}
      ${statusCard("image", "关键事件", number(keyTotal), "五大物理动作")}
      ${statusCard("activity", "正在处理", number(running), "上传与分析任务")}
    </section>
    <section class="panel"><header class="panel-heading"><div><p class="eyebrow">RECENT EXPERIMENTS</p><h2>最近实验</h2><p>直接从${archiveName}读取，不展示运行缓存。</p></div><a href="#/experiments">全部实验 ${icon("arrow")}</a></header>${archiveRows(recent)}</section>
    <section class="workflow-strip">
      <article><span>01</span><i>${icon("upload")}</i><div><strong>上传与原视频留存</strong><p>按实际路数创建机位</p></div></article>
      <article><span>02</span><i>${icon("clock")}</i><div><strong>对齐与有界筛选</strong><p>最近邻 + 视觉锚点</p></div></article>
      <article><span>03</span><i>${icon("boxes")}</i><div><strong>跨视角关键素材</strong><p>帧、片段、时间戳</p></div></article>
      <article><span>04</span><i>${icon("brain")}</i><div><strong>细粒度步骤理解</strong><p>当前步骤与下一步</p></div></article>
    </section>
  </div>`;
  document.querySelector("#rerun-benchmark")?.addEventListener("click", rerunBenchmark);
}

async function rerunBenchmark() {
  const button = document.querySelector("#rerun-benchmark");
  if (button) {
    button.disabled = true;
    button.textContent = "正在准备固定基准…";
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
      button.innerHTML = `${icon("activity")}重跑六路 3 小时基准`;
    }
  }
}

function renderExperiments(target = "experiments") {
  setChrome(target);
  const title = target === "materials" ? "关键素材库" : target === "reports" ? "实验室日报" : "实验记录";
  const copy = target === "materials"
    ? "按正式实验档案进入关键素材，查看五大类动作的对齐帧、对齐片段、时间戳与模型判断。"
    : target === "reports"
      ? "从已验收证据自动生成日报 JSON、Markdown、HTML 与 PDF，并保留人工复核状态。"
      : `按实验查看原视频留存、分析结果与${archiveLabel()}位置。`;
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">EVIDENCE ARCHIVE</p><h1>${title}</h1><p>${copy}</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header>${target === "experiments" ? `<section class="panel"><header class="panel-heading"><div><h2>采集批次处理账本</h2><p>直接说明每个 index 批次是否已处理并留存；不会让用户逐个筛选约 90 个分片。</p></div><a class="secondary-button" href="#/new">选择批次</a></header>${collectionLedger()}</section>` : ""}<section class="panel"><header class="panel-heading"><div><h2>${archiveLabel()}目录</h2><p>${state.health?.nas_archive_root ? `根目录：${esc(state.health.nas_archive_root)}` : "正在读取归档根目录"}</p></div></header>${archiveRows(filteredArchives(), target === "materials" ? "materials" : target === "reports" ? "reports" : "experiments")}</section></div>`;
}

function createSource(video = null, csv = null, index = state.sources.length) {
  const stem = video?.name?.replace(/\.[^.]+$/, "") || `view-${String(index + 1).padStart(2,"0")}`;
  const viewId = stem.replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-|-$/g, "").slice(0, 80) || `view-${String(index + 1).padStart(2,"0")}`;
  return { key: crypto.randomUUID(), viewId, role: index === 0 ? "first_person" : "third_person", video, csv };
}

function normalizePairName(filename) {
  return filename.toLocaleLowerCase().replace(/\.[^.]+$/, "").replace(/timestamps?|timecodes?|clock|frames?|pts|video|record/g, "").replace(/[^a-z0-9\u4e00-\u9fff]/g, "");
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
  const unusedCsvs = [...csvs];
  state.sources = videos.map((video, index) => {
    const normalized = normalizePairName(video.name);
    let matchIndex = unusedCsvs.findIndex((csv) => {
      const candidate = normalizePairName(csv.name);
      return candidate && normalized && (candidate.includes(normalized) || normalized.includes(candidate));
    });
    if (matchIndex < 0 && csvs.length === videos.length) matchIndex = 0;
    const csv = matchIndex >= 0 ? unusedCsvs.splice(matchIndex, 1)[0] : null;
    const previous = state.sources[index];
    const source = createSource(video, csv, index);
    if (previous?.role) source.role = previous.role;
    return source;
  });
  renderNew();
  toast(`已按实际选择生成 ${videos.length} 路机位${csvs.length ? `，识别 ${csvs.length} 个 CSV` : ""}。`);
}

function sourceCard(source, index) {
  return `<article class="source-card" data-source="${esc(source.key)}">
    <header class="source-card-head"><span>${icon("video")}</span><div><strong>机位 ${String(index + 1).padStart(2,"0")}${source.video ? ` · ${esc(source.video.name)}` : ""}</strong><small>${source.video ? formatBytes(source.video.size) : "等待选择该机位的视频"}</small></div><button class="remove-source" type="button" data-remove="${esc(source.key)}" aria-label="移除机位">${icon("x")}</button></header>
    <div class="source-config-grid">
      <label>机位 ID<input data-view-id="${esc(source.key)}" value="${esc(source.viewId)}" autocomplete="off" /></label>
      <label>拍摄视角<select data-role="${esc(source.key)}"><option value="first_person" ${source.role === "first_person" ? "selected" : ""}>第一人称 · 手部与精细接触</option><option value="third_person" ${source.role === "third_person" ? "selected" : ""}>第三人称 · 空间与移动轨迹</option></select></label>
    </div>
    <div class="file-grid">
      <label class="file-field ${source.video ? "has-file" : ""}"><input type="file" accept="video/*,.mp4,.mov,.m4v,.mkv,.avi,.webm" data-video="${esc(source.key)}"/><span class="file-icon">${icon("video")}</span><span><strong>${source.video ? esc(source.video.name) : "选择实验视频"}</strong><small>${source.video ? formatBytes(source.video.size) : "该机位的连续原视频"}</small></span>${source.video ? `<span class="file-check">${icon("check")}</span>` : ""}</label>
      <label class="file-field ${source.csv ? "has-file" : ""}"><input type="file" accept=".csv,text/csv" data-csv="${esc(source.key)}"/><span class="file-icon">${icon("clock")}</span><span><strong>${source.csv ? esc(source.csv.name) : "选择时间戳 CSV"}</strong><small>${source.csv ? formatBytes(source.csv.size) : "可选；用于最近邻对齐"}</small></span>${source.csv ? `<span class="file-check">${icon("check")}</span>` : ""}</label>
    </div>
  </article>`;
}

function roleGuidance() {
  const videos = state.sources.filter((source) => source.video).length;
  const first = state.sources.filter((source) => source.role === "first_person").length;
  const third = state.sources.filter((source) => source.role === "third_person").length;
  if (videos < 2) return { tone: "warning", text: `已选择 ${videos} 路视频；跨视角分析至少需要 2 路。路数按用户实际上传生成，不要求固定为 6 路。` };
  if (!first || !third) return { tone: "warning", text: `当前 ${videos} 路视频中，需要至少确认一路第一人称和一路第三人称。` };
  return { tone: "ok", text: `已准备 ${videos} 路视频：${first} 路第一人称、${third} 路第三人称。系统会按这 ${videos} 路实际输入并发调度。` };
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
  if (processing === "archived") return { label: "已处理并留存", tone: "archived", note: `正式归档：${collection.processing.archive_name || "已生成"}` };
  if (["queued", "processing"].includes(processing)) return { label: "处理中", tone: "processing", note: `任务 ${collection.processing.run_id || "已接管"}` };
  if (processing === "failed") return { label: "处理失败", tone: "failed", note: "正式归档未提升，可查看失败 staging 证据" };
  if (collection.status === "ready") return { label: "可开始分析", tone: "ready", note: "索引已封口，双视角角色完整" };
  if (collection.status === "recording") return { label: "采集中", tone: "recording", note: "等待所有相机写入结束时间" };
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
  if (!items.length) return `<div class="empty-state compact"><strong>没有匹配的采集批次</strong><p>无需手工选择 15 分钟分片；刷新索引或修改日期/批次搜索词。</p></div>`;
  return `<div class="collection-card-list">${items.map((collection) => {
    const status = collectionStatusCopy(collection);
    const views = collection.resolved_view_counts || {};
    const selected = collection.collection_id === state.selectedCollectionId;
    const issues = [...(collection.blocking_issue_codes || []), ...(collection.warning_codes || [])];
    return `<article class="collection-card ${status.tone} ${selected ? "selected" : ""}">
      <header><div><small>${esc(collection.collection_id)}</small><strong>${formatDate(collection.recording_start_time)}</strong></div><span class="collection-status ${status.tone}">${esc(status.label)}</span></header>
      <p>${esc(status.note)}${collection.approved_override_count ? ` · ${number(collection.approved_override_count)} 个有收据的角色覆盖` : ""}</p>
      <div class="collection-metrics"><span><b>${number(collection.camera_count)}</b> 路</span><span><b>${number(collection.video_segment_count)}</b> 个 MP4</span><span><b>${number(collection.clock_segment_count)}</b> 个 CSV</span><span><b>${number(views.first_person)}</b> 第一 + <b>${number(views.third_person)}</b> 第三</span></div>
      ${issues.length ? `<small class="collection-issues">${esc(issues.join(" · "))}</small>` : ""}
      <button class="${selected ? "secondary-button" : "primary-button"}" type="button" data-select-collection="${esc(collection.collection_id)}" ${collection.ready_to_analyze && !["queued", "processing"].includes(collection.processing?.state) ? "" : "disabled"}>${selected ? `${icon("check")}已选择` : collection.processing?.state === "archived" ? `${icon("check")}重新分析为新归档` : `${icon("server")}选择此批次`}</button>
    </article>`;
  }).join("")}</div>`;
}

function collectionLedger() {
  if (!state.collections.length) return `<div class="empty-state compact"><strong>索引中还没有采集批次</strong><p>系统只增量读取 index 字段表，不递归扫描 NAS 视频目录。</p></div>`;
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
  const videos = state.sources.filter((source) => source.video).length;
  const csvs = state.sources.filter((source) => source.csv).length;
  const first = state.sources.filter((source) => source.role === "first_person").length;
  const third = state.sources.filter((source) => source.role === "third_person").length;
  const ids = state.sources.map((source) => source.viewId.trim()).filter(Boolean);
  const unique = new Set(ids).size === ids.length;
  return { mode: "upload", videos, csvs, first, third, title, ready: Boolean(title && videos >= 2 && videos === state.sources.length && first && third && unique && ids.length === state.sources.length) };
}

function renderNew() {
  setChrome("new");
  const guide = roleGuidance();
  const currentTitle = document.querySelector("#experiment-name")?.value;
  if (currentTitle !== undefined) state.experimentName = currentTitle;
  const selected = selectedCollection();
  const readyCount = state.collections.filter((item) => item.status === "ready" && !["archived", "queued", "processing"].includes(item.processing?.state)).length;
  const recordingCount = state.collections.filter((item) => item.status === "recording").length;
  const attentionCount = state.collections.filter((item) => item.status === "attention").length;
  const archivedCount = state.collections.filter((item) => item.processing?.state === "archived").length;
  main.innerHTML = `<div class="page new-experiment-page">
    <header class="page-hero"><div><p class="eyebrow">实验分析 / 创建任务</p><h1>选择采集批次</h1><p>系统从采集索引自动聚合同一 experiment_id 下的全部相机、15 分钟 MP4 和时钟 CSV。用户选择一张批次卡片即可，不再逐个上传和配置约 90 个文件。</p></div><div class="hero-state"><small>NAS 发现策略</small><strong>索引增量读取 · 零复制</strong><p>只监控约 1.2MB 的索引变化；选择任务后才校验源文件并进入分析。</p></div></header>
    <nav class="setup-progress" aria-label="新建实验进度"><a href="#collection-source"><span>01</span><strong>选择批次</strong><small>自动聚合多路分片</small></a><a href="#experiment-info"><span>02</span><strong>归档名称</strong><small>英文安全命名</small></a><a href="#analysis-method"><span>03</span><strong>分析方式</strong><small>YOLO + 豆包</small></a><a href="#review-start"><span>04</span><strong>核对启动</strong><small>零复制自动分析</small></a></nav>
    <div class="new-layout"><div class="new-primary">
      <section class="form-section collection-picker" id="collection-source"><div class="section-heading"><span>01</span><div><h2>NAS 采集批次</h2><p>experiment_id 是批次主键；列表只读索引元数据，不扫描目录、不打开视频。</p></div></div>
        <div class="collection-summary"><span><b>${number(readyCount)}</b> 未处理可分析</span><span><b>${number(archivedCount)}</b> 已处理留存</span><span><b>${number(recordingCount)}</b> 采集中</span><span><b>${number(attentionCount)}</b> 需处理</span><span><b>${number(state.collections.length)}</b> 全部批次</span></div>
        <label class="collection-search">${icon("search")}<input id="collection-search" value="${esc(state.collectionQuery)}" placeholder="按日期或 experiment_id 查找" /></label>
        <div id="collection-card-results">${collectionCards()}</div>
        ${selected ? `<div class="selected-collection-note"><span>${icon("check")}</span><div><strong>已选择 ${esc(selected.collection_id)}</strong><p>${number(selected.camera_count)} 路 · ${number(selected.video_segment_count)} 个 MP4 · ${number(selected.clock_segment_count)} 个 CSV；启动前执行一次并发存在性校验，原视频复制 0 字节。</p></div><button class="secondary-button" id="clear-collection" type="button">改选批次</button></div>` : ""}
      </section>
      <section class="form-section" id="experiment-info"><div class="section-heading"><span>02</span><div><h2>归档名称</h2><p>仅命名 NAS 正式归档；模型理解完成后，内部实验文件夹仍按具体实验名称自动归档。</p></div></div><label class="field-label"><span>英文安全归档名称</span><input id="experiment-name" value="${esc(state.experimentName)}" maxlength="120" placeholder="例如：VisionCortex-Collection-20260810-e918b762" autocomplete="off" /></label></section>
      <section class="form-section manual-upload-fallback ${selected ? "is-secondary" : ""}" id="recorded-videos"><div class="section-heading"><span>备用</span><div><h2>索引外文件上传</h2><p>仅当采集批次尚未进入 index 表时使用；正常采集数据无需选择这些文件。</p></div></div>
        <div class="batch-import-panel"><div><strong>一键选择全部实验文件</strong><p>可选择任意实际路数。支持 MP4/MOV/MKV/AVI/WebM 和时间戳 CSV。</p></div><div class="batch-actions"><label class="primary-button batch-import-button">${icon("upload")}选择视频与 CSV<input id="batch-input" type="file" multiple accept="video/*,.mp4,.mov,.m4v,.mkv,.avi,.webm,.csv,text/csv" /></label><label class="secondary-button batch-import-button">${icon("folder")}选择实验文件夹<input id="folder-input" type="file" multiple webkitdirectory directory /></label></div></div>
        <div class="batch-notice">备用上传仍按用户实际路数生成，不固定为 6 路；选择 NAS 批次后，本区域不会参与任务。</div>
        <div class="role-guidance ${guide.tone}">${esc(guide.text)}</div>
        <div class="source-list">${state.sources.length ? state.sources.map(sourceCard).join("") : `<div class="empty-state"><strong>尚未选择视频</strong><p>点击上方“选择视频与 CSV”，选中几路就会出现几条机位配置。</p></div>`}</div>
        <button class="secondary-button add-source" id="add-source" type="button">${icon("plus")}手动添加一路</button>
      </section>
      <section class="form-section" id="analysis-method"><div class="section-heading"><span>03</span><div><h2>分析与归档方式</h2><p>当前任务会自动执行整条流水线，不需要手工逐阶段启动。</p></div></div><div class="analysis-method-overview"><article class="method-card"><span>${icon("gauge")}</span><div><small>CV FOUNDATION</small><strong>YOLO + 跟踪 + 时序规则</strong><p>多路并发粗扫与有界精扫，先精确找出真实实验边界和五类动作候选。</p></div></article><article class="method-card"><span>${icon("brain")}</span><div><small>MULTIMODAL UNDERSTANDING</small><strong>豆包 Seed 2.1 Pro</strong><p>仅阅读有界片段和代表素材，输出当前步骤、下一步骤、对象、事实与不确定项。</p></div></article></div></section>
    </div><aside class="review-card" id="review-start"><div class="section-heading"><span>04</span><div><h2>核对并启动</h2><p>${selected ? "源视频保持在 NAS 原位置；每完成一项就归档一项。" : "原视频先留存，之后每完成一项就归档一项。"}</p></div></div><div class="review-summary" id="review-summary"></div><div class="upload-progress hidden" id="upload-progress"><div class="progress-copy"><span id="progress-message">准备任务</span><strong id="progress-percent">0%</strong></div><span class="progress-track"><i id="progress-bar" style="width:0%"></i></span><div class="run-stage-list" id="run-stages"></div></div><button class="primary-button" id="start-run" type="button">${selected ? `${icon("server")}零复制开始分析` : `${icon("upload")}上传、留存并开始分析`}</button><div class="batch-notice">正式输出根目录：<strong>${esc(state.health?.nas_archive_root || "Y:\\VisionCortexExperimentArchive")}</strong><br/>${selected ? "源文件复制 0 字节，产出按阶段持续写入正式归档。" : "任务创建后会立即生成六类标准子目录。"}</div></aside></div>
  </div>`;
  bindNewPage();
  updateReview();
}

function updateReview() {
  const review = reviewState();
  const summary = document.querySelector("#review-summary");
  if (!summary) return;
  summary.innerHTML = review.mode === "nas_collection"
    ? `<div class="review-line"><span>采集批次</span><strong>${esc(review.collection.collection_id)}</strong></div><div class="review-line"><span>自动聚合</span><strong>${review.videos} 路 / ${number(review.collection.video_segment_count)} 个 MP4</strong></div><div class="review-line"><span>视角构成</span><strong>${review.first} 第一人称 + ${review.third} 第三人称</strong></div><div class="review-line"><span>源视频复制</span><strong>0 字节</strong></div><div class="review-line"><span>角色解析收据</span><strong>${number(review.collection.approved_override_count)} 个覆盖</strong></div><div class="review-line"><span>自动完整流水线</span><strong>启用</strong></div>`
    : `<div class="review-line"><span>实际视频路数</span><strong>${review.videos} 路</strong></div><div class="review-line"><span>时间戳 CSV</span><strong>${review.csvs} 个</strong></div><div class="review-line"><span>视角构成</span><strong>${review.first} 第一人称 + ${review.third} 第三人称</strong></div><div class="review-line"><span>NAS 原视频留存</span><strong>启用</strong></div><div class="review-line"><span>自动完整流水线</span><strong>启用</strong></div>`;
  const button = document.querySelector("#start-run");
  button.disabled = !review.ready || Boolean(state.activeRun && !["completed","failed"].includes(state.activeRun.state));
}

function updateSource(key, change) {
  const source = state.sources.find((item) => item.key === key);
  if (!source) return;
  Object.assign(source, change);
}

function bindNewPage() {
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
  document.querySelectorAll("[data-video]").forEach((input) => input.addEventListener("change", () => { state.selectedCollectionId = null; const file = input.files[0] || null; updateSource(input.dataset.video, { video: file }); renderNew(); }));
  document.querySelectorAll("[data-csv]").forEach((input) => input.addEventListener("change", () => { state.selectedCollectionId = null; updateSource(input.dataset.csv, { csv: input.files[0] || null }); renderNew(); }));
  document.querySelector("#start-run").addEventListener("click", submitRun);
}

function xhrUpload(formData, progress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/runs");
    request.responseType = "json";
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) progress(event.loaded / event.total);
    });
    request.addEventListener("load", () => {
      if (request.status >= 200 && request.status < 300) resolve(request.response);
      else reject(new Error(request.response?.detail || `上传失败：HTTP ${request.status}`));
    });
    request.addEventListener("error", () => reject(new Error("网络连接中断，上传未完成")));
    request.send(formData);
  });
}

function renderStages(activeStage, progressValue) {
  const stages = ["original_ingest","preflight","alignment","motion_probe","candidate_coarse","candidate_fine","candidate_audit","experiment_understanding","experiment_clips","key_materials","mllm","package","daily_report"];
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

async function submitRun() {
  const review = reviewState();
  if (!review.ready) { toast(review.mode === "nas_collection" ? "该批次尚未通过封口与视角质量门，或缺少归档名称。" : "请先填写实验名称，选择至少两路视频，并确认第一/第三人称。", "error"); return; }
  if (review.mode === "nas_collection") {
    await submitCollectionRun(review);
    return;
  }
  const formData = new FormData();
  formData.append("experiment_name", review.title);
  const specs = [];
  let csvIndex = 0;
  state.sources.forEach((source, videoIndex) => {
    formData.append("videos", source.video, source.video.name);
    const spec = { view_id: source.viewId, role: source.role, video_index: videoIndex, calibration_hint_ms: 0 };
    if (source.csv) {
      spec.csv_index = csvIndex;
      formData.append("timestamp_csvs", source.csv, source.csv.name);
      csvIndex += 1;
    }
    specs.push(spec);
  });
  formData.append("view_specs_json", JSON.stringify(specs));
  document.querySelector("#upload-progress").classList.remove("hidden");
  document.querySelector("#start-run").disabled = true;
  const nasOnly = state.health?.web_upload_retention_mode === "nas_only";
  setProgress(0, nasOnly ? "正在写入 NAS 原视频留存区并校验文件" : "正在写入本地运行区与 NAS 原视频留存区");
  try {
    const created = await xhrUpload(formData, (value) => setProgress(value * .15, `正在上传并留存 ${review.videos} 路原视频`));
    const archiveName = new URL(created.archive_url, location.origin).searchParams.get("archive") || review.title;
    state.activeRun = { run_id: created.run_id, state: "queued", progress: .15, experiment_id: archiveName };
    setPhase(state.activeRun);
    await pollRun(created.run_id, archiveName);
  } catch (error) {
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
    setProgress(.02, "任务已接管；正在并发校验 NAS 分片与时钟 CSV");
    await pollRun(created.run_id, created.archive_name);
  } catch (error) {
    toast(error.message, "error");
    setProgress(1, `失败：${error.message}`);
    document.querySelector("#start-run").disabled = false;
  }
}

async function pollRun(runId, archiveName) {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
    state.activeRun = run;
    setPhase(run);
    renderStages(run.state, .15 + Number(run.progress || 0) * .85);
    if (run.state === "completed") {
      setProgress(1, "分析完成，全部产出已归档到 NAS");
      toast("实验分析完成，正在打开正式 NAS 结果。", "");
      await loadAll();
      location.hash = `#/archive/${encodeURIComponent(archiveName)}/experiments`;
      return;
    }
    if (run.state === "failed") {
      toast(run.error || "分析失败", "error");
      setProgress(1, `失败：${run.error || "请检查服务日志"}`);
      document.querySelector("#start-run").disabled = false;
      return;
    }
  }
}

function renderTasks() {
  setChrome("tasks");
  const runs = state.runs;
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">GUIDED ARCHIVE PIPELINE</p><h1>任务进度与自动归档</h1><p>从原视频留存到日报 PDF，每完成一个环节就写入 NAS 并生成阶段回执。这里说明当前在做什么、已归档什么以及下一步会得到什么。</p></div><div class="hero-actions"><a class="primary-button" href="#/new">${icon("plus")}新建实验</a></div></header><section class="status-grid">${statusCard("activity","当前任务",number(runs.length),"含可恢复持久状态")}${statusCard("gauge","正在处理",number(runs.filter((run)=>!["completed","failed","interrupted"].includes(run.state)).length),"页面每 4 秒自动刷新")}${statusCard("check","已完成",number(runs.filter((run)=>run.state==="completed").length),"已进入正式档案")}${statusCard("file","中断/失败",number(runs.filter((run)=>["failed","interrupted"].includes(run.state)).length),"已完成产出仍保留在 NAS")}</section>${runs.length ? runs.map(runObservabilityCard).join("") : `<section class="panel"><div class="empty-state"><strong>当前没有运行任务</strong><p>从“新建实验”上传视频后，任务会立即出现在这里，并按七个环节逐项归档。</p></div></section>`}</div>`;
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

function guidedPipelineView(run) {
  const snapshot = run.observability || {};
  const receipts = new Map((snapshot.stage_receipts || []).map((item)=>[item.stage,item]));
  const rows = GUIDED_PIPELINE.map((definition,index)=>({ definition, ...guidedStageState(run,definition,receipts,index) }));
  const active = rows.find((item)=>["active","failed","interrupted"].includes(item.state)) || rows.find((item)=>item.state === "waiting") || rows.at(-1);
  const activeIndex = rows.indexOf(active);
  const next = rows.slice(activeIndex+1).find((item)=>item.state === "waiting");
  const status = snapshot.status || {};
  const stateLabel = { active:"正在进行", done:"已归档", "done-unreceipted":"已通过", waiting:"等待中", failed:"本环节失败", interrupted:"等待续跑" };
  const cards = rows.map(({definition,state:stageState,receipt})=>{
    const artifacts = (receipt?.artifacts || []).filter((item)=>item.available).map((item)=>item.name).filter(Boolean);
    const detail = stageState === "done"
      ? `${receipt?.stage_duration_seconds != null ? `耗时 ${duration(receipt.stage_duration_seconds)} · ` : ""}${artifacts.length ? `已确认 ${artifacts.slice(0,3).join("、")}` : "阶段回执已写入 NAS"}`
      : stageState === "active" ? definition.doing
      : stageState === "failed" ? `失败位置：${STAGE_LABELS[status.failed_stage] || status.failed_stage || run.error || "本环节"}`
      : stageState === "interrupted" ? "已完成产出保留，可从持久账本续跑。"
      : stageState === "done-unreceipted" ? "后续阶段已开始；该历史运行没有独立阶段回执。"
      : `完成后：${definition.outcome}`;
    return `<article class="journey-step ${stageState}"><span class="journey-number">${definition.number}</span><div class="journey-copy"><header><strong>${esc(definition.title)}</strong><span>${esc(stateLabel[stageState])}</span></header><p>${esc(detail)}</p><small>${icon("folder")} 归档到 ${esc(definition.folder)}</small></div></article>`;
  }).join("");
  return `<section class="current-guide ${active.state}"><div><span class="guide-kicker">${esc(stateLabel[active.state])} · 环节 ${esc(active.definition.number)}/07</span><h3>${esc(active.definition.title)}</h3><p>${esc(status.message || active.definition.doing)}</p></div><aside><small>${next ? "完成后进入" : "最终结果"}</small><strong>${esc(next?.definition.title || "完整 NAS 正式档案")}</strong><span>${esc(next?.definition.outcome || active.definition.outcome)}</span></aside></section><div class="pipeline-journey">${cards}</div>`;
}

function runObservabilityCard(run) {
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
  const archivePath = run.nas_staging || run.nas_output || "运行目录待登记";
  const gpuCompute = gpu["utilization.gpu"] ?? gpu.utilization_percent ?? "—";
  const nvdec = gpu["utilization.decoder"] ?? gpu.decoder_percent ?? "—";
  const memory = gpu["memory.used"] ?? gpu.memory_used_mib ?? "—";
  const freshnessNote = freshness.stale && !["completed","failed","interrupted"].includes(run.state)
    ? `<div class="freshness-warning">状态数据已 ${duration(freshness.age)} 未更新：NAS 遥测可能延迟，不能据此判定任务停止；页面仍会继续刷新。</div>` : "";
  const archiveLink = run.state === "completed" ? `<a class="secondary-button" href="#/archive/${encodeURIComponent(run.experiment_id || "")}/experiments">查看正式档案</a>` : "";
  return `<section class="panel live-run-card"><header class="panel-heading"><div><h2>${esc(run.experiment_id || run.run_id)}</h2><p>${esc(run.run_id)} · ${esc(archivePath)}</p></div><div class="run-heading-actions"><button class="secondary-button" type="button" data-copy-path="${esc(archivePath)}">${icon("copy")}复制当前归档路径</button>${archiveLink}<span class="queue-status ${esc(run.state)}"><i></i>${esc(STAGE_LABELS[current] || current)}</span></div></header>${freshnessNote}${guidedPipelineView(run)}<details class="technical-observability" ${!["completed"].includes(run.state) ? "open" : ""}><summary>查看实时性能、逐视角进度与 Token</summary><div class="live-metric-grid"><article><small>总体进度</small><strong>${Math.round(Number(run.progress ?? status.progress ?? 0)*100)}%</strong><span>${duration(elapsedForRun(run,status))} 已用</span></article><article><small>GPU / NVDEC（瞬时）</small><strong>${gpuCompute}% / ${nvdec}%</strong><span>单点样本 · ${memory} MiB 显存</span></article><article><small>CPU / 内存（瞬时）</small><strong>${live.cpu_percent ?? "—"}% / ${live.memory_percent ?? "—"}%</strong><span>单点主机采样</span></article><article><small>NAS/网络读取（瞬时）</small><strong>${network.received_mib_per_second ?? "—"} MiB/s</strong><span>进程树读 ${processIo.read_mib_per_second ?? "—"} MiB/s</span></article><article><small>模型 Token</small><strong>${number(tokens.total_tokens)}</strong><span>${number(tokens.input_tokens)} 输入 + ${number(tokens.output_tokens)} 输出</span></article><article><small>最近更新</small><strong>${freshness.value ? formatDate(freshness.value) : "待采样"}</strong><span>${freshness.age == null ? "正式运行账本" : `${duration(freshness.age)} 前`}</span></article></div>${views.length ? `<div class="view-runtime-grid">${views.map(([viewId,item])=>{ const done=item.completed_units ?? item.completed_work_units; const total=item.total_units ?? item.total_work_units; return `<article><span class="view-state-dot ${item.state === "completed" ? "completed" : ""}"></span><div><strong>${esc(viewId)}</strong><small>${esc(item.role || "待识别角色")} · ${esc(item.decode_backend || "待分配解码")} · ${esc(item.state || "waiting")} · ${total ? `${number(done)}/${number(total)} 单元` : `${number(item.segment_count)} 分片`}</small></div></article>`; }).join("")}</div>` : `<div class="empty-state compact-empty">等待逐视角运行账本；输入仍已纳入任务。</div>`}</details></section>`;
}

function renderOperations() {
  setChrome("operations");
  const health = state.health || {};
  const benchmark = health.fixed_benchmark || {};
  main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">SYSTEM HEALTH</p><h1>服务健康</h1><p>当前节点的 Web、NAS、模型凭据、固定基准和本地缓存位置。</p></div></header><section class="health-grid"><article class="health-item"><span>${icon("server")}</span><div><small>API 服务</small><strong>${esc(health.status || "不可用")}</strong></div></article><article class="health-item"><span>${icon("folder")}</span><div><small>NAS 正式归档根目录</small><strong>${esc(health.nas_archive_root || "未配置")}</strong></div></article><article class="health-item"><span>${icon("brain")}</span><div><small>多模态模型</small><strong>${esc(health.model || "未配置")} · Key ${health.ark_key_configured ? "已配置" : "未配置"}</strong></div></article><article class="health-item"><span>${icon("video")}</span><div><small>输入容量</small><strong>路数动态 · 已按至少 6 路×3 小时设计</strong></div></article></section><section class="panel"><header class="panel-heading"><div><h2>六路 3 小时固定基准</h2><p>后续重复验证直接复用同一正式档案，不再新建实验。</p></div><button class="primary-button" id="rerun-benchmark" type="button">${icon("activity")}立即重跑</button></header><table class="metric-table"><tbody><tr><td>NAS 字段索引</td><td>${esc(benchmark.index_csv || "未配置")}</td></tr><tr><td>固定归档</td><td>${esc(`${health.nas_archive_root || ""}\\${benchmark.archive_name || ""}`)}</td></tr><tr><td>原视频输入</td><td>${esc(benchmark.input_mode || "NAS 分片虚拟时间轴")}</td></tr><tr><td>运行控制与日志</td><td>${esc(benchmark.local_runtime_root || "未配置")}</td></tr><tr><td>YOLO/断点缓存</td><td>${esc(benchmark.local_cache_root || "未配置")}</td></tr></tbody></table></section></div>`;
  document.querySelector("#rerun-benchmark")?.addEventListener("click", rerunBenchmark);
}

async function loadArchive(name) {
  if (!state.archiveCache.has(name)) state.archiveCache.set(name, await api(`/api/archives/${encodeURIComponent(name)}`));
  return state.archiveCache.get(name);
}

function resultHeader(data, tab) {
  const tokens = data.metrics?.tokens?.run_total || {};
  const performance = data.metrics?.preprocessing_display || {};
  const fullColdStart = performance.full_cold_start || {};
  const preprocessing = fullColdStart.measured ? fullColdStart.seconds : data.metrics?.display_preprocessing_seconds;
  const endToEnd = data.metrics?.web_end_to_end?.total_duration_seconds ?? data.metrics?.fixed_benchmark_end_to_end?.total_duration_seconds;
  const quality = data.quality_acceptance || {};
  const boundary = quality.experiment_boundaries || {};
  const materials = quality.key_materials || {};
  const boundaryNote = boundary.evaluated
    ? `边界通过率 ${percent(boundary.boundary_pass_rate)}`
    : boundary.evidence_package_eval_passed
      ? `${number(boundary.structural_group_count ?? data.experiments.length)} 组结构/媒体已验收；无人工边界基线`
      : "未提供人工边界基线，不推算准确率";
  const explicitCount = materials.cross_view_or_explicit_uncertainty_count;
  const materialNote = materials.event_count
    ? `${number(materials.dual_view_material_count)}/${number(materials.event_count)} 双视角成套；${number(materials.cross_view_supported_count)} 项双侧共同佐证${explicitCount != null ? `；${number(explicitCount)} 项均有可审计关联` : ""}`
    : "暂无关键事件";
  const preprocessingLabel = fullColdStart.measured ? "历史完整冷启动预处理" : "本次预处理";
  const preprocessingNote = fullColdStart.measured ? "独立全量基准；不含模型理解" : "不含模型理解";
  return `<div class="result-header"><header class="page-hero compact"><div><p class="eyebrow">EVIDENCE ARCHIVE</p><h1>${esc(data.name)}</h1><p>第一人称与第三人称统一时间轴产出；仅保留通过有界审计的真实实验片段。</p></div><div class="hero-actions"><button class="secondary-button" type="button" data-copy-path="${esc(data.path)}">${icon("copy")}复制归档路径</button><button class="primary-button" type="button" data-open-folder="${esc(data.name)}">${icon("folder")}在资源管理器打开</button></div></header><div class="current-work"><span class="current-icon">${icon("folder")}</span><div><small>${archiveLabel()}位置（资源管理器直接粘贴）</small><h2 class="archive-path">${esc(data.path)}</h2><p>存储位置：${esc(data.network_path || data.path)} · 原视频、实验片段、关键素材、模型 JSON、耗时与 Token 均在此目录。</p></div><span class="badge">${archiveShortLabel()}</span></div><section class="status-grid">${statusCard("flask","有界实验",number(data.experiments.length),boundaryNote)}${statusCard("image","关键事件",number(data.key_events.length),materialNote)}${statusCard("clock",preprocessingLabel,duration(preprocessing),preprocessingNote)}${statusCard("token","总 Token",number(tokens.total_tokens),endToEnd != null ? `端到端 ${duration(endToEnd)}` : `${number(tokens.input_tokens)} 输入 + ${number(tokens.output_tokens)} 输出`)}</section><nav class="result-tabs"><a class="result-tab ${tab==="experiments"?"active":""}" href="#/archive/${encodeURIComponent(data.name)}/experiments">实验片段与步骤理解</a><a class="result-tab ${tab==="materials"?"active":""}" href="#/archive/${encodeURIComponent(data.name)}/materials">关键素材与当前/下一步</a><a class="result-tab ${tab==="reports"?"active":""}" href="#/archive/${encodeURIComponent(data.name)}/reports">实验室日报</a><a class="result-tab ${tab==="metrics"?"active":""}" href="#/archive/${encodeURIComponent(data.name)}/metrics">耗时、Token、验收与资源</a></nav></div>`;
}

function experimentCard(experiment) {
  return `<article class="experiment-card"><header><div><h2>${esc(experiment.name)}</h2><div class="timecode">${timecode(experiment.start_ms)} → ${timecode(experiment.end_ms)}</div></div><span class="badge">${experiment.continuity_type === "continuous" ? "连续实验" : "独立实验"}</span></header><div class="video-understanding"><div class="video-frame">${experiment.aligned_video_url ? `<video controls preload="metadata" src="${esc(experiment.aligned_video_url)}"></video>` : `<div class="empty-state">暂无对齐视频</div>`}</div><div class="understanding-panel"><div><strong>模型整体理解</strong><p>${esc(experiment.summary || "暂无模型摘要")}</p></div><div><strong>步骤粒度</strong><p>${number(experiment.steps?.length)} 个带时间边界的细粒度步骤；每步明确当前正在做什么和可支持的下一步。</p></div>${experiment.uncertainties?.length ? `<div><strong>不确定项</strong><p>${esc(experiment.uncertainties.join("；"))}</p></div>` : ""}</div></div><details ${experiment.steps?.length <= 10 ? "open" : ""}><summary class="secondary-button">查看 ${number(experiment.steps?.length)} 个细粒度步骤</summary><div class="step-list">${(experiment.steps || []).map((step,index)=>`<article class="step-card"><span class="step-number">步骤 ${esc(step.step_index ?? index+1)}<small>${timecode(step.start_global_ms)}<br/>${timecode(step.end_global_ms)}</small></span><div class="step-body"><strong>当前：${esc(step.current_step || step.observed_action || "未描述")}</strong><p class="step-next"><b>下一步：</b>${esc(step.next_step || "没有足够证据支持下一步")}</p><span class="step-meta">对象：${esc((step.objects || []).join("、") || "未明确")} · 置信度：${step.confidence ?? "—"}</span></div></article>`).join("")}</div></details></article>`;
}

function materialCard(event) {
  const mllm = event.provenance?.mllm || {};
  const facts = event.decision?.observed_facts || [];
  const next = mllm.next_step || event.decision?.supported_inferences?.[0] || "没有足够证据支持下一步";
  const cross = event.cross_view_associations?.[0];
  const dualView = eventHasDualViewSupport(event);
  return `<article class="material-card"><header><div><h2>${esc(event.event_id)} · ${esc(ACTION_LABELS[event.action_type] || event.action_type)}</h2><div class="timecode">${timecode(Number(event.start_us)/1000)} → ${timecode(Number(event.end_us)/1000)} · 峰值 ${timecode(Number(event.peak_timestamp_us)/1000)}</div></div><div class="material-badges"><span class="badge">双视角成套</span><span class="badge ${dualView ? "trusted" : "uncertain"}">${dualView ? "双侧共同佐证" : "双视角对齐 · 单侧动作清晰"}</span></div></header><div class="video-understanding"><div><div class="video-frame">${event.aligned_frame_url ? `<img loading="lazy" src="${esc(event.aligned_frame_url)}" alt="${esc(event.event_id)} 第一/第三人称并排对齐关键帧"/>` : ""}</div>${event.aligned_clip_url ? `<div class="video-frame" style="margin-top:8px"><video controls preload="none" src="${esc(event.aligned_clip_url)}"></video></div>` : ""}</div><div class="understanding-panel"><div><strong>当前步骤</strong><p>${esc(mllm.current_step || facts[0] || "未知")}</p></div><div><strong>下一步骤</strong><p>${esc(next)}</p></div><div><strong>双视角证据强度</strong><p>${esc(cross ? `${cross.consistency} · ${dualView ? "第一/第三人称均能共同佐证该动作" : "第一/第三人称素材均已对齐归档，但该动作只在其中一侧清晰可见"}` : "双视角素材已归档，但未建立动作级关联")}</p></div><div><strong>对象</strong><p>${esc(Object.entries(event.objects || {}).map(([key,value])=>`${key}: ${value}`).join("；") || "未明确")}</p></div>${facts.length ? `<div><strong>可观察事实</strong><ul class="fact-list">${facts.slice(0,3).map((fact)=>`<li>${esc(fact)}</li>`).join("")}</ul></div>` : ""}<div class="usage-box">Token：${number(mllm.usage?.input_tokens)} 输入 + ${number(mllm.usage?.output_tokens)} 输出 = ${number(mllm.usage?.total_tokens)}</div></div></div></article>`;
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
  if (!sections.length) return `<div class="empty-state"><strong>没有符合条件的关键素材</strong><p>可放宽实验、动作类型、跨视角状态或关键词筛选。</p></div>`;
  return `<div class="material-result-summary"><strong>显示 ${number(events.length)} / ${number(formalTotal)} 个双视角关键事件</strong><span>默认按实验分组；视频仅在需要播放时加载。</span></div>${sections.map(([groupId,items])=>{ const group = groupMap.get(groupId) || items[0].experiment_group || {}; const index = group.index == null ? "—" : String(group.index + 1).padStart(2,"0"); return `<section class="material-group"><header><span class="material-group-index">${index}</span><div><h2>${esc(group.name || groupId)}</h2><p>${timecode(group.start_ms)} → ${timecode(group.end_ms)} · ${group.continuity_type === "continuous" ? "连续实验" : "独立实验"}</p></div><span class="badge">${number(items.length)} 个事件</span></header><div class="material-grid">${items.map(materialCard).join("")}</div></section>`; }).join("")}`;
}

function materialsView(data) {
  ensureMaterialFilters(data);
  const filters = state.materialFilters;
  const formalEvents = data.key_events.filter(eventHasAlignedDualViewMaterial);
  const quarantinedCount = data.key_events.length - formalEvents.length;
  const actionTypes = [...new Set(formalEvents.map((event)=>event.action_type).filter(Boolean))];
  const actionCounts = Object.fromEntries(actionTypes.map((type)=>[type,formalEvents.filter((event)=>event.action_type===type).length]));
  return `<section class="material-workspace"><header class="material-toolbar-heading"><div><p class="eyebrow">DUAL-VIEW KEY MATERIAL EVIDENCE</p><h2>按实验查阅双视角关键素材</h2><p>正式素材必须同时包含第一人称、第三人称及其并排对齐帧/片段；缺少任一视角的候选不会进入正式素材库。</p></div><span class="badge">${number(formalEvents.length)} 个双视角事件</span></header>${quarantinedCount ? `<div class="freshness-warning"><strong>${number(quarantinedCount)} 个候选缺少成套双视角素材，已从正式关键素材库隔离。</strong></div>` : ""}<div class="material-filter-bar"><label><span>实验片段</span><select id="material-group-filter"><option value="all" ${filters.group==="all"?"selected":""}>全部实验（${number(formalEvents.length)}）</option>${(data.experiment_groups||[]).map((group,index)=>`<option value="${esc(group.group_id)}" ${filters.group===group.group_id?"selected":""}>${String(index+1).padStart(2,"0")} · ${esc(group.name)}（${number(group.key_event_count)}）</option>`).join("")}</select></label><label><span>动作类型</span><select id="material-action-filter"><option value="all">全部五类动作</option>${actionTypes.map((type)=>`<option value="${esc(type)}" ${filters.action===type?"selected":""}>${esc(ACTION_LABELS[type]||type)}（${number(actionCounts[type])}）</option>`).join("")}</select></label><label><span>双视角证据强度</span><select id="material-support-filter"><option value="all">全部双视角事件</option><option value="dual" ${filters.support==="dual"?"selected":""}>双侧共同佐证动作</option><option value="partial" ${filters.support==="partial"?"selected":""}>双视角已对齐 · 单侧动作清晰</option></select></label><label class="material-query"><span>搜索步骤、对象或事件 ID</span><input id="material-query" value="${esc(filters.query)}" placeholder="例如：移液器、开盖、EVT-000010" /></label></div><div id="material-results">${materialResults(data)}</div></section>`;
}

function bindMaterialFilters(data) {
  const rerender = () => { document.querySelector("#material-results").innerHTML = materialResults(data); };
  document.querySelector("#material-group-filter")?.addEventListener("change", (event)=>{ state.materialFilters.group=event.target.value; rerender(); });
  document.querySelector("#material-action-filter")?.addEventListener("change", (event)=>{ state.materialFilters.action=event.target.value; rerender(); });
  document.querySelector("#material-support-filter")?.addEventListener("change", (event)=>{ state.materialFilters.support=event.target.value; rerender(); });
  document.querySelector("#material-query")?.addEventListener("input", (event)=>{ state.materialFilters.query=event.target.value; rerender(); });
}

function dailyReportView(data) {
  const report = data.daily_report || {};
  const overview = report.overview || {};
  const alignment = report.alignment_summary || {};
  const performance = report.performance || {};
  const actions = report.action_summary || [];
  const timeline = report.experiment_timeline || [];
  const fullRunPreprocessing = performance.full_run_preprocessing || {};
  if (!report.report_id) return `<section class="panel"><div class="empty-state"><strong>该历史档案尚未生成实验室日报</strong><p>下一次运行流水线会自动生成；也可使用 CLI 对已验收档案补生成。</p></div></section>`;
  const links = [
    ["daily_report_pdf", "正式 PDF"],
    ["daily_report_html", "HTML 报告"],
    ["daily_report_markdown", "Markdown"],
    ["daily_report_json", "事实源 JSON"],
    ["daily_report_eval", "日报验收 JSON"],
  ].filter(([key]) => data.links?.[key]);
  return `<section class="daily-report-cover"><div><p class="eyebrow">EVIDENCE-BACKED DAILY REPORT</p><h2>实验室日报 · ${esc(report.report_date)}</h2><p>${esc(report.experiment_id)}</p></div><span class="badge">${overview.evidence_package_eval_passed ? "证据包已验收" : "待验收"}</span><div class="daily-report-links">${links.map(([key,label])=>`<a class="secondary-button" target="_blank" href="${esc(data.links[key])}">${icon("file")}${label}</a>`).join("")}</div></section>
  <section class="status-grid">${statusCard("video","输入视角",number(overview.input_view_count),`${number(overview.first_person_views)} 第一人称 + ${number(overview.third_person_views)} 第三人称`)}${statusCard("flask","有界实验",number(overview.experiment_group_count),"独立/连续实验组")}${statusCard("image","关键事件",number(overview.key_event_count),`${number(overview.physical_change_count)} 项状态变化`)}${statusCard("token","日报新增 Token",number(report.source_policy?.additional_model_tokens?.total_tokens),"复用已有模型理解")}</section>
  <section class="panel"><header class="panel-heading"><div><h2>实验时间线与步骤理解</h2><p>时间均为统一实验相对时间；每项保留证据事件与跨视角素材链接。</p></div></header><div class="daily-timeline">${timeline.map((group,index)=>`<article><span class="daily-index">${String(index+1).padStart(2,"0")}</span><div><header><h3>${esc(group.experiment_name)}</h3><span class="badge">${group.continuity_type === "continuous" ? "连续实验" : "独立实验"}</span></header><p class="timecode">${esc(group.start_timecode)} → ${esc(group.end_timecode)}</p><p>${esc(group.overall_summary || "无模型摘要")}</p><details><summary>查看 ${number(group.steps?.length)} 个细粒度步骤</summary><div class="step-list">${(group.steps||[]).map((step)=>`<article class="step-card"><span class="step-number">步骤 ${esc(step.step_index)}<small>${esc(step.start_timecode)}<br/>${esc(step.end_timecode)}</small></span><div class="step-body"><strong>当前：${esc(step.current_step || "未说明")}</strong><p class="step-next"><b>下一步：</b>${esc(step.next_step || "证据不足")}</p><span class="step-meta">对象：${esc((step.objects||[]).join("、") || "未明确")} · ${number(step.supporting_views?.length)} 路证据</span></div></article>`).join("")}</div></details></div></article>`).join("")}</div></section>
  <section class="panel"><header class="panel-heading"><div><h2>五类动作、质量与成本</h2><p>观察事实、证据支持的模型理解和不确定项在 JSON 中分别保存。</p></div></header><div class="daily-action-grid">${actions.map((item)=>`<article><strong>${number(item.event_count)}</strong><span>${esc(item.action_label)}</span></article>`).join("")}</div><table class="metric-table"><tbody><tr><td>时间对齐</td><td>${number(alignment.aligned)}/${number(alignment.view_count)} 路 aligned，平均置信度 ${alignment.mean_confidence ?? "—"}</td></tr><tr><td>不确定性 / 矛盾</td><td>${number(report.uncertainties?.length)} 组 / ${number(report.contradictions?.length)} 项</td></tr><tr><td>本次流水线总耗时</td><td>${duration(performance.total_duration_seconds)}</td></tr><tr><td>本次流水线预处理</td><td>${duration(performance.preprocessing_sla?.actual_seconds)}（不含模型理解；可能复用已验收 CV 账本）</td></tr>${fullRunPreprocessing.seconds != null ? `<tr><td>全量六路预处理验收</td><td>${duration(fullRunPreprocessing.seconds)}（${esc(fullRunPreprocessing.includes || "完整预处理") }）</td></tr>` : ""}<tr><td>总 Token</td><td>${number(performance.total_input_tokens)} 输入 + ${number(performance.total_output_tokens)} 输出 = ${number(performance.total_tokens)}</td></tr><tr><td>人工复核</td><td>${esc(report.human_review?.status || "pending")}</td></tr></tbody></table></section>`;
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
  const retention = webIngest?.retention_mode === "nas_only" ? "浏览器上传并直接留存 NAS（本地不复制）" : "浏览器上传并留存本地与 NAS";
  const extraRows = `${webIngest?.duration_seconds != null ? `<tr><td>input_ingest</td><td>${metrics.web_ingest ? retention : "NAS 索引输入准备/复用"}</td><td>${duration(webIngest.duration_seconds)}</td></tr>` : ""}${endToEnd?.total_duration_seconds != null ? `<tr><td>end_to_end</td><td>请求进入至最终 NAS 归档完成</td><td>${duration(endToEnd.total_duration_seconds)}</td></tr>` : ""}`;
  const quality = data.quality_acceptance || {};
  const boundary = quality.experiment_boundaries || {};
  const materials = quality.key_materials || {};
  const materialEvaluation = keyMaterialEvaluationSummary(data);
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
  };
  return `<section class="panel"><header class="panel-heading"><div><h2>耗时口径</h2><p>${performanceSummary} ${reuseNote}</p></div></header><div class="performance-compare"><article><small>历史完整冷启动预处理</small><strong>${duration(fullColdStart.seconds)}</strong><span>${esc(fullColdStart.includes || "预检 + 对齐 + 全量粗扫 + 有界精扫 + 边界审计；不含模型理解")}</span></article><article><small>当前归档运行总耗时</small><strong>${duration(currentRun.total_seconds ?? metrics.total_duration_seconds)}</strong><span>${currentRun.reused_validated_cv_ledgers ? "复用已验收 CV 账本，重新生成理解/媒体/证据包" : "以本次运行账本为准"}</span></article><article><small>当前运行预处理</small><strong>${duration(currentRun.preprocessing_seconds)}</strong><span>${currentRun.reused_validated_cv_ledgers ? "不是冷启动基准" : "当前运行实际值"}</span></article></div><table class="metric-table"><thead><tr><th>阶段</th><th>说明</th><th>耗时</th></tr></thead><tbody>${extraRows}${stages.map((stage)=>`<tr><td>${esc(stage.stage)}</td><td>${esc(STAGE_LABELS[stage.stage] || stage.stage)}</td><td>${duration(stage.duration_seconds)}</td></tr>`).join("")}</tbody></table></section><section class="panel"><header class="panel-heading"><div><h2>Token 用量</h2><p>CV、FFmpeg 与 TensorRT 不消耗模型 Token；断点复用的模型结果不重复计入本次实际消耗。</p></div></header><table class="metric-table"><thead><tr><th>阶段</th><th>执行 / 复用</th><th>输入 Token</th><th>输出 Token</th><th>总 Token</th></tr></thead><tbody><tr><td>实验片段步骤理解</td><td>${number(tokens.experiment_groups?.executed_call_count ?? tokens.experiment_groups?.call_count)} / ${number(tokens.experiment_groups?.reused_call_count)}</td><td>${number(tokens.experiment_groups?.input_tokens)}</td><td>${number(tokens.experiment_groups?.output_tokens)}</td><td>${number(tokens.experiment_groups?.total_tokens)}</td></tr><tr><td>关键素材理解</td><td>${number(tokens.key_materials?.executed_call_count ?? tokens.key_materials?.call_count)} / ${number(tokens.key_materials?.reused_call_count)}</td><td>${number(tokens.key_materials?.input_tokens)}</td><td>${number(tokens.key_materials?.output_tokens)}</td><td>${number(tokens.key_materials?.total_tokens)}</td></tr><tr><td><strong>全任务</strong></td><td>—</td><td><strong>${number(tokens.run_total?.input_tokens)}</strong></td><td><strong>${number(tokens.run_total?.output_tokens)}</strong></td><td><strong>${number(tokens.run_total?.total_tokens)}</strong></td></tr></tbody></table></section><section class="panel"><header class="panel-heading"><div><h2>质量验收</h2><p>${esc(materialEvaluation.displayNote)}</p></div></header><table class="metric-table"><tbody><tr><td>总体状态</td><td>${esc(statusLabel)}</td></tr><tr><td>实验检出 Precision / Recall</td><td>${boundaryAccuracy}</td></tr><tr><td>关键素材 Precision / Recall</td><td>${esc(materialEvaluation.recallLabel)}</td></tr><tr><td>边界通过率 / 连续性准确率</td><td>${boundaryContinuity}</td></tr><tr><td>证据包结构与媒体</td><td>${boundary.evidence_package_eval_passed || materials.evidence_package_eval_passed ? "通过自动验收" : "未通过或无验收记录"}</td></tr><tr><td>关键素材五类覆盖</td><td>${esc(materialEvaluation.categoryCoverageLabel)}</td></tr><tr><td>第一/第三人称成套素材</td><td>${number(materials.dual_view_material_count)} / ${number(materials.event_count)}（${percent(materials.dual_view_material_rate)}）</td></tr><tr><td>双侧共同佐证动作</td><td>${number(materials.cross_view_supported_count)} / ${number(materials.event_count)}（${percent(materials.cross_view_supported_rate)}）</td></tr><tr><td>双视角关联可审计</td><td>${number(materials.cross_view_or_explicit_uncertainty_count)} / ${number(materials.event_count)}</td></tr></tbody></table></section><section class="panel"><header class="panel-heading"><div><h2>资源遥测（按阶段）</h2><p>来源：resource_telemetry.json；主机网络包含其他流量，进程树 I/O 单独列出。</p></div></header><table class="metric-table"><thead><tr><th>阶段</th><th>GPU mean/max</th><th>NVDEC mean/max</th><th>CPU mean/max</th><th>网络接收 mean/max</th></tr></thead><tbody>${Object.entries(summaries).map(([stage,item])=>`<tr><td>${esc(STAGE_LABELS[stage]||stage)}</td><td>${metricStat(item.gpu_compute_percent)}/${metricStat(item.gpu_compute_percent,"max","%")}</td><td>${metricStat(item.nvdec_percent)}/${metricStat(item.nvdec_percent,"max","%")}</td><td>${metricStat(item.cpu_percent)}/${metricStat(item.cpu_percent,"max","%")}</td><td>${metricStat(item.host_network_receive_mib_s,"mean"," MiB/s")}/${metricStat(item.host_network_receive_mib_s,"max"," MiB/s")}</td></tr>`).join("")}</tbody></table></section><section class="panel"><header class="panel-heading"><div><h2>模型理解与验收文件</h2><p>这些链接直接指向当前 NAS 档案中的正式文件；历史档案没有的文件不会显示为可点击链接。</p></div></header><div class="result-links" style="padding:20px">${Object.entries(links).map(([key,label])=>data.links[key]?`<a target="_blank" href="${esc(data.links[key])}">${icon("file")}${label}</a>`:"").join("")}</div></section>`;
}

async function renderArchive(name, tab = "experiments") {
  setChrome("archive", name);
  main.innerHTML = `<div class="page-loading"><span class="spinner"></span><strong>正在读取${archiveLabel()}</strong></div>`;
  try {
    const data = await loadArchive(name);
    main.innerHTML = `<div class="page">${resultHeader(data,tab)}${tab === "materials" ? materialsView(data) : tab === "reports" ? dailyReportView(data) : tab === "metrics" ? metricsView(data) : `<section class="step-list">${data.experiments.map(experimentCard).join("")}</section>`}</div>`;
    bindArchiveActions();
    if (tab === "materials") bindMaterialFilters(data);
  } catch (error) {
    main.innerHTML = `<div class="empty-state"><strong>无法读取该实验档案</strong><p>${esc(error.message)}</p><a class="secondary-button" href="#/experiments">返回实验记录</a></div>`;
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
  main.innerHTML = `<div class="page-loading"><span class="spinner"></span><strong>正在读取 YOLO badcase 队列</strong></div>`;
  try {
    const data = await api(`/api/annotation-workspace?${query}`);
    const summary = data.summary || {};
    main.innerHTML = `<div class="page annotation-page"><header class="page-hero compact"><div><p class="eyebrow">MODEL QUALITY WORKBENCH</p><h1>YOLO badcase 标注工作台</h1><p>把模型漏检、误检与类别混淆转成可复核训练数据。CV 推理结果与人工真值严格分离，只有人工实际提交的框才进入导出。</p></div><div class="hero-actions"><a class="secondary-button" target="_blank" href="/api/annotation-workspace/export">${icon("file")}导出已审核真值</a></div></header>
      <section class="status-grid">${statusCard("target","badcase 总数",number(summary.total),"来自已冻结审计队列")}${statusCard("check","已审核",number(summary.reviewed),"决策已写本地可追溯账本")}${statusCard("clock","待审核",number(summary.pending),"按 P0/P1/P2 排序处理")}${statusCard("file","P0 高优先级",number(summary.priority_counts?.P0),"确认漏检和高风险问题")}</section>
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

function bindArchiveActions() {
  document.querySelectorAll("[data-copy-path]").forEach((button) => button.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(button.dataset.copyPath); toast("归档路径已复制。" ); }
    catch { toast(`请手工复制：${button.dataset.copyPath}`, "error"); }
  }));
  document.querySelectorAll("[data-open-folder]").forEach((button) => button.addEventListener("click", async () => {
    try { const result = await api(`/api/archives/${encodeURIComponent(button.dataset.openFolder)}/open`, { method: "POST" }); toast(`已打开：${result.path}`); }
    catch (error) { toast(error.message, "error"); }
  }));
}

function routeParts() {
  return location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
}

async function router() {
  const parts = routeParts();
  const route = parts[0] || "home";
  if (route === "archive" && parts[1]) return renderArchive(parts[1], parts[2] || "experiments");
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
  toast("NAS 档案与任务状态已刷新。" );
});
window.addEventListener("hashchange", router);
window.setInterval(refreshTaskSnapshots, 4000);

hydrateIcons();
const legacyArchive = new URLSearchParams(location.search).get("archive");
if (legacyArchive && !location.hash) location.hash = `#/archive/${encodeURIComponent(legacyArchive)}/experiments`;
loadAll().then(router).catch((error) => {
  main.innerHTML = `<div class="empty-state"><strong>无法连接分析服务</strong><p>${esc(error.message)}</p></div>`;
});
