"use strict";

// Explicit application ports keep domain modules independently testable.
window.VisionCortexUploads = Object.freeze({
createUploads(context) {
  const { delay, api, beginStageFollow, formatBytes, pollRun, renderStages, reviewState, setPhase, setProgress, state, submitCollectionRun, toast } = context;

const UPLOAD_SESSION_CACHE_KEY = "visioncortex.large-upload-session.v1";

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


  return Object.freeze({ buildUploadPlan, cachedUploadSession, rememberUploadSession, forgetUploadSession, createOrResumeUploadSession, validateBrowserPlan, sha256Fallback, chunkSha256, sendUploadChunk, uploadPlanFiles, cancelActiveUpload, submitRun });
}
});
