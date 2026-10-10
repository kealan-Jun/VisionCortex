/* Experiment-scoped recording playback, subtitles and literal transcript search. */
(() => {
  const clock = seconds => {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    return `${String(Math.floor(total / 3600)).padStart(2, "0")}:${String(Math.floor(total / 60) % 60).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
  };
  async function render({container, name, staging, release, experiments = [], api, esc, toast, isCurrent}) {
    const endpoint = `/api/${staging ? "staging-runs" : "archives"}/${encodeURIComponent(name)}/speech`;
    const route = new URLSearchParams(location.hash.split("?")[1] || "");
    let query = route.get("q") || "", selected = route.get("chunk") || "", data, requestId = 0, chunks = [];
    const current = () => container.isConnected && isCurrent();
    const fetchRows = (offset = 0) => {
      const params = new URLSearchParams({q: query, offset: String(offset), limit: "100", fold:String(container.querySelector("#speech-fold")?.checked ?? true), aliases:String(container.querySelector("#speech-aliases")?.checked ?? true)});
      const hint=container.querySelector("#speech-hint")?.value; if(hint) params.set("hint",hint);
      if (selected) params.set("chunk", selected);
      if (release && !staging) params.set("release", release);
      return api(`${endpoint}?${params}`);
    };
    container.innerHTML = '<p role="status">正在读取本实验的录音…</p>';
    try {
      data = await fetchRows();
      if (!current()) return;
      chunks = data.sources.flatMap(source => source.chunks.map(chunk => ({...chunk, source})));
      const sync = window.VisionCortexSpeechSync;
      const timeline = data.timeline;
      const videos = [...(timeline?.videos || []), ...experiments.filter(item => item.aligned_video_url && Number.isFinite(item.start_ms) && item.end_ms > item.start_ms).map(item => ({
        view_id:"aligned", role:"aligned", url:item.aligned_video_url, anchors:[[0,item.start_ms],[(item.end_ms-item.start_ms)/1000,item.end_ms]], name:item.name,
      }))];
      const views = [...new Set(videos.map(item => item.view_id))];
      const steps = experiments.flatMap(item => (item.steps || []).map(step => ({...step, experiment_name:item.name})));
      const statusLabels = {disabled:"本次分析未启用录音转写。", not_available:"该实验尚无录音转写产出。", running:"录音转写进行中，以下为已完成的部分。", failed:"录音处理未完成，以下内容仅为已保存的阶段产出。", completed:"录音处理已完成。"};
      container.innerHTML = `<header class="panel-heading"><div><h2>录音与转写</h2><p>${esc(statusLabels[data.status] || "录音状态待核验")}</p></div></header>
        <p>转写文字尚未经人工校对；录音中的说法不能直接证明实验动作已完成。</p>
        ${data.sources.filter(source => !source.available).map(source => `<p class="analysis-readiness-note">${esc(source.view_id)} · 第 ${source.segment_ordinal + 1} 段：${esc(source.message || "没有可用录音")}</p>`).join("")}
        ${chunks.length ? `<div class="speech-search"><label class="field-label">录音来源<select id="speech-source"><option value="">搜索本实验全部录音</option>${chunks.map(chunk=>`<option value="${esc(chunk.id)}">${esc(chunk.source.view_id)} · 视频第 ${chunk.source.segment_ordinal + 1} 段 · ${clock(chunk.start_seconds)}–${clock(chunk.end_seconds)}</option>`).join("")}</select></label></div>
        <div class="speech-timeline"><div class="speech-video-panel"><label class="field-label">同步画面（浏览器预览）<select id="speech-view">${views.map(view=>`<option value="${esc(view)}">${esc(view === "aligned" ? "实验片段 · 双视角" : view)}</option>`).join("")}</select></label><video id="speech-video" muted playsinline preload="metadata" hidden></video><p id="speech-sync-status" role="status">${timeline ? "选择文字即可跳到对应音画。" : "本次实验尚无同步索引。"}</p></div><div id="speech-player"></div></div><label class="field-label speech-global-seek"><span>实验时间 <output id="speech-global-time">—</output></span><input id="speech-global-seek" type="range" min="0" max="1" step="0.01" value="0" aria-label="实验时间轴" disabled /></label><div id="speech-steps" class="speech-steps"></div><form id="speech-search" class="speech-search"><label class="field-label"><span>搜索转写文字</span><input name="query" maxlength="200" placeholder="例如：拍照、加入、温度" /></label><button class="secondary-button" type="submit">搜索本实验</button><button class="secondary-button" type="button" id="speech-search-library">跨实验搜索</button><label><input type="checkbox" id="speech-fold" checked /> 折叠重复文字</label><label><input type="checkbox" id="speech-aliases" checked /> 扩展术语</label><label>声源提示<select id="speech-hint"><option value="">全部</option><option value="possible_device_playback">疑似设备播报</option><option value="possible_operation_narration">疑似操作口述</option><option value="unknown">声源未知</option></select></label></form>
        <p id="speech-count" role="status"></p><div id="speech-lines" class="speech-lines"></div><button class="secondary-button" id="speech-more" type="button" hidden>加载更多</button>` : ""}`;
      window.VisionCortexSpeechTools?.mount({container,data,endpoint,name,staging,api,esc,toast,current,clock});
      if (!chunks.length) return;
      const player = container.querySelector("#speech-player");
      const lines = container.querySelector("#speech-lines");
      const more = container.querySelector("#speech-more");
      let playingId = "", synchronizer = null, seekGeneration = 0;
      const video = container.querySelector("#speech-video");
      const videoView = container.querySelector("#speech-view");
      const syncStatus = container.querySelector("#speech-sync-status");
      const globalSeek = container.querySelector("#speech-global-seek");
      const globalTime = container.querySelector("#speech-global-time");
      const mapped = (timeline?.audio || []).filter(item => item.anchors.length > 1);
      globalSeek.disabled = !mapped.length;
      globalSeek.max = String(Math.max(1, ...mapped.map(item=>item.anchors.at(-1)[1]/1000)));
      videoView.value = views.includes("aligned") ? "aligned" : views.find(view => videos.find(item=>item.view_id===view)?.role === "first_person") || views[0] || "";
      videoView.addEventListener("change", ()=>synchronizer?.sync(true));
      function showTime(globalMs) {
        globalTime.textContent = globalMs === null ? "偏移未知或时钟缺口" : clock(globalMs/1000);
        if (globalMs !== null) globalSeek.value = String(globalMs/1000);
        const audio = player.querySelector("audio");
        for (const row of lines.querySelectorAll(".speech-row")) row.classList.toggle("active", row.dataset.chunk === playingId && audio && audio.currentTime >= Number(row.dataset.start) && audio.currentTime < Number(row.dataset.end));
        for (const button of container.querySelectorAll("[data-speech-step]")) button.classList.toggle("active", globalMs !== null && globalMs >= Number(button.dataset.start) && globalMs < Number(button.dataset.end));
      }
      function seekGlobal(globalMs, autoplay = false) {
        const found = sync.locateAudio(timeline?.audio, chunks, globalMs);
        if (found) { playChunk(found.chunk.id, found.seconds, autoplay); return; }
        player.querySelector("audio")?.pause(); video.pause();
        syncStatus.textContent = "此实验时刻没有已对齐的录音，无法联动回放。";
        globalTime.textContent = clock(globalMs/1000);
      }
      globalSeek.addEventListener("input", ()=>seekGlobal(Number(globalSeek.value)*1000));
      const stepList = container.querySelector("#speech-steps");
      if (!steps.length) stepList.textContent = "本次尚无已确认实验步骤；可按转写文字和录音理解定位。";
      for (const [index, step] of steps.entries()) {
        const button = document.createElement("button");
        button.type = "button"; button.className = "secondary-button";
        button.dataset.speechStep = String(index); button.dataset.start = String(step.start_global_ms); button.dataset.end = String(step.end_global_ms);
        button.textContent = `${clock(step.start_global_ms/1000)} · ${step.current_step || step.title || step.description || `步骤 ${index+1}`}`;
        button.addEventListener("click", ()=>seekGlobal(step.start_global_ms, true)); stepList.append(button);
      }
      function playChunk(id, seek = null, autoplay = true) {
        const chunk = chunks.find(item => item.id === id);
        if (!chunk) return;
        if (playingId !== id) {
          synchronizer?.dispose();
          player.querySelector("audio")?.pause();
          playingId = id;
          globalSeek.disabled = !(timeline?.audio?.find(item=>item.chunk_id===id)?.anchors.length > 1);
          const labels = {"audio.m4a":"下载试听录音", "transcript.srt":"SRT 字幕", "transcript.vtt":"VTT 字幕", "transcript.txt":"转写文本", "transcript.json":"转写 JSON"};
          player.innerHTML = `<h3>${esc(chunk.source.view_id)} · 第 ${chunk.source.segment_ordinal + 1} 段录音</h3><p>${chunk.source.alignment === "NOT_PROVEN" ? "与视频的时间偏移未知；按录音时间播放。" : "已记录与视频的对齐依据，实际音画同步仍待复核。"}</p><audio controls preload="metadata" src="${esc(chunk.files['audio.m4a'].url)}"><track kind="subtitles" srclang="zh" label="自动转写" src="${esc(chunk.files['transcript.vtt'].url)}" /></audio><div class="speech-caption" aria-live="off">播放后显示字幕</div><div class="speech-downloads">${Object.entries(labels).map(([key, label])=>`<a class="secondary-button" href="${esc(chunk.files[key].url)}" download="${esc(chunk.id)}-${key}">${label}</a>`).join("")}</div>`;
          const audio = player.querySelector("audio");
          synchronizer = sync.bind({audio, video, mapping:timeline?.audio?.find(item=>item.chunk_id===id), videos,
            selectedView:()=>videoView.value, status:syncStatus, current:()=>current() && player.querySelector("audio")===audio, onTime:showTime});
          const caption = player.querySelector(".speech-caption");
          const track = audio.textTracks[0];
          if (track) {
            track.mode = "hidden";
            track.addEventListener("cuechange", () => {
              caption.textContent = Array.from(track.activeCues || []).map(cue => cue.getCueAsHTML().textContent).join(" ") || "…";
            });
          }
          player.querySelector("track").addEventListener("error", () => { caption.textContent = "字幕暂不可用，可查看下方文字或下载字幕。"; });
        }
        if (seek !== null) {
          if (autoplay) container.querySelector(".speech-timeline")?.scrollIntoView({behavior:"smooth", block:"start"});
          const audio = player.querySelector("audio");
          const generation = ++seekGeneration;
          const start = () => { if (!current() || generation !== seekGeneration || player.querySelector("audio") !== audio) return; audio.currentTime = Math.min(Math.max(0, seek), audio.duration || 0); synchronizer?.sync(true); if (autoplay) audio.play().catch(error=>{ if (error.name !== "AbortError" && current() && player.querySelector("audio") === audio) toast("请点击播放按钮收听录音。"); }); };
          if (audio.readyState) start(); else audio.addEventListener("loadedmetadata", start, {once:true});
        }
      }
      function showRows(result, append = false) {
        data = result;
        if (!append) lines.innerHTML = "";
        const firstIndex = lines.children.length;
        function rowButton(row, i) {
          const button = document.createElement("button");
          button.type = "button"; button.className = "speech-row";
          button.innerHTML = `<time>${clock(row.start_seconds)}</time><span>${esc(row.text)}<small>${esc(row.view_id)} · ${esc(row.source_hint?.label || "声源未知")} · 录音提及${row.occurrence_count > 1 ? ` · 重复 ${row.occurrence_count} 次（首条提示）` : ""}</small></span><span aria-hidden="true">▶</span>`;
          button.title = row.source_hint?.basis || "声源未确认";
          button.dataset.row = String(firstIndex + i); button.dataset.reference = row.reference_id || row.id;
          button.dataset.chunk = row.chunk_id; button.dataset.start = String(row.playback_start_seconds); button.dataset.end = String(row.playback_end_seconds);
          button.addEventListener("click", () => playChunk(row.chunk_id, row.playback_start_seconds));
          return button;
        }
        result.segments.forEach((row, i) => {
          if (!(row.occurrence_count > 1)) { lines.append(rowButton(row,i)); return; }
          const group=document.createElement("div"); group.className="speech-repeat-group";
          group.append(rowButton(row,i));
          const details=document.createElement("details"); details.innerHTML=`<summary>展开全部 ${row.occurrence_count} 次出现</summary><div></div><button type="button" class="secondary-button">加载其余出现位置</button>`;
          const occurrences=details.querySelector("div"), loadMore=details.querySelector("button");
          let next=0, loading=false;
          async function loadOccurrences() {
            if(loading || next===null) return; loading=true; loadMore.disabled=true;
            try {
              const params=new URLSearchParams({q:query,phrase:row.phrase_id,offset:String(next),limit:"100",fold:"false",aliases:String(container.querySelector("#speech-aliases").checked)});
              if(release && !staging) params.set("release",release);
              const hint=container.querySelector("#speech-hint").value; if(hint) params.set("hint",hint);
              const found=await api(`${endpoint}?${params}`);
              if(!current() || !details.isConnected) return;
              found.segments.forEach((item,j)=>occurrences.append(rowButton(item,j)));
              next=found.next_offset; loadMore.hidden=next===null;
            } catch(error) { if(current()) toast(error.message,"error"); }
            finally { loading=false; loadMore.disabled=false; }
          }
          details.addEventListener("toggle",()=>{if(details.open && !occurrences.children.length) loadOccurrences();});
          loadMore.addEventListener("click",loadOccurrences); group.append(details); lines.append(group);
        });
        const terms=(result.search?.expanded_terms || []).filter(Boolean);
        container.querySelector("#speech-count").textContent = `${result.occurrence_total ?? result.total} 次出现${result.folded ? `，折叠为 ${result.total} 组` : ""}，已显示 ${lines.children.length} ${result.folded ? "组" : "条"}${query && terms.length ? `；匹配词：${terms.join("、")}` : ""}。声源提示仅供核对。`;
        more.hidden = result.next_offset === null;
      }
      async function refresh(append = false) {
        const id = ++requestId;
        more.disabled = true;
        try {
          const result = await fetchRows(append ? data.next_offset : 0);
          if (current() && id === requestId) showRows(result, append);
        } catch (error) { if (current() && id === requestId) toast(error.message, "error"); }
        finally { if (current() && id === requestId) more.disabled = false; }
      }
      container.querySelector("#speech-source").addEventListener("change", event => {
        selected = event.target.value;
        if (selected) playChunk(selected);
        refresh();
      });
      container.querySelector("#speech-search").addEventListener("submit", event => {
        event.preventDefault(); query = new FormData(event.target).get("query").trim(); refresh();
      });
      for(const id of ["speech-fold","speech-aliases","speech-hint"]) container.querySelector(`#${id}`).addEventListener("change",()=>refresh());
      container.querySelector('#speech-search input[name="query"]').value=query;
      container.querySelector("#speech-search-library").addEventListener("click",()=>{
        const term=container.querySelector('#speech-search input[name="query"]').value.trim();
        if(!term) { toast("请先输入搜索词。"); return; }
        window.VisionCortexSpeechTools?.library({container,query:term,aliases:container.querySelector("#speech-aliases").checked,api,esc,current,clock});
      });
      more.addEventListener("click", () => refresh(true));
      if (selected && !chunks.some(chunk => chunk.id === selected)) selected = "";
      container.querySelector("#speech-source").value = selected;
      const seek = Number(route.get("t"));
      playChunk(selected || chunks[0].id, route.has("t") && Number.isFinite(seek) ? seek : null, false);
      showRows(data);
      if (route.has("g") && Number.isFinite(Number(route.get("g")))) seekGlobal(Number(route.get("g")));
      if (data.model_understanding?.parts?.length) {
        const understanding = document.createElement("section");
        const title = document.createElement("h3");
        title.textContent = "录音理解 · 结合抽样画面";
        const note = document.createElement("p");
        note.textContent = "未发现可确认的实验片段，以下仅解释口述内容与抽样画面的关系；不代表已确认实验步骤。";
        understanding.append(title, note);
        for (const part of data.model_understanding.parts) {
          if (part.status !== "completed") continue;
          const paragraph = document.createElement("p");
          paragraph.textContent = part.speech_interpretation.summary;
          understanding.append(paragraph);
          const refs = new Set(part.speech_interpretation.referenced_segment_ids);
          for (const row of part.speech_context.segments.filter(row => refs.has(row.id))) {
            const button = document.createElement("button");
            button.type = "button"; button.className = "secondary-button";
            button.textContent = `回听 ${clock(row.playback_start_seconds)} · ${row.text}`;
            button.addEventListener("click", () => playChunk(row.chunk_id, row.playback_start_seconds));
            understanding.append(button);
          }
        }
        container.insertBefore(understanding, container.querySelector("#speech-search"));
      }
    } catch (error) {
      if (current()) container.innerHTML = `<p role="alert">${esc(error.message)}</p>`;
    }
  }
  function attach({main, data, name, staging}) {
    const relations = {consistent:"与画面一致", contradiction:"与画面有冲突", unrelated:"与画面无关", uncertain:"与画面关系不确定"};
    for (const experiment of data.experiments || []) {
      const card = document.getElementById(`experiment-${experiment.folder}`);
      (experiment.steps || []).forEach((step, index) => {
        const panel = card?.querySelector(`[data-experiment-step-panel="${index}"]`);
        if (!panel || panel.querySelector(".speech-step-link") || !Number.isFinite(step.start_global_ms)) return;
        const link = document.createElement("a"); link.className = "secondary-button speech-step-link";
        link.href = `#/${staging ? "stage" : "archive"}/${encodeURIComponent(name)}/speech?g=${step.start_global_ms}`;
        link.textContent = "在音画时间轴中查看"; panel.append(link);
      });
      const interpretation = experiment.speech_interpretation;
      if (!card || !main.contains(card) || !interpretation?.summary || card.querySelector(".speech-understanding")) continue;
      const rows = new Map((experiment.speech_context?.segments || []).map(row => [row.id, row]));
      const section = document.createElement("section");
      section.className = "speech-understanding";
      const heading = document.createElement("h3");
      heading.textContent = "录音相关说明";
      const summary = document.createElement("p");
      summary.textContent = interpretation.summary;
      const note = document.createElement("p");
      note.className = "analysis-readiness-note";
      note.textContent = `${relations[interpretation.relation_to_visual] || "关系待核对"} · 机器转写、声源未识别；口述不作为动作完成证明。`;
      const references = ids => {
        const links = document.createElement("div");
        links.className = "speech-downloads";
        for (const id of ids || []) {
          const row = rows.get(id);
          if (!row) continue;
          const link = document.createElement("a");
          const params = new URLSearchParams({chunk: row.chunk_id, t: String(row.playback_start_seconds)});
          link.href = `#/${staging ? "stage" : "archive"}/${encodeURIComponent(name)}/speech?${params}`;
          link.className = "secondary-button";
          link.textContent = `回听 ${clock(row.playback_start_seconds)} · ${row.text}`;
          link.title = id;
          links.append(link);
        }
        return links;
      };
      section.append(heading, summary, note, references(interpretation.referenced_segment_ids));
      card.append(section);
      (experiment.steps || []).forEach((step, index) => {
        const panel = card.querySelector(`[data-experiment-step-panel="${index}"]`);
        if (panel && step.speech_segment_ids?.length) panel.append(references(step.speech_segment_ids));
      });
    }
  }
  window.VisionCortexSpeech = {render, attach};
})();
