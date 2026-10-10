/* Capture diagnostics, selected understanding refresh, baseline evaluation and library search. */
(() => {
  function mount({container, data, endpoint, name, staging, api, esc, toast, current, clock}) {
    const panel = document.createElement("section"); panel.className = "speech-tools";
    const capture = data.capture_quality;
    panel.innerHTML = `<details class="speech-quality" ${capture?.status === "warning" ? "open" : ""}><summary>采集质量预检 · ${capture ? (capture.status === "warning" ? "发现采集问题" : "抽样结果") : "尚未抽样"}</summary>
      <p>识别准确率尚未验证：暂无人工校对基线。以下只描述实际抽样，不能代表整段质量。</p>
      ${(capture?.records || []).map(row=>`<article><strong>${esc(row.view_id)} · 第 ${row.segment_ordinal+1} 段</strong><p>视频抽样 ${row.video_samples.length} 帧，可见样本 ${Number.isFinite(row.visible_sample_fraction) ? (row.visible_sample_fraction*100).toFixed(0)+"%" : "未核验"}；音频抽样 ${(row.sampled_audio_seconds || 0).toFixed(1)} 秒${Number.isFinite(row.audio_sample_coverage) ? `（覆盖 ${(row.audio_sample_coverage*100).toFixed(1)}%）` : ""}，可用短窗 ${Number.isFinite(row.usable_audio_sample_fraction) ? (row.usable_audio_sample_fraction*100).toFixed(0)+"%" : "未核验"}。</p>${row.warnings.map(w=>`<p class="analysis-readiness-note">${esc(w)}</p>`).join("")}</article>`).join("")}
      ${capture?.omitted_sources ? `<p>另有 ${capture.omitted_sources} 个来源未抽样。</p>` : ""}
      ${staging ? '<button type="button" class="secondary-button" data-capture-refresh>重新检查采集质量</button>' : ""}</details>
      ${staging && data.refresh_targets?.length ? `<details><summary>按片段重新理解录音</summary><p>使用已保留的画面和转写，更新所选片段的录音说明、步骤引用与报告。</p><select aria-label="重算片段" id="speech-refresh-target">${data.refresh_targets.map(item=>`<option value="${esc(item.id)}">${esc(item.label)} · ${clock(item.start_global_ms/1000)}–${clock(item.end_global_ms/1000)}</option>`).join("")}</select><button type="button" class="secondary-button" data-understanding-refresh>重算所选片段</button></details>` : ""}
      <details><summary>人工基线评测</summary><p>下载引用模板，由校对人填写参考文字后上传评测。这里计算样本字符错误率，不修改实验原始转写。</p><button class="secondary-button" type="button" data-baseline-template>下载基线模板</button><label class="field-label">选择人工校对 JSON<input type="file" accept="application/json,.json" data-baseline-file /></label><p data-evaluation-result role="status"></p></details>
      <p data-speech-task-status role="status"></p>`;
    container.querySelector(".panel-heading").after(panel);
    async function submit(scope, target) {
      const parameters = target ? "?"+new URLSearchParams({target:target.id, revision:target.revision}) : "";
      const buttons = panel.querySelectorAll("[data-capture-refresh],[data-understanding-refresh]");
      buttons.forEach(button=>button.disabled=true);
      const status = panel.querySelector("[data-speech-task-status]");
      try {
        const job = await api(`/api/runs/${encodeURIComponent(name)}/refresh/${scope}${parameters}`, {method:"POST"});
        status.textContent = "已排队；本页会显示任务结果。";
        while (current()) {
          const result = await api(job.status_url);
          if (!current()) return;
          if (["completed", "failed", "interrupted"].includes(result.state)) {
            if (result.state !== "completed") throw new Error(result.message || "片段任务未完成，原成果已保留");
            status.textContent = "所选任务已完成。";
            const reload = document.createElement("button"); reload.type="button"; reload.className="secondary-button";
            reload.textContent = "查看更新后的结果"; reload.addEventListener("click", ()=>location.reload()); status.append(reload);
            return;
          }
          status.textContent = result.message || "正在处理所选片段…";
          await new Promise(resolve=>setTimeout(resolve, 2000));
        }
      } catch (error) { if (current()) { status.textContent=error.message; toast(error.message,"error"); } }
      finally { buttons.forEach(button=>button.disabled=false); }
    }
    panel.querySelector("[data-capture-refresh]")?.addEventListener("click", ()=>submit("capture_quality"));
    panel.querySelector("[data-understanding-refresh]")?.addEventListener("click", ()=>submit("understanding",data.refresh_targets.find(item=>item.id===panel.querySelector("#speech-refresh-target").value)));
    panel.querySelector("[data-baseline-template]").addEventListener("click", async()=>{
      try {
        const result = await api(`${endpoint}?limit=500&fold=false`);
        const template = {speech_sha256:result.search.speech_sha256, human_reviewed:false, reviewer:"", segments:result.segments.map(row=>({reference_id:row.reference_id, text:""}))};
        const url=URL.createObjectURL(new Blob([JSON.stringify(template,null,2)],{type:"application/json"}));
        const link=document.createElement("a"); link.href=url; link.download="speech-human-reference.json"; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
      } catch(error) { if(current()) toast(error.message,"error"); }
    });
    panel.querySelector("[data-baseline-file]").addEventListener("change", async event=>{
      const output = panel.querySelector("[data-evaluation-result]");
      try {
        const file=event.target.files[0]; if(!file) return;
        if(file.size>2*1024*1024) throw new Error("基线文件需小于 2 MB，每次最多 500 条。");
        const reference=JSON.parse(await file.text());
        const result=await api(endpoint.replace(/\/speech$/, "/speech-evaluation"), {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(reference)});
        if(current()) output.textContent=`样本字符错误率 ${(result.character_error_rate*100).toFixed(2)}%；覆盖 ${result.evaluated_segments}/${result.total_segments} 条。${result.limitation}`;
      } catch(error) { if(current()) output.textContent=error.message; }
    });
  }
  function library({container, query, aliases, api, esc, current, clock}) {
    let next=0, request=0;
    const panel=document.createElement("section"); panel.className="speech-library-results";
    panel.innerHTML='<h3>跨实验录音搜索</h3><p role="status"></p><div></div><button class="secondary-button" type="button">搜索更多实验</button>';
    container.querySelector(".speech-library-results")?.remove(); container.querySelector("#speech-search").after(panel);
    const status=panel.querySelector("p"), list=panel.querySelector("div"), more=panel.querySelector("button");
    async function load() {
      const id=++request; more.disabled=true;
      try {
        const result=await api(`/api/speech-search?${new URLSearchParams({q:query,aliases:String(aliases),archive_offset:String(next)})}`);
        if(!current() || !panel.isConnected || id!==request) return;
        for(const row of result.segments) {
          const link=document.createElement("a"); link.className="speech-row"; link.href=row.href;
          link.innerHTML=`<time>${clock(row.playback_start_seconds)}</time><span>${esc(row.text)}<small>${esc(row.experiment)} · 录音提及，未确认动作 · ${esc(row.source_hint.label)}</small></span><span>回听</span>`; list.append(link);
        }
        status.textContent=`已显示 ${list.children.length} 条。每个实验最多展示 100 条；进入对应实验可检索其余文字。${result.unavailable.length ? `${result.unavailable.length} 个来源未通过完整性检查。` : ""}`;
        next=result.next_archive_offset; more.hidden=next===null;
      } catch(error) { if(current()) status.textContent=error.message; }
      finally { more.disabled=false; }
    }
    more.addEventListener("click",load); load();
  }
  window.VisionCortexSpeechTools={mount,library};
})();
