window.VisionCortexDayTimeline = (() => {
  const clock = us => new Date(us/1000).toLocaleTimeString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});
  async function render({main,api,esc,setChrome,state}, day) {
    setChrome('day-timeline');
    const initialProgress=await api('/api/device-day-progress').catch(()=>null);
    const dates=[...new Set((state.deviceDayArchives||[]).map(x=>x.archive.slice(0,10)))].sort().reverse();
    day=day||initialProgress?.focus_date||dates[0]||new Date().toLocaleDateString('sv-SE',{timeZone:'Asia/Shanghai'});
    const hash=location.hash;
    main.innerHTML=`<div class="page"><header class="page-hero compact"><div><p class="eyebrow">当天实验观察记录</p><h1>${esc(day)} 时间线</h1><p>按真实采集时间查看全部设备；同一时刻的多路画面不重复计算时长。</p></div></header><section class="panel processing-overview" data-processing-overview aria-live="polite">正在读取后台处理进度…</section><section class="panel"><label>查看采集日期 <input type="date" value="${esc(day)}" data-timeline-date></label> <button class="secondary-button" data-timeline-refresh>刷新</button> <a href="#/device-days">设备日归档</a> · <a href="/archive-overview" target="_blank" rel="noopener">实验数据总览</a></section><section class="panel" data-timeline-body>正在读取当天产出…</section></div>`;
    main.querySelector('[data-timeline-date]').onchange=e=>{location.hash=`#/day-timeline/${encodeURIComponent(e.target.value)}`;};
    main.querySelector('[data-timeline-refresh]').onclick=()=>render({main,api,esc,setChrome,state},day);
    const progressPanel=main.querySelector('[data-processing-overview]');
    const stageNames={retention:'原片与资料归档',vision:'YOLO 预处理',stt:'录音识别',understanding:'多模态理解',report:'日报更新'};
    let lastProgress=null;
    function showProgress(p) {
      const delayed=!p;
      if(p)lastProgress=p;
      p=p||lastProgress;
      if(!p){progressPanel.innerHTML='<p role="alert">处理状态暂时读取不到，后台任务可能仍在运行。页面会自动重试；也可打开<a href="/archive-overview" target="_blank" rel="noopener">实验数据总览</a>查看后台定期发布的处理状态与分片时延。</p>';return;}
      const chip=document.querySelector("#phase-chip span");
      if(chip)chip.textContent=delayed?'处理状态刷新延迟':`后台处理：${p.running.length} 个阶段任务运行中`;
      const selected=p.days[day];
      const visionJobs=p.running.filter(j=>j.stage==='vision');
      const visionDates=[...new Set(visionJobs.map(j=>j.date))];
      const waitLabels={upstream_failed:'上游失败待恢复',upstream_pending:'上游未完成',provider_blocked:'云端账户不可用',night_window:'等待夜间窗口',lease_recovery:'租约过期待恢复',pending_validation:'待校验调度'};
      const pendingLabels={retention:'尚未进入归档队列',vision:'尚未入队（待原片归档或校验）',stt:'尚未入队（待原片与录音归档或校验）',understanding:'尚未入队（待预处理、录音结果或校验）',report:'尚未入队（待理解结果或校验）'};
      progressPanel.innerHTML=`${delayed?`<p role="alert">刷新暂时延迟，以下保留 ${new Date(p.observed_at*1000).toLocaleTimeString('zh-CN')} 的处理快照，并非当前实时状态。正在自动重试。</p>`:''}<div class="processing-heading"><div><p class="eyebrow">${delayed?'最近一次处理快照':'后台正在做什么'}</p><h2>${p.running.length?`${p.running.length} 个阶段任务${delayed?'在快照时运行':'正在处理'}`:'快照中没有可确认的运行任务'}</h2><p>历史补跑优先日期：<a href="#/day-timeline/${esc(p.focus_date||day)}">${esc(p.focus_date||'未指定')}</a> · 新采集优先，其他历史数据随后</p></div><small>更新于 ${new Date(p.observed_at*1000).toLocaleTimeString('zh-CN')}<br>每 5 秒自动更新处理状态</small></div>
      <p><strong>YOLO ${delayed?'快照':'当前'}：${visionJobs.length} 个分片任务运行中</strong>${visionDates.length?` · 采集日期 ${visionDates.map(d=>`<a href="#/day-timeline/${esc(d)}">${esc(d)}</a>`).join('、')}`:''}${visionJobs.length&&!visionJobs.some(j=>j.date===day)?`。${delayed?'快照中':'当前'}没有处理所选日期 ${esc(day)} 的分片，下面的进度仅统计所选日期。`:''}</p>
      ${p.night_schedule?.enabled?`<p><strong>多模态与日报：${p.night_schedule.open?'夜间处理窗口已开启':'等待今晚 '+esc(p.night_schedule.start)}</strong> · ${esc(p.night_schedule.timezone)} ${esc(p.night_schedule.start)}–次日 ${esc(p.night_schedule.end)}。白天原片归档、YOLO、录音识别继续运行；已开始的夜间任务会完成后结束。</p>`:''}
      ${p.capture_link_cleanup&&!p.capture_link_cleanup.enabled?'<p class="muted">采集端原片替换软链接尚未启用：NAS 原生链接及采集读取兼容性待验证，原片继续保留。预处理正常运行。</p>':''}
      ${p.provider?.active?'<p role="alert">云端账户仍不可用，录音识别与多模态等待自动恢复；归档和 YOLO 不受云端开关阻塞。</p>':''}
      <div class="processing-latency">${p.latency_html||''}</div>
      <h3>你正在查看 ${esc(day)}${selected?` · ${selected.camera_count} 台设备 · ${selected.total} 个已发现分片`:''}</h3>
      ${selected?`<p>${selected.missing_input_count||0} 个输入缺失已从任务完成率分母中排除，保留失败记录；以后恢复时自动处理。</p><div class="processing-stage-grid">${Object.entries(stageNames).map(([key,label])=>{const c=selected.stages[key];return `<article class="processing-stage"><h4>${label}</h4><strong>${c.completed}<small> / ${selected.processing_total??selected.total} 分片任务完成</small></strong><progress value="${c.completed}" max="${Math.max(1,selected.processing_total??selected.total)}" aria-label="${label}任务完成比例"></progress><p><b>${c.running} 正在处理</b> · ${c.queued} 入队未完成 · <span class="${c.failed?'processing-failure':''}">${c.failed} 失败</span></p><small>${Object.entries(p.waiting?.[day]?.[key]||{}).map(([reason,n])=>`${n} ${waitLabels[reason]||esc(reason)}`).join(' · ')}</small><small>${c.not_enqueued?`${c.not_enqueued} ${pendingLabels[key]}`:'没有未入队分片'}${c.expired?` · ${c.expired} 待恢复租约`:''}</small></article>`;}).join('')}</div>`:'<p>该日期尚无采集队列记录，不能判断当天没有实验。</p>'}
      <p class="muted">以上是队列任务进度；旧版本完成记录不代表当前全流程已验收。录音阶段包含无录音的正常结束，日报是逐分片更新同一份报告。</p>
      <details open><summary>${delayed?'快照中':'现在'}具体在跑哪些分片（全部日期）</summary><div class="processing-table-wrap"><table class="processing-table"><thead><tr><th>采集日期</th><th>设备</th><th>正在执行</th><th>分片采集时段</th></tr></thead><tbody>${p.running.map(j=>`<tr><td><a href="#/day-timeline/${esc(j.date)}">${esc(j.date)}</a></td><td>${esc(j.camera)}</td><td><strong>${stageNames[j.stage]}</strong><br><small>${esc(j.phase||'等待阶段详情')}${j.phase_elapsed_seconds!=null?` · ${Math.floor(j.phase_elapsed_seconds)} 秒`:''}${j.frame_counts?`<br>推理帧：粗扫 ${j.frame_counts.coarse||0} · 精扫 ${j.frame_counts.fine||0}`:''}</small></td><td>${clock(j.start_us)}—${j.end_us?clock(j.end_us):'结束时间未知'}</td></tr>`).join('')||'<tr><td colspan="4">暂无有效运行租约；其余分片可能在等待上游、重试或调度。</td></tr>'}</tbody></table></div></details>
      ${p.errors.length?`<p role="alert">${p.errors.map(esc).join('；')}，上述状态可能不完整。</p>`:''}`;
    }
    showProgress(initialProgress);
    async function pollProgress(){
      if(hash!==location.hash||!progressPanel.isConnected)return;
      if(!document.hidden){const p=await api('/api/device-day-progress').catch(()=>null);if(hash!==location.hash||!progressPanel.isConnected)return;showProgress(p);}
      setTimeout(pollProgress,5000);
    }
    setTimeout(pollProgress,5000);
    const body=main.querySelector('[data-timeline-body]');
    try {
      const data=await api(`/api/day-timeline/${encodeURIComponent(day)}`);
      if(hash!==location.hash)return;
      const entries=new Map(data.entries.map(e=>[e.id,e]));
      const link=(url,label)=>url?`<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>`:'';
      const sourceAt=(e,us)=>`${e.source_url}#t=${Math.max(0,(e.start_ms+(us-e.start_us)/1000)/1000).toFixed(3)}`;
      const views=edge=>[entries.get(edge.first_person),entries.get(edge.third_person)];
      const labels={active:'有活动观察时段',inactive:'未检出实验活动',missing:'处理结果尚未覆盖此时段'};
      let shown=30;
      function draw() {
        body.innerHTML=`<p><strong>${data.camera_count} 台设备</strong> · 已有 ${data.indexed_recordings} 个归档分片 · ${data.pending_recordings} 个分片尚无处理结果</p><p>有活动观察时长 ${(data.activity_seconds/60).toFixed(1)} 分钟（重叠去重，不代表已确认实验时长）。</p>${data.errors.length||data.discovery_errors.length?`<p role="alert">还有 ${data.errors.length} 个索引和 ${data.discovery_errors.length} 条采集读取异常，缺失画面不作无活动判断。</p>`:''}<p>多视角按采集时间对照；相机时钟偏差、同台仪器及同一次操作尚待核实。</p>${data.periods.slice(0,shown).map(p=>{
          const relevant=data.cross_view_links.filter(e=>e.start_us<p.end_us&&e.end_us>p.start_us);
          return `<article class="panel"><h2>${clock(p.start_us)}—${clock(p.end_us)} · ${labels[p.state]}</h2>${relevant.length?`<details><summary>第一／第三人称对照 · ${relevant.length} 组候选</summary>${relevant.map(edge=>{
            const pair=views(edge);if(pair.some(e=>!e))return '';
            return `<section class="panel"><p>${clock(edge.start_us)}—${clock(edge.end_us)} · ${edge.status==='action_candidate_pending_verification'?'共用算法找到动作候选，身份与时钟待核实':'时间重叠，待核实'} · ${edge.shared_action_audit.status==='completed'?'动作审核已执行':'等待异步动作审核'}</p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">${pair.map((e,i)=>`<div><h3>${i?'第三人称':'第一人称'} · ${esc(e.camera)}</h3>${e.source_url?`<video controls preload="none" style="width:100%" src="${esc(sourceAt(e,edge.start_us))}"></video>`:'原片暂不可用'}<p>${link(e.json_url,'片段与关键帧索引')} · ${e.activity==='active'?'该视角检出活动':'该视角未检出活动，仍保留对照'}</p></div>`).join('')}</div><p>两个播放器定位到同一采集时刻，精确同步偏差尚未校验。</p></section>`;
          }).join('')}</details>`:''}${p.state==='missing'?'<p>这里暂时没有已发布的视频处理结果。可能尚未处理或没有采集，不能当作没有实验活动。</p>':''}<details><summary>查看 ${p.entry_ids.length} 路／段来源与操作记录</summary>${p.entry_ids.map(id=>entries.get(id)).filter(Boolean).map(e=>`<section><h3>${clock(e.start_us)}—${clock(e.end_us)} · ${esc(e.camera)}</h3><p>${link(e.clip_url||e.source_url,'打开视频')} · ${link(e.json_url,'片段 JSON')} ${e.cross_view_status==='other_view_missing'?'· 暂无另一视角':''}</p>${e.steps.map(s=>`<p><strong>${clock(s.start_us)}</strong> ${esc(s.description)} <small>${esc(s.basis||'')}</small></p>`).join('')||e.summaries.map(t=>`<p>${esc(t)}</p>`).join('')||'<p>该时段的具体操作理解尚未产出。</p>'}<div class="frames">${e.key_frames.map(f=>link(f.url,`${clock(f.capture_us||e.start_us)} 动作关键帧`)).join(' · ')}</div></section>`).join('')}</details></article>`;
        }).join('')||'<p>当天尚无已处理结果；不能据此判断没有实验。</p>'}${shown<data.periods.length?'<button class="secondary-button" data-more-periods>显示更多时段</button>':''}`;
        body.querySelector('[data-more-periods]')?.addEventListener('click',()=>{shown+=30;draw();});
      }
      draw();
    } catch {
      if(hash===location.hash)body.innerHTML='<p role="alert">当天时间线暂时无法读取，请刷新重试。原有后台处理继续运行。</p>';
    }
  }
  return {render};
})();
