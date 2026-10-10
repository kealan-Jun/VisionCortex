/* Task history is separate from current machine state. Values come from receipts. */
(() => {
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const valid = value => typeof value === 'number' && Number.isFinite(value);
  const num = (value, digits=0) => valid(value) ? value.toLocaleString('zh-CN',{maximumFractionDigits:digits}) : '未记录';
  const pct = value => valid(value) ? `${num(value,1)}%` : '未记录';
  const stamp = value => value ? new Date(value).toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}) : '未记录';
  const elapsed = seconds => valid(seconds) ? `${Math.floor(Math.round(seconds)/60)}:${String(Math.round(seconds)%60).padStart(2,'0')}` : '未记录';
  const extraLabels = {gap_review:"缺口补充分析",candidate_coarse:'全量粗扫',candidate_fine:'候选窗口精扫',candidate_audit:'候选边界审计',operation_review:'操作步骤整理',operation_review_refresh:'操作步骤重新整理',experiment_group_understanding_pre_curation:'实验步骤初步理解',experiment_group_understanding:'实验步骤理解',key_material_understanding:'关键素材理解',participant_visual_review:'动作参与对象复核',experiment_boundary_review:'实验边界复核',workflow_boundary_review:'连续实验边界复核',workflow_group_understanding:'连续实验步骤理解',interrupted_group_refresh:'中断的步骤刷新',recording_speech_understanding:'录音语义理解'};
  const stageName = (key, labels) => extraLabels[key] || labels?.[key] || key;
  const stat = (scope,key) => scope?.[key] || {};
  const tile = (title,value,note,color='green') => `<article class="insight-kpi ${color}"><span>${escape(title)}</span><strong>${escape(value)}</strong><small>${escape(note)}</small></article>`;
  const hardwareTiles = (scope,compact=false) => {
    const gpu=stat(scope,'gpu_percent'), vram=stat(scope,'vram_percent'), cpu=stat(scope,'cpu_percent'), ram=stat(scope,'ram_percent');
    const used=stat(scope,'vram_used_mib'), total=stat(scope,'vram_total_mib');
    const capacity=valid(used.max)&&valid(total.max)?`峰值 ${num(used.max/1024,2)} / ${num(total.max/1024,2)} GiB`:'显存容量未记录';
    return tile('GPU 计算利用率 · 平均',pct(gpu.mean),`峰值 ${pct(gpu.max)}`)
      +tile('显存占用率 · 平均',pct(vram.mean),capacity,'blue')
      +tile('CPU 占用率 · 平均',pct(cpu.mean),`整机 · 峰值 ${pct(cpu.max)}`,'amber')
      +(compact?'':tile('内存占用率 · 平均',pct(ram.mean),`整机 · 峰值 ${pct(ram.max)}`,'slate'));
  };
  const usageNotice = tokens => !tokens.available ? '未保存可用的 Token 用量记录。' : tokens.coverage==='partial'
    ? `已知用量；${num(tokens.unknown_attempts)} 次请求尝试用量未知，实际总消耗可能更高。`
    :tokens.coverage==='aggregate_only'?'历史记录仅有汇总，无法核对逐次请求。':'按已保存的服务商回执统计，包含有回执的重试。';
  function chart(points,key,title,color) {
    const available=points.some(p=>valid(p[key]?.mean));
    if(!available)return `<article class="insight-chart"><h3>${title}</h3><p class="insight-empty">该阶段未保存有效采样</p></article>`;
    const first=points[0]?.offset_seconds||0,last=points.at(-1)?.offset_seconds||first;
    const x=p=>36+(p.offset_seconds-first)/Math.max(1,last-first)*564;
    const y=v=>120-Math.min(100,Math.max(0,v));
    let segments=[], segment=[];
    for(const p of points){if(valid(p[key]?.mean)){segment.push(`${x(p).toFixed(2)},${y(p[key].mean).toFixed(2)}`);}else if(segment.length){segments.push(segment);segment=[];}}
    if(segment.length)segments.push(segment);
    return `<article class="insight-chart"><h3><i style="background:${color}"></i>${title}</h3><svg viewBox="0 0 620 152" role="img" aria-label="${title}，折线为采样均值，圆点为采样峰值"><g class="chart-grid">${[0,50,100].map(v=>`<line x1="36" y1="${y(v)}" x2="600" y2="${y(v)}"/><text x="3" y="${y(v)+4}">${v}%</text>`).join('')}</g>${segments.map(s=>`<polyline fill="none" stroke="${color}" stroke-width="2.2" points="${s.join(' ')}"/>`).join('')}${points.filter(p=>valid(p[key]?.max)).map(p=>`<circle cx="${x(p)}" cy="${y(p[key].max)}" r="2" fill="${color}" opacity=".6"><title>${escape(stamp(p.timestamp))} · 均值 ${pct(p[key].mean)} · 峰值 ${pct(p[key].max)} · ${p.sample_count} 次采样</title></circle>`).join('')}<text class="chart-time" x="36" y="145">${elapsed(first)}</text><text class="chart-time" text-anchor="end" x="600" y="145">${elapsed(last)}</text></svg></article>`;
  }
  function hardwareBody(hardware,selected,labels) {
    const scope=selected==='all'?hardware.overall:hardware.stages?.[selected];
    const points=(hardware.timeline||[]).filter(p=>selected==='all'||p.stage===selected);
    return `<div class="insight-kpis">${hardwareTiles(scope)}</div><div class="insight-charts">${chart(points,'gpu_percent','GPU 计算利用率','#207e69')}${chart(points,'vram_percent','显存占用率','#4c79b6')}${chart(points,'cpu_percent','CPU 占用率','#b58034')}</div><p class="insight-caption">折线为区间采样均值，圆点为区间峰值；横轴为距首次采样的分钟:秒。${selected==='all'?'全阶段':escape(stageName(selected,labels))}有效 GPU 采样 ${num(scope?.gpu_percent?.count)} / ${num(scope?.sample_count)}。</p>`;
  }
  function tokenTable(rows,labels,models=false) {
    return `<div class="insight-table-wrap"><table class="insight-table"><thead><tr><th>${models?'厂商 / 模型':'调用阶段'}</th><th>调用 / 复用</th><th>输入 Token</th><th>输出 Token</th><th>总 Token</th><th>用量未知尝试</th></tr></thead><tbody>${rows.map(r=>`<tr><th>${escape(models?`${r.provider} · ${r.model}`:stageName(r.stage,labels))}</th><td>${num(r.executed_calls)} / ${num(r.reused_calls)}</td><td>${num(r.totals?.input_tokens)}</td><td>${num(r.totals?.output_tokens)}</td><td>${num(r.totals?.total_tokens)}</td><td>${num(r.unknown_attempts)}</td></tr>`).join('')||'<tr><td colspan="6">未保存逐次调用记录</td></tr>'}</tbody></table></div>`;
  }
  function componentView(timing) {
    const scans=timing.scans||[];
    const fields={read_decode_supply_seconds:'读取与解码供帧',frame_preparation_seconds:'CPU 画面预处理',producer_emit_seconds:'送帧与队列背压',model_preprocess_seconds:'模型预处理',model_inference_seconds:'模型推理',model_postprocess_seconds:'模型后处理',tracking_serialization_seconds:'跟踪与序列化',ledger_write_seconds:'结果写入与缓冲刷新'};
    return `<details class="insight-stage-details"><summary>预处理分项诊断与优化方向</summary><p>以下为各工作线程累计值，会并行重叠。读取和解码由现有后端合并执行；磁盘读取时间无法单独确认。模型分项采用后端计时（含固定 Batch 填充），写入计时不代表存储设备已完成持久化。</p>${scans.map(scan=>{
      const c=scan.component_timings||{}, entries=Object.entries(fields);
      const present=Object.keys(c).length>0;
      const suggestion=scan.classification==='decode_or_source_starved'?'供帧不足：优先对比输入存储和解码供给，确认 Batch 是否频繁空等。':scan.classification==='cpu_postprocess_bound'?'跟踪与写入耗时较高：分别查看序列化和结果写入，再决定优化项。':scan.classification==='gpu_inference_bound'?'模型调用耗时较高：用分项计时区分预处理、推理和后处理，再比较 Batch 与引擎。':'当前多环节共同影响耗时，结合分项和队列等待进行对照。';
      return `<h4>${escape({coarse:'粗扫',fine_scout:'候选探测',fine:'精扫'}[scan.phase]||scan.phase)}</h4>${present?`<p>分项计时覆盖 ${num(scan.component_profiled_workers)} / ${num(scan.component_expected_workers)} 个工作单元。${suggestion}</p><div class="insight-table-wrap"><table class="insight-table"><tbody>${entries.map(([key,label])=>`<tr><th>${label}</th><td>${valid(c[key])?num(c[key],3)+' 秒':'未记录'}</td></tr>`).join('')}</tbody></table></div><p>实际平均 Batch ${num(scan.actual_batch_size_mean,1)}，填充率 ${pct(valid(scan.batch_fill_ratio)?scan.batch_fill_ratio*100:null)}。解码采样帧 ${num(c.decoded_frames)}；模型推理计时覆盖 ${num(c.model_inference_profiled_frames)} 帧（含填充）。</p>`:'<p>这次历史运行没有分项计时，不能从总耗时反推。新的分析会记录可测分项。</p>'}`;
    }).join('')||'<p>本次没有扫描阶段记录。</p>'}</details>`;
  }

  function timingView(timing,labels) {
    if(!timing.available)return '';
    const largest=timing.largest_stage, model=timing.model||{},rows=timing.stages||[];
    const seconds=value=>valid(value)?`${num(value,1)} 秒`:'未记录';
    return `<section class="insight-section insight-timing"><header class="insight-section-heading"><div><h3>时间花在哪里</h3><p>本次分析 ${elapsed(timing.wall_seconds)} · ${largest?`耗时最多：${escape(stageName(largest.stage,labels))}，${num(largest.share_percent,1)}%`:''}</p></div></header><div class="timing-bars">${rows.map(row=>`<div class="timing-row"><span>${escape(stageName(row.stage,labels))}</span><i><b style="width:${Math.min(100,Math.max(0,row.share_percent||0))}%"></b></i><strong>${elapsed(row.seconds)}</strong></div>`).join('')}</div><details class="insight-stage-details"><summary>查看解码供帧、推理与模型并发明细</summary><p class="insight-caption">以下为并行工作累计时间，不能与上方阶段耗时相加。供帧等待包含读取、解码与调度，历史记录无法把这三者单独拆开。</p><div class="insight-table-wrap"><table class="insight-table"><thead><tr><th>扫描阶段</th><th>等待供帧</th><th>推理累计</th><th>跟踪与落盘</th><th>实际平均 Batch</th></tr></thead><tbody>${(timing.scans||[]).filter(r=>r.available).map(r=>`<tr><th>${escape({coarse:'粗扫',fine_scout:'候选探测',fine:'精扫'}[r.phase]||r.phase)}</th><td>${seconds(r.queue_wait_seconds)}</td><td>${seconds(r.inference_seconds)}</td><td>${seconds(r.tracking_and_ledger_seconds)}</td><td>${num(r.actual_batch_size_mean,1)}</td></tr>`).join('')||'<tr><td colspan="5">本次未保留扫描明细</td></tr>'}</tbody></table></div>${componentView(timing)}<div class="timing-model"><p><strong>模型请求峰值并发</strong> ${num(model.peak_concurrency)} · 有效时间回执 ${num(model.timed_attempts)} / ${num(model.expected_attempts)}${model.concurrency_coverage!=='complete'?'，不能据此确认整次并发上限':''}</p><p><strong>已记录的失败请求累计等待</strong> ${seconds(model.failed_attempt_seconds_sum)} · <strong>已记录的重试退避累计</strong> ${seconds(model.backoff_seconds_sum)}（${num(model.backoff_recorded_attempts)} / ${num(model.expected_attempts)} 次尝试有记录）</p><p>已记录的限流响应：${num(model.rate_limit_attempts)} 次。旧回执未保留 HTTP 状态时无法核实其限流情况。</p><p>模型明细包含账本中后续复核。旧请求没有起止时间或退避记录时显示未记录，不反推实际并发。</p></div></details></section>`;
  }
  function render(data,options={}) {
    const insights=data.observability?.insights||{},hardware=insights.hardware||{},tokens=insights.tokens||{};
    const totals=tokens.totals||{},compact=options.compact,labels=options.labels||{};
    const timestamp=hardware.started_at?`${stamp(hardware.started_at)} — ${stamp(hardware.ended_at)}`:'未保存任务历史采样';
    const header=`<header class="insight-heading"><div><p class="insight-eyebrow">${compact?'RUN OVERVIEW':'PERFORMANCE & USAGE'}</p><h2>${compact?'资源与模型消耗':'性能与用量'}</h2><p>硬件历史采样 · ${timestamp}</p></div>${compact?`<a class="insight-detail-link" href="${escape(options.href)}">查看完整数据 <span>↗</span></a>`:'<button class="secondary-button" type="button" data-export-insights>导出性能与用量 JSON</button>'}</header>`;
    if(compact)return `<section class="run-insights is-compact">${header}<div class="insight-kpis">${hardwareTiles(hardware.overall,true)}${tile(tokens.coverage==='partial'?'已知 Token 总消耗':'Token 总消耗',num(totals.total_tokens),`输入 ${num(totals.input_tokens)} · 输出 ${num(totals.output_tokens)}`,'slate')}</div><p class="insight-caption">${escape(usageNotice(tokens))} 硬件为任务运行时的整机采样；Token 含账本中后续复核。</p></section>`;
    const stages=Object.entries(hardware.stages||{});
    const latest=hardware.latest||{}, latestAge=(Date.now()-Date.parse(latest.timestamp))/1000;
    const liveNow=hardware.live_status==='running' && latestAge>=0 && latestAge<20 && !['failed','completed','partial','cancelled','interrupted'].includes(data.observability?.status?.stage);
    const g=latest.gpu||{};
    const liveBar=liveNow?`<p class="insight-live">最新采样 · ${stamp(latest.timestamp)} · GPU ${pct(g['utilization.gpu'])} · 显存 ${valid(g['memory.used'])&&g['memory.total']>0?pct(g['memory.used']/g['memory.total']*100):'未记录'} · CPU ${pct(latest.cpu_percent)}</p>`:'';
    const sourceLinks=[['resource_telemetry','原始硬件采样 JSON'],['resource_telemetry_journal','逐次采样 JSONL'],['metrics','模型调用与耗时账本']].filter(([key])=>data.links?.[key]);
    return `<section class="run-insights" id="run-insights">${header}${timingView(insights.timing||{},labels)}${liveBar}<section class="insight-section"><header class="insight-section-heading"><div><h3>硬件运行情况</h3><p>${num(hardware.sample_count)} 次历史采样 · 设定间隔 ${num(hardware.sampling_interval_seconds,2)} 秒</p></div><label>处理阶段<select data-insight-stage aria-label="选择性能统计阶段"><option value="all">全阶段</option>${stages.map(([key])=>`<option value="${escape(key)}">${escape(stageName(key,labels))}</option>`).join('')}</select></label></header>${hardware.available?'':'<p class="insight-empty">本次记录没有可用的历史采样，无法还原平均值、峰值和趋势。</p>'}<div data-hardware-insights>${hardwareBody(hardware,'all',labels)}</div><details class="insight-definitions"><summary>指标口径与采样范围</summary><p>GPU 计算利用率表示采样周期内执行计算内核的时间比例，100% 不等于已达到理论算力上限。显存占用率 = 已用显存 ÷ 显存总容量，与显存读写忙碌率不同。CPU / 内存为整机占用，GPU 为设备 0，均可能包含其他程序。</p><p>均值按有效原始采样点计算；峰值来自所有有效采样，缺失值不按 0 处理。采样范围 ${timestamp}。${hardware.excluded_sample_count?`另有 ${hardware.excluded_sample_count} 条旧尝试或无效时间采样未计入。`:''}</p></details><details class="insight-stage-details"><summary>按阶段查看硬件数据</summary><div class="insight-table-wrap"><table class="insight-table"><thead><tr><th>阶段</th><th>GPU 平均 / 峰值</th><th>显存 平均 / 峰值</th><th>CPU 平均 / 峰值</th><th>解码器 平均 / 峰值</th><th>功耗峰值</th></tr></thead><tbody>${stages.map(([stage,s])=>`<tr><th>${escape(stageName(stage,labels))}</th>${['gpu_percent','vram_percent','cpu_percent','nvdec_percent'].map(key=>`<td>${pct(s[key]?.mean)} / ${pct(s[key]?.max)}</td>`).join('')}<td>${num(s.power_w?.max,1)} W</td></tr>`).join('')}</tbody></table></div></details></section><section class="insight-section"><header class="insight-section-heading"><div><h3>模型 Token 消耗</h3><p>已保存账本中的全部阶段，含后续复核；采样时段可能与后续调用时段不同</p></div><span class="insight-status">${tokens.coverage==='reported'?'回执齐全':'部分用量待确认'}</span></header><div class="insight-kpis token-kpis">${tile('输入 Token',num(totals.input_tokens),`其中缓存输入 ${num(totals.cached_input_tokens)}`)}${tile('输出 Token',num(totals.output_tokens),'以服务商返回的用量为准','blue')}${tile(tokens.coverage==='partial'?'已知总 Token':'总 Token',num(totals.total_tokens),`${num(tokens.attempts)} 次请求尝试 · ${num(tokens.reused_calls)} 次本地复用`,'slate')}</div><p class="insight-usage-note">${escape(usageNotice(tokens))} 本地结果复用不重复计入；缓存输入已包含在输入 Token 中。${tokens.reconciled===false?'汇总账本与逐次回执存在差异，此处采用逐次回执；请核对原始文件。':''}</p>${tokenTable(tokens.stages||[],labels)}<details class="insight-stage-details"><summary>按厂商与模型查看消耗</summary>${tokenTable(tokens.models||[],labels,true)}</details></section><footer class="insight-sources"><span>原始记录</span>${sourceLinks.map(([key,label])=>`<a href="${escape(data.links[key])}" target="_blank" rel="noopener">${label} ↗</a>`).join('')}</footer></section>`;
  }
  function bind(root,data,options={}) {
    const hardware=data.observability?.insights?.hardware||{};
    root.querySelector('[data-insight-stage]')?.addEventListener('change',event=>{
      root.querySelector('[data-hardware-insights]').innerHTML=hardwareBody(hardware,event.target.value,options.labels||{});
    });
    root.querySelector('[data-export-insights]')?.addEventListener('click',()=>{
      const payload={schema_version:'visioncortex-run-insights/1',exported_at:new Date().toISOString(),experiment:data.name,run_id:data.staging_run_id||null,insights:data.observability?.insights||{},sources:data.observability?.sources||{}};
      const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
      const a=document.createElement('a');a.href=url;a.download='VisionCortex-性能与用量.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });
  }
  globalThis.VisionCortexRunInsights={render,bind};
})();
