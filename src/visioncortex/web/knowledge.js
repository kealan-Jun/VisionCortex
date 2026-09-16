window.VisionCortexKnowledge = (() => {
  async function render({main, api, esc, setChrome}) {
    setChrome('knowledge');
    main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">统一实验记录</p><h1>证据检索与任务追踪</h1><p>检索离线分析与 NAS 设备日结果；引用保留来源、时间和版本。</p></div></header><section class="panel"><form data-search-form><label>关键词或问题 <input name="q" required maxlength="2000" placeholder="例如：移液器、称量、某天的操作步骤"></label> <label>采集日期 <input name="day" type="date"></label> <label>相机 <input name="camera" placeholder="全部设备"></label><p><button class="primary-button" type="submit">查找证据</button> <button class="secondary-button" type="button" data-ask>依据证据回答（调用已配置模型）</button></p></form><p data-search-status aria-live="polite">检索只读取本地索引，不启动分析任务。</p><div data-answer></div><div data-evidence></div></section><section class="panel"><h2>任务事件</h2><p>展示持久化的入队、执行和结果记录；重启后仍可查阅。</p><form data-task-form><label>任务 ID <input name="task" placeholder="分片 recording_id 或离线 run_id"></label> <button class="secondary-button" type="submit">查询历史</button></form><div data-events aria-live="polite"></div><button class="secondary-button" data-more-events>读取后续事件</button></section></div>`;
    const form = main.querySelector('[data-search-form]');
    const status = main.querySelector('[data-search-status]');
    const results = main.querySelector('[data-evidence]');
    const answer = main.querySelector('[data-answer]');
    const showHits = hits => {
      results.innerHTML = hits.map(h=>`<article class="panel"><h3>${esc(h.archive)} · ${esc(h.kind)}</h3><p>${esc(h.text.slice(0,1000))}</p><p><a href="${esc(h.citation_url)}" target="_blank" rel="noopener">证据与版本回执</a> · <a href="${esc(h.evidence.url)}" target="_blank" rel="noopener">查看来源</a></p><small>${esc(h.evidence.evidence_status)} · 检索命中不等于确认实验结论</small></article>`).join('');
    };
    async function submit(ask) {
      const data = Object.fromEntries(new FormData(form));
      if (!data.q.trim()) return;
      form.querySelectorAll('button').forEach(b=>b.disabled=true);
      status.textContent = ask ? '正在按证据生成回答…' : '正在查找…'; answer.textContent='';
      try {
        if (ask) {
          const r = await api('/api/knowledge/ask', {method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':crypto.randomUUID()},body:JSON.stringify({question:data.q,day:data.day||null,camera:data.camera||null})});
          const names={answered:'回答已保存；请结合证据核对',insufficient_evidence:'当前没有足够证据',provider_unavailable:'模型服务暂不可用，仍可查阅检索证据',rejected_untraceable_answer:'回答缺少有效引用，已拒绝作为结论',evidence_changed:'证据刚发生更新，请重新检索',submission_pending:'相同问题已在处理中，请稍后重试'};
          status.textContent=names[r.status]||r.status;
          showHits(r.citations||[]);
          const refs = new Map((r.citations||[]).map(h=>[h.id,h]));
          answer.innerHTML=(r.claims||[]).map(c=>`<p>${esc(c.text)} ${(c.citations||[]).map(id=>`<a target="_blank" rel="noopener" href="${esc(refs.get(id)?.citation_url||'#')}">[证据]</a>`).join(' ')}</p>`).join('')+(r.id?`<p><a href="/api/knowledge/answers/${encodeURIComponent(r.id)}" target="_blank" rel="noopener">回答回执</a></p>`:'');
        } else {
          const r=await api('/api/knowledge/search?'+new URLSearchParams(Object.entries(data).filter(([,v])=>v)));
          showHits(r.items); status.textContent=`显示 ${r.items.length} 条证据；索引由后台增量更新。`;
        }
      } catch(e) {status.textContent=`读取失败：${e.message}`;}
      finally {form.querySelectorAll('button').forEach(b=>b.disabled=false);}
    }
    form.onsubmit=e=>{e.preventDefault();submit(false);};
    main.querySelector('[data-ask]').onclick=()=>submit(true);
    const events = main.querySelector('[data-events]');
    let cursor=0, task='';
    async function readEvents(reset=false) {
      if(reset){cursor=0;events.textContent='';}
      try {
        const r=await api((task?`/api/tasks/${encodeURIComponent(task)}/history`:'/api/task-events')+`?after=${cursor}&limit=30`);
        cursor=r.next_cursor;
        events.insertAdjacentHTML('beforeend',r.events.map(e=>`<p><time>${esc(new Date(e.at*1000).toLocaleString('zh-CN'))}</time> · ${esc(e.source)} · ${esc(e.state)}<br><small>${esc(e.task_id)} · 第 ${esc(e.attempt)} 次执行 · ${esc(e.data.stage||'')}</small></p>`).join('')||'<p>没有后续事件。历史任务从本版本上线后开始记录，不伪造此前事件。</p>');
      }catch(e){events.textContent=`事件读取失败：${e.message}`;}
    }
    main.querySelector('[data-task-form]').onsubmit=e=>{e.preventDefault();task=new FormData(e.target).get('task').trim();readEvents(true);};
    main.querySelector('[data-more-events]').onclick=()=>readEvents();
    await readEvents();
  }
  return {render};
})();
