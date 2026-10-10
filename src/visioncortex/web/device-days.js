/* Device/day evidence stays inside the existing application shell. */
window.VisionCortexDeviceDays = (() => {
  let selectedDate = "all", selectedCamera = "all";
  const route = (name, tab = "report") => `#/device-day/${encodeURIComponent(name)}/${tab}`;
  const file = (name, path) => `/api/device-days/${encodeURIComponent(name)}/files/${path.split('/').map(encodeURIComponent).join('/')}`;
  const watch = (name, path) => `/device-days/${encodeURIComponent(name)}/watch/${path.split('/').map(encodeURIComponent).join('/')}`;
  const tabs = {report:"整合日报",clips:"片段与帧索引",media:"原视频与采集数据",understanding:"步骤级理解",comment:"录音与备注"};
  function renderLibrary(ctx) {
    const {main,state,esc,setChrome,deviceDaySection,deviceDayQueueSection} = ctx;
    setChrome('device-days');
    const all = state.deviceDayArchives || [];
    const errors=(state.deviceDayErrors||[]).filter(row=>{const parts=String(row.path||'').split('/');return (selectedDate==='all'||parts[1]===selectedDate)&&(selectedCamera==='all'||parts[0]===selectedCamera);});
    const dates = [...new Set(all.map(row=>row.archive.slice(0,10)))].sort().reverse();
    const cameras = [...new Set(all.map(row=>row.archive.slice(11)))].sort();
    const rows = all.filter(row=>(selectedDate==='all'||row.archive.startsWith(selectedDate+'_')) &&
      (selectedCamera==='all'||row.archive.slice(11)===selectedCamera));
    const options = (values, selected) => values.map(value=>`<option value="${esc(value)}" ${value===selected?'selected':''}>${esc(value)}</option>`).join('');
    main.innerHTML = `<div class="page"><header class="page-hero compact"><div><p class="eyebrow">实验成果</p><h1>设备日归档</h1><a class="primary-button" href="/archive-overview" target="_blank" rel="noopener">实验数据总览 · 按日期找实验视频</a><a class="secondary-button" href="#/day-timeline/${selectedDate==='all'?dates[0]||'':selectedDate}">查看当天统一时间线</a><p>先选日期和相机，再查看整合日报或打开对应视频、CSV 与录音。</p></div></header><section class="panel"><div class="archive-filter-toolbar"><label><span>采集日期</span><select data-device-date><option value="all">全部日期</option>${options(dates,selectedDate)}</select></label><label><span>相机设备</span><select data-device-camera><option value="all">全部设备</option>${options(cameras,selectedCamera)}</select></label></div></section>${errors.length?`<section class="panel" role="alert"><h2>采集读取失败 · 尚未完成识别</h2><p>以下分片尚未恢复读取，不能判定为无实验活动。</p><ul>${errors.map(row=>`<li><strong>${esc(row.path)}</strong><details><summary>读取失败原因</summary>${esc(row.message||'采集说明无法读取')}</details></li>`).join('')}</ul></section>`:''}${deviceDaySection(false,Infinity,rows)}<details class="panel"><summary>查看全局处理队列</summary>${deviceDayQueueSection()}</details></div>`;
    main.querySelector('[data-device-date]').addEventListener('change',event=>{selectedDate=event.target.value;renderLibrary(ctx);});
    main.querySelector('[data-device-camera]').addEventListener('change',event=>{selectedCamera=event.target.value;renderLibrary(ctx);});
  }
  async function renderDetail(ctx,name,tab='report',selectedRecording='') {
    const {main,api,esc,number,setChrome,formatDate} = ctx;
    const requested = location.hash;
    setChrome('device-days');
    main.innerHTML='<div class="page"><section class="panel" role="status">正在读取设备日归档…</section></div>';
    try {
      const data = await api(`/api/device-days/${encodeURIComponent(name)}/index`);
      if(location.hash!==requested)return;
      if(!tabs[tab])tab='report';
      const records=data.recordings||[],segments=data.segments||[];
      const {completed:complete,preprocessed}=ctx.deviceDaySummary(data);
      main.innerHTML=`<div class="page device-day-detail"><header class="page-hero compact"><div><p class="eyebrow"><a href="#/device-days">设备日归档</a> / ${esc(name.slice(0,10))}</p><h1>${esc(name.slice(11))}</h1><p>已归档 ${number(records.length)} 个采集分片 · ${number(preprocessed)} 个预处理完成 · ${number(complete)} 个完成全流程 · ${number(segments.length)} 个保留区间</p></div></header><nav class="archive-quick-filters" aria-label="设备日内容">${Object.entries(tabs).map(([key,label])=>`<a class="secondary-button ${key===tab?'active':''}" ${key===tab?'aria-current="page"':''} href="${route(name,key)}">${label}</a>`).join('')}</nav><section class="panel device-day-body"></section></div>`;
      const body=main.querySelector('.device-day-body');
      const link=(ref,label)=>{const path=typeof ref==='string'?ref:ref.path;return `<a href="${esc((path.toLowerCase().endsWith('.mp4')?watch:file)(name,path))}">${esc(label)}</a>`;};
      if(tab==='report'||tab==='understanding') {
        const path=tab==='report'?'LaboratoryDailyReport/LaboratoryDailyReport.html':'MultimodalUnderstanding/UnderstandingReport.html';
        body.innerHTML=`${complete<records.length?'<p role="status">仍有分片处理中，下方展示已经落盘的内容。</p>':''}<iframe class="device-day-document" title="${tabs[tab]}" src="${file(name,path)}"></iframe>`;
      } else if(tab==='media') {
        body.innerHTML=records.map(record=>{
          const sources=record.sources||[],media=sources.filter(source=>['video','audio_audio'].includes(source.kind));
          const metadata=sources.filter(source=>!['video','audio_audio'].includes(source.kind));
          return `<article class="device-day-source"><h2>${esc(formatDate(record.start_us/1000))}</h2><p>${media.map(source=>link(source.retained,source.kind==='video'?'打开原视频':'播放原录音')).join(' · ')}</p><details data-capture-files="${esc(record.recording_id)}"><summary>CSV、标定、时间戳与采集日志</summary><ul>${metadata.map(source=>`<li>${link(source.retained,source.retained.path.split('/').pop())}</li>`).join('')}</ul></details></article>`;
        }).join('') || '<p>正在归档采集文件。</p>';
        body.querySelectorAll('[data-capture-files]').forEach(details=>details.addEventListener('toggle',async()=>{
          const record=records.find(row=>row.recording_id===details.dataset.captureFiles);
          if(!details.open||details.dataset.loaded||!record?.capture_manifest)return;
          try {
            const catalog=await api(file(name,record.capture_manifest.path));
            if(!details.isConnected)return;
            const refs=(catalog.files||[]).filter(row=>!record.sources.some(source=>['video','audio_audio'].includes(source.kind)&&source.retained.path===row.retained.path));
            details.querySelector('ul').innerHTML=refs.map(row=>`<li>${link(row.retained,row.retained.path.split('/').pop())}</li>`).join('')+`<li>${link(record.capture_manifest,'CaptureFiles.json')}</li>`;
            details.dataset.loaded='true';
          } catch {details.querySelector('ul').insertAdjacentHTML('beforeend','<li>补充文件清单暂时无法读取，已归档的 CSV 仍可打开。</li>');}
        }));
      } else if(tab==='clips') {
        if(!segments.length){body.innerHTML='<p>YOLO 正在处理，片段结果落盘后可在这里查看。</p>';return;}
        const preferred=segments.findIndex(s=>s.recording_id===selectedRecording&&s.activity==='active');
        const initial=preferred>=0?preferred:Math.max(0,segments.findIndex(s=>s.recording_id===selectedRecording));
        body.innerHTML=`<label class="device-day-selector">选择时间片段<select data-device-segment>${segments.map((s,index)=>`<option value="${index}" ${index===initial?'selected':''}>${esc(formatDate(s.start_us/1000))} — ${esc(formatDate(s.end_us/1000))} · ${esc(s.activity_label)}</option>`).join('')}</select></label><div data-device-segment-view></div>`;
        const show=index=>{
          const s=segments[index],ref=s.video||s.source_ref;
          const start=s.video?0:s.start_ms/1000,end=s.video?(s.end_ms-s.start_ms)/1000:s.end_ms/1000;
          const frames=(rows,label)=>`<div class="device-day-frame-grid">${rows.map(f=>`<figure><a href="${file(name,f.path)}"><img loading="lazy" src="${file(name,f.path)}" alt="${label}"></a><figcaption>${esc(f.understanding_text||label)} · 原片 ${(f.local_ms/1000).toFixed(2)} 秒</figcaption></figure>`).join('')}</div>`;
          body.querySelector('[data-device-segment-view]').innerHTML=`<h2>${esc(s.activity_label)}</h2><iframe class="device-day-video" title="实验视频播放器" style="width:100%;height:76vh;border:0" allowfullscreen src="${watch(name,ref.path)}#t=${start},${end}"></iframe><p>${link(s.json_path,'本片段 JSON 与溯源索引')} · ${s.video?'切分后的实验视频':'引用原视频的对应区间'}</p><h3>动作关键帧 · ${number(s.key_frames.length)}</h3>${s.key_frames.length?frames(s.key_frames,'动作关键帧'):'<p>本区间没有通过五类物理动作筛选的关键帧。</p>'}<details><summary>场景采样帧 · ${number(s.scene_frames.length)}</summary>${frames(s.scene_frames,'场景采样帧')}</details>`;
        };
        show(initial);body.querySelector('[data-device-segment]').addEventListener('change',event=>show(Number(event.target.value)));
      } else {
        const status={no_audio:'该分片没有录音',no_speech:'未检出语音',no_transcript:'模型未识别出文字',transcribed:'已识别'};
        body.innerHTML=`<h2>录音与 STT 识别</h2><p>仅展示该设备、该采集日的录音。点击句子时间可播放对应原录音区间。</p>${records.map(record=>{
          const transcript=record.transcription||{},comments=transcript.comments||[],chunks=transcript.chunks||[];
          const models=[...new Set(chunks.map(c=>c.model?.model||c.model?.repository).filter(Boolean))];
          const audio=(record.sources||[]).find(s=>s.kind==='audio_audio')?.retained;
          return `<article class="device-day-source"><h3>${esc(formatDate(record.start_us/1000))}</h3><p>${esc(status[transcript.outcome]||'等待识别')} ${models.length?' · '+esc(models.join('、')):''}</p>${audio?`<audio controls preload="none" src="${file(name,audio.path)}"></audio>`:''}${transcript.outcome==='no_transcript'?'<p>未识别出文字不能证明录音里没有人说话，原始返回已留存。</p>':''}${comments.map(c=>`<p><a href="${file(name,c.audio_ref.path)}#t=${c.audio_start_seconds},${c.audio_end_seconds}">${esc(formatDate(c.start_us/1000))}</a> ${esc(c.text)} · ${link(c.transcript_path,'识别来源')}</p>`).join('')}<details><summary>模型执行与分段记录</summary><ul>${chunks.map(c=>`<li>${c.start_seconds.toFixed(1)}–${c.end_seconds.toFixed(1)} 秒 · ${esc(status[c.outcome]||c.outcome)} · ${link(c.receipt,'执行回执')}</li>`).join('')||'<li>暂无执行回执。</li>'}</ul></details></article>`;
        }).join('')}<section data-device-context><h2>Comment 与 Protocol</h2><p>正在读取该设备当天的记录…</p></section>`;
        const context=body.querySelector('[data-device-context]');
        const results=await Promise.allSettled([fetch(file(name,'Comment/Comment.jsonl'),{cache:'no-store'}).then(async response=>{if(!response.ok)throw new Error('Comment unavailable');return (await response.text()).split('\n').filter(line=>line.trim()).map(line=>JSON.parse(line));}),api(file(name,'Comment/Protocol.json'))]);
        if(!context.isConnected)return;
        context.innerHTML='<h2>Comment 与 Protocol</h2>'+results.map((result,index)=>{
          const label=index?'Protocol':'人工 Comment';
          return `<h3>${label}</h3>${result.status==='fulfilled'?(index?`<p class="device-day-context-text">${esc(result.value.text||'')}</p>`:result.value.map(row=>`<p><strong>${esc(formatDate(row.start_us/1000))} · ${esc(row.author||'')}</strong> ${esc(row.text)}</p>`).join('')||'<p>尚无人工 Comment。</p>'):'<p>当前没有可读取的记录。</p>'}`;
        }).join('');
      }
    } catch(error) {
      if(location.hash!==requested)return;
      main.innerHTML=`<div class="page"><a href="#/device-days">返回设备日归档</a><section class="panel" role="alert">读取失败：${esc(error.message)}</section></div>`;
    }
  }
  return {renderLibrary,renderDetail};
})();
