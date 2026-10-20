/* Device/day evidence stays inside the existing application shell. */
window.VisionCortexDeviceDays = (() => {
  let selectedDate = "all", selectedCamera = "all";
  let detailGeneration = 0, detailController = null, detailRoot = null, detailHash = null;
  const route = (name, tab = "report") => `#/device-day/${encodeURIComponent(name)}/${tab}`;
  const file = (name, path) => `/api/device-days/${encodeURIComponent(name)}/files/${path.split('/').map(encodeURIComponent).join('/')}`;
  const watch = (name, path) => `/device-days/${encodeURIComponent(name)}/watch/${path.split('/').map(encodeURIComponent).join('/')}`;
  const tabs = {report:"整合日报",clips:"片段与帧索引",media:"原视频与采集数据",understanding:"步骤级理解",comment:"录音与备注"};
  function stopPlayers(root) {
    root?.querySelectorAll?.('video,audio,iframe').forEach(player=>{
      player.pause?.();player.removeAttribute?.('src');player.load?.();
      if(player.tagName==='IFRAME')player.src='about:blank';
    });
  }
  function clearDetail() {
    detailGeneration++;detailController?.abort();detailController=null;
    stopPlayers(detailRoot);detailRoot=null;
  }
  window.addEventListener?.('hashchange',()=>{if(detailRoot&&location.hash!==detailHash)clearDetail();});
  function renderLibrary(ctx) {
    clearDetail();
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
    clearDetail();
    const generation=detailGeneration;
    detailController=new AbortController();detailRoot=main;
    const options={signal:detailController.signal,readTimeoutMs:15000};
    const requested = location.hash;
    detailHash=requested;
    const current=()=>generation===detailGeneration&&location.hash===requested;
    setChrome('device-days');
    main.innerHTML='<div class="page"><section class="panel" role="status">正在读取设备日归档…</section></div>';
    try {
      let data={},timeIndex=null;
      try {timeIndex=await api(`/api/device-days/${encodeURIComponent(name)}/time-index`,options);} catch {
        if(!current())return;
        data=await api(`/api/device-days/${encodeURIComponent(name)}/index`,options);
      }
      if(!current())return;
      if(!tabs[tab])tab='report';
      const recordMap=new Map((timeIndex?.recordings||[]).map(row=>[row.recording_id,row]));
      (data.recordings||[]).forEach(row=>recordMap.set(row.recording_id,{...recordMap.get(row.recording_id),...row}));
      // Older published time indexes have direct media refs but no sources list.
      // Keep these existing files usable before a legacy canonical view is read.
      const addSource=(rid,kind,ref)=>{
        if(!rid||!ref)return;
        const record=recordMap.get(rid)||{recording_id:rid};
        const sources=[...(record.sources||[])];
        if(!sources.some(source=>source.kind===kind&&source.retained?.path===ref.path))sources.push({kind,retained:ref});
        recordMap.set(rid,{...record,sources});
      };
      (timeIndex?.streams?.video||[]).filter(row=>row.kind==='original_video').forEach(row=>addSource(row.recording_id,'video',row.file));
      (timeIndex?.streams?.audio||[]).forEach(row=>{
        addSource(row.recording_id,'audio_audio',row.file);
        (row.metadata_files||[]).forEach(ref=>addSource(row.recording_id,'audio_metadata',ref));
      });
      const records=[...recordMap.values()];
      const intervals=(timeIndex?.streams?.video||[]).filter(row=>row.kind==='video_segment'&&row.record_scope==='retention_interval');
      const chosen=intervals.filter(row=>row.video_value==='selected');
      const pathOf=ref=>{
        const path=typeof ref==='string'?ref:ref?.path;
        return typeof path==='string'&&(!ref?.path_base||ref.path_base==='device_day')&&
          /^(MetaData|MetaVideo|ProcessedClips|MultimodalUnderstanding|LaboratoryDailyReport|Comment)\//.test(path)&&
          !path.split('/').includes('..')&&!path.includes('\\')?path:null;
      };
      const playable=row=>{
        const path=pathOf(row.file);
        return row.video_value==='selected'&&row.processing_status==='completed'&&path&&
          /^ProcessedClips\/Clips\/\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(ExperimentActivity|NoExperimentActivity)_[A-Za-z0-9_-]{1,64}\/\1\.mp4$/.test(path)?path:null;
      };
      const published=chosen.filter(playable);
      const {preprocessed}=ctx.deviceDaySummary({...data,recordings:records});
      main.innerHTML=`<div class="page device-day-detail"><header class="page-hero compact"><div><p class="eyebrow"><a href="#/device-days">设备日归档</a> / ${esc(name.slice(0,10))}</p><h1>${esc(name.slice(11))}</h1><p>${number(records.length)} 个采集记录 · ${number(preprocessed)} 个预处理完成 · ${number(intervals.length)} 个筛选区间 · ${number(chosen.length)} 个选中（${number(published.length)} 个已有视频发布索引）</p><p>理解与文件分别以各区间实际状态为准。</p></div></header><nav class="archive-quick-filters" aria-label="设备日内容">${Object.entries(tabs).map(([key,label])=>`<a class="secondary-button ${key===tab?'active':''}" ${key===tab?'aria-current="page"':''} href="${route(name,key)}">${label}</a>`).join('')}</nav><section class="panel device-day-body"></section></div>`;
      const body=main.querySelector('.device-day-body');
      const link=(ref,label)=>{const path=pathOf(ref);return path?`<a href="${esc((path.toLowerCase().endsWith('.mp4')?watch:file)(name,path))}">${esc(label)}</a>`:`<span>${esc(label)}：尚未发布</span>`;};
      const values={selected:'选中',confirmed_no_value:'确认无价值',pending:'待判定'};
      const activities={experiment:'实验',non_experiment:'非实验',unknown:'未明确'};
      const processing={queued:'排队',running:'处理中',partial:'部分完成',completed:'完成',failed:'失败',
        waiting_for_value_review:'等待价值判断',waiting_for_evidence:'等待证据核验',submitted_unknown:'结果未知',waiting_for_capture:'等待输入'};
      const states=row=>`活动：${esc(activities[row.activity_category]||'未明确')} · 视频价值：${esc(values[row.video_value]||'待判定')} · 视频处理：${esc(processing[row.processing_status]||row.processing_status||'状态未知')}`;
      if(tab==='report') {
        const path='LaboratoryDailyReport/LaboratoryDailyReport.json';
        body.innerHTML=`<p>${link(path,'打开完整日报 JSON')} · 报告持续更新；输入覆盖、缺失与来源以该文件为准。</p><pre data-device-report>正在读取日报 JSON…</pre>`;
        const preview=body.querySelector('[data-device-report]');
        try {
          const report=await api(file(name,path),options);
          if(!current()||!preview.isConnected)return;
          const text=JSON.stringify(report,null,2);
          preview.textContent=text.length>200000?text.slice(0,200000)+'\n预览已截断；完整内容请打开日报 JSON。':text;
        } catch {
          if(current()&&preview.isConnected)preview.textContent='日报 JSON 尚未发布或暂时无法读取，已完成片段仍可在其他页查看。';
        }
      } else if(tab==='understanding') {
        const meanings=new Map((timeIndex?.streams?.multimodal||[]).map(row=>[row.segment_id,row]));
        body.innerHTML=intervals.map(row=>{
          const result=meanings.get(row.segment_id);
          return `<article class="device-day-source"><h2>${esc(formatDate(row.start_us/1000))} — ${esc(formatDate(row.end_us/1000))}</h2><p>${states(row)}</p><p>理解处理：${esc(processing[result?.status]||result?.status||'状态未知')}。实际动作、结果与不确定性以理解 JSON 为准。</p><p>${link(result?.content_file,'Understanding.json')} · ${link(row.metadata_file,'片段 JSON 与原片映射')}</p></article>`;
        }).join('')||'<p>对应时段的理解 JSON 尚未发布；原片和已有历史文件仍可独立查阅。</p>';
      } else if(tab==='media') {
        body.innerHTML=records.map(record=>{
          const sources=record.sources||(record.video?[{kind:'video',retained:record.video}]:[]),media=sources.filter(source=>['video','audio_audio'].includes(source.kind));
          const metadata=sources.filter(source=>!['video','audio_audio'].includes(source.kind));
          return `<article class="device-day-source"><h2>${esc(formatDate(record.start_us/1000))}</h2><p>${media.map(source=>link(source.retained,source.kind==='video'?'打开原视频':'播放原录音')).join(' · ')}</p><details data-capture-files="${esc(record.recording_id)}"><summary>CSV、标定、时间戳与采集日志</summary><ul>${metadata.map(source=>`<li>${link(source.retained,source.retained.path.split('/').pop())}</li>`).join('')}</ul></details></article>`;
        }).join('') || '<p>正在归档采集文件。</p>';
        body.querySelectorAll('[data-capture-files]').forEach(details=>details.addEventListener('toggle',async()=>{
          const record=records.find(row=>row.recording_id===details.dataset.captureFiles);
          if(!details.open||details.dataset.loaded||!record)return;
          try {
            const catalog=record.capture_manifest
              ?await api(file(name,record.capture_manifest.path),options)
              :{files:((await api(`/api/device-days/${encodeURIComponent(name)}/index`,options)).recordings||[])
                .find(row=>row.recording_id===record.recording_id)?.sources||[]};
            if(!current()||!details.isConnected)return;
            const refs=(catalog.files||[]).filter(row=>row.retained&&!record.sources?.some(source=>['video','audio_audio'].includes(source.kind)&&source.retained?.path===row.retained.path));
            details.querySelector('ul').innerHTML=refs.map(row=>`<li>${link(row.retained,row.retained.path.split('/').pop())}</li>`).join('')+(record.capture_manifest?`<li>${link(record.capture_manifest,'CaptureFiles.json')}</li>`:'');
            details.dataset.loaded='true';
          } catch {if(current()&&details.isConnected)details.querySelector('ul').insertAdjacentHTML('beforeend','<li>补充文件清单暂时无法读取，已归档的 CSV 仍可打开。</li>');}
        }));
      } else if(tab==='clips') {
        if(!intervals.length){body.innerHTML=timeIndex?'<p>尚无已发布的筛选区间。预处理候选与原片保留在采集数据页，不代表已经选中。</p>':'<p>筛选时间索引暂时无法读取，当前无法确认选中视频；可独立查阅原片与采集数据。</p>';return;}
        const preferred=intervals.findIndex(s=>(!selectedRecording||s.recording_id===selectedRecording)&&playable(s));
        const initial=preferred>=0?preferred:Math.max(0,intervals.findIndex(s=>s.recording_id===selectedRecording));
        body.innerHTML=`<p>发布索引与实际可读性分别核验；播放器只打开选中且已完成发布的视频，不自动改播候选整段或原片。</p><label class="device-day-selector">选择筛选区间<select data-device-segment>${intervals.map((s,index)=>`<option value="${index}" ${index===initial?'selected':''}>${esc(formatDate(s.start_us/1000))} — ${esc(formatDate(s.end_us/1000))} · ${esc(activities[s.activity_category]||'未明确')} · ${esc(values[s.video_value]||'待判定')}</option>`).join('')}</select></label><div data-device-segment-view></div>`;
        const show=index=>{
          if(!current())return;
          const row=intervals[index],view=body.querySelector('[data-device-segment-view]');
          if(!row||!view.isConnected)return;
          stopPlayers(view);
          const video=playable(row),source=pathOf(row.source_video);
          const mapping=Number.isFinite(row.source_start_ms)&&Number.isFinite(row.source_end_ms)?`${(row.source_start_ms/1000).toFixed(3)}–${(row.source_end_ms/1000).toFixed(3)} 秒`:'映射待核验';
          const trace=source?`<a href="${esc(watch(name,source))}${Number.isFinite(row.source_start_ms)&&Number.isFinite(row.source_end_ms)?`#t=${row.source_start_ms/1000},${row.source_end_ms/1000}`:''}">原片追溯（${esc(mapping)}）</a>`:'原片引用待发布';
          view.innerHTML=`<h2>${states(row)}</h2>${video?`<video class="device-day-video" controls preload="metadata" src="${file(name,video)}"></video><p data-device-playback-status>已发布索引；实际播放与画面质量以打开文件后的核验为准。</p>`:'<p>本区间没有可播放的已发布选中视频。待判定、失败或结果未知不代表无价值。</p>'}<p>${esc(row.selection_reason||'判断依据待发布')}</p><p>${link(row.metadata_file,'本片段 JSON 与溯源索引')} · ${trace}</p>`;
          view.querySelector('video')?.addEventListener('error',()=>{
            if(current()&&view.isConnected)view.querySelector('[data-device-playback-status]').textContent='视频暂时无法播放；保留该索引与原片追溯，不自动回退播放原片。';
          });
        };
        show(initial);body.querySelector('[data-device-segment]').addEventListener('change',event=>show(Number(event.target.value)));
      } else {
        const status={no_audio:'该分片没有录音',no_speech:'未检出语音',no_transcript:'模型未识别出文字',transcribed:'已识别'};
        const independent=timeIndex?.streams?.audio||[];
        const audioEntries=independent.map(row=>{
          const stt=row.transcription||{},audio=pathOf(row.file);
          const existingPlayer=records.some(record=>(record.sources||[]).some(source=>source.kind==='audio_audio'&&pathOf(source.retained)===audio));
          return `<article class="device-day-source"><h3>独立录音 · ${esc(formatDate(row.start_us/1000))}</h3><p>${esc(status[stt.outcome]||processing[stt.status]||'等待识别')} · 以该录音自己的时间与来源为准。</p>${audio&&!existingPlayer?`<audio controls preload="none" src="${file(name,audio)}"></audio>`:''}<p>${link(stt.content_file,'Transcript.json')} · ${link(stt.readable_text_file,'Transcript.txt')}</p></article>`;
        }).join('');
        body.innerHTML=`<h2>录音与 STT 识别</h2><p>仅展示该设备、该采集日的录音。点击句子时间可播放对应原录音区间。</p>${timeIndex?.field_events_file?`<p>${link(timeIndex.field_events_file,'现场 OCR 事件 JSON')} · 读数与操作结论分别记录。</p>`:''}${audioEntries}${records.map(record=>{
          const transcript=record.transcription||{},comments=transcript.comments||[],chunks=transcript.chunks||[];
          const models=[...new Set(chunks.map(c=>c.model?.model||c.model?.repository).filter(Boolean))];
          const audio=(record.sources||[]).find(s=>s.kind==='audio_audio')?.retained;
          return `<article class="device-day-source"><h3>${esc(formatDate(record.start_us/1000))}</h3><p>${esc(status[transcript.outcome]||'等待识别')} ${models.length?' · '+esc(models.join('、')):''}</p>${audio?`<audio controls preload="none" src="${file(name,audio.path)}"></audio>`:''}${transcript.outcome==='no_transcript'?'<p>未识别出文字不能证明录音里没有人说话，原始返回已留存。</p>':''}${comments.map(c=>`<p><a href="${file(name,c.audio_ref.path)}#t=${c.audio_start_seconds},${c.audio_end_seconds}">${esc(formatDate(c.start_us/1000))}</a> ${esc(c.text)} · ${link(c.transcript_path,'识别来源')}</p>`).join('')}<details><summary>模型执行与分段记录</summary><ul>${chunks.map(c=>`<li>${c.start_seconds.toFixed(1)}–${c.end_seconds.toFixed(1)} 秒 · ${esc(status[c.outcome]||c.outcome)} · ${link(c.receipt,'执行回执')}</li>`).join('')||'<li>暂无执行回执。</li>'}</ul></details></article>`;
        }).join('')}<section data-device-context><h2>Comment 与历史参考材料</h2><p>正在读取该设备当天的记录…</p></section>`;
        const context=body.querySelector('[data-device-context]');
        const results=await Promise.allSettled([fetch(file(name,'Comment/Comment.jsonl'),{cache:'no-store',signal:options.signal}).then(async response=>{if(!response.ok)throw new Error('Comment unavailable');return (await response.text()).split('\n').filter(line=>line.trim()).map(line=>JSON.parse(line));}),api(file(name,'Comment/Protocol.json'),options)]);
        if(!current()||!context.isConnected)return;
        context.innerHTML='<h2>Comment 与历史参考材料</h2>'+results.map((result,index)=>{
          const label=index?'历史 Protocol 参考材料（不代表已观察步骤）':'人工 Comment';
          return `<h3>${label}</h3>${result.status==='fulfilled'?(index?`<p class="device-day-context-text">${esc(result.value.text||'')}</p>`:result.value.map(row=>`<p><strong>${esc(formatDate(row.start_us/1000))} · ${esc(row.author||'')}</strong> ${esc(row.text)}</p>`).join('')||'<p>尚无人工 Comment。</p>'):'<p>当前没有可读取的记录。</p>'}`;
        }).join('');
      }
    } catch(error) {
      if(!current())return;
      main.innerHTML=`<div class="page"><a href="#/device-days">返回设备日归档</a><section class="panel" role="alert">读取失败：${esc(error.message)}</section></div>`;
    }
  }
  return {renderLibrary,renderDetail};
})();
