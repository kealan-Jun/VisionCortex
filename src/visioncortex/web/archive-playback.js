/* Browse sealed media using the shared offline alignment when available. */
const VisionCortexArchivePlayback = (() => {
  function position(recordings, us) {
    const r = recordings.find(x => x.start_us <= us && us < x.end_us && x.source_url);
    if (!r) return null;
    const p = r.seek_points;
    let i = 1;
    while (i < p.length - 1 && p[i][0] < us) i++;
    const a = p[i - 1], b = p[i];
    return {recording:r, seconds:a[1] + (us - a[0]) / (b[0] - a[0]) * (b[1] - a[1]),
      rate:(b[1]-a[1])*1e6/(b[0]-a[0])};
  }
  function mediaChoices(p, us, start) {
    const r=p.recording, choices=[];
    const clip=(r.clips||[]).find(c=>c.start_us<=us&&us<c.end_us&&c.source_url);
    if(clip)choices.push({url:clip.source_url,seconds:p.seconds-clip.offset_seconds,kind:'clip'});
    // Use range playback of an available original immediately. A disposable
    // transcode is a codec fallback, not a prerequisite for every camera.
    if(r.source_url && (!r.source_status || r.source_status==='available'))choices.push({url:r.source_url,seconds:p.seconds,kind:'original'});
    if(r.preview_url && (!r.source_status || r.source_status==='available')){
      const window=Math.floor((us-start)/30000000), left=Math.max(r.start_us,start+window*30000000);
      choices.push({url:r.preview_url+'&window='+window,seconds:p.seconds-position([r],left).seconds,kind:'preview'});
    }
    return choices;
  }
  function mediaPosition(p, us, start) {
    const choice=mediaChoices(p,us,start)[0];
    return choice?{url:choice.url,seconds:choice.seconds}:null;
  }
  function canAdvance(videos) {
    return videos.some(v=>!v.hidden&&!v.error&&v.readyState>=2&&!v.seeking);
  }
  function install() {
    const dialog = document.getElementById('Playback'), body = document.getElementById('PlaybackBody');
    const title = document.getElementById('PlaybackTitle');
    const esc = x => String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const clock = us => new Date(us/1000).toLocaleTimeString('zh-CN', {timeZone:'Asia/Shanghai',hour12:false});
    let stop = () => {}, generation = 0;
    function clear() {generation++;stop();body.querySelectorAll('video').forEach(v=>{v.pause();v.removeAttribute('src');v.load();});body.replaceChildren();}
    dialog.addEventListener('close', clear);
    function open(label) {clear();title.textContent=label;dialog.showModal();}
    document.querySelectorAll('[data-watch]').forEach(a=>a.addEventListener('click',e=>{
      e.preventDefault();open('实验片段 · '+a.dataset.archive);
      const player=document.createElement('iframe');player.title='实验视频播放器';
      player.src='/device-days/'+encodeURIComponent(a.dataset.archive)+'/watch/'+a.dataset.path.split('/').map(encodeURIComponent).join('/')+(a.dataset.fragment||'');
      player.style.cssText='width:100%;height:76vh;border:0';player.allowFullscreen=true;
      body.append(player);
    }));
    document.querySelectorAll('[data-copy-path]').forEach(b=>b.addEventListener('click',async()=>{
      const input=b.parentElement.querySelector('input');input.focus();input.select();
      try {await navigator.clipboard.writeText(input.value);b.textContent='已复制';}
      catch {b.textContent='路径已选中，请复制';}
    }));
    document.querySelectorAll('[data-multiview]').forEach(b=>b.addEventListener('click',async()=>{
      open(b.dataset.dayValue+' · 同一时段多视角');const token=generation;
      body.textContent='正在读取该时段各相机的归档与处理状态…';
      try {
        const q=new URLSearchParams({archive:b.dataset.archive,segment_id:b.dataset.segment});
        const response=await fetch('/api/day-timeline/'+encodeURIComponent(b.dataset.dayValue)+'/playback?'+q);
        if(!response.ok)throw new Error('读取失败（'+response.status+'），请刷新后重试');
        const data=await response.json();if(token!==generation||!dialog.open)return;
        const roles={first_person:'第一人称',third_person:'第三人称',unknown:'视角待配置'};
        const actions={hand_object_contact:'手与物体接触',object_movement:'物体移动',liquid_movement:'液体移动',container_state_change:'容器状态变化',device_panel_operation:'设备面板操作'};
        const available=data.cameras.filter(c=>c.recordings.some(r=>mediaChoices({recording:r,seconds:0},r.start_us,data.start_us).length));
        const unavailable=data.cameras.filter(c=>!available.includes(c));
        const alignment=data.alignment_ready?'已使用共享离线对齐结果定位各路画面':'尚无通过门控的共享对齐结果，以下按采集时间对照';
        const missingDual=(data.grouping_decisions||[]).some(d=>d.decision==='quarantined_missing_dual_view');
        const association=data.experiments?.length?'共享离线流程已形成动作关联；这不代表完整实验的起止边界已经确认。':'该时段尚未形成通过离线分组门控的实验配对；其他机位供核查。';
        body.innerHTML=`<p><strong>${clock(data.start_us)}—${clock(data.end_us)}</strong> · ${alignment}</p><p>${association}${missingDual&&!data.experiments?.length?' 当前片段的动作证据只有一种视角，未通过第一、第三人称共同支持条件。':''}</p>
          ${data.experiments?.length?`<details><summary>查看 ${data.experiments.length} 组动作关联与边界依据</summary>`:''}${(data.experiments||[]).map(e=>`<section><h3>双视角动作关联 · ${((e.group.global_end_ms-e.group.global_start_ms)/1000).toFixed(2)} 秒</h3><p>${esc(e.group.continuity_reason||'共享动作分组')}。完整实验边界待核验。</p>${e.status==='partial_view_coverage'?'<p>该关联区间包含未核验的第三人称时段，未生成占位对齐视频；下方可以播放各设备的实际同期画面。</p>':''}<div>${(e.outputs||[]).filter(o=>o.aligned_video).map(o=>`<h4>第一、第三人称对齐视频</h4><video controls preload="metadata" src="/api/device-days/${encodeURIComponent(o.archive)}/files/${o.aligned_video.path.split('/').map(encodeURIComponent).join('/')}"></video>`).join('')}</div><div class="view-grid">${(e.outputs||[]).map(o=>`<article><h4>${esc(roles[o.role])} · ${esc(o.camera)}</h4><video controls preload="metadata" src="/api/device-days/${encodeURIComponent(o.archive)}/files/${o.video.path.split('/').map(encodeURIComponent).join('/')}"></video><a href="/api/device-days/${encodeURIComponent(o.archive)}/files/${o.json_path.split('/').map(encodeURIComponent).join('/')}">实验边界、对应视角与关键事件</a></article>`).join('')}</div></section>`).join('')}${data.experiments?.length?'</details>':''}
          ${(data.actions||[]).length?`<p>本片段动作候选关键帧：${data.actions.map(a=>`<a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(actions[a.action_type]||'待核验动作')}</a>`).join(' · ')}。动作结论仍以完整证据核验为准。</p>`:''}
          <div class="playback-controls"><button data-play-all>播放可用视角</button><button data-pause-all>暂停</button><input type="range" min="0" max="${(data.end_us-data.start_us)/1e6}" value="0" step="0.1" aria-label="共同时间"><output></output></div>
          <div class="view-grid" data-synchronized-views>${available.map(c=>`<article><h3>${esc(roles[c.role]||roles.unknown)} · ${esc(c.camera)}</h3><p>${c.selected?'所选片段':c.association==='offline_group_selected'?'共享离线分组对应视角':'同期对照 · 实验对应待核验'}</p><video muted playsinline preload="none"></video><p data-view-status role="status"></p><small data-view-clock></small><button data-retry-view hidden>重试本路</button></article>`).join('')}</div>
          ${unavailable.length?`<details><summary>该时段另有 ${unavailable.length} 路无可用画面</summary><ul>${unavailable.map(c=>`<li>${esc(c.camera)}：${c.recordings.length?'已记录原片引用，但文件缺失或不可读':c.unknown_time_recordings?'记录缺少有效时间范围':'没有覆盖该时段的已归档录像'}</li>`).join('')}</ul></details>`:''}
          ${data.errors.length?`<details><summary>${data.errors.length} 个设备日索引读取失败</summary><ul>${data.errors.map(e=>`<li>${esc(e.archive)}：${esc(e.reason)}</li>`).join('')}</ul></details>`:''}
          <p><small>某一路准备或失败时，其余可用视角继续播放。原片分片边界不作为实验结束依据。</small></p>`;
        const slider=body.querySelector('input[type=range]'), output=body.querySelector('output');
        const cards=[...body.querySelectorAll('[data-synchronized-views] article')];
        let running=false, anchor=0, base=0, timer=null;
        const playing=new WeakSet(), failures=cards.map(()=>new Set());
        function pause(){running=false;clearInterval(timer);cards.forEach(c=>c.querySelector('video').pause());}
        function draw(us, seek=false) {
          output.textContent=clock(us);
          available.forEach((c,i)=>{
            const card=cards[i], video=card.querySelector('video'), state=card.querySelector('[data-view-status]');
            const retry=card.querySelector('[data-retry-view]'), mapping=card.querySelector('[data-view-clock]');
            const p=position(c.recordings,us);
            if(!p){video.pause();video.hidden=true;retry.hidden=true;state.textContent='该时刻没有覆盖画面，其余视角继续播放';mapping.textContent='';return;}
            const r=p.recording, media=mediaChoices(p,us,data.start_us).find(m=>!failures[i].has(m.url));
            if(!media){video.pause();video.hidden=true;retry.hidden=false;state.textContent='本路片段与原片预览均未能读取；可重试本路，其余视角不受影响';return;}
            video.hidden=false;retry.hidden=true;
            video.onerror=()=>{
              if(token!==generation||!dialog.open)return;
              failures[i].add(media.url);draw(data.start_us+Number(slider.value)*1e6,true);
            };
            if(video.dataset.source!==media.url){video.dataset.source=media.url;video.preload='auto';video.src=media.url;video.load();}
            const apply=()=>{
              if(token!==generation)return;
              if(seek||(!video.seeking&&Math.abs(video.currentTime-media.seconds)>.4))video.currentTime=Math.max(0,media.seconds);
              video.playbackRate=Math.max(.25,Math.min(4,p.rate));
              if(running&&!playing.has(video)&&video.paused){playing.add(video);video.play().catch(()=>{}).finally(()=>playing.delete(video));}
            };
            if(video.readyState>=1)apply();
            else video.onloadedmetadata=()=>{if(token===generation)draw(data.start_us+Number(slider.value)*1e6,true);};
            const missingClip=(r.clips||[]).some(c=>c.status&&c.status!=='available');
            const origin=media.kind==='clip'?'切分后的活动视频':missingClip||failures[i].size?'片段不可用，已切换到原片对应区间':'原片对照';
            state.textContent=origin+(video.readyState<2?' · 正在准备本路画面':'');
            mapping.textContent=r.time_basis==='shared_offline_alignment'?`共享离线对齐${r.alignment?.state==='aligned'?'':'估计（该区间待验证）'} · 估计不确定度 ${Number(r.alignment?.uncertainty_ms||0).toFixed(1)} 毫秒`:r.time_basis==='recorder_csv_interpolation'?'依据采集 CSV 定位 · 精确同步偏差未验证':'依据分片开始时间估算定位';
          });
        }
        cards.forEach((card,i)=>card.querySelector('[data-retry-view]').onclick=()=>{failures[i].clear();card.querySelector('video').dataset.source='';draw(data.start_us+Number(slider.value)*1e6,true);});
        stop=pause;
        slider.oninput=()=>{base=Number(slider.value);anchor=performance.now();draw(data.start_us+base*1e6,true);};
        body.querySelector('[data-play-all]').onclick=()=>{
          pause();if(Number(slider.value)>=Number(slider.max))slider.value=0;
          running=true;anchor=performance.now();base=Number(slider.value);draw(data.start_us+base*1e6,true);
          timer=setInterval(()=>{
            const now=performance.now();
            if(canAdvance(cards.map(c=>c.querySelector('video'))))base=Math.min(Number(slider.max),base+(now-anchor)/1000);
            anchor=now;slider.value=base;draw(data.start_us+base*1e6);if(base>=Number(slider.max))pause();
          },100);
        };
        body.querySelector('[data-pause-all]').onclick=pause;
        draw(data.start_us,true);
      } catch(e) {if(token===generation)body.textContent=e.message;}
    }));
  }
  return {position,mediaPosition,mediaChoices,canAdvance,install};
})();
if(typeof module==='object')module.exports=VisionCortexArchivePlayback;
