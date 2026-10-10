/* Piecewise mapping: [audio/video seconds, experiment milliseconds]. */
(() => {
  function mapTime(mapping, value, reverse = false) {
    const points = mapping?.anchors || [];
    if (!Number.isFinite(value) || points.length < 2) return null;
    const axis = reverse ? 1 : 0, output = 1-axis;
    let low = 0, high = points.length;
    while (low < high) { const mid = (low+high) >> 1; if (points[mid][axis] < value) low = mid+1; else high = mid; }
    if (low < points.length && points[low][axis] === value) return points[low][output];
    if (!low || low === points.length) return null;
    const a = points[low-1], b = points[low];
    if (b[axis] <= a[axis] || (mapping.max_gap_seconds != null && b[0]-a[0] > mapping.max_gap_seconds)) return null;
    return a[output] + (value-a[axis])/(b[axis]-a[axis])*(b[output]-a[output]);
  }
  function locateAudio(mappings, chunks, globalMs) {
    for (const mapping of mappings || []) {
      const seconds = mapTime(mapping, globalMs, true);
      const chunk = chunks.find(item => item.id === mapping.chunk_id);
      if (seconds !== null && chunk && seconds >= 0 && seconds <= chunk.end_seconds-chunk.start_seconds) return {chunk, seconds};
    }
    return null;
  }
  function bind({audio, video, mapping, videos, selectedView, status, current, onTime}) {
    let active = null, disposed = false;
    const listeners = [];
    const listen = (target, event, fn) => { target.addEventListener(event, fn); listeners.push(()=>target.removeEventListener(event, fn)); };
    function sync(force = false) {
      if (disposed) return;
      if (!current()) { dispose(); return; }
      const globalMs = mapTime(mapping, audio.currentTime);
      onTime(globalMs);
      const selected = videos.find(item => item.view_id === selectedView() && mapTime(item, globalMs, true) !== null);
      if (globalMs === null || !selected) {
        video.pause(); video.hidden = true;
        status.textContent = globalMs === null ? "此处没有可靠的音画时间映射，录音可独立播放。" : "此时刻没有所选视角画面，可切换视角。";
        return;
      }
      video.hidden = false;
      const seconds = mapTime(selected, globalMs, true);
      if (active?.url !== selected.url) {
        active = selected; video.pause(); video.src = selected.url; video.load(); force = true;
      }
      if (video.readyState) {
        if (force || (video.readyState >= 2 && !video.seeking && Math.abs(video.currentTime-seconds) > .25)) video.currentTime = seconds;
        const nextGlobal = mapTime(mapping, audio.currentTime+.25);
        const nextVideo = nextGlobal === null ? null : mapTime(selected, nextGlobal, true);
        video.playbackRate = nextVideo === null ? audio.playbackRate : Math.max(.25, Math.min(4, (nextVideo-seconds)/.25*audio.playbackRate));
        if (audio.paused || audio.ended) video.pause();
        else if (video.paused) video.play().catch(()=> { status.textContent = "视频播放暂不可用，录音仍可收听。"; });
      }
      status.textContent = `已按采集时钟同步 · 音画实际误差待复核${selected.uncertainty_ms ? ` · 对齐估计误差 ±${Math.round(selected.uncertainty_ms)} 毫秒` : ""}`;
    }
    function dispose() { if (disposed) return; disposed = true; listeners.forEach(fn=>fn()); audio.pause(); video.pause(); }
    for (const event of ["timeupdate", "play", "pause", "seeking", "ratechange", "ended"]) listen(audio, event, ()=>sync(event === "seeking"));
    listen(video, "loadedmetadata", ()=>sync(true));
    listen(video, "error", ()=> { status.textContent = "同步视频加载失败；请刷新或检查该批次来源。"; });
    listen(window, "hashchange", dispose);
    sync(true);
    return {sync, dispose};
  }
  window.VisionCortexSpeechSync = {mapTime, locateAudio, bind};
})();
