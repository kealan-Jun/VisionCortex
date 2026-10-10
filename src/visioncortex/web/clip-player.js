/* Paint actual decoded frames to avoid native MP4 overlay blackouts in embedded browsers. */
(() => {
  const video=document.getElementById('Media'), canvas=document.getElementById('Picture');
  const play=document.getElementById('Play'), seek=document.getElementById('Seek');
  const status=document.getElementById('Status'), retry=document.getElementById('Retry');
  const context=canvas.getContext('2d', {alpha:false});
  let frames=0, paintedTime=-1, timer=null;
  const match=location.hash.match(/^#t=(\d+(?:\.\d+)?)(?:,(\d+(?:\.\d+)?))?$/);
  let left=0, right=Infinity;
  const clock=t=>Number.isFinite(t)?`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`:'--:--';
  function message(text,error=false){status.textContent=text;status.className=error?'error':'';}
  function update(){
    seek.disabled=!Number.isFinite(video.duration);seek.min=left;seek.max=Number.isFinite(video.duration)?Math.min(right,video.duration):0;
    seek.value=video.currentTime;document.getElementById('Time').textContent=`${clock(video.currentTime)} / ${clock(video.duration)}`;
    play.textContent=video.paused?'播放':'暂停';
  }
  function paint(){
    if(video.readyState>=2&&video.videoWidth&&video.currentTime!==paintedTime){
      if(canvas.width!==video.videoWidth||canvas.height!==video.videoHeight){canvas.width=video.videoWidth;canvas.height=video.videoHeight;}
      try{context.drawImage(video,0,0,canvas.width,canvas.height);paintedTime=video.currentTime;frames++;canvas.dataset.paintedFrames=String(frames);canvas.dataset.mediaTime=String(paintedTime);}
      catch{message('画面绘制失败，请重新载入。',true);retry.hidden=false;}
    }
    update();
  }
  function watch(){paint();if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(watch);}
  if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(watch);
  else timer=setInterval(paint,1000/30);
  play.onclick=async()=>{
    if(!video.paused){video.pause();return;}
    if(video.currentTime>=right)video.currentTime=left;
    try{await video.play();}catch(e){message(e.name==='NotAllowedError'?'请再次点击播放。':'无法播放此视频，请重新载入。',true);retry.hidden=false;}
  };
  video.addEventListener('loadedmetadata',()=>{
    left=match?Math.min(Number(match[1]),video.duration):0;
    right=match?.[2]?Math.min(Number(match[2]),video.duration):video.duration;
    if(right<=left){left=0;right=video.duration;}
    video.currentTime=left;update();
  });
  video.addEventListener('loadeddata',()=>{paint();message('首帧已就绪，点击播放。');});
  video.addEventListener('playing',()=>{message('正在播放');retry.hidden=true;update();});
  video.addEventListener('pause',()=>{update();if(!video.ended)message('已暂停');});
  video.addEventListener('timeupdate',()=>{update();if(!video.paused&&video.currentTime>=right){video.pause();message('所选片段播放结束');}});
  video.addEventListener('seeked',paint);
  video.addEventListener('waiting',()=>message('正在缓冲视频，请稍候…'));
  video.addEventListener('stalled',()=>{message('视频传输暂时停滞，可重新载入。',true);retry.hidden=false;});
  video.addEventListener('ended',()=>{update();message('播放结束');});
  video.addEventListener('error',()=>{message(`视频读取或解码失败（${video.error?.code||'未知'}），请重新载入。`,true);retry.hidden=false;});
  seek.oninput=()=>{video.currentTime=Number(seek.value);update();};
  document.getElementById('Speed').onchange=e=>{video.playbackRate=Number(e.target.value);};
  document.getElementById('Sound').onclick=e=>{video.muted=!video.muted;e.target.textContent=video.muted?'取消静音':'静音';};
  document.getElementById('Fullscreen').onclick=()=>document.getElementById('Screen').requestFullscreen?.().catch(()=>message('当前窗口不支持全屏。'));
  retry.onclick=()=>{paintedTime=-1;retry.hidden=true;message('正在重新载入…');video.load();};
  window.addEventListener('pagehide',()=>{video.pause();if(timer)clearInterval(timer);});
})();
