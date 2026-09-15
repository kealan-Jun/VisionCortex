(() => {
  const filter=document.getElementById('DateFilter');
  const key='VisionCortexArchiveOverviewClipsV2';
  let saved={};
  try {saved=JSON.parse(sessionStorage.getItem(key)||'{}');} catch {}
  const hashDate=location.hash.startsWith('#Day')?location.hash.slice(4):null;
  const defaultDate=document.querySelector('[data-activity-count]:not([data-activity-count="0"])')?.dataset.day||'';
  const date=hashDate??saved.date??defaultDate;
  filter.value=[...filter.options].some(o=>o.value===date)?date:'';
  const apply=()=>document.querySelectorAll('[data-day]').forEach(el=>{el.hidden=!!filter.value&&el.dataset.day!==filter.value;});
  apply();
  filter.addEventListener('change',apply);
  if(location.protocol==='http:'||location.protocol==='https:') {
    document.querySelectorAll('[data-artifact]').forEach(a=>{
      const prefix=a.hasAttribute('data-watch')?'/device-days/':'/api/device-days/';
      const action=a.hasAttribute('data-watch')?'/watch/':'/files/';
      a.href=prefix+encodeURIComponent(a.dataset.archive)+action+a.dataset.path.split('/').map(encodeURIComponent).join('/')+(a.dataset.fragment||'');
    });
  } else {document.querySelectorAll('[data-app]').forEach(a=>{a.hidden=true;});}
  const allDetails=[...document.querySelectorAll('details')];
  allDetails.forEach((d,i)=>{d.open=(saved.open||[]).includes(i);});
  if(!location.hash&&saved.scroll)window.scrollTo(0,saved.scroll);
  function refresh(){
    try {sessionStorage.setItem(key,JSON.stringify({date:filter.value,scroll:window.scrollY,open:allDetails.flatMap((d,i)=>d.open?[i]:[])}));} catch {}
    location.reload();
  }
  document.getElementById('Refresh').onclick=refresh;
  document.querySelectorAll('a[href^="#Day"]').forEach(a=>a.addEventListener('click',()=>{
    filter.value=a.hash.slice(4);apply();
  }));
  addEventListener('hashchange',()=>{if(location.hash.startsWith('#Day')){filter.value=location.hash.slice(4);apply();document.getElementById('Day'+filter.value)?.scrollIntoView();}});
  if(hashDate)requestAnimationFrame(()=>document.getElementById('Day'+filter.value)?.scrollIntoView());
  VisionCortexArchivePlayback.install();
  const age=Date.now()/1000-Number(document.querySelector('[data-updated]').dataset.updated);
  if(age>120){const p=document.createElement('p');p.className='error';p.textContent='总览超过 2 分钟未更新，以下是上次快照，请检查后台服务。';document.querySelector('header').append(p);}
  setInterval(()=>{if(!document.hidden&&!document.getElementById('Playback').open&&document.activeElement?.tagName!=='SELECT')refresh();},60000);
})();
