"""Device-day views consume published interval files rather than CV candidates."""
import shutil
import subprocess

import pytest

from repo_paths import WEB


def run_node(script):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for browser characterization')
    source = (WEB/'device-days.js').read_text()
    result = subprocess.run([node], input='const assert=require("node:assert/strict");\n'+DOM+'\n'+source+'\n'+script,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout+result.stderr


DOM = r'''
const listeners={};const window={addEventListener:(name,fn)=>{listeners[name]=fn;}};
global.fetch=async()=>({ok:false,text:async()=>''});
const location={hash:'#/device-day/2026-10-20_cam/clips'};
function player(){return {tagName:'VIDEO',paused:0,removed:0,loaded:0,
 pause(){this.paused++;},removeAttribute(){this.removed++;},load(){this.loaded++;},addEventListener(){}};}
function node(){return {isConnected:true,innerHTML:'',textContent:'',handlers:{},
 addEventListener(name,fn){this.handlers[name]=fn;},querySelector(){return null;},querySelectorAll(){return [];}};}
const pages=[];let html='',body=null;
const main={get innerHTML(){return html;},set innerHTML(value){
 if(body){body.isConnected=false;Object.values(body.nodes).forEach(n=>{n.isConnected=false;});}
 html=value;body=node();body.nodes={};
 Object.defineProperty(body,'innerHTML',{get(){return this.html||'';},set(value){
  this.html=value;this.nodes={};
  if(value.includes('data-device-segment-view')){
   const view=node();Object.defineProperty(view,'innerHTML',{get(){return this.html||'';},set(value){
    this.html=value;this.video=value.includes('<video ')?player():null;this.status=node();
   }});view.querySelector=s=>s==='video'?view.video:s==='[data-device-playback-status]'?view.status:null;
   view.querySelectorAll=()=>view.video?[view.video]:[];
   this.nodes['[data-device-segment-view]']=view;this.nodes['[data-device-segment]']=node();
  }
  if(value.includes('data-device-report'))this.nodes['[data-device-report]']=node();
  if(value.includes('data-device-context'))this.nodes['[data-device-context]']=node();
 }});
 body.querySelector=s=>body.nodes[s]||null;
 body.querySelectorAll=()=>body.nodes['[data-device-segment-view]']?.querySelectorAll()||[];
 pages.push(body);
},querySelector:()=>body,querySelectorAll:()=>body?.querySelectorAll()||[]};
const requests=[];
const api=(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject}));
const ctx={main,api,esc:x=>String(x??''),number:x=>String(x),setChrome(){},formatDate:x=>String(x),
 deviceDaySummary:()=>({completed:0,preprocessed:1})};
const ref=path=>({path,path_base:'device_day'});
const original='MetaData/Capture/rgb.mp4';
const candidate={kind:'video_segment',record_scope:'preprocessing_candidate',segment_id:'candidate',
 recording_id:'r',file:ref('ProcessedClips/Clips/LegacyWhole/ExperimentActivity.mp4'),start_us:1000000,end_us:901000000};
const folder=id=>`00-00-01_00-00-02_NoExperimentActivity_${id}`;
const interval=(id,value='selected',status='completed')=>({kind:'video_segment',record_scope:'retention_interval',
 recording_id:'r',segment_id:id,start_us:1000000,end_us:2000000,source_start_ms:1000,source_end_ms:2000,
 activity_category:'non_experiment',video_value:value,processing_status:status,
 file:value==='selected'?ref(`ProcessedClips/Clips/${folder(id)}/NoExperimentActivity.mp4`):null,
 source_video:ref(original),metadata_file:ref(`ProcessedClips/Clips/${folder(id)}/NoExperimentActivity.json`)});
const first=interval('Selected1'),second=interval('Selected2'),pending=interval('Pending','pending','waiting_for_value_review');
const noValue=interval('NoValue','confirmed_no_value','completed');
const source={recordings:[{recording_id:'r',start_us:1000000,sources:[{kind:'video',retained:ref(original)}]}],
 segments:[{...candidate,video:candidate.file.path,activity:'active',source_ref:ref(original)}]};
const timeline={recordings:[],streams:{video:[{kind:'original_video',file:ref(original)},candidate,first,second,pending,noValue],
 multimodal:[candidate,...[first,second,pending,noValue].map(r=>({segment_id:r.segment_id,
 content_file:ref(`MultimodalUnderstanding/${folder(r.segment_id)}/Understanding.json`)}))]}};
const tick=()=>new Promise(setImmediate);
async function resolvePage(tab='clips',time=timeline,sourceValue=source){
 const offset=requests.length;location.hash=`#/device-day/2026-10-20_cam/${tab}`;
 const render=window.VisionCortexDeviceDays.renderDetail(ctx,'2026-10-20_cam',tab);
 assert.ok(requests[offset].url.endsWith('/time-index'));
 if(time instanceof Error){
  requests[offset].reject(time);await tick();
  assert.ok(requests[offset+1].url.endsWith('/index'));
  if(sourceValue instanceof Error)requests[offset+1].reject(sourceValue);else requests[offset+1].resolve(sourceValue);
 }else requests[offset].resolve(time);
 if(tab==='report'){await tick();return {render,offset};}
 if(tab==='comment'){await tick();requests[offset+1].resolve({text:''});}
 await render;return {render,offset};
}
'''


def test_only_published_retention_intervals_are_counted_or_played_and_pending_keeps_json():
    run_node(r'''
(async()=>{
 await resolvePage();
 assert.ok(main.innerHTML.includes('4 个筛选区间'));
 assert.equal(main.innerHTML.includes('完成全流程'),false);
 assert.ok(main.innerHTML.includes('2 个选中（2 个已有视频发布索引）'));
 assert.equal(body.innerHTML.includes('LegacyWhole'),false);
 const view=body.nodes['[data-device-segment-view]'];
 assert.ok(view.innerHTML.includes('NoExperimentActivity_Selected1/NoExperimentActivity.mp4'));
 assert.equal(view.innerHTML.includes('src="/api/device-days/2026-10-20_cam/files/MetaData/'),false);
 assert.ok(view.innerHTML.includes('视频价值：选中'));
 assert.ok(view.innerHTML.includes('活动：非实验'));
 assert.ok(view.innerHTML.includes('处理：完成'));
 const oldPlayer=view.video;
 body.nodes['[data-device-segment]'].handlers.change({target:{value:'2'}});
 assert.equal(oldPlayer.paused,1);assert.equal(oldPlayer.removed,1);assert.equal(oldPlayer.loaded,1);
 assert.equal(view.innerHTML.includes('<video'),false);
 assert.ok(view.innerHTML.includes('NoExperimentActivity_Pending/NoExperimentActivity.json'));
 assert.ok(view.innerHTML.includes('原片追溯（1.000–2.000 秒）'));
 assert.ok(view.innerHTML.includes('视频价值：待判定'));
 body.nodes['[data-device-segment]'].handlers.change({target:{value:'3'}});
 assert.equal(view.innerHTML.includes('<video'),false);assert.ok(view.innerHTML.includes('确认无价值'));
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


@pytest.mark.parametrize('time', ['missing', 'selected_without_verified_publication'])
def test_missing_selection_or_incomplete_publication_never_falls_back_to_candidate(time):
    run_node(f'const scenario="{time}";\n'+r'''
(async()=>{
 const data=scenario==='missing'?Error('missing TimeIndex'):{streams:{video:[interval('NotPublished','selected','queued')]}};
 await resolvePage('clips',data);
 assert.equal(body.innerHTML.includes('<video'),false);
 assert.equal(body.innerHTML.includes('LegacyWhole'),false);
 assert.equal(main.innerHTML.includes('1 个已有视频发布索引'),false);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_understanding_opens_corresponding_json_without_candidate_or_html_substitute():
    run_node(r'''
(async()=>{
 await resolvePage('understanding');
 assert.ok(body.innerHTML.includes('MultimodalUnderstanding/00-00-01_00-00-02_NoExperimentActivity_Selected1/Understanding.json'));
 assert.ok(body.innerHTML.includes('MultimodalUnderstanding/00-00-01_00-00-02_NoExperimentActivity_Pending/Understanding.json'));
 assert.equal(body.innerHTML.includes('<iframe'),false);
 assert.equal(body.innerHTML.includes('.html'),false);
 assert.equal(body.innerHTML.includes('LegacyWhole'),false);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_time_index_still_delivers_selected_files_when_canonical_index_is_unavailable():
    run_node(r'''
(async()=>{
 await resolvePage('clips',timeline,Error('source index unavailable'));
 assert.equal(requests.length,1);
 assert.ok(requests[0].url.endsWith('/time-index'));
 assert.ok(body.nodes['[data-device-segment-view]'].innerHTML.includes('NoExperimentActivity_Selected1/NoExperimentActivity.mp4'));
 assert.ok(main.innerHTML.includes('2 个选中'));
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_published_sources_and_ocr_do_not_wait_for_canonical_index():
    run_node(r'''
(async()=>{
 const time={recordings:[{recording_id:'r',start_us:1000000,sources:[
  {kind:'video',retained:ref(original)},{kind:'frames',retained:ref('MetaData/Capture/frames.csv')}],
  capture_manifest:ref('Comment/.Evidence/CaptureFiles/abc.json')}],streams:{video:[],audio:[]},
  field_events_file:ref('Comment/FieldEvents.json')};
 await resolvePage('media',time);
 assert.ok(body.innerHTML.includes('MetaData/Capture/frames.csv'));
 assert.ok(body.innerHTML.includes('MetaData/Capture/rgb.mp4'));
 assert.equal(requests.length,1);assert.ok(requests[0].url.endsWith('/time-index'));
 await resolvePage('comment',time);
 assert.ok(body.innerHTML.includes('Comment/FieldEvents.json'));
 assert.equal(requests.some(r=>r.url.endsWith('/index')),false);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_replaced_detail_cannot_render_a_late_time_index_or_start_legacy_reads():
    run_node(r'''
(async()=>{
 const old=window.VisionCortexDeviceDays.renderDetail(ctx,'2026-10-20_cam','clips');
 const currentRender=window.VisionCortexDeviceDays.renderDetail(ctx,'2026-10-20_cam','clips');
 requests[1].resolve(timeline);await currentRender;
 const latest=body.nodes['[data-device-segment-view]'].innerHTML;
 requests[0].resolve({streams:{video:[interval('Obsolete')]}});await old;
 assert.equal(body.nodes['[data-device-segment-view]'].innerHTML,latest);
 assert.equal(requests[0].options.signal.aborted,true);
 assert.equal(requests.length,2);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_report_reads_json_and_same_route_refresh_rejects_stale_response():
    run_node(r'''
(async()=>{
 const old=await resolvePage('report');
 const oldBody=body,oldPreview=body.nodes['[data-device-report]'];
 assert.ok(oldBody.innerHTML.includes('LaboratoryDailyReport/LaboratoryDailyReport.json'));
 assert.equal(oldBody.innerHTML.includes('<iframe'),false);
 const oldReport=requests[old.offset+1];
 const fresh=await resolvePage('report');
 assert.equal(requests[old.offset].options.signal.aborted,true);
 requests[fresh.offset+1].resolve({生成时间:'new',输入覆盖:'actual'});await fresh.render;
 const latest=body.nodes['[data-device-report]'].textContent;
 oldReport.resolve({生成时间:'obsolete'});await old.render;
 assert.equal(body.nodes['[data-device-report]'].textContent,latest);
 assert.equal(oldPreview.textContent,'');
 assert.ok(latest.includes('new'));
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_route_leave_aborts_reads_and_releases_active_player():
    run_node(r'''
(async()=>{
 await resolvePage();const active=body.nodes['[data-device-segment-view]'].video;
 location.hash='#/knowledge';listeners.hashchange();
 assert.equal(active.paused,1);assert.equal(active.removed,1);assert.equal(active.loaded,1);
 assert.equal(requests[0].options.signal.aborted,true);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_audio_first_transcript_files_remain_visible_without_canonical_video_record():
    run_node(r'''
(async()=>{
 const audio=ref('MetaData/Capture/audio.opus');
 const time={recordings:[{recording_id:'audio-first',start_us:1000000,end_us:2000000}],streams:{video:[],audio:[{
  recording_id:'audio-first',start_us:1000000,end_us:2000000,file:audio,transcription:{status:'completed',outcome:'transcribed',
  content_file:ref('Comment/FixedAudioSpan/Transcript.json'),readable_text_file:ref('Comment/FixedAudioSpan/Transcript.txt')}
 }]}};
 await resolvePage('comment',time,{recordings:[],segments:[]});
 assert.ok(body.innerHTML.includes('Comment/FixedAudioSpan/Transcript.json'));
 assert.ok(body.innerHTML.includes('Comment/FixedAudioSpan/Transcript.txt'));
 assert.ok(body.innerHTML.includes('<audio controls preload="none"'));
 assert.ok(main.innerHTML.includes('1 个采集记录'));
 assert.equal(body.innerHTML.includes('<video'),false);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


@pytest.mark.parametrize('path', ['MetaData/Capture/rgb.mp4', 'MetaVideo/Capture/rgb.mp4',
    'Comment/Unsupported.mp4', 'ProcessedClips/Clips/LegacyWhole/ExperimentActivity.mp4'])
def test_mislabelled_raw_or_noncontract_file_never_becomes_selected_player(path):
    import json
    run_node('const mislabelled='+json.dumps(path)+';\n'+r'''
(async()=>{
 const row=interval('Mislabelled');row.file=ref(mislabelled);
 await resolvePage('clips',{streams:{video:[row]}});
 assert.equal(body.nodes['[data-device-segment-view]'].innerHTML.includes('<video'),false);
 assert.ok(main.innerHTML.includes('1 个选中（0 个已有视频发布索引）'));
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
