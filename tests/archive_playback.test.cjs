const test=require('node:test');
const assert=require('node:assert/strict');
const {position,mediaPosition}=require('../src/visioncortex/web/archive-playback.js');
test('seek across recording boundaries with a gap and nonuniform clock',()=>{
  const a={start_us:0,end_us:2000000,source_url:'/first',seek_points:[[0,0],[1000000,1.01],[2000000,2.03]]};
  const b={start_us:3000000,end_us:4000000,source_url:'/second',seek_points:[[3000000,0],[4000000,1]]};
  assert.equal(position([a,b],1500000).seconds,1.52);
  assert.equal(position([a,b],2000000),null);
  assert.equal(position([a,b],2500000),null);
  assert.equal(position([a,b],3000000).seconds,0);
  assert.equal(position([a,b],4000000),null);
  assert.equal(position([{...a,source_url:null}],500000),null);
});
test('reuse a published clip and keep preview times relative to their own window',()=>{
 const r={start_us:0,end_us:90000000,source_url:'/raw',preview_url:'/preview?x=1',
  seek_points:[[0,0],[90000000,90]],clips:[{start_us:0,end_us:10000000,source_url:'/clip',offset_seconds:0}]};
 assert.deepEqual(mediaPosition(position([r],5000000),5000000,0),{url:'/clip',seconds:5});
 assert.deepEqual(mediaPosition(position([r],35000000),35000000,0),{url:'/raw',seconds:35});
 const fallback=mediaChoices(position([r],35000000),35000000,0).find(m=>m.kind==='preview');
 assert.deepEqual(fallback,{url:'/preview?x=1&window=1',seconds:5,kind:'preview'});
});
const {mediaChoices,canAdvance}=require('../src/visioncortex/web/archive-playback.js');
test('a missing derived clip falls back to its own original; missing originals are not requested',()=>{
 const r={start_us:0,end_us:10000000,source_url:'/raw',preview_url:'/preview?x=1',source_status:'available',seek_points:[[0,0],[10000000,10]],clips:[{start_us:0,end_us:10000000,source_url:null,status:'missing'}]};
 assert.deepEqual(mediaChoices(position([r],5000000),5000000,0).map(m=>m.kind),['original','preview']);
 r.source_status='missing';assert.deepEqual(mediaChoices(position([r],5000000),5000000,0),[]);
});
test('one buffering camera cannot freeze ready cameras',()=>{
 assert.equal(canAdvance([{readyState:2,hidden:false,seeking:false},{readyState:0,hidden:false,seeking:false}]),true);
 assert.equal(canAdvance([{readyState:0,hidden:false,seeking:false}]),false);
 assert.equal(canAdvance([{readyState:4,hidden:true,seeking:false}]),false);
});
