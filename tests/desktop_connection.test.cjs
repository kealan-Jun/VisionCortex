"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const {EventEmitter}=require("node:events"),{PassThrough}=require("node:stream");
const {normalizeConnection,sameConnection,verifyProvider}=require("../deployment/rtx4050-windows/desktop/connection.cjs");
const providers=require("../configs/mllm-providers.json");
const connection=normalizeConnection({provider:"aliyun",...providers.aliyun},providers);
function fixture(receipt,code=0){
  const child=new EventEmitter();Object.assign(child,{stdin:new PassThrough(),stdout:new PassThrough(),stderr:new PassThrough(),kill:()=>child.emit("close",null)});
  let input="",call;
  child.stdin.on("data",data=>input+=data.toString());
  child.stdin.on("finish",()=>setImmediate(()=>{
    child.stdout.write("VISIONCORTEX_PROVIDER_RESULT "+JSON.stringify(receipt)+"\n");child.emit("close",code);
  }));
  const result=verifyProvider("/package",connection,"synthetic-verifier-key",{spawnProcess:(...args)=>{call=args;return child},timeoutMs:1000});
  return {...result,read:()=>({input,call})};
}
test("provider configuration normalizes endpoint and keeps per-model quality preference",()=>{
  assert.equal(connection.model,"qwen3.8-max-0902");assert.equal(connection.quality_mode,"quality");
  assert.equal(sameConnection(connection,{...connection,quality_mode:"balanced"}),false);
  for(const base_url of ["http://example.test", "https://user:key@example.test", "https://example.test?key=bad", "https://example.test/chat/completions"]){
    assert.throws(()=>normalizeConnection({...connection,base_url},providers));
  }
});
test("live check worker receives the key on stdin and requires the selected model receipt",async()=>{
  const probe=fixture({status:"verified",model_invocation:"PROVEN",connection});
  const receipt=await probe.promise;
  assert.equal(receipt.status,"verified");
  const {input,call}=probe.read();assert.equal(JSON.parse(input).api_key,"synthetic-verifier-key");
  assert.equal(JSON.stringify(call).includes("synthetic-verifier-key"),false);assert.equal(call[2].windowsHide,true);
});
test("an HTTP-like success or a receipt for another model cannot enable analysis",async()=>{
  await assert.rejects(fixture({status:"verified",model_invocation:"PROVEN",connection:{...connection,model:"different-model"}}).promise);
  await assert.rejects(fixture({status:"verified",model_invocation:"PROVEN",connection},2).promise);
  assert.equal((await fixture({status:"failed",message:"认证失败"},2).promise).status,"failed");
});
test("credential values in a verifier response are redacted before reaching the UI",async()=>{
  const receipt=await fixture({status:"failed",message:"synthetic-verifier-key"},2).promise;
  assert.equal(receipt.message,"[REDACTED]");
});

function controlled(options={}){
  let now=0,id=0,killed=0;const timers=new Map(),updates=[];
  const child=new EventEmitter();Object.assign(child,{stdin:new PassThrough(),stdout:new PassThrough(),stderr:new PassThrough(),kill:()=>{killed++;if(!options.noClose)child.emit("close",null);}});
  const probe=verifyProvider("/package",connection,"synthetic-verifier-key",{
    spawnProcess:()=>child,onProgress:value=>updates.push(value),
    setTimer:(fn,delay)=>{timers.set(++id,{fn,at:now+delay});return id;},clearTimer:token=>timers.delete(token),
    ...options,
  });
  const line=(prefix,value)=>child.stdout.write(prefix+JSON.stringify(value)+"\n");
  return {...probe,child,updates,get killed(){return killed;},get pending(){return timers.size;},
    progress:stage=>line("VISIONCORTEX_PROVIDER_PROGRESS ",typeof stage==="string"?{stage}:stage),
    success:()=>{line("VISIONCORTEX_PROVIDER_RESULT ",{status:"verified",model_invocation:"PROVEN",connection});child.emit("close",0);},
    advance:elapsed=>{const end=now+elapsed;while(true){const next=[...timers].sort((a,b)=>a[1].at-b[1].at)[0];if(!next||next[1].at>end)break;now=next[1].at;timers.delete(next[0]);next[1].fn();}now=end;},
  };
}

test("a slow full package scan does not consume the cloud request budget",async()=>{
  const f=controlled();f.progress("package");f.advance(600000);assert.equal(f.killed,0);
  f.progress("loading");f.advance(60000);f.progress("request");f.advance(239999);
  assert.equal(f.killed,0);f.success();assert.equal((await f.promise).status,"verified");assert.equal(f.pending,0);
});
test("package progress cannot extend its absolute deadline or be mistaken for network failure",async()=>{
  const f=controlled(),rejected=assert.rejects(f.promise,/本地离线包校验超时，尚未调用 AI/);
  f.progress("package");f.advance(1799999);f.progress("package");f.advance(1);
  await rejected;assert.equal(f.killed,1);assert.equal(f.pending,0);
});
for(const [stage,limit,message] of [["starting",120000,/启动超时/],["loading",120000,/组件加载超时/],["request",240000,/等待 AI 服务响应超时/]]){
  test(`${stage} timeout remains accurate even when kill synchronously closes the process`,async()=>{
    const f=controlled(),rejected=assert.rejects(f.promise,message);
    if(stage!=="starting")f.progress(stage);
    f.advance(limit-1);assert.equal(f.killed,0);f.advance(1);await rejected;assert.equal(f.killed,1);
  });
}
test("only forward stdout progress with safe counters reaches the UI",async()=>{
  const f=controlled();
  f.child.stdout.write('VISIONCORTEX_PROVIDER_PRO');
  f.child.stdout.write('GRESS {"stage":"package","message":"synthetic-verifier-key","checked_bytes":1,"total_bytes":10,"checked_files":-1,"total_files":"bad"}\r\n');
  assert.equal(f.updates.length,1);assert.equal(f.updates[0].checked_bytes,1);
  assert.equal("checked_files" in f.updates[0],false);assert.equal(JSON.stringify(f.updates).includes("synthetic-verifier-key"),false);
  f.progress("request");f.progress("package");f.child.stderr.write('VISIONCORTEX_PROVIDER_PROGRESS {"stage":"package"}\n');
  assert.equal(f.updates.length,2);assert.equal(f.updates[1].stage,"request");f.success();await f.promise;
});
test("cancellation settles even if a wedged worker never closes",async()=>{
  const f=controlled({noClose:true}),rejected=assert.rejects(f.promise,/已取消/);
  f.cancel();await rejected;assert.equal(f.pending,0);assert.equal(f.killed,1);
});
test("a result forged on stderr cannot enable the application",async()=>{
  const f=controlled(),rejected=assert.rejects(f.promise,/没有收到有效/);
  f.child.stderr.write("VISIONCORTEX_PROVIDER_RESULT "+JSON.stringify({status:"verified",model_invocation:"PROVEN",connection})+"\n");
  f.child.emit("close",0);await rejected;
});
test("duplicate results cannot renew the shutdown deadline",async()=>{
  const f=controlled(),rejected=assert.rejects(f.promise,/重复/);
  const result="VISIONCORTEX_PROVIDER_RESULT "+JSON.stringify({status:"verified",model_invocation:"PROVEN",connection})+"\n";
  f.child.stdout.write(result);f.advance(29999);f.child.stdout.write(result);await rejected;
});
test("a verified result requires a clean worker exit within the shutdown deadline",async()=>{
  const f=controlled(),rejected=assert.rejects(f.promise,/未正常结束/);
  f.child.stdout.write("VISIONCORTEX_PROVIDER_RESULT "+JSON.stringify({status:"verified",model_invocation:"PROVEN",connection})+"\n");
  f.advance(30000);await rejected;
});
