"use strict";
const {spawn}=require("node:child_process");
const path=require("node:path");
const {validateApiKey}=require("./controller.cjs");

function normalizeConnection(value, providers) {
  const result={};
  for(const field of ["provider","base_url","model","api_protocol"]){
    const item=value?.[field];
    if(typeof item!=="string" || !item.trim() || item.length>2048 || /[\x00-\x1f]/u.test(item)) throw new Error("请填写完整的厂商、接口地址和图像模型。");
    result[field]=item.trim();
  }
  if(!providers[result.provider] || !["ark_responses","chat_completions"].includes(result.api_protocol)) throw new Error("请选择支持的服务和接口类型。");
  let url;
  try {url=new URL(result.base_url);} catch {throw new Error("请输入完整的 HTTPS Base URL。");}
  if(url.protocol!=="https:" || url.username || url.password || url.search || url.hash) throw new Error("接口地址须使用 HTTPS，不能包含密钥或查询参数。");
  const pathname=url.pathname.replace(/\/+$/u,"");
  if(pathname.endsWith("/responses") || pathname.endsWith("/chat/completions")) throw new Error("请填写 Base URL，不要包含 /responses 或 /chat/completions。");
  result.base_url=url.origin+pathname;
  result.quality_mode=value.quality_mode||"quality";
  if(!["quality","balanced"].includes(result.quality_mode)) throw new Error("请选择有效的使用偏好。");
  return result;
}

function sameConnection(left,right){
  return ["provider","base_url","model","api_protocol","quality_mode"].every(field=>left?.[field]===right?.[field]);
}

const STAGES=["starting","package","loading","request","finishing"];
const STAGE_TIMEOUTS={starting:120000,package:1800000,loading:120000,request:240000,finishing:30000};
const TIMEOUT_MESSAGES={
  starting:"本地验证程序启动超时，尚未调用 AI 服务。请检查包内 Python 与 Windows 运行组件。",
  package:"本地离线包校验超时，尚未调用 AI 服务。请确认压缩包已完整解压到本地 SSD，并检查磁盘状态。",
  loading:"本地验证组件加载超时，尚未调用 AI 服务。请检查运行组件及诊断记录。",
  request:"等待 AI 服务响应超时。本地文件校验已通过，请检查所选接口、模型服务状态与网络后重试。",
  finishing:"已收到验证结果，但本地验证程序未正常结束，请查看诊断记录后重试。",
};
function verifyProvider(root, connection, key, {spawnProcess=spawn, timeoutMs=240000,
  stageTimeouts={},onProgress=()=>{},setTimer=setTimeout,clearTimer=clearTimeout}={}) {
  key=validateApiKey(key);
  const child=spawnProcess(path.join(root,"python/python.exe"),["-u","-B",path.join(root,"tools/verify_mllm_connection.py")],{
    cwd:root,windowsHide:true,shell:false,stdio:["pipe","pipe","pipe"],
    env:{...process.env,VISIONCORTEX_DESKTOP_MODE:"1",PYTHONUTF8:"1",PYTHONIOENCODING:"utf-8",PYTHONDONTWRITEBYTECODE:"1"},
  });
  const limits={...STAGE_TIMEOUTS,request:timeoutMs,...stageTimeouts};
  let buffer="",bytes=0,settled=false,stage="starting",timer,receipt,abort;
  const promise=new Promise((resolve,reject)=>{
    function finish(error,value){
      if(settled)return;
      settled=true;clearTimer(timer);if(error)reject(error);else resolve(value);
    }
    abort=message=>{finish(new Error(message));child.kill();};
    function arm(){
      clearTimer(timer);
      timer=setTimer(()=>abort(TIMEOUT_MESSAGES[stage]),limits[stage]);
    }
    function progress(value){
      const index=STAGES.indexOf(value?.stage);
      if(index<1||index>3||index<STAGES.indexOf(stage))return;
      // Progress within a phase must not renew its absolute deadline.
      if(value.stage!==stage){stage=value.stage;arm();}
      const counts={};
      for(const name of ["checked_bytes","total_bytes","checked_files","total_files"]){
        if(Number.isSafeInteger(value[name])&&value[name]>=0)counts[name]=value[name];
      }
      const labels={package:"正在校验本地离线包，完成后再连接 AI 服务…",loading:"正在加载本地验证组件，尚未调用 AI 服务…",request:"本地校验已通过，正在实际调用所选模型验证多图理解…"};
      let message=labels[stage];
      if(stage==="package"&&counts.total_bytes>0&&counts.checked_bytes<=counts.total_bytes){
        message=`正在校验本地离线包：${(counts.checked_bytes/1024**3).toFixed(2)} / ${(counts.total_bytes/1024**3).toFixed(2)} GiB；${counts.checked_files??0} / ${counts.total_files??0} 个文件。完成后再连接 AI 服务。`;
      }
      try{onProgress({stage,message,...counts});}catch{/* UI/log failure must not alter verification. */}
    }
    function consume(line){
      if(line.startsWith("VISIONCORTEX_PROVIDER_PROGRESS ")){
        try{progress(JSON.parse(line.slice("VISIONCORTEX_PROVIDER_PROGRESS ".length)));}catch{/* Ignore non-protocol diagnostics. */}
      }else if(line.startsWith("VISIONCORTEX_PROVIDER_RESULT ")){
        if(stage==="finishing")return abort("收到重复的模型调用回执，连接验证已中止。");
        try{
          receipt=JSON.parse(line.slice("VISIONCORTEX_PROVIDER_RESULT ".length).split(key).join("[REDACTED]"));
          stage="finishing";arm();
        }catch{abort("没有收到有效的模型调用回执，请检查运行组件与诊断记录。");}
      }
    }
    arm();
    child.stdin.on("error",()=>{});
    child.stdout.setEncoding("utf8");child.stderr.setEncoding("utf8");
    child.stdout.on("data",chunk=>{
      if(settled)return;
      bytes+=Buffer.byteLength(chunk);if(bytes>1024*1024)return abort("本地验证输出异常，连接验证已中止。");
      buffer+=chunk;
      let end;while((end=buffer.indexOf("\n"))>=0){const line=buffer.slice(0,end).replace(/\r$/u,"");buffer=buffer.slice(end+1);consume(line);if(settled)return;}
    });
    // stderr cannot forge a successful result or reach the UI with credentials.
    child.stderr.on("data",chunk=>{bytes+=Buffer.byteLength(chunk);if(!settled&&bytes>1024*1024)abort("本地验证输出异常，连接验证已中止。");});
    child.once("error",()=>finish(new Error("无法启动连接验证，请检查应用运行组件。")));
    child.once("close",code=>{
      if(settled)return;
      if(buffer)consume(buffer);
      if(settled)return;
      if(!receipt||!["verified","failed"].includes(receipt.status)||
        receipt.status==="verified"&&(code!==0||!sameConnection(receipt.connection,connection)||receipt.model_invocation!=="PROVEN")){
        return finish(new Error("没有收到有效的模型调用回执，请检查运行组件与诊断记录。"));
      }
      finish(null,receipt);
    });
    child.stdin.end(JSON.stringify({connection,api_key:key})+"\n");
  });
  return {promise,cancel:()=>{if(!settled)abort("连接验证已取消。");}};
}

module.exports={normalizeConnection,sameConnection,verifyProvider};
