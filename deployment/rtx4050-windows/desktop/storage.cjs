"use strict";
const path = require("node:path");
const {spawn} = require("node:child_process");

function storageRequest(root, value, {spawnProcess=spawn, timeoutMs=12000}={}) {
  return new Promise((resolve,reject)=>{
    const child=spawnProcess(path.join(root,"python/python.exe"),["-B",path.join(root,"tools/desktop_storage.py")],
      {cwd:root,windowsHide:true,shell:false,stdio:["pipe","pipe","ignore"],env:{...process.env,PYTHONUTF8:"1",PYTHONDONTWRITEBYTECODE:"1"}});
    let output="",settled=false;
    const done=(error,value)=>{if(settled)return;settled=true;clearTimeout(timer);error?reject(error):resolve(value);};
    const timer=setTimeout(()=>{child.kill();done(new Error("保存位置检测超时，请检查网络盘连接或选择本地目录。"));},timeoutMs);
    child.stdout.setEncoding("utf8");
    child.stdout.on("data",chunk=>{output+=chunk;if(output.length>65536){child.kill();done(new Error("保存位置检测返回异常。"));}});
    child.once("error",()=>done(new Error("保存位置检测无法启动。")));
    child.stdin.on("error",()=>{});
    child.once("close",()=>{
      try{const result=JSON.parse(output);if(!result.ok)throw new Error(result.message||"目录不可用。");done(null,result);}
      catch(error){done(new Error(error instanceof SyntaxError?"保存位置检测未完成。":error.message));}
    });
    child.stdin.end(JSON.stringify(value)+"\n");
  });
}

function settingsChangeAllowed(runs, health) {
  if(!Array.isArray(runs?.runs)||!health?.large_uploads?.sessions) return false;
  const terminal=new Set(["completed","partial","failed","interrupted"]);
  return runs.runs.every(run=>terminal.has(run.state)) && Number(health.large_uploads.sessions.open)===0;
}
module.exports={storageRequest,settingsChangeAllowed};
