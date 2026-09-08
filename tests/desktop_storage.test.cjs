"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict");
const {EventEmitter}=require("node:events"),{PassThrough}=require("node:stream");
const {storageRequest,settingsChangeAllowed}=require("../deployment/rtx4050-windows/desktop/storage.cjs");

test("settings changes reject active work, incomplete uploads and unknown status",()=>{
  const health={large_uploads:{sessions:{open:0}}};
  assert.equal(settingsChangeAllowed({runs:[]},health),true);
  assert.equal(settingsChangeAllowed({runs:[{state:"completed"},{state:"partial"},{state:"failed"}]},health),true);
  for(const state of ["running","queued","preparing",undefined])assert.equal(settingsChangeAllowed({runs:[{state}]},health),false);
  assert.equal(settingsChangeAllowed({runs:[]},{large_uploads:{sessions:{open:1}}}),false);
  assert.equal(settingsChangeAllowed({runs:[]},{}),false);
});

test("directory path is passed through stdin without shell interpretation",async()=>{
  let invocation,received;
  const child=new EventEmitter();Object.assign(child,{stdin:new PassThrough(),stdout:new PassThrough(),kill(){}});
  child.stdin.on("data",chunk=>{received=JSON.parse(chunk.toString());});
  const pending=storageRequest("/package with spaces",{action:"probe",data_root:"D:\\实验 files"},{spawnProcess:(...args)=>{invocation=args;return child;}});
  child.stdout.end(JSON.stringify({ok:true,data_root:"D:\\实验 files",kind:"local"}));child.emit("close",0);
  assert.equal((await pending).kind,"local");
  assert.equal(invocation[2].shell,false);assert.equal(invocation[2].windowsHide,true);
  assert.equal(invocation[1].some(value=>value.includes("实验")),false);
  assert.equal(received.data_root,"D:\\实验 files");
});

test("an unavailable network path times out and does not return a local fallback",async()=>{
  let killed=false;
  const child=new EventEmitter();Object.assign(child,{stdin:new PassThrough(),stdout:new PassThrough(),kill(){killed=true;}});
  await assert.rejects(storageRequest("/package",{action:"probe",data_root:"\\\\nas\\offline"},{spawnProcess:()=>child,timeoutMs:10}),/超时/);
  assert.equal(killed,true);
});
