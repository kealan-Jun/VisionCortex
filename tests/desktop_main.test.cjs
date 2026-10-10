"use strict";
const {test}=require("node:test"),assert=require("node:assert/strict"),vm=require("node:vm");
const fs=require("node:fs"),os=require("node:os"),path=require("node:path"),{EventEmitter}=require("node:events"),{createHash}=require("node:crypto");
const source=path.resolve(__dirname,"..");
const tick=()=>new Promise(resolve=>setImmediate(resolve));

async function fixture(t){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),"vc-main-"));t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  fs.mkdirSync(path.join(root,"configs"));fs.copyFileSync(path.join(source,"configs/mllm-providers.json"),path.join(root,"configs/mllm-providers.json"));
  fs.mkdirSync(path.join(root,"src/visioncortex"),{recursive:true});
  const hash=createHash("sha256");for(const name of ["mllm.py","mllm_provider.py","provider_connection.py"]){const bytes=fs.readFileSync(path.join(source,"src/visioncortex",name));hash.update(bytes);fs.writeFileSync(path.join(root,"src/visioncortex",name),bytes);}
  const adapter=hash.digest("hex"),handlers={},starts=[],notices=[];let window,menu,failProbe=false,runs={runs:[]};
  class Window extends EventEmitter{
    constructor(){super();window=this;this.webContents=new EventEmitter();Object.assign(this.webContents,{mainFrame:{url:""},send(){},getURL:()=>this.webContents.mainFrame.url,session:{setPermissionRequestHandler(){}},setWindowOpenHandler(){}});}
    async loadFile(file){this.webContents.mainFrame.url=require("node:url").pathToFileURL(file).href;}
    async loadURL(url){this.webContents.mainFrame.url=url;}
    isDestroyed(){return false;}show(){}isMinimized(){return false;}focus(){}
  }
  class Controller extends EventEmitter{constructor(){super();this.child=null;this.origin="http://127.0.0.1:8199";}start(...args){starts.push(args);this.child={};}async stop(){this.child=null;}}
  const app=new EventEmitter();Object.assign(app,{setName(){},disableHardwareAcceleration(){},setPath(){},getPath:()=>path.join(root,"UserConfig"),requestSingleInstanceLock:()=>true,whenReady:()=>Promise.resolve(),quit(){},exit(){}});
  const electron={app,BrowserWindow:Window,ipcMain:{handle:(name,handler)=>handlers[name]=handler},safeStorage:{isEncryptionAvailable:()=>true,encryptString:value=>Buffer.from(value),decryptString:value=>value.toString()},shell:{openPath:async()=>"",openExternal:async()=>{}},Menu:{buildFromTemplate:items=>items,setApplicationMenu:items=>menu=items},dialog:{showMessageBox:async(_w,value)=>notices.push(value),showErrorBox(){},showOpenDialog:async()=>({canceled:false,filePaths:[path.join(root,"Chosen")]})}};
  const controlledRequire=name=>{
    if(name==="electron")return electron;
    if(name==="./controller.cjs")return {...require(path.join(source,"deployment/rtx4050-windows/desktop/controller.cjs")),BackendController:Controller};
    if(name==="./storage.cjs")return {...require(path.join(source,"deployment/rtx4050-windows/desktop/storage.cjs")),storageRequest:async(_root,value)=>{
      if(value.action==="discover")return {candidates:[{data_root:path.join(root,"NAS"),kind:"network",label:"已连接网络盘"}],message:"synthetic mapped drive"};
      if(failProbe)throw new Error("测试：网络盘不可用");
      return {source_root:value.source_root||"",data_root:value.data_root,archive_root:path.join(value.data_root,"Archives"),kind:value.data_root.endsWith("NAS")?"network":"local",write_rename_read:"PROVEN"};
    }};
    if(name==="./connection.cjs")return {...require(path.join(source,"deployment/rtx4050-windows/desktop/connection.cjs")),verifyProvider:(_root,connection,key,options)=>{options.onProgress({stage:"package",message:"正在校验本地离线包",checked_bytes:1,total_bytes:2});options.onProgress({stage:"request",message:"正在等待模型响应"});return {promise:Promise.resolve(key==="synthetic-bad"?{status:"failed",message:"synthetic authentication rejection"}:{status:"verified",model_invocation:"PROVEN",connection,adapter_sha256:adapter}),cancel(){}};}};
    if(name==="node:child_process")return {spawn:()=>{const child=new EventEmitter();setImmediate(()=>child.emit("close",0));return child;}};
    if(name==="node:net")return {createServer:()=>({once(){},listen(_port,_host,cb){cb();},address:()=>({port:8199}),close(cb){cb();}})};
    return require(name);
  };
  vm.runInNewContext(fs.readFileSync(path.join(source,"deployment/rtx4050-windows/desktop/main.cjs"),"utf8"),{
    require:controlledRequire,__dirname:path.join(root,"resources/app"),process,console,URL,AbortSignal,
    fetch:async url=>({ok:true,json:async()=>url.endsWith("/api/runs")?runs:{large_uploads:{sessions:{open:0}}}}),
  });
  for(let i=0;i<8;i++)await tick();
  const call=(name,value)=>handlers["desktop:"+name]({sender:window.webContents,senderFrame:window.webContents.mainFrame},value);
  return {root,call,starts,notices,get menu(){return menu;},setFailProbe:value=>{failProbe=value;},setRuns:value=>{runs=value;}};
}

test("desktop first launch recommends NAS; failed selection is not saved; selected folder reaches owned backend",async t=>{
  const f=await fixture(t);
  assert.equal(f.call("get-state").status,"needs-storage");
  assert.equal(f.call("get-state").storageSuggestion,path.join(f.root,"NAS"));
  f.setFailProbe(true);await assert.rejects(f.call("save-storage",path.join(f.root,"NAS")),/不可用/);
  assert.equal(fs.existsSync(path.join(f.root,"Runtime/Desktop/storage.json")),false);
  f.setFailProbe(false);await f.call("save-storage",path.join(f.root,"Chosen"));
  const connection={provider:"aliyun",base_url:"https://example.test/v1",model:"vision-model",api_protocol:"chat_completions",quality_mode:"quality"};
  await f.call("save-connection",{connection,api_key:"synthetic-bad"});assert.equal(f.starts.length,0);
  await f.call("save-connection",{connection,api_key:"synthetic-good"});for(let i=0;i<4;i++)await tick();
  assert.equal(f.starts.length,1);assert.equal(f.starts[0][3].data_root,path.join(f.root,"Chosen"));
  assert.equal(JSON.stringify(f.call("get-state")).includes("synthetic-good"),false);
  const log=fs.readFileSync(path.join(f.root,"Runtime/Logs/connection-verification.jsonl"),"utf8");
  assert.deepEqual(log.trim().split("\n").map(line=>JSON.parse(line).stage),["package","request","request"]);
  assert.equal(log.includes("synthetic-good"),false);
  f.setRuns({runs:[{state:"running"}]});await f.menu[0].submenu.find(item=>item.label==="保存位置设置").click();
  assert.equal(f.notices.length,1);assert.equal(f.call("get-state").status,"starting");
  f.setRuns({runs:[]});await f.menu[0].submenu.find(item=>item.label==="保存位置设置").click();
  await f.call("save-storage",path.join(f.root,"New data"));f.call("continue");for(let i=0;i<4;i++)await tick();
  assert.equal(f.starts.length,2);assert.equal(f.starts[1][3].data_root,path.join(f.root,"New data"));
});

test("desktop preserves provider-specific keys without exposing them to its renderer",async t=>{
  const f=await fixture(t);await f.call("save-storage",path.join(f.root,"Chosen"));
  for(const provider of ["aliyun","zhipu"]){
    if(f.starts.length)await f.menu[0].submenu.find(item=>item.label==="AI 服务设置").click();
    await f.call("save-connection",{connection:{provider,base_url:"https://example.test/v1",model:"vision",api_protocol:"chat_completions"},api_key:"synthetic-"+provider});
    for(let i=0;i<4;i++)await tick();
  }
  await f.menu[0].submenu.find(item=>item.label==="AI 服务设置").click();
  const state=f.call("get-state");assert.deepEqual(Object.keys(state.savedProfiles).sort(),["aliyun","zhipu"]);
  await f.call("save-connection",{connection:state.savedProfiles.aliyun.connection,api_key:""});for(let i=0;i<4;i++)await tick();
  assert.equal(f.starts.at(-1)[0],"synthetic-aliyun");
  assert.equal(JSON.stringify(f.call("get-state")).includes("synthetic-aliyun"),false);
});


test("desktop retains the selected original directory across save and launch",async t=>{
  const f=await fixture(t);
  const sourceRoot=path.join(f.root,"Original recordings");
  await f.call("save-storage",{data_root:path.join(f.root,"Chosen"),source_root:sourceRoot});
  const connection={provider:"aliyun",base_url:"https://example.test/v1",model:"vision-model",api_protocol:"chat_completions",quality_mode:"quality"};
  await f.call("save-connection",{connection,api_key:"synthetic-source-selection"});
  for(let i=0;i<4;i++)await tick();
  assert.equal(f.starts[0][3].source_root,sourceRoot);
  assert.equal(JSON.parse(fs.readFileSync(path.join(f.root,"Runtime/Desktop/storage.json"),"utf8")).source_root,sourceRoot);
});
