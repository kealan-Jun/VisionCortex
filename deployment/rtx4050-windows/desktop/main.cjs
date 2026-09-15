"use strict";

const {app,BrowserWindow,ipcMain,safeStorage,shell,Menu,dialog}=require("electron");
const fs=require("node:fs"),path=require("node:path"),net=require("node:net");
const {createHash}=require("node:crypto");
const {spawn}=require("node:child_process");
const {pathToFileURL}=require("node:url");
const {BackendController,serviceUrlAllowed,validateApiKey}=require("./controller.cjs");
const {normalizeConnection,sameConnection,verifyProvider}=require("./connection.cjs");
const {storageRequest,settingsChangeAllowed}=require("./storage.cjs");

const root=path.resolve(__dirname,"../..");
const setupFile=path.join(__dirname,"setup.html"),setupUrl=pathToFileURL(setupFile).href;
const desktopRoot=path.join(root,"Runtime/Desktop"),logRoot=path.join(root,"Runtime/Logs");
const providers=JSON.parse(fs.readFileSync(path.join(root,"configs/mllm-providers.json"),"utf8"));
fs.mkdirSync(desktopRoot,{recursive:true});fs.mkdirSync(logRoot,{recursive:true});
app.setName("VisionCortex");
// Reserve the small GPU for video models; the desktop interface uses system RAM.
app.disableHardwareAcceleration();
app.setPath("userData",desktopRoot);app.setPath("sessionData",path.join(desktopRoot,"Session"));
let window,settings=null,verifier=null,exiting=false,launching=false,storage=null,storageBusy=false;
let profiles={},storageCandidates=[],storageMessage="",storageSuggestion=path.join(root,"Runtime");
let state={status:"preparing",message:"正在打开 VisionCortex…"};
const backend=new BackendController({root,writeLog:line=>fs.appendFileSync(path.join(logRoot,"desktop.log"),`${line}\n`,"utf8")});

function publicState(){
  return {...state,providers,connection:state.connection||settings?.connection||null,
    savedConnection:settings?.connection||null,hasSavedKey:!!settings?.api_key,verification:settings?.verification||null,
    savedProfiles:Object.fromEntries(Object.entries(profiles).map(([id,value])=>[id,{connection:value.connection,hasSavedKey:!!value.api_key}])),
    storage,storageCandidates,storageMessage,storageSuggestion,canContinue:!!storage&&verifiedSettings(settings)};
}
function update(next){state=next;if(window&&!window.isDestroyed()) window.webContents.send("desktop:state",publicState());}
function assertSetup(event){
  if(!window||event.sender!==window.webContents||event.senderFrame!==window.webContents.mainFrame||event.senderFrame.url!==setupUrl) throw new Error("设置请求来源无效。");
}
function secretFile(){return path.join(app.getPath("appData"),"VisionCortex/Secrets/ai-connection.encrypted");}
function adapterIdentity(){
  const hash=createHash("sha256");
  for(const name of ["mllm.py","mllm_provider.py","provider_connection.py"]) hash.update(fs.readFileSync(path.join(root,"src/visioncortex",name)));
  return hash.digest("hex");
}
function readSettings(){
  if(!safeStorage.isEncryptionAvailable()) throw new Error("Windows 密钥保护不可用。");
  const file=secretFile();if(!fs.existsSync(file)) return null;
  const value=JSON.parse(safeStorage.decryptString(fs.readFileSync(file)));
  if(value.schemaVersion===2) profiles={[value.connection.provider]:value};
  else if(value.schemaVersion===3) profiles=value.profiles||{};
  else return null;
  for(const profile of Object.values(profiles)){
    profile.connection=normalizeConnection(profile.connection,providers);profile.api_key=validateApiKey(profile.api_key);
  }
  return profiles[value.activeProvider||value.connection?.provider]||null;
}
function verifiedSettings(value){
  return value?.verification?.status==="verified"&&value.verification.model_invocation==="PROVEN"&&
    sameConnection(value.connection,value.verification.connection)&&value.verification.adapter_sha256===adapterIdentity();
}
function writeSettings(value){
  if(!safeStorage.isEncryptionAvailable()) throw new Error("Windows 密钥保护不可用，配置未保存。");
  const file=secretFile();fs.mkdirSync(path.dirname(file),{recursive:true});
  const nextProfiles={...profiles,[value.connection.provider]:value};
  fs.writeFileSync(`${file}.partial`,safeStorage.encryptString(JSON.stringify({schemaVersion:3,activeProvider:value.connection.provider,profiles:nextProfiles})),{mode:0o600});
  fs.renameSync(`${file}.partial`,file);
  profiles=nextProfiles;
}
function runHidden(executable,args){return new Promise((resolve,reject)=>{
  const child=spawn(executable,args,{cwd:root,windowsHide:true,shell:false,stdio:"ignore"});
  child.once("error",reject);child.once("close",resolve);
});}
async function ensureRuntime(){
  const code=await runHidden(path.join(root,"python/python.exe"),["-I","-c","import ctypes; ctypes.WinDLL('msvcp140.dll'); ctypes.WinDLL('vcruntime140_1.dll')"]);
  if(code===0) return;
  update({status:"preparing",message:"正在准备 Windows 运行组件，请完成系统弹出的安装提示。"});
  const powershell=path.join(process.env.SystemRoot||"C:\\Windows","System32/WindowsPowerShell/v1.0/powershell.exe");
  const installed=await runHidden(powershell,["-NoLogo","-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-File",path.join(__dirname,"ensure-vc-runtime.ps1"),"-PackageRoot",root]);
  if(installed===3010) throw new Error("Windows 组件已准备完成，请重启电脑后再次打开应用。");
  if(![0,1638].includes(installed)) throw new Error("Windows 组件安装未完成，请重试并允许微软安装器运行。");
}
function freePort(){return new Promise((resolve,reject)=>{
  const server=net.createServer();server.once("error",reject);
  server.listen(0,"127.0.0.1",()=>{const port=server.address().port;server.close(()=>resolve(port));});
});}
async function start(value){
  if(backend.child||launching||exiting) return;
  if(!storage){update({status:"needs-storage",message:"请先选择实验产出的保存位置。"});return;}
  if(!verifiedSettings(value)){update({status:"needs-key",message:"服务配置尚未验证，或应用的调用方式已更新，请先验证连接。"});return;}
  launching=true;update({status:"starting",message:"正在准备应用，首次使用可能需要几分钟。"});
  try{
    await ensureRuntime();
    try{storage=await storageRequest(root,{action:"probe",data_root:storage.data_root,source_root:storage.source_root||""});}
    catch(error){update({status:"needs-storage",message:"上次保存位置暂不可用，请恢复连接或选择其他位置。",formError:error.message});return;}
    const port=await freePort();if(!exiting) backend.start(value.api_key,port,value,storage);
  }
  catch(error){update({status:"error",message:error.message});}
  finally{launching=false;}
}
backend.on("state",async next=>{
  if(exiting||!window||window.isDestroyed()) return;
  update(next);
  try{
    if(next.status==="ready") await window.loadURL(next.url);
    else if(next.status==="error"&&window.webContents.getURL()!==setupUrl) await window.loadFile(setupFile);
  }catch{if(!exiting){update({status:"error",message:"应用界面暂时无法打开，请重试。"});await window.loadFile(setupFile);}}
});
ipcMain.handle("desktop:get-state",event=>{assertSetup(event);return publicState();});
ipcMain.handle("desktop:scan-storage",async event=>{
  assertSetup(event);if(launching||storageBusy||backend.child)throw new Error("请稍候再检测。");
  storageBusy=true;
  try{
    const detected=await storageRequest(root,{action:"discover"});
    storageCandidates=detected.candidates;storageMessage=detected.message;
    if(!storage) storageSuggestion=storageCandidates[0]?.data_root||path.join(root,"Runtime");
    return publicState();
  }finally{storageBusy=false;}
});
ipcMain.handle("desktop:choose-storage",async event=>{
  assertSetup(event);if(launching||storageBusy||backend.child)throw new Error("请稍候再选择。");
  const result=await dialog.showOpenDialog(window,{title:"选择 VisionCortex 数据保存目录",defaultPath:storage?.data_root||storageSuggestion,properties:["openDirectory","createDirectory"]});
  return result.canceled?null:result.filePaths[0];
});
ipcMain.handle("desktop:save-storage",async(event,value)=>{
  assertSetup(event);if(launching||storageBusy||backend.child)throw new Error("请稍候再保存。");
  storageBusy=true;
  try{
    const checked=await storageRequest(root,{action:"probe",data_root:typeof value==="string"?value:value?.data_root,source_root:typeof value==="string"?"":value?.source_root||""});
    const file=path.join(desktopRoot,"storage.json");
    fs.writeFileSync(`${file}.partial`,JSON.stringify(checked,null,2),"utf8");fs.renameSync(`${file}.partial`,file);
    storage=checked;
    update({status:"needs-key",message:verifiedSettings(settings)?"保存位置已更新，可使用已验证的 AI 配置打开应用。":"保存位置已就绪，请设置并验证 AI 服务。"});
    return publicState();
  }finally{storageBusy=false;}
});
ipcMain.handle("desktop:choose-source",async event=>{
  assertSetup(event);if(launching||storageBusy||backend.child)throw new Error("请稍候再选择。");
  const result=await dialog.showOpenDialog(window,{title:"选择已有原视频目录",defaultPath:storage?.source_root||undefined,properties:["openDirectory"]});
  return result.canceled?null:result.filePaths[0];
});
ipcMain.handle("desktop:continue",event=>{assertSetup(event);if(verifiedSettings(settings)&&storage)void start(settings);});
ipcMain.handle("desktop:save-connection",async(event,value)=>{
  assertSetup(event);
  if(backend.child||launching||verifier) throw new Error("应用正在准备或运行，请稍后重试。");
  const connection=normalizeConnection(value?.connection,providers);
  let key=value?.api_key;
  // A saved secret can only be reused at the same provider and exact endpoint.
  const saved=profiles[connection.provider];
  if(!key&&saved?.connection.base_url===connection.base_url) key=saved.api_key;
  key=validateApiKey(key);launching=true;
  const diagnostic=path.join(logRoot,"connection-verification.jsonl");
  const logVerification=value=>{
    try{fs.appendFileSync(diagnostic,JSON.stringify({at:new Date().toISOString(),...value})+"\n","utf8");}
    catch{/* Keep the on-screen result available if the diagnostic disk is full. */}
  };
  update({status:"verifying",verificationStage:"starting",connection,message:"正在准备本地验证程序，尚未调用 AI 服务…"});
  try{
    await ensureRuntime();if(exiting) return;
    update({status:"verifying",verificationStage:"starting",connection,message:"正在启动本地文件校验，完成后再连接 AI 服务…"});
    fs.writeFileSync(diagnostic,"","utf8");
    verifier=verifyProvider(root,connection,key,{onProgress:progress=>{
      logVerification(progress);
      if(!exiting)update({status:"verifying",verificationStage:progress.stage,connection,message:progress.message});
    }});
    const receipt=await verifier.promise;
    logVerification({status:receipt.status,stage:receipt.verification_stage||state.verificationStage,
      phase_seconds:receipt.phase_seconds,verification_elapsed_seconds:receipt.verification_elapsed_seconds});
    if(exiting) return;
    if(receipt.status!=="verified"){
      update({status:"needs-key",connection,message:"连接尚未验证，请修正配置后重试。",formError:receipt.message,detail:receipt.detail||""});
      return {verified:false};
    }
    const next={schemaVersion:2,connection:receipt.connection,api_key:key,verification:receipt};
    writeSettings(next);settings=next;
    fs.writeFileSync(path.join(desktopRoot,"connection-verification.json"),JSON.stringify(receipt,null,2),"utf8");
    return {verified:true};
  }catch(error){
    const message=String(error.message).split(key).join("[REDACTED]");
    logVerification({status:"failed",stage:state.verificationStage||"starting",message});
    if(!exiting)update({status:"needs-key",connection,message:"连接或配置保存未完成。",formError:message});
    return {verified:false};
  }
  finally{verifier=null;launching=false;if(!exiting&&state.status==="verifying"&&verifiedSettings(settings)) void start(settings);}
});
ipcMain.handle("desktop:retry",async event=>{
  assertSetup(event);if(launching||verifier||state.status!=="error") return;
  await backend.stop();if(settings) void start(settings);else update({status:"needs-key",message:"请先设置并验证 AI 服务。"});
});
ipcMain.handle("desktop:open-logs",event=>{assertSetup(event);return shell.openPath(logRoot);});
async function openSettings(section){
  if(launching||storageBusy||exiting) return;
  if(backend.child){
    try{
      const read=async endpoint=>{const response=await fetch(backend.origin+endpoint,{signal:AbortSignal.timeout(4000)});if(!response.ok)throw new Error();return response.json();};
      const [runs,health]=await Promise.all([read("/api/runs"),read("/api/health")]);
      if(!settingsChangeAllowed(runs,health)){
        await dialog.showMessageBox(window,{type:"info",message:"请等分析任务完成，并完成或取消尚未完成的视频上传后，再修改设置。"});return;
      }
    }catch{await dialog.showMessageBox(window,{type:"info",message:"暂时无法确认任务状态，请稍后再修改设置。"});return;}
  }
  update({status:"preparing",message:"正在打开设置…"});await backend.stop();
  update({status:section==="storage"?"needs-storage":"needs-key",message:section==="storage"?"选择产出保存位置；已有文件保留原处，切回原位置即可继续查看。":"选择服务、接口地址和图像模型，验证后启用。"});
  await window.loadFile(setupFile);
}
ipcMain.handle("desktop:open-model-doc",(event,provider,model)=>{
  assertSetup(event);const entry=providers[provider]?.models?.find(item=>item.id===model);
  if(!entry?.source||new URL(entry.source).protocol!=="https:") throw new Error("未找到官方模型说明。");
  return shell.openExternal(entry.source);
});
if(!app.requestSingleInstanceLock()) app.quit();
else{
  app.on("second-instance",()=>{if(window){if(window.isMinimized()) window.restore();window.show();window.focus();}});
  app.whenReady().then(async()=>{
    window=new BrowserWindow({width:1320,height:960,minWidth:960,minHeight:720,title:"VisionCortex",backgroundColor:"#f3f7f6",show:false,
      webPreferences:{preload:path.join(__dirname,"preload.cjs"),contextIsolation:true,nodeIntegration:false,sandbox:true,webSecurity:true}});
    window.once("ready-to-show",()=>window.show());
    window.webContents.session.setPermissionRequestHandler((_contents,_permission,callback)=>callback(false));
    window.webContents.on("will-navigate",(event,url)=>{if(url!==setupUrl&&!serviceUrlAllowed(url,backend.origin)) event.preventDefault();});
    window.webContents.setWindowOpenHandler(({url})=>serviceUrlAllowed(url,backend.origin)?{action:"allow",overrideBrowserWindowOptions:{autoHideMenuBar:true,webPreferences:{nodeIntegration:false,contextIsolation:true,sandbox:true,webSecurity:true}}}:{action:"deny"});
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      {label:"应用",submenu:[{label:"AI 服务设置",click:()=>openSettings("ai")},
        {label:"保存位置设置",click:()=>openSettings("storage")},
        {label:"打开产出文件夹",click:()=>{if(storage)shell.openPath(storage.archive_root);}},
        {label:"打开诊断目录",click:()=>shell.openPath(logRoot)},{type:"separator"},{role:"quit",label:"退出"}]},
      {label:"编辑",submenu:[{role:"undo",label:"撤销"},{role:"redo",label:"重做"},{type:"separator"},{role:"cut",label:"剪切"},{role:"copy",label:"复制"},{role:"paste",label:"粘贴"},{role:"selectAll",label:"全选"}]},
      {label:"显示",submenu:[{role:"resetZoom",label:"实际大小"},{role:"zoomIn",label:"放大"},{role:"zoomOut",label:"缩小"}]},
    ]));
    window.on("close",event=>{if(!exiting){event.preventDefault();app.quit();}});
    await window.loadFile(setupFile);
    try{
      await ensureRuntime();settings=readSettings();
      const file=path.join(desktopRoot,"storage.json");
      if(fs.existsSync(file))storage=JSON.parse(fs.readFileSync(file,"utf8"));
      if(!storage){
        const detected=await storageRequest(root,{action:"discover"});storageCandidates=detected.candidates;storageMessage=detected.message;
        storageSuggestion=storageCandidates[0]?.data_root||path.join(root,"Runtime");
        update({status:"needs-storage",message:"先选择实验产出的保存位置。已连接的网络盘优先推荐，也可以选择本地目录。"});
      }else if(verifiedSettings(settings)) void start(settings);
      else update({status:"needs-key",message:"选择你使用的 AI 服务，完成真实连接验证后开始使用。"});
    }
    catch{update({status:"needs-key",message:"请在这台电脑重新设置并验证 AI 服务。"});}
  }).catch(()=>{dialog.showErrorBox("VisionCortex 无法打开","请完整解压压缩包到可写的本地目录后重新打开。");app.quit();});
  app.on("before-quit",event=>{
    if(exiting) return;event.preventDefault();exiting=true;verifier?.cancel();
    update({status:"preparing",message:"正在关闭应用…"});backend.stop().finally(()=>app.exit(0));
  });
}
