"use strict";
const element=id=>document.getElementById(id);
const bridge=window.visioncortexDesktop;
let providers={},initialized=false,savedConnection=null,hasSavedKey=false,lastAppliedConnection="";
let savedProfiles={},storageInitialized=false;
function readConnection(){return {provider:element("provider").value,model:element("model").value,base_url:element("base-url").value,api_protocol:element("protocol").value,quality_mode:element("quality-mode").value};}
function explainModel(){
  const provider=providers[element("provider").value]||{};
  const model=(provider.models||[]).find(item=>item.id===element("model").value);
  element("model-reason").textContent=model?.reason||"自定义模型需要支持多图输入与 JSON 响应，启用前会实际调用验证。";
  element("model-doc").hidden=!model?.source;
  const date=provider.catalog_checked_at;
  const stale=date&&(Date.now()-Date.parse(date)>30*86400000);
  element("catalog-note").textContent=date?`推荐依据：官方能力文档与任务适配；核验于 ${date}。实验质量排名待实测。${stale?"目录已有一段时间未更新，请核对控制台最新模型。":""}`:"";
}
function modelChoices(selected){
  const provider=providers[element("provider").value]||{};
  element("model-choice").replaceChildren();
  for(const item of provider.models||[]){const option=document.createElement("option");option.value=item.id;option.textContent=item.label;element("model-choice").append(option);}
  const custom=document.createElement("option");custom.value="__custom__";custom.textContent="自定义视觉模型 / 部署 ID";element("model-choice").append(custom);
  const known=(provider.models||[]).some(item=>item.id===selected);
  element("model-choice").value=known?selected:"__custom__";
  element("model").value=selected||"";element("model").hidden=known;
  explainModel();
}
function endpointHint(){
  element("endpoint").textContent=element("base-url").value?`请求发送到：${element("base-url").value}`:"请在接口设置中填写服务地址。";
  const saved=savedProfiles[element("provider").value]||{connection:savedConnection,hasSavedKey};
  const reusable=saved.hasSavedKey&&saved.connection?.provider===element("provider").value&&saved.connection?.base_url===element("base-url").value;
  element("api-key").placeholder=reusable?"密钥已加密保存，留空复用":"输入所选厂商的 API 密钥";
}
function fillConnection(value){
  for(const [field,id] of [["provider","provider"],["model","model"],["base_url","base-url"],["api_protocol","protocol"]]) element(id).value=value[field]||"";
  element("quality-mode").value=value.quality_mode||"quality";
  modelChoices(value.model);
  endpointHint();
}
function showState(state){
  savedProfiles=state.savedProfiles||{};
  const choosingStorage=state.status==="needs-storage";
  element("storage-form").hidden=!choosingStorage;
  if(choosingStorage&&!storageInitialized){element("storage-path").value=state.storage?.data_root||state.storageSuggestion||"";element("source-path").value=state.storage?.source_root||"";storageInitialized=true;}
  element("storage-candidates").replaceChildren(new Option("手动选择目录",""));
  for(const item of state.storageCandidates||[])element("storage-candidates").append(new Option(item.label,item.data_root));
  element("storage-scan-note").textContent=state.storageMessage||"自动检测当前用户已连接的网络盘，不遍历文件内容。未映射的 NAS 可通过文件夹选择器连接。";
  element("storage-current").hidden=!state.storage||choosingStorage;
  element("storage-current").textContent=state.storage?`产出文件夹：${state.storage.archive_root}`:"";
  element("continue").hidden=!state.canContinue||state.status!=="needs-key";
  if(state.providers&&!initialized){
    providers=state.providers;
    for(const [id,provider] of Object.entries(providers)){
      const option=document.createElement("option");option.value=id;option.textContent=provider.label;element("provider").append(option);
    }
    fillConnection(state.connection||{provider:Object.keys(providers)[0],...Object.values(providers)[0]});initialized=true;
  }
  if(state.connection&&JSON.stringify(state.connection)!==lastAppliedConnection){fillConnection(state.connection);lastAppliedConnection=JSON.stringify(state.connection);}
  savedConnection=state.savedConnection||null;
  hasSavedKey=!!state.hasSavedKey;
  const busy=state.status==="verifying";
  element("title").textContent=choosingStorage?"选择保存位置":state.status==="needs-key"?"连接你的 AI 服务":state.status==="error"?"需要处理一个问题":busy?(["starting","package","loading"].includes(state.verificationStage)?"正在检查本地应用":"正在验证 AI 服务"):"正在准备应用";
  element("message").textContent=state.message||"正在准备…";
  element("key-form").hidden=!["needs-key","verifying"].includes(state.status);
  for(const field of element("key-form").elements) field.disabled=busy;
  element("error-actions").hidden=state.status!=="error";
  element("progress").hidden=["error","needs-key","verifying","needs-storage"].includes(state.status);
  element("form-error").textContent=[state.formError,state.detail].filter(Boolean).join("\n");
  const receipt=state.verification;
  element("verification-receipt").hidden=!receipt;
  if(receipt){
    const provider=providers[receipt.connection?.provider]?.label||receipt.connection?.provider;
    const usage=receipt.usage?.total_tokens;
    element("verification-detail").textContent=[`${provider} · ${receipt.connection?.model}`,`服务返回模型：${receipt.response_model||"服务未返回"}`,`验证时间：${receipt.checked_at}`,`请求编号：${receipt.request_id||"服务未返回"}`,`Token 用量：${usage??"服务未返回"}`,"已验证多图理解及 JSON 响应。真实实验质量仍需任务验收。"].join("\n");
  }
  endpointHint();
}
element("provider").addEventListener("change",()=>{
  const provider=element("provider").value,quality_mode=element("quality-mode").value;
  fillConnection(savedProfiles[provider]?.connection||{provider,...providers[provider],quality_mode,model:providers[provider].recommendations?.[quality_mode]||providers[provider].model});element("api-key").value="";
  if(provider==="custom") document.querySelector(".advanced").open=true;
});
element("quality-mode").addEventListener("change",()=>{const provider=providers[element("provider").value];modelChoices(provider?.recommendations?.[element("quality-mode").value]||element("model").value);});
element("model-choice").addEventListener("change",()=>{
  const selected=element("model-choice").value;
  element("model").hidden=selected!=="__custom__";element("model").value=selected==="__custom__"?"":selected;explainModel();
});
element("model").addEventListener("input",explainModel);
element("model-doc").addEventListener("click",()=>bridge.openModelDoc(element("provider").value,element("model").value));
element("base-url").addEventListener("input",endpointHint);
element("key-form").addEventListener("submit",async event=>{
  event.preventDefault();element("start").disabled=true;
  try{
    const value={connection:readConnection(),api_key:element("api-key").value};element("api-key").value="";
    await bridge.saveConnection(value);
  }catch(error){element("form-error").textContent=String(error.message||"请检查配置后重试。").replace(/^Error invoking remote method '[^']+': Error: /u,"");}
  finally{element("start").disabled=false;}
});
element("retry").addEventListener("click",async()=>{try{await bridge.retry();}catch{showState({status:"needs-key",message:"请重新设置并验证 AI 服务。"});}});
element("logs").addEventListener("click",()=>bridge.openLogs());
function displayError(error){element("form-error").textContent=String(error.message||"操作未完成，请重试。").replace(/^Error invoking remote method '[^']+': Error: /u,"");}
element("storage-candidates").addEventListener("change",()=>{if(element("storage-candidates").value)element("storage-path").value=element("storage-candidates").value;});
element("choose-storage").addEventListener("click",async()=>{try{const selected=await bridge.chooseStorage();if(selected)element("storage-path").value=selected;}catch(error){displayError(error);}});
element("choose-source").addEventListener("click",async()=>{try{const selected=await bridge.chooseSource();if(selected)element("source-path").value=selected;}catch(error){displayError(error);}});
element("scan-storage").addEventListener("click",async()=>{
  element("scan-storage").disabled=true;
  try{showState(await bridge.scanStorage());}catch(error){displayError(error);}finally{element("scan-storage").disabled=false;}
});
element("storage-form").addEventListener("submit",async event=>{
  event.preventDefault();element("form-error").textContent="";
  for(const field of element("storage-form").elements)field.disabled=true;
  try{await bridge.saveStorage({data_root:element("storage-path").value,source_root:element("source-path").value.trim()});}catch(error){displayError(error);}
  finally{for(const field of element("storage-form").elements)field.disabled=false;}
});
element("continue").addEventListener("click",async()=>{try{await bridge.continue();}catch(error){displayError(error);}});
if(bridge){bridge.onState(showState);bridge.getState().then(showState);}
