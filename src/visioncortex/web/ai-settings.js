"use strict";
window.VisionCortexAISettings = (() => {
  let pending = false;
  let updateBusy = () => {};
  async function render({main, esc, icon, setChrome, api, state}) {
    setChrome("ai-settings");
    main.innerHTML = '<div class="page ai-settings-page"><p role="status">正在读取 AI 服务设置…</p></div>';
    let info;
    try { info = await api("/api/ai-settings"); }
    catch (error) {
      if (!location.hash.startsWith("#/ai-settings")) return;
      main.innerHTML = `<div class="page"><header class="page-hero compact"><h1>AI 服务设置</h1></header><section class="form-section"><p role="alert">${esc(error.message)}</p><p>请从这台电脑的本机分析工作台打开此页面。</p></section></div>`;
      return;
    }
    if (!location.hash.startsWith("#/ai-settings")) return;
    const providers = {...info.providers};
    for (const profile of [info.active, ...Object.values(info.profiles)]) {
      const c = profile?.connection;
      if (c && !providers[c.provider]) providers[c.provider] = {label:c.provider, ...c, models:[]};
    }
    let discovered = null;
    function modelList() {return discovered || providers[field("ai-provider").value]?.models || []; }
    main.innerHTML = `<div class="page ai-settings-page">
      <header class="page-hero compact"><div><p class="eyebrow">连接你的 AI 服务</p><h1>AI 服务设置</h1><p>选择服务并填写密钥，自动读取模型；验证通过后用于新任务。</p></div></header>
      <div class="ai-settings-layout"><section class="form-section ai-settings-form-section">
        <form id="ai-form" autocomplete="off">
          <div class="ai-field-row"><div><label for="ai-provider">AI 服务厂商</label><select id="ai-provider">${Object.entries(providers).map(([id,p])=>`<option value="${esc(id)}">${esc(p.label)}</option>`).join("")}</select></div>
          <div><label for="ai-quality">使用偏好</label><select id="ai-quality"><option value="quality">质量优先（推荐）</option><option value="balanced">速度与成本优先</option></select></div></div>
          <details class="ai-advanced"><summary>接口地址与协议</summary><label for="ai-endpoint">接口地址（Base URL）</label><input id="ai-endpoint" type="url" required maxlength="2048" spellcheck="false"><label for="ai-protocol">接口类型</label><select id="ai-protocol"><option value="chat_completions">Chat Completions 兼容接口</option><option value="ark_responses">火山引擎 Responses</option></select></details>
          <p class="ai-request-endpoint" id="ai-request-endpoint"></p>
          <label for="ai-key">API 密钥</label><input id="ai-key" type="password" autocomplete="off" maxlength="4096" spellcheck="false" placeholder="输入所选厂商的 API 密钥">
          <div class="ai-discovery-actions"><button class="secondary-button" id="ai-discover" type="button">读取可用模型</button><p class="ai-help">只向上方地址查询模型列表；也可直接填写模型 ID。</p></div>
          <p class="ai-form-result" id="ai-discovery-result" role="status" aria-live="polite"></p>
          <label for="ai-model-choice">图像理解模型</label><select id="ai-model-choice"></select>
          <input id="ai-model" aria-label="自定义视觉模型 ID" placeholder="填写控制台中的视觉模型或部署 ID" maxlength="256" spellcheck="false" hidden>
          <p class="ai-model-reason" id="ai-model-reason"></p><p class="ai-catalog-note" id="ai-catalog-note"></p>
          <a id="ai-model-doc" target="_blank" rel="noopener noreferrer" hidden>查看官方模型说明 ↗</a>
          <p class="ai-help">验证会联网发送两张随机测试图，产生少量 API 费用。通过后保存并启用；密钥不会显示或写入实验报告。</p>
          <button class="primary-button" id="ai-save" type="submit">验证并启用 ${icon("arrow")}</button>
          <p class="ai-form-result" id="ai-result" role="status" aria-live="polite"></p>
        </form>
      </section><aside class="ai-settings-aside"><section class="form-section" id="ai-active"></section>
        <section class="form-section ai-explanation"><h2>接入你的模型服务</h2><p>已收录的服务自动填写接口地址。其他厂商可使用兼容接口；无需等待应用增加厂商名单。</p><p>读取模型后，优先显示厂商声明支持图像的模型；能力未标注的模型会保留并在启用前验证。部分厂商需要手动填写部署 ID。</p><p>密钥无法可靠识别所有厂商。请选择密钥所属服务；没有兼容接口的服务需要专用适配。</p><p>连接验证确认账号权限、图片理解和响应格式；实验步骤识别的准确率仍需要真实视频验收。</p><p>切换厂商或模型后，已有任务继续使用提交时的配置。每家厂商可以分别保存密钥。</p></section>
      </aside></div></div>`;
    const form = document.getElementById("ai-form");
    const field = id => form.querySelector(`#${id}`);
    function showActive() {
      const target = document.getElementById("ai-active");
      if (!target) return;
      const active = info.active, selected = active?.connection || info.legacy, receipt = active?.verification;
      target.innerHTML = `<h2>当前使用</h2><p class="ai-active-provider">${esc(providers[selected.provider]?.label || selected.provider)}</p><p class="ai-active-model">${esc(selected.model)}</p><span class="ai-status ${active?.verified ? "verified" : ""}">${active?.verified ? "真实连接已验证" : "已配置 · 尚未通过本页验证"}</span>${receipt ? `<dl class="ai-receipt"><dt>验证时间</dt><dd>${esc(new Date(receipt.checked_at).toLocaleString("zh-CN", {hour12:false}))}</dd><dt>服务返回模型</dt><dd>${esc(receipt.response_model || "服务未返回")}</dd><dt>请求编号</dt><dd>${esc(receipt.request_id || "服务未返回")}</dd><dt>Token 用量</dt><dd>${esc(receipt.usage?.total_tokens ?? "服务未返回")}</dd></dl>` : '<p class="ai-help">现有服务配置保持可用。可以在左侧重新验证，或选择另一家厂商。</p>'}<a class="secondary-button" href="#/new">新建实验 ${icon("arrow")}</a>`;
    }
    function selection() {return {provider:field("ai-provider").value, model:field("ai-model").value, quality_mode:field("ai-quality").value, base_url:field("ai-endpoint").value, api_protocol:field("ai-protocol").value};}
    function hints() {
      const current = selection(), provider = providers[current.provider];
      const model = modelList().find(item => item.id === current.model);
      field("ai-model-reason").textContent = model?.reason || (model?.vision_declared === true ? "厂商声明支持图像输入；启用前仍会验证多图理解和 JSON 响应。" : "模型需支持多图输入和 JSON 响应，启用前会实际调用验证。");
      const stale = Date.now() - Date.parse(provider.catalog_checked_at) > 30 * 86400000;
      field("ai-catalog-note").textContent = discovered ? "来源：当前接口返回的模型列表。列表与模型名称不代表真实实验质量。" : !provider.models?.length ? "先读取模型列表，或填写控制台中的视觉模型 / 部署 ID。" : `推荐依据：官方能力与任务适配；核验于 ${provider.catalog_checked_at}。实验质量排名待实测。${stale ? "请核对控制台最新模型。" : ""}`;
      const doc = field("ai-model-doc");const source = model?.source || provider.source;doc.hidden = !source?.startsWith("https://");
      if (source?.startsWith("https://")) doc.href = source; else doc.removeAttribute("href");
      field("ai-request-endpoint").textContent = current.base_url ? `请求发送到：${current.base_url}` : "请填写接口地址。";
      const saved = info.profiles[current.provider];
      const reusable = saved?.has_saved_key && saved.connection.base_url === current.base_url || info.legacy.has_key && info.legacy.provider === current.provider && info.legacy.base_url === current.base_url;
      field("ai-key").placeholder = reusable ? "密钥已保存，留空可复用" : "输入所选厂商的 API 密钥";
    }
    function models(selected) {
      const list = modelList();
      field("ai-model-choice").innerHTML = list.map(item=>`<option value="${esc(item.id)}">${esc(item.label)}${discovered ? item.vision_declared === true ? " · 支持图像" : " · 能力待验证" : ""}</option>`).join("") + '<option value="__custom__">自定义视觉模型 / 部署 ID</option>';
      const known = list.some(item=>item.id === selected);
      field("ai-model-choice").value = known ? selected : "__custom__";
      field("ai-model").value = selected || "";field("ai-model").hidden = known;
      hints();
    }
    function fill(connection) {
      discovered = null;field("ai-discovery-result").textContent = "";
      field("ai-provider").value = connection.provider;field("ai-quality").value = connection.quality_mode || "quality";
      field("ai-endpoint").value = connection.base_url || "";field("ai-protocol").value = connection.api_protocol || "chat_completions";
      models(connection.model);field("ai-key").value = "";
      form.querySelector("details").open = connection.provider === "custom";
    }
    function busy(value, discovering=false) {
      if (!form.isConnected) return;
      for (const input of form.elements) input.disabled = value;
      field("ai-save").textContent = value && !discovering ? "正在验证真实调用…" : "验证并启用";
      field("ai-discover").textContent = value && discovering ? "正在读取模型…" : "读取可用模型";
    }
    updateBusy = busy;
    field("ai-discover").addEventListener("click",async()=>{
      if (pending) return;
      if (!field("ai-endpoint").checkValidity() || !field("ai-endpoint").value.trim()) {
        form.querySelector("details").open = true;field("ai-endpoint").reportValidity();return;
      }
      pending=true;busy(true,true);field("ai-discovery-result").textContent="正在向所选接口读取模型列表…";
      try {
        const result=await api("/api/ai-settings/models",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({connection:selection(),api_key:field("ai-key").value})});
        if (!form.isConnected) return;
        discovered=result.models || [];
        const selected=discovered.some(m=>m.id===field("ai-model").value) ? field("ai-model").value : discovered[0]?.id || field("ai-model").value;
        models(selected);
        field("ai-discovery-result").textContent=discovered.length ? `已读取 ${discovered.length} 个模型。${result.excluded_non_image ? `已排除 ${result.excluded_non_image} 个明确不支持图像的模型。` : ""}${result.truncated ? "列表未完整返回，可手动填写未列出的模型。" : ""}选择后点击“验证并启用”。` : "没有返回可用图像模型。可手动填写视觉模型或部署 ID 后验证。";
        field("ai-discovery-result").dataset.tone="success";
      } catch(error) {if(form.isConnected) {field("ai-discovery-result").textContent=error.message;field("ai-discovery-result").dataset.tone="error";}}
      finally {pending=false;updateBusy(false);}
    });
    field("ai-provider").addEventListener("change",()=>{
      const id=field("ai-provider").value, provider=providers[id], quality_mode=field("ai-quality").value;
      fill(info.profiles[id]?.connection || {provider:id, ...provider, quality_mode, model:provider.recommendations?.[quality_mode] || provider.model});
      field("ai-result").textContent = "";
    });
    field("ai-quality").addEventListener("change",()=>{const p=providers[field("ai-provider").value];models(p.recommendations?.[field("ai-quality").value] || field("ai-model").value);});
    field("ai-model-choice").addEventListener("change",()=>{const id=field("ai-model-choice").value;field("ai-model").hidden=id!=="__custom__";field("ai-model").value=id==="__custom__"?"":id;hints();});
    field("ai-model").addEventListener("input",hints);
    function resetDiscovery() {discovered=null;field("ai-discovery-result").textContent="";models(field("ai-model").value);}
    field("ai-endpoint").addEventListener("input",resetDiscovery);
    field("ai-protocol").addEventListener("change",resetDiscovery);
    field("ai-key").addEventListener("input",resetDiscovery);
    form.addEventListener("submit",async event=>{
      event.preventDefault();if (pending) return;
      if (!field("ai-model").value.trim()) {field("ai-result").textContent="请填写图像理解模型。";return;}
      pending=true;busy(true);field("ai-result").textContent="正在实际调用所选模型，请稍候。通常需要几秒到几分钟。";
      const body=JSON.stringify({connection:selection(),api_key:field("ai-key").value});field("ai-key").value="";
      try {
        const result=await api("/api/ai-settings/verify",{method:"POST",headers:{"Content-Type":"application/json"},body});
        if (result.activated) {
          info=await api("/api/ai-settings");state.health=await api("/api/health");
          if (form.isConnected) {showActive();hints();field("ai-result").textContent="验证通过，已启用。新提交的分析任务将使用当前厂商和模型。";field("ai-result").dataset.tone="success";}
        } else if (form.isConnected) {
          field("ai-result").textContent=[result.verification?.message,result.verification?.detail,"配置未启用，已有设置保持不变。"].filter(Boolean).join("\n");field("ai-result").dataset.tone="error";
        }
      } catch(error) {if (form.isConnected) {field("ai-result").textContent=error.message;field("ai-result").dataset.tone="error";}}
      finally {pending=false;updateBusy(false);}
    });
    const defaultProvider = providers[info.legacy.provider] ? info.legacy.provider : Object.keys(providers)[0];
    fill(info.active?.connection || {provider:defaultProvider, ...providers[defaultProvider], quality_mode:"quality"});
    showActive();busy(pending);
  }
  return {render};
})();
