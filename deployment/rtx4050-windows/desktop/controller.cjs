"use strict";

const { EventEmitter } = require("node:events");
const { spawn } = require("node:child_process");
const path = require("node:path");

const STATE_PREFIX = "VISIONCORTEX_DESKTOP_STATE ";

function serviceUrlAllowed(value, origin) {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" && parsed.hostname === "127.0.0.1" && parsed.origin === origin && !parsed.username && !parsed.password;
  } catch { return false; }
}

function validateApiKey(value) {
  if (typeof value !== "string" || !value.trim() || value.length > 4096 || /[\r\n\0]/u.test(value)) {
    throw new Error("请输入有效的 API 密钥。");
  }
  return value.trim();
}

class BackendController extends EventEmitter {
  constructor({ root, spawnProcess = spawn, writeLog = () => {}, platform = process.platform, stopTimeoutMs = 8000 }) {
    super();
    this.root = root;
    this.spawnProcess = spawnProcess;
    this.writeLog = writeLog;
    this.platform = platform;
    this.child = null;
    this.stopping = false;
    this.origin = null;
    this.stopTimeoutMs = stopTimeoutMs;
  }

  start(key, port, settings, storage) {
    if (this.child) throw new Error("应用正在准备或运行中。");
    key = validateApiKey(key);
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("应用端口无效。");
    if (!settings?.connection || settings.verification?.status !== "verified") throw new Error("请先验证 AI 服务连接。");
    this.stopping = false;
    this.lastState = null;
    this.origin = `http://127.0.0.1:${port}`;
    const executable = path.join(this.root, "python", "python.exe");
    const arguments_ = ["-u", "-B", path.join(this.root, "tools", "rtx4050_portable.py"),
      "--desktop", "--parent-pid", String(process.pid), "--port", String(port), "--no-browser"];
    const child = this.spawnProcess(executable, arguments_, {
      cwd: this.root, windowsHide: true, shell: false, stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, MLLM_API_KEY: key, VISIONCORTEX_DESKTOP_MODE: "1",
        VISIONCORTEX_DESKTOP_CONNECTION: JSON.stringify({connection:settings.connection,verification:settings.verification}),
        VISIONCORTEX_DESKTOP_STORAGE: JSON.stringify(storage||{}),
        PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8", PYTHONDONTWRITEBYTECODE: "1" },
    });
    this.child = child;
    child.stdin.on("error", () => {}); // The child may exit before a stop command reaches its pipe.
    const scrub = (text) => String(text).split(key).join("[已隐藏]");
    const consume = (stream) => {
      let pending = "";
      stream.setEncoding("utf8");
      stream.on("data", (chunk) => {
        pending += chunk;
        let end;
        while ((end = pending.indexOf("\n")) >= 0) {
          const line = scrub(pending.slice(0, end).trim());
          pending = pending.slice(end + 1);
          this.writeLog(line);
          if (!line.startsWith(STATE_PREFIX)) continue;
          try {
            const state = JSON.parse(line.slice(STATE_PREFIX.length));
            if (!state || typeof state.message !== "string" || !["preparing", "ready", "error"].includes(state.status)) continue;
            if (state.status === "ready" && !serviceUrlAllowed(state.url, this.origin)) continue;
            this.lastState = state;
            this.emit("state", state);
          } catch { /* Unstructured model output stays in the diagnostic log. */ }
        }
        if (pending.length > 65536) { this.writeLog(scrub(pending)); pending = ""; }
      });
    };
    consume(child.stdout);
    consume(child.stderr);
    child.on("error", () => this.emit("state", { status: "error", message: "应用运行环境无法启动，请查看诊断记录。" }));
    child.on("close", (code) => {
      this.child = null;
      if (!this.stopping && this.lastState?.status !== "error") this.emit("state", { status: "error", message: `应用后台已退出（${code ?? "异常"}）。请重新启动或查看诊断记录。` });
      this.emit("stopped");
    });
    return child;
  }

  async stop() {
    const child = this.child;
    if (!child) return;
    this.stopping = true;
    await new Promise((resolve) => {
      let timer;
      let fallback;
      const done = () => { clearTimeout(timer); clearTimeout(fallback); child.removeListener("close", done); resolve(); };
      child.once("close", done);
      if (child.stdin.writable) child.stdin.write("stop\n");
      timer = setTimeout(() => {
        if (this.child !== child || child.exitCode !== null) return done();
        // Only the process tree created by this controller may be terminated.
        // The Python Windows Job Object also cleans up if Electron crashes.
        if (this.platform === "win32") {
          fallback = setTimeout(done, this.stopTimeoutMs);
          const killer = this.spawnProcess("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"],
            { windowsHide: true, shell: false, stdio: "ignore" });
          killer.once("close", done);
          killer.once("error", done);
        } else { child.kill("SIGTERM"); done(); }
      }, this.stopTimeoutMs);
    });
  }
}

module.exports = { BackendController, STATE_PREFIX, serviceUrlAllowed, validateApiKey };
