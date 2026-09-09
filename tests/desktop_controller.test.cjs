"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { EventEmitter } = require("node:events");
const { PassThrough } = require("node:stream");
const { BackendController, STATE_PREFIX, serviceUrlAllowed } = require("../deployment/rtx4050-windows/desktop/controller.cjs");

const settings={connection:{provider:"aliyun",base_url:"https://example.test/v1",model:"vision-model",api_protocol:"chat_completions",quality_mode:"quality"},verification:{status:"verified"}};

function fixture(options = {}) {
  const calls = [], logs = [], states = [];
  let now = 0, nextTimer = 0;
  const timers = new Map();
  const watchdogTimers = {
    setTimeout(callback, delay) { const id = ++nextTimer; timers.set(id, { callback, at: now + delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
  };
  const advance = (delay) => {
    const target = now + delay;
    while (true) {
      const due = [...timers].filter(([, timer]) => timer.at <= target).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      timers.delete(due[0]); now = due[1].at; due[1].callback();
    }
    now = target;
  };
  const child = new EventEmitter();
  Object.assign(child, { pid: 321, exitCode: null, stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough() });
  const controller = new BackendController({ root: "/package with spaces", platform: "win32", stopTimeoutMs: 15,
    watchdogTimers, ...options, writeLog: (line) => logs.push(line), spawnProcess: (...args) => {
      calls.push(args);
      return calls.length === 1 ? child : new EventEmitter();
    } });
  controller.on("state", (state) => states.push(state));
  const state = value => child.stdout.write(STATE_PREFIX + JSON.stringify(value) + "\n");
  return { controller, child, calls, logs, states, advance, state, timers };
}

test("starts only its bundled Python, hides console, and passes the secret only in environment", () => {
  const { controller, calls } = fixture();
  controller.start("synthetic-test-key", 8123, settings);
  const [executable, args, options] = calls[0];
  assert.equal(executable, "/package with spaces/python/python.exe");
  assert.equal(args.includes("synthetic-test-key"), false);
  assert.equal(options.env.MLLM_API_KEY, "synthetic-test-key");
  assert.equal(options.windowsHide, true);
  assert.equal(options.shell, false);
  assert.equal(args.includes("--desktop"), true);
  assert.throws(() => controller.start("test", 8123, settings));
  assert.equal(calls.length, 1);
});

test("handles split UTF8 lines, redacts secrets, and rejects a readiness URL outside its own service", () => {
  const { controller, child, logs, states } = fixture();
  controller.start("synthetic-test-key", 8123, settings);
  const message = STATE_PREFIX + JSON.stringify({ status: "preparing", message: "模型 synthetic-test-key" }) + "\n";
  const bytes = Buffer.from(message);
  child.stdout.write(bytes.subarray(0, 63));
  child.stdout.write(bytes.subarray(63));
  for (const url of ["https://example.com/", "http://127.0.0.1:9000/", "http://user:pass@127.0.0.1:8123/"]) {
    child.stdout.write(STATE_PREFIX + JSON.stringify({ status: "ready", message: "ready", url }) + "\n");
  }
  child.stdout.write(STATE_PREFIX + "invalid\n");
  child.stdout.write(STATE_PREFIX + JSON.stringify({ status: "ready", message: "ready", url: "http://127.0.0.1:8123/#/home" }) + "\n");
  assert.deepEqual(states.map((state) => state.status), ["preparing", "ready"]);
  assert.equal(states[0].message, "模型 [已隐藏]");
  assert.equal(logs.join("").includes("synthetic-test-key"), false);
  assert.equal(serviceUrlAllowed("file:///etc/passwd", controller.origin), false);
});

test("retains the actionable preparation failure when the process then exits", () => {
  const { controller, child, states } = fixture();
  controller.start("test", 8123, settings);
  child.stderr.write(STATE_PREFIX + JSON.stringify({ status: "error", message: "文件校验失败" }) + "\n");
  child.emit("close", 1);
  assert.equal(states.length, 1);
  assert.equal(states[0].message, "文件校验失败");
  assert.equal(controller.child, null);
});

test("normal shutdown asks its child to stop and never kills another process", async () => {
  const { controller, child, calls, states } = fixture();
  controller.start("test", 8123, settings);
  child.stdin.on("data", (data) => { assert.equal(data.toString(), "stop\n"); setImmediate(() => child.emit("close", 0)); });
  await controller.stop();
  assert.equal(calls.length, 1);
  assert.equal(states.length, 0);
  await controller.stop();
});

test("unresponsive child gets a bounded shutdown targeting only the owned PID tree", async () => {
  const { controller, calls } = fixture();
  controller.start("test", 8123, settings);
  await controller.stop();
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[1][1], ["/PID", "321", "/T", "/F"]);
  assert.equal(calls[1][2].windowsHide, true);
});

test("invalid setup values cannot launch a process", () => {
  const { controller, calls } = fixture();
  for (const key of ["", " \n", "abc\nxyz", null]) assert.throws(() => controller.start(key, 8123, settings));
  assert.throws(() => controller.start("test", 0));
  assert.equal(calls.length, 0);
});

test("desktop stops a silent hardware supervisor without depending on any Python output", async () => {
  const f = fixture();
  f.controller.start("synthetic-test-key", 8123, settings);
  f.state({ status: "preparing", stage: "hardware", message: "正在检查本机运行环境…" });
  f.advance(239999);
  assert.equal(f.states.at(-1).status, "preparing");
  f.advance(1);
  assert.equal(f.states.at(-1).reason, "desktop_hardware_timeout");
  assert.equal(f.states.at(-1).operation, "hardware_supervisor");
  // Late output cannot reopen the workspace after a failed startup.
  f.state({ status: "ready", message: "ready", url: "http://127.0.0.1:8123/" });
  assert.equal(f.states.at(-1).status, "error");
  await f.controller.stop();
  assert.equal(f.calls.length, 2);
  assert.deepEqual(f.calls[1][1], ["/PID", "321", "/T", "/F"]);
  f.child.emit("close", 1);
  assert.equal(f.states.filter(state => state.status === "error").length, 1);
  assert.equal(f.timers.size, 0);
  assert.ok(f.logs.some(line => line.includes("desktop_hardware_timeout")));
  assert.equal(f.logs.join("").includes("synthetic-test-key"), false);
});

test("hardware progress advances its deadline but repeated messages and ordinary logs do not", async () => {
  const f = fixture();
  f.controller.start("test", 8123, settings);
  f.state({ status: "preparing", stage: "hardware", message: "检查环境" });
  f.advance(180000);
  const loading = { status: "preparing", stage: "hardware", operation: "torch_import", message: "加载 PyTorch" };
  f.state(loading);
  f.advance(180000);
  f.state(loading);
  f.child.stdout.write("native library still loading\n");
  f.advance(59999);
  assert.equal(f.states.at(-1).status, "preparing");
  f.advance(1);
  assert.equal(f.states.at(-1).operation, "torch_import");
  assert.equal(f.states.at(-1).status, "error");
  f.child.emit("close", 1);
  await f.controller.stop();
});

test("hardware has a total deadline even if it keeps claiming new operations", async () => {
  const f = fixture();
  f.controller.start("test", 8123, settings);
  for (let i = 0; i < 7; i++) {
    f.state({ status: "preparing", stage: "hardware", operation: `step_${i}`, message: "检查环境" });
    f.advance(180000);
  }
  assert.equal(f.states.at(-1).reason, "desktop_hardware_timeout");
  f.child.emit("close", 1);
  await f.controller.stop();
});

for (const next of [
  { status: "preparing", stage: "engines", message: "准备引擎" },
  { status: "ready", message: "ready", url: "http://127.0.0.1:8123/" },
  { status: "error", message: "CUDA 检查失败" },
]) {
  test(`hardware deadline is removed after ${next.stage || next.status}`, () => {
    const f = fixture();
    f.controller.start("test", 8123, settings);
    f.state({ status: "preparing", stage: "hardware", message: "检查环境" });
    f.state(next);
    f.advance(3600000);
    assert.deepEqual(f.states.at(-1), next);
    assert.equal(f.timers.size, 0);
    assert.equal(f.calls.length, 1);
  });
}

test("normal shutdown cancels the hardware deadline", async () => {
  const f = fixture();
  f.controller.start("test", 8123, settings);
  f.state({ status: "preparing", stage: "hardware", message: "检查环境" });
  f.child.stdin.on("data", () => setImmediate(() => f.child.emit("close", 0)));
  await f.controller.stop();
  f.advance(3600000);
  assert.equal(f.states.some(state => state.status === "error"), false);
  assert.equal(f.calls.length, 1);
  assert.equal(f.timers.size, 0);
});
