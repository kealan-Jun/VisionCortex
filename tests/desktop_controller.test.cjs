"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { EventEmitter } = require("node:events");
const { PassThrough } = require("node:stream");
const { BackendController, STATE_PREFIX, serviceUrlAllowed } = require("../deployment/rtx4050-windows/desktop/controller.cjs");

const settings={connection:{provider:"aliyun",base_url:"https://example.test/v1",model:"vision-model",api_protocol:"chat_completions",quality_mode:"quality"},verification:{status:"verified"}};

function fixture() {
  const calls = [], logs = [], states = [];
  const child = new EventEmitter();
  Object.assign(child, { pid: 321, exitCode: null, stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough() });
  const controller = new BackendController({ root: "/package with spaces", platform: "win32", stopTimeoutMs: 15,
    writeLog: (line) => logs.push(line), spawnProcess: (...args) => {
      calls.push(args);
      return calls.length === 1 ? child : new EventEmitter();
    } });
  controller.on("state", (state) => states.push(state));
  return { controller, child, calls, logs, states };
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
