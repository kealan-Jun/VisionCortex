const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/visioncortex/web/app.js', 'utf8');
function library(archives, search = '') {
  const context = vm.createContext({state:{deviceDayArchives:archives, search},
    esc:value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;'),
    icon:()=>'', number:value=>String(value || 0), encodeURIComponent});
  vm.runInContext(source.slice(source.indexOf('function deviceDaySummary('), source.indexOf('function renderHome(')), context);
  return context;
}
test('full workflow readiness requires all current stage receipts', () => {
  const completed = {current_results_ready:true,stages:Object.fromEntries(['retention','vision','stt','understanding','report'].map(stage=>[stage,{status:'completed'}]))};
  const ready = {archive:'2026-09-01_lubancat-52d2ef0c_cam01',segment_count:3,
    recordings:[completed,completed]};
  const waiting = {archive:'2026-09-09_rk3588-ubuntu_cam01',segment_count:0,
    recordings:[{stages:{retention:{status:'completed'},vision:{status:'running'}}}]};
  const ui = library([ready,waiting]);
  assert.equal(ui.deviceDaySummary(waiting).reportReady,false);
  assert.equal(ui.deviceDaySummary(waiting).running,true);
  assert.equal(ui.deviceDaySummary(ready).completed,2);
  assert.equal(ui.deviceDaySummary({recordings:[{...completed,current_results_ready:false}]}).completed,0);
  assert.equal(ui.deviceDaySummary({recordings:[{stages:{report:{status:'completed'}}}]}).completed,0,
    'A report from an older run must not count as a completed current workflow');
  assert.equal(ui.deviceDaySummary({recordings:[{stages:{...completed.stages,vision:{status:'running'}}}]}).completed,0);
  const reports = ui.deviceDaySection(true);
  assert.match(reports,/2026-09-01/);
  assert.doesNotMatch(reports,/2026-09-09/);
  assert.match(reports,/#\/device-day\/2026-09-01_lubancat-52d2ef0c_cam01\/report/);
  assert.match(ui.deviceDaySection(),/CSV/);
  assert.match(ui.deviceDaySection(),/处理中/);
});
test('archive identity is escaped, URLs encoded, and date search applied', () => {
  const row = {archive:'2026-09-09_<img onerror=attack>',recordings:[]};
  const ui = library([row]);
  assert.doesNotMatch(ui.deviceDaySection(), /<img/);
  assert.match(ui.deviceDaySection(),/%3Cimg/);
  assert.equal(library([row],'2026-09-01').deviceDaySection(), '');
});
test('preprocessing remains visible while cloud stages are unavailable', () => {
  const row = {archive:'2026-09-09_camera',segment_count:2,recordings:[{
    preprocessing_ready:true,current_results_ready:false,
    stages:{retention:{status:'completed'},vision:{status:'completed'},understanding:{status:'failed'}}}]};
  const ui = library([row]);
  assert.equal(ui.deviceDaySummary(row).preprocessed,1);
  assert.equal(ui.deviceDaySummary(row).completed,0);
  assert.match(ui.deviceDaySection(),/1 个预处理完成/);
});
