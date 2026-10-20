"""Device-day library cards separate source, selection and processing grains."""
import shutil
import subprocess

import pytest

from repo_paths import WEB


def run_node(script):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for browser characterization')
    source = (WEB/'app.js').read_text()
    functions = []
    for name in ('deviceDaySummary', 'deviceDaySection'):
        prefix = f'function {name}('
        functions.append(prefix + source.split(prefix, 1)[1].split('\n}\n', 1)[0] + '\n}\n')
    prelude = r'''
const assert=require('node:assert/strict');
const state={search:'',deviceDayArchives:[],deviceDaySyncError:false};
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number=value=>String(value),icon=()=>'';
const row={archive:'2026-10-20_camera-1',read_status:'available',segment_count:35,preprocessing_candidate_count:35,
 selected_interval_count:2,pending_interval_count:3,confirmed_no_value_interval_count:0,
 report_exists:true,report_path:'LaboratoryDailyReport/LaboratoryDailyReport.json',
 report_url:'/old/LaboratoryDailyReport.html',recordings:[
 {recording_id:'one',preprocessing_ready:true,current_results_ready:false,stages:{vision:{status:'completed'},understanding:{status:'partial'},report:{status:'partial'}}},
 {recording_id:'two',preprocessing_ready:false,current_results_ready:false,stages:{stt:{status:'failed'}}}]};
'''
    result = subprocess.run([node], input=prelude + '\n'.join(functions) + script,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_card_does_not_count_candidates_as_retained_or_demand_full_stage_completion():
    run_node(r'''
const summary=deviceDaySummary(row),html=deviceDaySection(false,Infinity,[row]);
assert.equal(summary.count,2);assert.equal(summary.completed,0);assert.equal(summary.preprocessed,1);
assert.equal(summary.candidateCount,35);assert.equal(summary.selectedCount,2);assert.equal(summary.pendingCount,3);
assert.equal(summary.reportReady,true);
assert.ok(html.includes('原始录像 2 条')&&html.includes('预处理完成 1 条')&&html.includes('预处理候选 35 个'));
assert.ok(html.includes('选中计划 2 个')&&html.includes('待判定 3 个')&&html.includes('确认无价值 0 个'));
assert.equal(html.includes('全流程完成'),false);assert.equal(html.includes('保留 35'),false);
assert.ok(html.includes('候选数量和选中计划不等于实际已留存视频'));
assert.ok(html.includes('/files/LaboratoryDailyReport/LaboratoryDailyReport.json'));
assert.equal(html.includes('LaboratoryDailyReport.html'),false);
assert.ok(deviceDaySection(true,Infinity,[row]).includes('查看日报 JSON（已有内容）'));
''')


def test_historical_missing_selection_fields_are_unknown_not_zero_or_candidate_count():
    run_node(r'''
const legacy={...row};for(const key of ['selected_interval_count','pending_interval_count','confirmed_no_value_interval_count','preprocessing_candidate_count'])delete legacy[key];
const summary=deviceDaySummary(legacy),html=deviceDaySection(false,Infinity,[legacy]);
assert.equal(summary.selectedCount,null);assert.equal(summary.pendingCount,null);assert.equal(summary.noValueCount,null);
assert.equal(summary.candidateCount,35);
assert.ok(html.includes('选中计划 待索引 个')&&html.includes('待判定 待索引 个'));
assert.equal(html.includes('选中计划 0 个'),false);assert.equal(html.includes('选中计划 35 个'),false);
''')


@pytest.mark.parametrize('value', ['null', 'undefined', 'true', '-1', '1.5', 'NaN', 'Infinity', '"2"'])
def test_invalid_or_absent_counts_cannot_invent_selected_success(value):
    run_node('row.selected_interval_count=' + value + r''';
assert.equal(deviceDaySummary(row).selectedCount,null);
assert.ok(deviceDaySection(false,Infinity,[row]).includes('选中计划 待索引 个'));
''')


def test_explicit_zero_selection_is_different_from_unknown_and_not_no_value():
    run_node(r'''
row.selected_interval_count=0;row.pending_interval_count=5;row.confirmed_no_value_interval_count=0;
const html=deviceDaySection(false,Infinity,[row]);
assert.ok(html.includes('选中计划 0 个')&&html.includes('待判定 5 个')&&html.includes('确认无价值 0 个'));
assert.equal(html.includes('没有有价值'),false);
''')


@pytest.mark.parametrize('status', ['unavailable', 'deferred'])
def test_unreadable_index_does_not_display_zero_inputs_or_completed_status(status):
    run_node('row.read_status=' + repr(status) + r''';
row.recordings=[];row.preprocessing_candidate_count=null;row.selected_interval_count=null;
const summary=deviceDaySummary(row),html=deviceDaySection(false,Infinity,[row]);
assert.equal(summary.count,null);assert.equal(summary.preprocessed,null);
assert.ok(html.includes('原始录像 待索引 条')&&html.includes('预处理完成 待索引 条'));
assert.equal(html.includes('已归档分片处理完成'),false);
assert.equal(html.includes('全流程完成'),false);
assert.equal(summary.reportReady,true,'an existing report remains readable despite deferred CV metadata');
''')


def test_completed_stage_rows_without_actual_json_report_do_not_create_report_button():
    run_node(r'''
row.recordings=[{current_results_ready:true,stages:Object.fromEntries(['retention','vision','stt','understanding','report'].map(stage=>[stage,{status:'completed'}]))}];
delete row.report_exists;
assert.equal(deviceDaySummary(row).completed,1);assert.equal(deviceDaySummary(row).reportReady,false);
assert.equal(deviceDaySection(true,Infinity,[row]),'');
assert.equal(deviceDaySection(false,Infinity,[row]).includes('查看日报 JSON'),false);
row.report_exists=true;row.report_path='LaboratoryDailyReport/LaboratoryDailyReport.html';
assert.equal(deviceDaySummary(row).reportReady,false);
''')


def test_unknown_call_and_search_filter_do_not_promote_completion_or_hide_matching_partial_report():
    run_node(r'''
row.recordings=[{stages:{understanding:{status:'submitted_unknown'}}}];
state.search='CAMERA-1';
const html=deviceDaySection(true,Infinity,[row]);
assert.ok(html.includes('有调用结果未知')&&html.includes('查看日报 JSON'));
assert.equal(html.includes('全流程完成'),false);
state.search='missing';assert.equal(deviceDaySection(true,Infinity,[row]),'');
''')
