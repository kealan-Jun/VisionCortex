"""Metrics must reconcile with receipts and preserve missing-data boundaries."""
import json
import subprocess
from pathlib import Path

from visioncortex import api
from visioncortex.run_insights import hardware_summary, timing_summary, token_summary


def test_timing_separates_wall_clock_from_overlapping_request_work():
    def attempt(start,end):
        return {'started_at':f'2026-09-09T00:00:{start:02d}+00:00',
                'ended_at':f'2026-09-09T00:00:{end:02d}+00:00','latency_seconds':end-start,
                'status':'completed','retry_wait_seconds':0}
    result=timing_summary({'total_duration_seconds':30,
        'stage_durations':[{'stage':'mllm','duration_seconds':20}],
        'mllm_calls':[call(attempt_receipts=[attempt(0,10)]),call(attempt_receipts=[attempt(5,15)]),
                      call(cache_reused=True,attempt_receipts=[attempt(0,50)])]})
    assert result['model']['peak_concurrency']==2
    assert result['model']['request_busy_wall_seconds']==15
    assert result['model']['attempt_seconds_sum']==20
    assert result['model']['mean_active_concurrency']==1.333
    assert result['wall_seconds']==30
    assert result['model']['concurrency_coverage']=='complete'
    assert result['model']['backoff_recorded_attempts']==2


def test_historical_timing_does_not_invent_concurrency_or_decode_time():
    result=timing_summary({'mllm_calls':[call(attempts=3,latency_seconds=90)]})
    assert result['model']['peak_concurrency'] is None
    assert result['model']['backoff_seconds_sum'] is None
    assert result['model']['backoff_recorded_attempts']==0
    assert result['model']['expected_attempts']==3
    assert result['read_decode_separately_measured'] is False


def test_followup_usage_is_identical_in_ui_and_export_and_not_counted_twice(tmp_path):
    from visioncortex.partial_delivery import write_partial_delivery
    from visioncortex.run_insights import with_operation_refreshes
    directory = tmp_path / 'JSON-Config-Files/Stage-Refreshes'
    directory.mkdir(parents=True)
    metrics = {'total_duration_seconds': 30, 'mllm_calls': [call()]}
    (directory.parent / 'run_metrics.json').write_text(json.dumps(metrics))
    (directory / '1-operations.json').write_text(json.dumps({'scope': 'operations',
        'model_calls': [call(), call(cache_reused=True)]}))
    (directory / '2-previous-operations.json').write_text(json.dumps({'groups': {}}))
    merged = with_operation_refreshes(tmp_path, metrics)
    assert len(merged['mllm_calls']) == 3
    assert merged == with_operation_refreshes(tmp_path, merged)
    assert len(metrics['mllm_calls']) == 1
    assert api._merged_run_metrics(tmp_path) == merged
    write_partial_delivery(tmp_path, merged)
    exported = json.loads((tmp_path / 'Partial-Results/Analysis-Result.json').read_text())
    assert exported['run_metrics'] == merged
    assert merged['total_duration_seconds'] == 30


def sample(second, gpu=None, used=None, total=None, cpu=20, stage='fine'):
    return {'timestamp':f'2026-09-09T00:00:{second:02d}+00:00','stage':stage,
            'cpu_percent':cpu,'memory_percent':40,
            'gpu':{'utilization.gpu':gpu,'memory.used':used,'memory.total':total}}


def test_hardware_percent_uses_capacity_and_preserves_missing_samples():
    report=hardware_summary({'samples':[sample(1,100,2048,4096),sample(2,None,1024,4096),sample(3,0,None,4096)]})
    s=report['overall']
    assert s['gpu_percent']=={'count':2,'mean':50.0,'max':100.0,'p95':100.0}
    assert s['vram_percent']['mean']==37.5
    assert s['vram_percent']['count']==2
    assert s['vram_used_mib']['max']==2048
    assert len(report['timeline'])==3
    assert report['timeline'][1]['gpu_percent']['mean'] is None


def test_hardware_does_not_reuse_previous_attempt_as_current_history():
    old=sample(1,99,4,8)
    new=sample(11,10,2,8)
    result=hardware_summary({'samples':[old]}, {'latest':new,'status':'running'},'2026-09-09T00:00:10+00:00')
    assert result['available'] is False
    assert result['excluded_sample_count']==1
    assert result['latest']==new
    assert result['overall']['gpu_percent']['mean'] is None
    assert hardware_summary({}, {'latest':old},'2026-09-09T00:00:10+00:00')['latest']=={}


def test_downsampling_keeps_peaks_and_never_blends_stage_statistics():
    rows=[sample(i,100 if i==19 else 10,2,8,stage='fine' if i<30 else 'mllm') for i in range(1,59)]
    result=hardware_summary({'samples':rows},max_points=5)
    assert len(result['timeline'])<=5+len(result['stages'])
    assert max(p['gpu_percent']['max'] for p in result['timeline'])==100
    assert result['stages']['mllm']['gpu_percent']['max']==10
    assert sum(p['sample_count'] for p in result['timeline'])==len(rows)
    assert rows[0]['gpu']['memory.used']==2


def call(stage='steps',input_tokens=10,output_tokens=20,total_tokens=30,**kwargs):
    return {'stage':stage,'provider':'example','model':'vision','attempts':1,
            'usage':{'input_tokens':input_tokens,'output_tokens':output_tokens,'total_tokens':total_tokens},**kwargs}


def test_tokens_count_all_stages_once_and_reconcile_excluding_local_reuse():
    metrics={'tokens':{'run_total':{'input_tokens':20,'output_tokens':40,'total_tokens':60}},
             'mllm_calls':[call(),call('boundary'),call(cache_reused=True)]}
    result=token_summary(metrics)
    assert result['totals']['total_tokens']==60
    assert result['executed_calls']==2 and result['reused_calls']==1
    assert result['reconciled'] is True
    assert result['coverage']=='reported'
    assert sum(r['totals']['total_tokens'] for r in result['stages'])==60
    assert sum(r['totals']['total_tokens'] for r in result['models'])==60
    assert metrics['mllm_calls'][2]['usage']['total_tokens']==30


def test_unknown_retries_missing_fields_and_server_cached_input_are_not_free():
    receipt=call(attempts=3)
    receipt['usage'].update(unknown_attempt_count=1,cached_input_tokens=5)
    result=token_summary({'mllm_calls':[receipt,call(total_tokens=None,input_tokens=None)]})
    assert result['attempts']==4
    assert result['unknown_attempts']==2
    assert result['totals']['total_tokens']==30
    assert result['totals']['input_tokens']==10
    assert result['totals']['cached_input_tokens']==5
    assert result['coverage']=='partial'
    assert result['missing_usage_calls']['input_tokens']==1
    assert token_summary({'mllm_calls':[call(attempts=3)]})['unknown_attempts']==2


def test_usage_only_cache_has_zero_new_consumption_but_absent_ledger_is_unknown():
    assert token_summary({'mllm_calls':[call(cache_reused=True)]})['totals']['total_tokens']==0
    assert token_summary({})['totals']['total_tokens'] is None
    assert token_summary({})['available'] is False
    aggregate=token_summary({'tokens':{'run_total':{'total_tokens':42}}})
    assert aggregate['coverage']=='aggregate_only'
    assert aggregate['totals']['total_tokens']==42
    mismatch=token_summary({'tokens':{'run_total':{'total_tokens':999}},'mllm_calls':[call()]})
    assert mismatch['reconciled'] is False
    assert mismatch['totals']['total_tokens']==30


def test_snapshot_exposes_history_and_receipt_sources_without_rewriting_them(tmp_path):
    folder=tmp_path/'JSON-Config-Files'
    folder.mkdir()
    payloads={'resource_telemetry.json':{'samples':[sample(1,50,4,8)]},
              'run_metrics.json':{'mllm_calls':[call()]}}
    for name,value in payloads.items():
        (folder/name).write_text(json.dumps(value))
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    result=api._run_snapshot_from_root(tmp_path)
    assert result['insights']['hardware']['overall']['vram_percent']['mean']==50
    assert result['insights']['tokens']['totals']['total_tokens']==30
    assert {p.name:p.read_bytes() for p in folder.iterdir()}==before


def test_frontend_preserves_unknowns_escapes_provider_text_and_exposes_full_breakdown():
    script=Path(__file__).parents[1]/'src/visioncortex/web/run-insights.js'
    payload={'observability':{'insights':{'hardware':hardware_summary({}),
              'tokens':token_summary({'mllm_calls':[call(model='<script>bad</script>')]})}}}
    code="const assert=require('node:assert/strict');\n"+script.read_text()+f"\nconst data={json.dumps(payload)};"
    code+='''
const html=VisionCortexRunInsights.render(data);
assert.ok(html.includes('未保存有效采样'));
assert.ok(html.includes('输入 Token') && html.includes('输出 Token') && html.includes('总 Token'));
assert.ok(html.includes('&lt;script&gt;bad&lt;/script&gt;'));
assert.ok(!html.includes('<script>bad'));
assert.ok(html.includes('整机占用'));
assert.ok(html.includes('data-export-insights'));
assert.ok(VisionCortexRunInsights.render({}, {compact:true}).includes('未保存可用的 Token'));
'''
    result=subprocess.run(['node','-e',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_live_readings_expire_and_export_contains_the_same_source_values():
    script=Path(__file__).parents[1]/'src/visioncortex/web/run-insights.js'
    code="const assert=require('node:assert/strict');\n"+script.read_text()+'''
const data={name:'run',staging_run_id:'R',observability:{status:{stage:'fine'},
 insights:{hardware:{live_status:'running',latest:{timestamp:new Date().toISOString(),cpu_percent:25,gpu:{'utilization.gpu':50,'memory.used':4,'memory.total':8}}},tokens:{totals:{total_tokens:123}}},sources:{metrics:'JSON-Config-Files/run_metrics.json'}}};
assert.ok(VisionCortexRunInsights.render(data).includes('class="insight-live"'));
data.observability.insights.hardware.latest.timestamp='2000-01-01T00:00:00Z';
assert.ok(!VisionCortexRunInsights.render(data).includes('class="insight-live"'));
let click,contents;
const button={addEventListener:(event,fn)=>{click=fn;}};
const root={querySelector:s=>s==='[data-export-insights]'?button:null};
globalThis.Blob=class{constructor(parts){contents=parts.join('');}};
globalThis.URL={createObjectURL:()=>'/blob',revokeObjectURL:()=>{}};
globalThis.document={createElement:()=>({click:()=>{}})};
globalThis.setTimeout=fn=>fn();
VisionCortexRunInsights.bind(root,data);click();
const saved=JSON.parse(contents);
assert.equal(saved.run_id,'R');assert.equal(saved.insights.tokens.totals.total_tokens,123);
assert.deepEqual(saved.sources,data.observability.sources);
'''
    result=subprocess.run(['node','-e',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
