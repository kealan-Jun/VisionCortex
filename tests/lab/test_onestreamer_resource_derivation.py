"""Resource reporting must preserve unknowns and exact native answer provenance."""
from ._support import research_helpers
import importlib.util
import json

import pytest


PATH = research_helpers() / 'plot_onestreamer_resources.py'
SPEC = importlib.util.spec_from_file_location('one_resource_derivation', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_ram_cache_and_process_values_remain_unknown(tmp_path):
    path = tmp_path/'Resources.jsonl'
    value = dict(wall_seconds=12, utc='2026-10-27T10:00:00+00:00',
                 gpu_csv='12000, 49140, 98, 400', progress=dict(query_branch_processed_time=424))
    path.write_text(json.dumps(value)+'\n')
    row, = MODULE.resource_rows(path)
    assert row['rss_bytes'] is None
    assert row['all_recorded_cache_bytes'] is None
    assert row['evaluation_gpu_used_mib'] is None
    assert row['worker_video_time'] is None
    assert row['causal_query_video_time'] == 424
    assert row['source_sha256'] == MODULE.digest(path)
    assert row['source_line'] == 1


def test_worker_and_external_gpu_telemetry_are_separated(tmp_path):
    path = tmp_path/'Resources.jsonl'
    value = dict(wall_seconds=1, at_unix=0, gpu_csv='23000, 49140, 90, 400',
                 worker_pid=22, compute_processes=['22, 20000', '99, 2000'],
                 process=dict(VmRSS_bytes=123), cache_bytes=dict(media=512),
                 progress=dict(worker_latest_materialized_video_time=5400, query_branch_processed_time=365))
    path.write_text(json.dumps(value)+'\n')
    row, = MODULE.resource_rows(path)
    assert row['evaluation_gpu_used_mib'] == 20000
    assert row['other_gpu_process_used_mib'] == 2000
    assert row['gpu_used_mib'] == 23000  # Driver overhead is not attributed to either process.
    assert row['worker_video_time'] == 5400
    assert row['causal_query_video_time'] == 365
    assert row['rss_bytes'] == 123


def test_unknown_external_nvml_value_does_not_become_zero(tmp_path):
    path = tmp_path/'Resources.jsonl'
    path.write_text(json.dumps(dict(wall_seconds=1, at_unix=0,
        gpu_csv='2000, 49140, 90, 300', worker_pid=22,
        compute_processes=['22, 1500', '99, N/A']))+'\n')
    row, = MODULE.resource_rows(path)
    assert row['other_gpu_process_used_mib'] is None
    assert row['evaluation_gpu_used_mib'] == 1500


def test_carried_previous_query_fields_do_not_become_an_active_query(tmp_path):
    path = tmp_path/'Resources.jsonl'
    path.write_text(json.dumps(dict(wall_seconds=1, at_unix=0,
        gpu_csv='2000, 49140, 90, 300', progress=dict(phase='native_extension',
        query='previous_A_question', query_branch_processed_time=2223,
        latest_processed_time=3810)))+'\n')
    row, = MODULE.resource_rows(path)
    assert row['worker_video_time'] == 3810
    assert row['causal_query_video_time'] is None
    assert row['query'] is None
    assert row['raw_progress_query_field'] == 'previous_A_question'


def test_query_index_rejects_changed_native_answer_and_unknown_stages(tmp_path):
    lane = tmp_path/'streams-2'/'A'
    lane.mkdir(parents=True)
    path = lane/'Q.json'
    answer = dict(id='Q', case='A', question_time=424, history_interval_seconds=1,
                  query_end_to_end_seconds=4,
                  stages_seconds=dict(answer_end_to_end=3, model_generate=2),
                  selection=dict(memory_tokens=8, actual_generated_events=2,
                                 actual_selected_events=2, truncated=False))
    path.write_text(json.dumps(answer))
    index = lane/'AnswerIndex.jsonl'
    index.write_text(json.dumps(dict(file='Q.json', sha256=MODULE.digest(path)))+'\n')
    row, = MODULE.latency_rows(tmp_path)
    assert row['raw_model_generate_seconds'] == 2
    assert row['vision_encoder_nested_seconds'] is None
    assert row['state_validation_after_seconds'] is None
    assert row['e2e_excluding_pre_validation_seconds'] is None
    assert row['scope'] == 'streams-2/A'
    path.write_text(json.dumps(dict(answer, text='tampered')))
    with pytest.raises(ValueError, match='hash changed'):
        MODULE.answer_rows(tmp_path)
