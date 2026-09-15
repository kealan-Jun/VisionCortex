from visioncortex.device_day_acceptance import observe, render


def test_idle_weekend_and_long_observation_are_not_load_acceptance(tmp_path):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    first = observe(cfg, {'observed_at': 1000, 'days': {}})
    final = observe(cfg, {'observed_at': 1000 + 9*3600, 'days': {}})
    assert first['samples'] == 1
    assert final['observed_hours'] == 9
    assert final['max_sample_gap_seconds'] == 9*3600
    assert final['live_inputs'] == 0
    assert final['evidence_status'] == 'PARTIAL_EVIDENCE'
    assert '不能证明生产吞吐能力' in render(final)


def test_exact_live_revision_counted_once_and_old_backfill_excluded(tmp_path):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    row = {'recording_id': 'a', 'source_signature': 'v1', 'revision_observed_at': 1001,
           'cohort': 'live_observation', 'vision_started_at': 1002, 'preprocessing_completed_at': None}
    observe(cfg, {'observed_at': 1000, 'days': {}})
    p = {'observed_at': 1010, 'days': {}, 'latency': {'recent': [row,
         row | {'recording_id': 'historical', 'cohort': 'historical_backfill'}]}}
    assert observe(cfg, p)['live_inputs'] == 1
    row['preprocessing_completed_at'] = 1020
    p['observed_at'] = 1030
    final = observe(cfg, p)
    assert final['live_inputs'] == final['live_published'] == 1
