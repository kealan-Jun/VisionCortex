"""Foreground newest-first selection precedes the bounded history LIMIT."""
import pytest

from test_device_day_backfill_policy import NOW, LocalState, recording
from visioncortex.device_day_backfill_policy import historical_candidates
from visioncortex.device_day_retention_worker import foreground_history_config


@pytest.mark.parametrize('stage', ['understanding', 'report'])
def test_foreground_selects_newest_ready_before_limit_and_keeps_fair_lane(tmp_path, stage):
    state = LocalState(tmp_path / 'runtime')
    state.config['device_day']['backfill'] = {'mode': 'fair'}
    state.config['device_day']['latest_first'] = True
    rows = [recording(str(i), age=200+i, audio={'source_signature': 'audio'}) for i in range(70)]
    state.observe(*rows)
    for row in rows:
        state.complete(row, ('retention', 'vision', 'stt', 'understanding') if stage == 'report'
                       else ('retention', 'vision', 'stt'))
    def selected(*, foreground=False, **options):
        return [r['recording_id'] for r in historical_candidates(
            foreground_history_config(state.config) if foreground else state.config,
            stage, limit=1, now=NOW, **options)]
    assert selected() == ['69']
    assert selected(foreground=True) == ['0']
    assert state.config['device_day']['backfill']['mode'] == 'fair'
    state.unavailable(rows[0])
    assert selected(foreground=True) == ['1']
    assert selected(foreground=True, exclude={'1'}) == ['2']
    state.monitor(age=1000)
    assert selected(foreground=True) == []


@pytest.mark.parametrize('stage', ['understanding', 'report'])
@pytest.mark.parametrize('latest', [True, False])
def test_foreground_passes_configured_order_to_history_selector(tmp_path, monkeypatch, stage, latest):
    from test_device_day_retention_worker import record, setup
    from visioncortex.device_day_retention_worker import RetentionWorker
    runner = setup(tmp_path, monkeypatch, [record('outside-recent-window', 1)])
    runner.settings.update(latest_first=latest, backfill={
        'enabled': True, 'stages': [stage], 'paid_history_authorized': True})
    runner.config['device_day'] = runner.settings
    seen = []
    def selector(config, *args, **kwargs):
        seen.append(config['device_day']['backfill'].get('mode'))
        return []
    monkeypatch.setattr('visioncortex.device_day_backfill_policy.historical_candidates', selector)
    assert RetentionWorker(stage=stage, recent_seconds=14400).candidates(runner) == []
    assert seen == ['idle' if latest else None]
    assert 'mode' not in runner.settings['backfill']
