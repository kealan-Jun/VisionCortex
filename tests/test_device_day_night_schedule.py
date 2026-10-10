from datetime import datetime
import pytest
from visioncortex.device_day_night_schedule import stage_admitted, night_schedule

CONFIG = {'device_day': {'night_processing': {'enabled': True}}}


@pytest.mark.parametrize('local,allowed', [('2026-09-11T20:29:59+08:00',False),
 ('2026-09-11T20:30:00+08:00',True),('2026-09-12T00:00:00+08:00',True),
 ('2026-09-12T07:59:59+08:00',True),('2026-09-12T08:00:00+08:00',False),
 ('2026-09-11T12:30:00+00:00',True)])
def test_night_admission_does_not_gate_realtime_stages(local, allowed):
    now = datetime.fromisoformat(local)
    for stage in ['understanding','report']:
        assert stage_admitted(CONFIG,stage,now) is allowed
    for stage in ['retention','vision','stt']:
        assert stage_admitted(CONFIG,stage,now)


def test_disabled_legacy_profile_and_next_start():
    now=datetime.fromisoformat('2026-09-11T18:00:00+08:00')
    assert stage_admitted({},'understanding',now)
    assert night_schedule(CONFIG,now)['next_start']=='2026-09-11T20:30:00+08:00'


@pytest.mark.parametrize('hour', [0, 8, 12, 21])
def test_explicit_semantic_pause_keeps_stt_admitted_all_day(hour):
    config = {'device_day': {'paused_stages': ['understanding', 'report'],
                             'night_processing': {'enabled': False}}}
    now = datetime.fromisoformat(f'2026-09-20T{hour:02d}:00:00+08:00')
    assert all(stage_admitted(config, stage, now) for stage in ('retention', 'vision', 'stt'))
    assert all(not stage_admitted(config, stage, now) for stage in ('understanding', 'report'))


@pytest.mark.parametrize('value', ['understanding', None, ['unknown'], [True]])
def test_invalid_pause_configuration_is_rejected(value):
    from visioncortex.device_day_night_schedule import paused_stages
    with pytest.raises(ValueError, match='paused_stages'):
        paused_stages({'device_day': {'paused_stages': value}})
