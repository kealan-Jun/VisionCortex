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
