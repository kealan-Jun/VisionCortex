from visioncortex.device_day_activity import job,phase,observations
from visioncortex.performance_stages import StageTimings


def test_live_phases_nest_and_cleanup_on_failure():
    try:
        with job('vision','a'):
            with phase('prerequisites'):
                assert observations()[('vision','a')]['phase']=='校验原片及上游产物'
            with StageTimings().measure('fine_scan_seconds'):
                assert '精扫' in observations()[('vision','a')]['phase']
                raise ValueError('failed')
    except ValueError:
        pass
    assert ('vision','a') not in observations()
