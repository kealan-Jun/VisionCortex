"""Unknown-clock evidence is publishable without inventing an event date."""
import pytest
from datetime import timedelta
from psycopg.types.json import Jsonb

from visioncortex.lab.timeline_policy import ZONE
from visioncortex.lab.timeline_views import TimelineViews
from visioncortex.lab.timeline_report import render_report
from visioncortex.lab.timeline_clock import source_clock_supported, source_clock_sql

from .test_timeline import hj as base_scene, media, activity


@pytest.fixture
def hj(core, settings):
    return base_scene.__wrapped__(core, settings)


def test_receipt_day_dirties_untimed_evidence_and_late_revision_without_dating_it(hj):
    core, timeline = hj['core'], hj['timeline']
    source = media(hj, 'untimed-receipt-partition', captured=False, duration=4)
    with core.db.transaction(readonly=True) as conn:
        received = conn.execute('SELECT received_at FROM capture WHERE customer_id=%s AND id=%s',
                                (core.customer, source['capture_id'])).fetchone()['received_at']
        day = received.astimezone(ZONE).date()
        assert conn.execute('SELECT day FROM hj_timeline_dirty_day WHERE customer_id=%s',
                            (core.customer,)).fetchall() == [dict(day=day)]
    views = TimelineViews(timeline)
    views.refresh_dirty()
    observed = activity(hj, 'untimed-speech', source, source_type='speech', start=1, end=2)
    assert observed['start_at'] is None and observed['end_at'] is None
    assert observed['person_id'] is None and observed['run_ids'] == []
    refreshed = views.refresh_dirty()['reports']
    assert refreshed == [dict(day=str(day), revision=2)]
    document = views.snapshot(str(day))['document']
    assert document['activity_count'] == 0 and not document['unassigned_activities']
    pending = document['unknown_time_activities']
    assert [a['id'] for a in pending['activities']] == [observed['id']]
    assert pending['receipt_day_is_archive_partition_only']
    assert pending['attribution'] == 'customer_pending_time_not_assigned_to_requested_day_or_range'
    assert '录音转写' in render_report(document)
    timeline.associate_activity('untimed-reviewed-unassigned', observed['id'], [],
                                actor='fixture-reviewer', reason='remain unassigned')
    assert views.refresh_dirty()['reports'] == [dict(day=str(day), revision=3)]
    revised = views.snapshot(str(day))['document']['unknown_time_activities']['activities'][0]
    assert revised['start_at'] is None and revised['person_id'] is None
    assert revised['current_association_status'] == 'reviewed_association'


def test_reported_but_uncalibrated_legacy_time_never_reenters_timed_day_or_person(hj):
    core, timeline = hj['core'], hj['timeline']
    source = media(hj, 'legacy-unverified-reported-clock', duration=4)
    observation = activity(hj, 'legacy-unverified-observation', source, source_type='ocr')
    basis = dict(basis='historical source label', uncertainty_us=0, clock_sync_verified=False)
    with core.db.transaction() as conn:
        received = conn.execute('SELECT received_at FROM capture WHERE customer_id=%s AND id=%s',
                               (core.customer, source['capture_id'])).fetchone()['received_at']
        old = received - timedelta(days=7)
        # Simulate an immutable legacy observation admitted before the clock gate.
        conn.execute('UPDATE hj_timeline_media SET start_at=%s,end_at=%s,time_basis=%s WHERE customer_id=%s AND media_id=%s',
                     (old, old+timedelta(seconds=4), Jsonb(basis), core.customer, source['media_id']))
        conn.execute('UPDATE hj_activity SET start_at=%s,end_at=%s WHERE customer_id=%s AND id=%s',
                     (old, old+timedelta(seconds=1), core.customer, observation['id']))
    views = TimelineViews(timeline)
    for day in [old.astimezone(ZONE).date(), received.astimezone(ZONE).date()]:
        document = views.snapshot(str(day))['document']
        assert document['activity_count'] == 0 and not document['unassigned_activities']
        assert not document['person_timelines']['persons']
        pending = document['unknown_time_activities']['activities'][0]
        assert pending['id'] == observation['id'] and pending['start_at'] is None
        assert pending['source_reported_interval']['start_at']
    newest = activity(hj, 'new-observation-on-legacy-source', source, source_type='speech')
    assert newest['start_at'] is None and newest['run_ids'] == []
    with core.db.transaction(readonly=True) as conn:
        original = conn.execute('SELECT start_at FROM hj_activity WHERE customer_id=%s AND id=%s',
                                (core.customer, observation['id'])).fetchone()
        assert original['start_at'] == old  # Projection did not rewrite the legacy row.


@pytest.mark.parametrize('basis,kind,expected', [
    (dict(basis='clock', uncertainty_us=0, clock_sync_verified=True), 'customer_capture', True),
    (dict(basis='clock', uncertainty_us=0, clock_sync_verified=False), 'customer_capture', False),
    (dict(basis='clock', uncertainty_us=0), 'customer_capture', False),
    (dict(basis='clock', uncertainty_us=0.0, clock_sync_verified=True), 'customer_capture', False),
    (dict(basis='clock', uncertainty_us=False, clock_sync_verified=True), 'customer_capture', False),
    (dict(basis='synthetic_fixture_clock', uncertainty_us=0), 'fixture', True),
    (dict(basis='synthetic_fixture_clock', uncertainty_us=0), 'customer_capture', False),
])
def test_clock_json_type_boundaries_agree_in_python_and_postgresql(db, basis, kind, expected):
    assert source_clock_supported(basis, kind) is expected
    sql = source_clock_sql()
    with db.transaction(readonly=True) as conn:
        row = conn.execute(f'''SELECT COALESCE({sql},false) AS supported
            FROM (SELECT %s::jsonb AS time_basis) m,
                 (SELECT %s::jsonb AS payload) i''', (Jsonb(basis), Jsonb(dict(source_kind=kind)))).fetchone()
        assert row['supported'] is expected
