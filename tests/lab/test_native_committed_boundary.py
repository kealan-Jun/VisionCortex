"""One real PG commit-boundary counterexample, generated native documents only."""
import json
from types import SimpleNamespace

import pytest

from visioncortex.lab.native_linked_layout import module
from visioncortex.lab.native_refresh import NativeRefresh
from visioncortex.lab.native_timeline_links import (PendingCommittedAssociation,
    association_revision, verified_committed_association)

from .test_person_report import scene, media, event, report
from .test_native_refresh import prepare
from .test_operator_scan import scan
from .test_timeline_publication_recovery import publisher

__all__ = ['scene']


def test_new_native_builders_only_attach_exact_committed_context_after_rollback_and_race(scene, monkeypatch, tmp_path):
    s = scene
    event(s, 'commit-boundary-observation', media(s, 'commit-boundary-camera0'))
    snapshot = report(s)
    publication = publisher(s)
    native = prepare(s['settings'].incoming, s['core'].customer, 'fixture-camera-0')
    refresh = NativeRefresh(s['core'], publication.settings)
    root, archive = native['day'], native['archive']

    def direct_day():
        return module('device_day_reports').day_document(SimpleNamespace(root=root,name=archive),native['index'],
            {'status':'pending'},readable={'视频留存统计':{}},entries=[],report_generated_at='fixture')

    def interrupt(*args, **kwargs):
        raise OSError('fixture rollback after sidecar writes')

    with monkeypatch.context() as patch:
        patch.setattr(publication, '_write_latest', interrupt)
        with pytest.raises(OSError, match='rollback'):
            publication.tick()
    assert list(root.glob('LaboratoryDailyReport/Timeline/Associations/r*.json'))
    assert '关联时间线' not in direct_day()
    assert association_revision(root,archive) == ()
    assert refresh.tick()['status'] == 'awaiting_committed_timeline_publication'
    committed = publication.tick()
    first = refresh.tick()
    assert first['refreshed'] == 1 and first['errors'] == 0
    main = root/'LaboratoryDailyReport/LaboratoryDailyReport.json'
    first_bytes = main.read_bytes()
    assert json.loads(first_bytes)['关联时间线']['snapshot_revision'] == snapshot['revision']
    assert '关联时间线' not in direct_day()  # Permission ended with refresh.
    receipt = next(f for f in committed['files'] if f['relative_path'].startswith(
        archive+'/LaboratoryDailyReport/Timeline/Associations/r'))
    scan(s,'commit-boundary-late-scan')
    later_snapshot = report(s)
    assert later_snapshot['revision'] > snapshot['revision']
    with verified_committed_association(root,archive,customer_id=s['core'].customer,receipt=receipt):
        assert direct_day()['关联时间线']['snapshot_revision'] == snapshot['revision']
        with monkeypatch.context() as patch:
            patch.setattr(publication,'_write_latest',interrupt)
            with pytest.raises(OSError,match='rollback'):
                publication.tick()
        with pytest.raises(PendingCommittedAssociation,match='changed'):
            direct_day()  # Higher valid bytes appeared after receipt verification.
    pending = refresh.tick()
    assert pending['pending'] == 1 and pending['refreshed'] == pending['errors'] == 0
    assert main.read_bytes() == first_bytes
    assert '关联时间线' not in direct_day() and association_revision(root,archive) == ()
    later_committed = publication.tick()
    final = refresh.tick()
    assert final['refreshed'] == 1 and final['errors'] == 0
    assert json.loads(main.read_text())['关联时间线']['snapshot_revision'] == later_snapshot['revision']
    assert '关联时间线' not in direct_day()
    (tmp_path/'NativeCommittedBoundaryProof.json').write_text(json.dumps(dict(
        first_committed=committed,first_refresh=first,uncommitted_higher_pending=pending,
        later_committed=later_committed,final_refresh=final,actual_model_calls=0,
        database_rollback_sidecar_not_included=True,context_restored_after_exit=True),indent=2)+'\n')
