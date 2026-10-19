"""A shutdown finishes its current atomic recording, not the whole backlog."""
from types import SimpleNamespace
import threading

from visioncortex.device_day_publication_worker import PublicationReplay
from visioncortex.publication_journal import PublicationJournal


def test_shutdown_retains_unvisited_publications_and_does_not_start_field_calls(tmp_path):
    journal = PublicationJournal(tmp_path)
    for rid in ('first', 'second', 'third'):
        journal.begin({'recording_id': rid, 'camera_key': 'owned_cam01',
                       'recording_start_us': 1_800_000_000_000_000})
    stop = threading.Event()
    visited = []
    def publish(layout, **kwargs):
        visited.append(kwargs['recording_id'])
        stop.set()
        return True
    runner = SimpleNamespace(runtime_root=tmp_path, backend_root=tmp_path, settings={
        'field_recognition': {'enabled': True}}, layout=lambda record: 'owned-layout', refresh_index=publish)
    result = PublicationReplay(stop=stop, field_fetch=lambda *a: (_ for _ in ()).throw(
        AssertionError('External field calls started during shutdown'))).tick(runner)
    assert result == {'status': 'stopping', 'published': 1, 'pending_publications': True}
    assert visited == ['first']
    assert [record['recording_id'] for _, record in journal.pending()] == ['second', 'third']
