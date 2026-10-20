import json

from visioncortex.device_day_queue import DeviceDayQueue


def test_publication_contention_does_not_exhaust_failure_budget(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr('visioncortex.device_day_queue.time.time', lambda: clock[0])
    q = DeviceDayQueue(tmp_path/'Queue.sqlite3')
    q.enqueue({'recording_id':'slice','configured_role':'first_person'}, 'v1')
    for n in range(4):
        owner = str(n)
        assert q.claim(owner, max_attempts=3)
        q.finish(owner,'slice',{'status':'running_elsewhere'}, .1)
        # Contention refunds the attempt but keeps a durable increasing wait.
        # Another poll cannot claim the same slice until that wait expires.
        assert q.claim('before-retry', max_attempts=3) is None
        with q.connect() as db:
            row = db.execute('SELECT status,attempts,result FROM recordings').fetchone()
            wait = json.loads(row['result'])
            assert (row['status'], row['attempts']) == ('queued', 0)
            assert wait['wait_count'] == n + 1
            assert wait['retry_at'] == clock[0] + 5 * 2 ** n
        clock[0] = wait['retry_at']
    with q.connect() as db:
        row = db.execute('select status,attempts,revision from recordings').fetchone()
        assert tuple(row) == ('queued',0,'v1')
    assert q.claim('real')
    q.finish('real','slice',{'status':'failed','message':'real failure'},1)
    with q.connect() as db:
        assert tuple(db.execute('select status,attempts from recordings').fetchone()) == ('failed',1)
def test_service_shutdown_waits_for_inflight_publication():
    from threading import Thread, Event
    from visioncortex.device_day_service import DeviceDayService
    service = DeviceDayService(lambda: {}, None)
    release, stopped = Event(), Event()
    def publishing():
        assert release.wait(10)
    service.thread = Thread(target=publishing, daemon=True)
    service.thread.start()
    def stop():
        service.stop()
        stopped.set()
    waiter = Thread(target=stop)
    waiter.start()
    try:
        assert service.stop_event.wait(2)
        assert not stopped.wait(5.1), 'shutdown returned while publication was still running'
    finally:
        release.set()
        waiter.join(2)
    assert stopped.is_set()
