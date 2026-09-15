from visioncortex.device_day_queue import DeviceDayQueue


def test_publication_contention_does_not_exhaust_failure_budget(tmp_path):
    q = DeviceDayQueue(tmp_path/'Queue.sqlite3')
    q.enqueue({'recording_id':'slice','configured_role':'first_person'}, 'v1')
    for n in range(4):
        owner = str(n)
        assert q.claim(owner, max_attempts=3)
        q.finish(owner,'slice',{'status':'running_elsewhere'}, .1)
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
