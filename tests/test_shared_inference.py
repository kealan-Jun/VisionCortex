from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident
import pytest
from visioncortex.shared_inference import InferenceBroker


class Scanner:
    batch_size = 4
    def __init__(self):
        self.owner = get_ident()
        self.last_engine_batch_sizes = []
    def infer(self, packets):
        assert get_ident() == self.owner
        self.last_engine_batch_sizes = [len(packets)]
        return [('box',p) for p in packets]


def test_cross_camera_batch_keeps_source_order_and_model_owner():
    broker = InferenceBroker(Scanner,wait_seconds=.1)
    barrier = Barrier(4)
    def submit(camera):
        barrier.wait()
        return broker.submit([(camera,1)])
    with ThreadPoolExecutor(max_workers=4) as pool:
        out=list(pool.map(submit,range(4)))
    assert out == [[('box',(i,1))] for i in range(4)]
    assert broker.stats()['model_calls']==1
    assert broker.stats()['mixed_request_calls']==1
    assert broker.submit([])==[]


def test_error_propagation_does_not_drop_next_request():
    class FailsOnce(Scanner):
        def infer(self, packets):
            if packets==['bad']:
                raise ValueError('invalid input')
            return super().infer(packets)
    broker=InferenceBroker(FailsOnce)
    with pytest.raises(ValueError,match='invalid input'):
        broker.submit(['bad'])
    assert broker.submit(['good']) == [('box','good')]


def test_oversized_request_and_late_tail_are_not_lost():
    broker=InferenceBroker(Scanner)
    assert broker.submit(list(range(7))) == [('box',n) for n in range(7)]
    assert broker.submit([8]) == [('box',8)]


def test_contexts_grow_with_active_cameras_and_reuse_released_models(monkeypatch):
    import visioncortex.shared_inference as shared
    monkeypatch.setattr(shared, '_POOLS', {})
    created = []
    def factory(*args, **kwargs):
        scanner = Scanner()
        created.append(scanner)
        return scanner
    config = {'models': {}, 'performance': {'shared_inference_enabled': True,
                                            'shared_inference_contexts_per_pool': 2}}
    first = shared.acquire_scanner(factory, 'first_person', config, 640, 16)
    assert len(created) == 1
    second = shared.acquire_scanner(factory, 'first_person', config, 640, 16)
    assert len(created) == 2
    third = shared.acquire_scanner(factory, 'first_person', config, 640, 16)
    assert len(created) == 2
    assert first.infer([1]) == [('box', 1)]
    second.close()
    fourth = shared.acquire_scanner(factory, 'first_person', config, 640, 16)
    assert fourth.broker is second.broker
    assert len(created) == 2
    for lease in (first, third, fourth):
        lease.close()
