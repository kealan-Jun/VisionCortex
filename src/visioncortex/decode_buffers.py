"""Bound decoded pixel buffers by bytes as well as configured frame counts."""
import queue
import time

from .runtime_control import check_cancelled

DEFAULT_FRAME_BUFFER_BYTES = 64 * 1024 * 1024


class CancellableQueue(queue.Queue):
    """Wake blocked producers/consumers when their scan is cancelled."""
    def __init__(self, maxsize, stop):
        super().__init__(maxsize)
        self.stop = stop

    def _operation(self, operation, args, block, timeout, empty_or_full):
        check_cancelled(self.stop)
        if not block:
            return operation(*args, block=False)
        if timeout is not None and timeout < 0:
            raise ValueError('timeout must be non-negative')
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            check_cancelled(self.stop)
            delay = .05 if deadline is None else max(0, min(.05, deadline - time.monotonic()))
            try:
                return operation(*args, timeout=delay)
            except empty_or_full:
                if deadline is not None and time.monotonic() >= deadline:
                    raise

    def put(self, item, block=True, timeout=None):
        return self._operation(super().put, (item,), block, timeout, queue.Full)

    def get(self, block=True, timeout=None):
        return self._operation(super().get, (), block, timeout, queue.Empty)


def frame_queue_limit(requested, width, height, *, max_bytes=DEFAULT_FRAME_BUFFER_BYTES,
                      queues=1, bytes_per_pixel=5):
    """Conservative native-size limit; one oversized frame must still progress.

    A detector packet may retain BGR, gray and previous-gray uint8 pixels.
    Prefetch callers pass three bytes per pixel and share one budget across
    their admitted chunk queues. Decoder-internal buffers, active batches and
    model memory are outside this bound; this is not a process RSS limit.
    """
    for value in (width, height, max_bytes, queues, bytes_per_pixel):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Decode buffer dimensions and byte budget must be positive integers")
    return max(1, min(int(requested), max_bytes // (width * height * bytes_per_pixel * queues)))
