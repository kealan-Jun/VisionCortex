"""Bound decoded pixel buffers by bytes as well as configured frame counts."""

DEFAULT_FRAME_BUFFER_BYTES = 64 * 1024 * 1024


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
