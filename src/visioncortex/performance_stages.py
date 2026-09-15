"""Measured component work; overlapping workers are never added to wall time."""

import threading
import time
from contextlib import contextmanager


class StageTimings:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def add(self, key, value):
        with self.lock:
            self.values[key] = self.values.get(key, 0) + value

    @contextmanager
    def measure(self, key):
        started = time.perf_counter()
        try:
            from .device_day_activity import phase
            with phase(key):
                yield
        finally:
            self.add(key, time.perf_counter() - started)

    def frames(self, frames):
        iterator = iter(frames)
        try:
            while True:
                with self.measure("read_decode_supply_seconds"):
                    try:
                        frame = next(iterator)
                    except StopIteration:
                        return
                self.add("decoded_frames", 1)
                self.add("decoded_pixel_bytes", frame[2].nbytes)
                yield frame
        finally:
            if hasattr(iterator, "close"):
                iterator.close()

    def snapshot(self):
        with self.lock:
            return {key: round(value, 6) for key, value in self.values.items()}


class MeasuredWriter:
    def __init__(self, writer, timings):
        self.writer, self.timings = writer, timings

    def write(self, text):
        with self.timings.measure("ledger_write_seconds"):
            return self.writer.write(text)

    def flush(self):
        with self.timings.measure("ledger_write_seconds"):
            return self.writer.flush()

    def close(self):
        with self.timings.measure("ledger_write_seconds"):
            return self.writer.close()


def prediction_timings(predictions, timings):
    """Backend-reported milliseconds per image, including engine padding."""
    for prediction in predictions:
        speed = getattr(prediction, "speed", None) or {}
        for key in ("preprocess", "inference", "postprocess"):
            value = speed.get(key)
            if isinstance(value, (int, float)) and 0 <= value < float("inf"):
                timings.add(f"model_{key}_seconds", value / 1000)
                timings.add(f"model_{key}_profiled_frames", 1)
