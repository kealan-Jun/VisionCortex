"""Compose fair vision admission with the unchanged device-day model owner."""
from .runtime_control import CURRENT, execution_context


class FairVisionBackend:
    """Only vision uses ordinary bounded pools during a fair history turn.

    The adapter keeps model arguments/results and every other public port with
    its original owner. It adds no model, receipt, cache or provider algorithm.
    """

    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def vision(self, layout, retention, key):
        parent = CURRENT.get()
        if parent.background_fair and parent.source == 'device_day_backfill':
            # Preserve the exact stop/yield/resource context. The different
            # admission source selects the existing bounded foreground pool;
            # the model still owns validation, scanning and output identity.
            with execution_context(job_id=parent.job_id, source='device_day_fair_vision',
                                   priority=parent.priority, stop=parent.stop,
                                   yield_signal=parent.yield_signal):
                return self.backend.vision(layout, retention, key)
        return self.backend.vision(layout, retention, key)

    def transcribe(self, layout, retention, key):
        return self.backend.transcribe(layout, retention, key)

    def understand(self, layout, recording, vision, context, key):
        return self.backend.understand(layout, recording, vision, context, key)


def create_backend(config):
    from .device_day_models import DeviceDayModels
    return FairVisionBackend(DeviceDayModels(config))
