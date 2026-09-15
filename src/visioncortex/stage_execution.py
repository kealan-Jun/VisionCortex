"""One stage boundary and dependency vocabulary, independent of archive layout."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
import time

from .runtime_control import check_cancelled, resource_slot


@dataclass(frozen=True)
class Stage:
    parents: tuple[str, ...]
    resource: str | None = None


STAGES = {
    'retention': Stage((), 'storage'),
    'alignment': Stage(('retention',), 'cpu'),
    'vision': Stage(('retention',)),
    'stt': Stage(('retention',), 'stt'),
    'understanding': Stage(('vision', 'stt')),
    'report': Stage(('understanding',), 'cpu'),
    'publication': Stage(('vision',), 'storage'),
}


def required_stages(stage):
    wanted = {stage}
    for _ in STAGES:
        wanted.update(parent for name in tuple(wanted) for parent in STAGES[name].parents)
    return [name for name in STAGES if name in wanted]


class StageExecutor:
    """Run the caller's existing stage implementation; never invent a receipt."""
    def __init__(self, config, observer=None):
        self.config, self.observer = config, observer or (lambda event: None)
        self.pool = None

    def run(self, stage, function, *args, **kwargs):
        check_cancelled()
        begun = time.monotonic()
        self.observer({'stage': stage, 'status': 'running'})
        try:
            resource = STAGES[stage].resource
            with resource_slot(self.config, resource):
                result = function(*args, **kwargs)
            check_cancelled()
        except Exception as exc:
            self.observer({'stage': stage, 'status': 'failed', 'error_type': type(exc).__name__})
            raise
        self.observer({'stage': stage, 'status': 'completed', 'wall_seconds': time.monotonic()-begun})
        return result

    def submit(self, stage, function, *args, **kwargs):
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='analysis-stage')
        context = copy_context()
        return self.pool.submit(context.run, self.run, stage, function, *args, **kwargs)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
