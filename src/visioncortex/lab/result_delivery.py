"""Explicit presentation-only composition; no alternate model execution owner."""
import importlib.util
from pathlib import Path
import threading

from .delivery import NativeDelivery

_LOCK = threading.RLock()
_BRIDGE = None


def refresh_class():
    """Reuse the pinned refresh algorithm with checked I/O dependency bindings.

    No module globals are replaced. A private module namespace loads the same
    bridge and uses the readable native I/O namespace. Semantic files on disk
    and their rule fingerprints remain untouched.
    """
    global _BRIDGE
    with _LOCK:
        if _BRIDGE is not None: return _BRIDGE.NativeRefresh
        path = Path(__file__).with_name('native_refresh.py')
        source = path.read_text()
        old = 'from .native_linked_layout import module'
        if source.count(old) != 1: raise ValueError('Native refresh import hook drift')
        source = source.replace(old, 'from .native_result_layout import module')
        if '    value = canonical(value)\n' not in source:
            old = '    return value, hashlib.sha256(raw).hexdigest(), len(raw)'
            if source.count(old) != 1: raise ValueError('Native refresh JSON hook drift')
            source = source.replace(old, '    from ..result_readability import canonical\n'
                '    value = canonical(value)\n' + old)
        spec = importlib.util.spec_from_loader('visioncortex.lab._readable_native_refresh', loader=None)
        bridge = importlib.util.module_from_spec(spec)
        exec(compile(source, str(path), 'exec', dont_inherit=True), bridge.__dict__)
        _BRIDGE = bridge
        return bridge.NativeRefresh


class ReadableNativeDelivery(NativeDelivery):
    def tick(self, limit=20):
        if self.settings.processing_profile != 'timeline-v1':
            return super().tick(limit)
        result = dict(created=0, dependencies_ready=0, profile='timeline-v1')
        if self.settings.timeline_publish:
            result['native_refresh'] = refresh_class()(self.core, self.settings).tick(limit)
            result['reports'] = result['native_refresh']['refreshed']
            result['errors'] = result['native_refresh']['errors']
        return result
