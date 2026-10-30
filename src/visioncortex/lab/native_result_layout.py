"""Optional result presentation namespace over the unchanged linked publisher.

The semantic adapter and release stay unchanged. This namespace has its own
build identity and only adds validated display fields at the existing I/O port.
"""
import importlib
import importlib.abc
import importlib.util
import sys
import threading
from pathlib import Path

from .native_comment_layout import SOURCE
from .native_linked_layout import bind_source as bind_linked

PACKAGE = 'visioncortex_lab_readable_results'
_LOCK = threading.RLock()


def bind_source(module, source):
    source = bind_linked(module, source)
    if module == 'device_day_write_guard':
        # Association validation and the alternate I/O namespace must see the
        # same held pin. Share only the context variable, never bypass checks.
        old = "CURRENT = ContextVar('device_day_publication_storage_guard', default=None)"
        if source.count(old) != 1:
            raise ValueError('Pinned result guard hook drift')
        source = source.replace(old,
            'from visioncortex.lab.native_linked_layout import module as _guard_owner\n'
            'CURRENT = _guard_owner("device_day_write_guard").CURRENT')
    if module == 'device_day_contract':
        edits = [
            ('        return json.load(handle)',
             '        from visioncortex.result_readability import canonical\n'
             '        return canonical(json.load(handle))'),
            ('def atomic_json(path: Path, value: Any) -> None:\n',
             'def atomic_json(path: Path, value: Any) -> None:\n'
             '    import os\n'
             '    if os.environ.get("VISIONCORTEX_READABLE_RESULTS") == "1":\n'
             '        from visioncortex.result_readability import kind, present\n'
             '        category = kind(path)\n'
             '        if category:\n'
             '            value = present(value, category)\n')]
        for old, new in edits:
            if source.count(old) != 1:
                raise ValueError('Pinned result presentation hook drift: ' + module)
            source = source.replace(old, new)
    return source


class _Loader(importlib.abc.SourceLoader):
    def __init__(self, filename, name): self.filename, self.name = filename, name
    def get_filename(self, fullname): return str(self.filename)
    def get_data(self, path): return Path(path).read_bytes()
    def get_code(self, fullname):
        return compile(bind_source(self.name, self.filename.read_text()), str(self.filename), 'exec', dont_inherit=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != PACKAGE and not fullname.startswith(PACKAGE + '.'):
            return None
        name = fullname.removeprefix(PACKAGE).removeprefix('.')
        candidate = SOURCE.joinpath(*name.split('.')) if name else SOURCE
        package = candidate.is_dir()
        filename = candidate/'__init__.py' if package else candidate.with_suffix('.py')
        if not filename.is_file(): return None
        return importlib.util.spec_from_file_location(fullname, filename, loader=_Loader(filename, name),
            submodule_search_locations=[str(candidate)] if package else None)


def module(name):
    with _LOCK:
        if not any(isinstance(f, _Finder) for f in sys.meta_path): sys.meta_path.insert(0, _Finder())
        return importlib.import_module(PACKAGE + '.' + name)
