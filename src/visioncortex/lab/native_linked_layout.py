"""Explicit builder-hook release over immutable native publisher sources.

The path-only namespace remains unchanged. Only pure document builders and the
readable revision token gain evidence sidecar references; their existing writer
and model/selection code remain the owners of native outputs.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import threading
from pathlib import Path

from .native_comment_layout import SOURCE, bind_source as bind_paths

PACKAGE = 'visioncortex_lab_linked'
_LOCK = threading.RLock()
EDITS = {
    'device_day_content': [
        ('    return tuple(revisions)\n','    from visioncortex.lab.native_timeline_links import association_revision\n    return tuple(revisions) + association_revision(root, index["archive"])\n',1),
        ('    return value, retention\n\n\ndef visual_entry',
         '    from visioncortex.lab.native_timeline_links import attach_segment\n    value = attach_segment(value, root, index, segment)\n    return value, retention\n\n\ndef visual_entry',1)],
    'device_day_reports': [
        ("    return {'全天综合理解': synthesis", "    document = {'全天综合理解': synthesis",1),
        ("'input_index_digest': digest(index)}\n\n\ndef publish_day",
         "'input_index_digest': digest(index)}\n    from visioncortex.lab.native_timeline_links import attach_day\n    return attach_day(document, layout.root, layout.name)\n\n\ndef publish_day",1)],
}


def bind_source(module, source):
    source = bind_paths(module,source)
    for old,new,count in EDITS.get(module,[]):
        if source.count(old)!=count:raise ValueError('Pinned native builder hook no longer matches: '+module)
        source=source.replace(old,new)
    return source


class _Loader(importlib.abc.SourceLoader):
    def __init__(self,filename,module):self.filename,self.module=filename,module
    def get_filename(self,fullname):return str(self.filename)
    def get_data(self,path):return Path(path).read_bytes()
    def get_code(self,fullname):
        return compile(bind_source(self.module,self.filename.read_text()),str(self.filename),'exec',dont_inherit=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=PACKAGE and not fullname.startswith(PACKAGE+'.'):return None
        name=fullname.removeprefix(PACKAGE).removeprefix('.')
        candidate=SOURCE.joinpath(*name.split('.')) if name else SOURCE
        package=candidate.is_dir();filename=candidate/'__init__.py' if package else candidate.with_suffix('.py')
        if not filename.is_file():return None
        return importlib.util.spec_from_file_location(fullname,filename,loader=_Loader(filename,name),
            submodule_search_locations=[str(candidate)] if package else None)


def module(name):
    with _LOCK:
        if not any(isinstance(finder,_Finder) for finder in sys.meta_path):sys.meta_path.insert(0,_Finder())
        return importlib.import_module(PACKAGE+'.'+name)
