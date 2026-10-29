"""Path-only binding of pinned VC publishers to Comment/Audio.

The source components remain byte-for-byte unchanged. A separate import
namespace compiles those same sources with explicit, checked path literals;
recognition, selection and schemas are not replaced.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import threading
from pathlib import Path

from .components import ROOT

PACKAGE = 'visioncortex_lab_comment'
SOURCE = ROOT / 'components/lab/visioncortex_pipeline/src/visioncortex_lab_pipeline'
_LOCK = threading.RLock()

# Each edit is an output path or its owning-directory allowlist. The pinned
# source manifest and required match counts reject upstream drift.
EDITS = {
    'device_day_contract': [('    "Comment",\n', '    "Comment/Audio",\n', 1)],
    'device_day_content': [
        ("root/'Comment'/name", "root/'Comment/Audio'/name", 1),
        ("relative = 'Comment/' +", "relative = 'Comment/Audio/' +", 1),
        ("[('Comment', audio_entries)", "[('Comment/Audio', audio_entries)", 1)],
    'device_day_stt': [("private_folder(layout, 'Comment', 'Recognition'", "private_folder(layout, 'Comment/Audio', 'Recognition'", 1)],
    'device_day_content_paths': [
        ("if directory not in {'ProcessedClips', 'MultimodalUnderstanding', 'LaboratoryDailyReport', 'Comment'}:",
         "if directory not in {'ProcessedClips', 'MultimodalUnderstanding', 'LaboratoryDailyReport', 'Comment', 'Comment/Audio'}:", 1)],
    'device_day_file_index': [
        ("root/'Comment'/'TimeIndex.json'", "root/'Comment/Audio'/'TimeIndex.json'", 1),
        ("'Comment/TimeIndex.json'", "'Comment/Audio/TimeIndex.json'", 1),
        ("'Comment/' + time_folder", "'Comment/Audio/' + time_folder", 1),
        ("'Comment/FieldEvents.json'", "'Comment/OCR/FieldEvents.json'", 2),
        ("'Comment/.Evidence/FieldRecognition'", "'Comment/OCR/.Evidence/FieldRecognition'", 1)],
    'application.field_archive_context': [
        ("name + '/Comment'", "name + '/Comment/OCR'", 1),
        ("'Comment/FieldEvents.json'", "'Comment/OCR/FieldEvents.json'", 1)],
    'device_day_synthesis': [("'Comment/Protocol.json'", "'Comment/Audio/Protocol.json'", 1)],
    'device_day_historical_publication': [
        ("'Comment/Index.json'", "'Comment/Audio/Index.json'", 1),
        ("'Comment/TimeIndex.json'", "'Comment/Audio/TimeIndex.json'", 1),
        ("[('Comment', comment_index)", "[('Comment/Audio', comment_index)", 1),
        ("'Comment/'+time_folder", "'Comment/Audio/'+time_folder", 1),
        ("'Comment/.Evidence/FieldRecognition/'", "'Comment/OCR/.Evidence/FieldRecognition/'", 1)],
}


def bind_source(module, source):
    for old, new, expected in EDITS.get(module, []):
        if source.count(old) != expected:
            raise ValueError('Pinned VC path binding no longer matches: '+module)
        source = source.replace(old, new)
    return source


class _Loader(importlib.abc.SourceLoader):
    def __init__(self, filename, module):
        self.filename, self.module = filename, module
    def get_filename(self, fullname):
        return str(self.filename)
    def get_data(self, path):
        return Path(path).read_bytes()
    def get_code(self, fullname):
        source = bind_source(self.module, self.filename.read_text())
        return compile(source, str(self.filename), 'exec', dont_inherit=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != PACKAGE and not fullname.startswith(PACKAGE+'.'):
            return None
        module = fullname.removeprefix(PACKAGE).removeprefix('.')
        candidate = SOURCE.joinpath(*module.split('.')) if module else SOURCE
        package = candidate.is_dir()
        filename = candidate/'__init__.py' if package else candidate.with_suffix('.py')
        if not filename.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, filename, loader=_Loader(filename, module),
                    submodule_search_locations=[str(candidate)] if package else None)


def module(name):
    with _LOCK:
        if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
        return importlib.import_module(PACKAGE+'.'+name)
