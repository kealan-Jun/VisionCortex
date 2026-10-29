from ._support import research_helpers
import hashlib
import importlib.util
import sys

import pytest


SCRIPTS = research_helpers()
SPEC = importlib.util.spec_from_file_location('research_archive_guard', SCRIPTS / 'archive_memory_research.py')
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(SCRIPTS))


def test_owned_public_progress_history_can_be_exported(tmp_path):
    raw = b'{"streams":1,"phase":"native_caption","processed_video_time":30}\n'
    path = tmp_path / 'streams-1' / 'ProgressHistory.jsonl'
    path.parent.mkdir()
    path.write_bytes(raw)
    result = MODULE.inventory(tmp_path, 1024)
    assert result[0]['path'] == 'streams-1/ProgressHistory.jsonl'
    assert result[0]['sha256'] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('relative', [
    '.ssh/ProgressHistory.jsonl',
    'SSHConfig.json',
    'runtime-ssh/ProgressHistory.jsonl',
    'ProgressHistory.jsonl/Config.json',
    'ProgressHistorySSH.jsonl',
    '.env.json',
    'weights/ProgressHistory.jsonl',
    'source.mp4',
])
def test_public_telemetry_exception_never_admits_private_sources(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='public-result allowlist'):
        MODULE.inventory(tmp_path, 1024)
