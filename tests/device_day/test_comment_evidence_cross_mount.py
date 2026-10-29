import errno
import importlib.util
import os

import pytest

from tests.repo_paths import ROOT
from visioncortex.comment_evidence import backend_path
from visioncortex.device_day_contract import DeviceDayLayout, artifact, atomic_json, read_json
from visioncortex.device_day_verification import verify_artifact_cached


def tool():
    spec = importlib.util.spec_from_file_location('comment_move', ROOT/'tools/migrations/move_comment_evidence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('interrupt', [False, True])
def test_separate_pinned_mounts_preserve_backend_and_atomic_recovery_copy(tmp_path, monkeypatch, interrupt):
    day = DeviceDayLayout(tmp_path/'Archives', 'camera', 1793232000000000, tmp_path/'Backend')
    source = day.comments/'.Evidence/Recognition/r/Result.json'
    atomic_json(source, {'actual': True})
    ref = artifact(day.root, source)
    rename = os.rename
    moves = []
    def separate_mounts(left, right):
        if right.is_relative_to(day.backend_root):
            raise OSError(errno.EXDEV, 'owned separate bind mount fixture')
        rename(left, right)
        moves.append((left, right))
        if interrupt and len(moves) == 1:
            raise OSError(errno.EIO, 'interrupt after durable archive retirement')
    monkeypatch.setattr(os, 'rename', separate_mounts)
    module = tool()
    if interrupt:
        with pytest.raises(OSError, match='interrupt'):
            module.migrate_compatible(day, max_files=100, max_bytes=1024**2)
    value = module.migrate_compatible(day, max_files=100, max_bytes=1024**2)
    assert value[0]['status'] == 'completed' and not (day.comments/'.Evidence').exists()
    assert len(moves) == 1
    assert verify_artifact_cached(day.root, ref, backend_root=day.backend_root)
    recovery = day.root/value[0]['retirement_copy']['path']/'Recognition/r/Result.json'
    assert read_json(recovery) == {'actual': True}
    assert read_json(backend_path(day.root, ref['path'], day.backend_root)) == {'actual': True}
