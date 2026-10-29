from contextlib import contextmanager, nullcontext
import importlib.util
from pathlib import Path
import sqlite3

from repo_paths import ROOT


def test_historical_time_labels_do_not_serialize_unrelated_device_days(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('comment_scope', ROOT/'tools/migrations/move_comment_evidence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = '2026-10-29_camera'
    runtime = tmp_path/'Runtime/device-day'
    runtime.mkdir(parents=True)
    label = '12-14-00.000000_12-29-00.000000'
    real = 'a'*24
    for name in (label, real):
        folder = tmp_path/'Archives'/archive/'Comment/.Evidence/Recognition'/name
        folder.mkdir(parents=True)
        (folder/'Transcript.txt').write_text('fixture')
    with sqlite3.connect(runtime/'receipt-projection-v2.sqlite3') as db:
        db.execute('CREATE TABLE receipts(day TEXT, id TEXT)')
        db.execute('INSERT INTO receipts VALUES (?,?)', (archive, real))
    locks = []
    @contextmanager
    def exclusive(path):
        locks.append(Path(path).name)
        yield
    monkeypatch.setattr(module, 'exclusive', exclusive)
    monkeypatch.setattr(module, 'storage_guard', lambda _: nullcontext())
    config = {'storage': {'archive_root': str(tmp_path/'Archives'),
        'local_cache_root': str(tmp_path/'Backend'), 'local_runtime_root': str(tmp_path/'Runtime')}}
    module.run(config, archive)
    assert locks == [archive+'.index.lock', *[real+'.'+stage+'.lock' for stage in
        ('all', 'retention', 'vision', 'stt', 'understanding', 'report')]]
    # A real identity in the authoritative projection always keeps its lock.
    with sqlite3.connect(runtime/'receipt-projection-v2.sqlite3') as db:
        db.execute('INSERT INTO receipts VALUES (?,?)', (archive, label))
    locks.clear()
    module.run(config, archive)
    assert label+'.stt.lock' in locks
