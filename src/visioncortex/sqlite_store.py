"""Owned SQLite connections: commit/rollback AND close at every boundary."""
from contextlib import contextmanager
import sqlite3


@contextmanager
def connection(path, *, timeout=30.0, readonly=False):
    target = path.absolute().as_uri() + '?mode=ro' if readonly else str(path)
    db = sqlite3.connect(target, uri=readonly, timeout=timeout)
    try:
        db.row_factory = sqlite3.Row
        db.execute(f'PRAGMA busy_timeout={int(timeout * 1000)}')
        db.execute('PRAGMA foreign_keys=ON')
        if not readonly:
            db.execute('PRAGMA synchronous=FULL')
        if readonly:
            db.execute('BEGIN')
        with db:
            yield db
    finally:
        db.close()
