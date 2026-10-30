"""Opt-in Comment-only presentation publisher with recoverable byte receipts.

The semantic publisher, rule identity, event paths and model results are reused.
Old publication receipts describe their historical bytes, retained in this local
journal; they are never rewritten to pretend that formatting did not change.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from visioncortex.comment_readability import readings_document, readings_text, transcript_text

from .timeline_publication import TimelinePublication


REVISION = 'comment-readable-20261030.1'
MAX_BYTES = 16 * 1024**2
MAX_HISTORY_BYTES = 128 * 1024**2


def supported(relative):
    parts = PurePosixPath(relative).parts
    return (not PurePosixPath(relative).is_absolute() and '..' not in parts and '\\' not in relative and len(parts) >= 5
            and parts[1] == 'Comment'
            and ((parts[2] == 'Audio' and parts[-1] == 'Transcript.txt')
                 or (len(parts) >= 6 and parts[2] == 'OCR' and parts[3] in ('VoicePhotoReadings', 'VideoReadings')
                     and parts[-1] in ('Readings.json', 'UnboundReadings.json'))))


class CommentPresentation:
    def __init__(self, history):
        self.history = Path(history)
        if str(self.history).startswith('/mnt/'):
            raise ValueError('Active Comment presentation journal must be local')
        self.history.mkdir(parents=True, exist_ok=True)

    def write(self, root, relative, canonical, guard):
        if not supported(relative):
            raise ValueError('Presentation writes are restricted to existing Comment members')
        if len(canonical) > MAX_BYTES:
            raise ValueError('Comment presentation exceeds byte bound')
        if relative.endswith('Transcript.txt'):
            source = TimelinePublication._read_bytes(root, relative[:-3] + 'json', guard)
            if source is None:
                raise ValueError('Transcript presentation requires its saved JSON')
            presented = transcript_text(json.loads(source))
        else:
            source = None
            original = json.loads(canonical)
            value = readings_document(original)
            assert {k:v for k,v in value.items() if k != '中文摘要'} == {k:v for k,v in original.items() if k != '中文摘要'}
            presented = readings_text(original)
        with (self.history / 'writer.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = TimelinePublication._read_bytes(root, relative, guard)
            if previous == presented:
                return dict(relative_path=relative, sha256=hashlib.sha256(presented).hexdigest(), byte_size=len(presented),
                            presentation_revision=REVISION)
            if previous is not None and not relative.endswith('Transcript.txt'):
                old = json.loads(previous)
                if {k:v for k,v in old.items() if k != '中文摘要'} != {k:v for k,v in original.items() if k != '中文摘要'}:
                    raise ValueError('Presentation cannot replace changed machine evidence')
            if sum(p.stat().st_size for p in self.history.glob('*') if p.is_file()) + len(canonical) + len(previous or b'') > MAX_HISTORY_BYTES:
                raise ValueError('Comment presentation history capacity requires export')
            def save(raw):
                digest = hashlib.sha256(raw).hexdigest()
                path = self.history / (digest + '.bytes')
                if not path.exists():
                    with path.open('xb') as stream:
                        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
                if path.read_bytes() != raw:
                    raise ValueError('Comment history readback differs')
                return digest
            before = save(previous) if previous is not None else None
            canonical_sha = save(canonical)
            source_sha = save(source) if source is not None else None
            plan = dict(revision=REVISION, relative_path=relative, root=str(root),
                        previous_sha256=before, canonical_sha256=canonical_sha, transcript_json_sha256=source_sha,
                        sha256=hashlib.sha256(presented).hexdigest(), byte_size=len(presented),
                        at=datetime.now(timezone.utc).isoformat())
            self._append(dict(plan, status='prepared'))
            expected = (before, len(previous)) if previous is not None else None
            if source is not None and TimelinePublication._read_bytes(root, relative[:-3] + 'json', guard) != source:
                raise ValueError('Transcript source changed before presentation publication')
            result = TimelinePublication._write(root, relative, presented, guard, replace_expected=expected)
            self._append(dict(plan, status='completed'))
            return dict(result, presentation_revision=REVISION, canonical_sha256=canonical_sha)

    def _append(self, value):
        with (self.history / 'Journal.jsonl').open('ab') as stream:
            stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode())
            stream.flush(); os.fsync(stream.fileno())


class CommentPresentationPublication(TimelinePublication):
    """Explicit dependency injection for existing production publication only."""
    def __init__(self, runtime, settings, *, history):
        super().__init__(runtime, settings)
        self.presentation = CommentPresentation(history)

    def _write(self, root, relative, raw, guard, *, replace_expected=None):
        if supported(relative):
            return self.presentation.write(root, relative, raw, guard)
        return TimelinePublication._write(root, relative, raw, guard, replace_expected=replace_expected)
