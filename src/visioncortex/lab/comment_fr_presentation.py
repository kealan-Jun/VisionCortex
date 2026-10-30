"""Readability of the existing FR projection; original FR members are untouched."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import PurePosixPath

from visioncortex.comment_readability import readings_text
from .comment_presentation import CommentPresentation
from .timeline_publication import TimelinePublication


def evidence_body(document):
    """Only derived navigation checksums may differ after text presentation."""
    value=deepcopy(document)
    for key in ('readings','unbound_readings'):
        if isinstance(value.get(key),dict):
            for field in ('sha256','byte_size','size_bytes'):
                value[key].pop(field,None)
    return value


class CommentFRPublication:
    _read_bytes=staticmethod(TimelinePublication._read_bytes)

    def __init__(self, *, history):
        self.presentation=CommentPresentation(history)

    @staticmethod
    def _json(document):
        if isinstance(document,dict) and document.get('schema') in ('field-numeric-observations/1','field-unbound-readings/1'):
            return readings_text(document)
        return TimelinePublication._json(document)

    def _write(self,root,relative,raw,guard):
        if PurePosixPath(relative).name in ('Readings.json','UnboundReadings.json'):
            return self.presentation.write(root,relative,raw,guard)
        if PurePosixPath(relative).name!='Evidence.json':
            return TimelinePublication._write(root,relative,raw,guard)
        parts=PurePosixPath(relative).parts
        if len(parts)!=6 or parts[1:3]!=('Comment','OCR') or parts[3] not in ('VoicePhotoReadings','VideoReadings'):
            raise ValueError('FR evidence presentation escaped Comment')
        previous=TimelinePublication._read_bytes(root,relative,guard)
        if previous==raw:return dict(relative_path=relative,sha256=hashlib.sha256(raw).hexdigest(),byte_size=len(raw))
        if previous is not None and evidence_body(json.loads(previous))!=evidence_body(json.loads(raw)):
            raise ValueError('FR presentation cannot change source evidence or Result identity')
        # Verify navigation before replacing its current view; six-field Results
        # and original source receipts keep their exact historical bytes.
        document=json.loads(raw)
        for key in ('readings','unbound_readings'):
            item=document.get(key)
            if isinstance(item,dict) and item.get('sha256') and item.get('path'):
                member=TimelinePublication._read_bytes(root,item['path'],guard)
                if member is None or hashlib.sha256(member).hexdigest()!=item['sha256']:
                    raise ValueError('FR presented navigation checksum differs')
        before=hashlib.sha256(previous).hexdigest() if previous is not None else None
        if previous is not None:
            path=self.presentation.history/(before+'.bytes')
            if not path.exists():
                with path.open('xb') as stream:
                    stream.write(previous);stream.flush()
                    import os
                    os.fsync(stream.fileno())
            if path.read_bytes()!=previous:raise ValueError('FR evidence preimage differs')
        expected=(before,len(previous)) if previous is not None else None
        receipt=TimelinePublication._write(root,relative,raw,guard,replace_expected=expected)
        self.presentation._append(dict(status='completed_fr_navigation',relative_path=relative,previous_sha256=before,
            sha256=receipt['sha256'],at=datetime.now(timezone.utc).isoformat()))
        return receipt
