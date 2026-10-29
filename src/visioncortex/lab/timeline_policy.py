"""Policies for our customer VC Lab's opt-in evidence timeline (no detector settings)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class RecordPolicy:
    # Daily archive closes a record, never the physical experiment or occupancy.
    daily_archive: bool = True
    inactivity_seconds: int | None = None
    arrival_grace_seconds: int = 0
    handover: str = 'independent'

    def __post_init__(self):
        if type(self.daily_archive) is not bool:
            raise ValueError('daily_archive must be boolean')
        for name in ('inactivity_seconds', 'arrival_grace_seconds'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Invalid record policy duration')
        if self.inactivity_seconds == 0 or self.arrival_grace_seconds is None:
            raise ValueError('Inactivity must be positive and grace must be specified')
        if not self.daily_archive and self.inactivity_seconds is None:
            raise ValueError('A finite automatic record closure policy is required')
        if self.handover not in {'independent', 'exclusive_position'}:
            raise ValueError('Unknown handover policy')


class TimelinePolicies:
    def __init__(self, document=None):
        document = document or {'schema_version': 'vc-lab-timeline-policy/1'}
        if document.get('schema_version') not in {'vc-lab-timeline-policy/1','vc-lab-timeline-policy/2'}:
            raise ValueError('Unknown timeline policy version')
        if set(document) - {'schema_version', 'defaults', 'instruments', 'experiments', 'window_seconds', 'context_seconds','video_input_unit','preceding_context_seconds'}:
            raise ValueError('Unknown timeline policy setting')
        self.document = document
        self.video_input_unit=document.get('video_input_unit','complete_retained_clip' if document['schema_version']=='vc-lab-timeline-policy/2' else 'window')
        self.preceding_context_seconds=document.get('preceding_context_seconds',600)
        if self.video_input_unit not in {'window','complete_retained_clip'} or (document['schema_version']=='vc-lab-timeline-policy/1' and self.video_input_unit!='window'):
            raise ValueError('Complete-clip policy requires the new version')
        if type(self.preceding_context_seconds) is not int or not 0<=self.preceding_context_seconds<=3600:
            raise ValueError('Preceding evidence context must be 0..3600 seconds')
        self.window_seconds = document.get('window_seconds', 30)
        self.context_seconds = document.get('context_seconds', 2)
        if type(self.window_seconds) is not int or not 1 <= self.window_seconds <= 900:
            raise ValueError('Window duration must be 1..900 seconds')
        if type(self.context_seconds) is not int or not 0 <= self.context_seconds <= 30:
            raise ValueError('Context duration must be 0..30 seconds')
        self.record()
        for key in document.get('experiments', {}):
            self.record(experiment_type=key)
        for key in document.get('instruments', {}):
            self.record(instrument_id=key)

    @classmethod
    def load(cls, path: Path | None):
        return cls(json.loads(path.read_text()) if path else None)

    def record(self, *, experiment_type=None, instrument_id=None):
        values = dict(self.document.get('defaults', {}))
        values.update(self.document.get('experiments', {}).get(experiment_type, {}))
        values.update(self.document.get('instruments', {}).get(str(instrument_id), {}))
        return asdict(RecordPolicy(**values))


ZONE = ZoneInfo('Asia/Shanghai')
