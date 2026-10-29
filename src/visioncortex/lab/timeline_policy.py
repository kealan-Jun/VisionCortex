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
        if set(document) - {'schema_version', 'defaults', 'instruments', 'experiments', 'window_seconds', 'context_seconds','video_input_unit','preceding_context_seconds', 'multimodal_prior_context', 'operation_segments', 'automatic_equipment_qr'}:
            raise ValueError('Unknown timeline policy setting')
        prior = document.get('multimodal_prior_context')
        if prior is not None:
            if (type(prior) is not dict or set(prior) != {
                    'enabled', 'max_items', 'max_utf8_bytes', 'lookback_seconds'}
                    or type(prior['enabled']) is not bool
                    or type(prior['max_items']) is not int or not 1 <= prior['max_items'] <= 64
                    or type(prior['max_utf8_bytes']) is not int or not 1024 <= prior['max_utf8_bytes'] <= 32768
                    or type(prior['lookback_seconds']) is not int or not 1 <= prior['lookback_seconds'] <= 3600):
                raise ValueError('Explicit bounded multimodal prior policy required')
        segments = document.get('operation_segments')
        if segments is not None and (type(segments) is not dict
                or set(segments) != {'enabled', 'anchor_pairing_seconds'}
                or type(segments['enabled']) is not bool
                or type(segments['anchor_pairing_seconds']) is not int
                or not 1 <= segments['anchor_pairing_seconds'] <= 300):
            raise ValueError('Explicit bounded operation anchor pairing policy required')
        automatic = document.get('automatic_equipment_qr')
        if automatic is not None:
            required = {'enabled', 'revision', 'photo', 'video', 'video_window_seconds',
                        'max_packet_seconds', 'max_packet_tasks', 'max_admissions_per_tick',
                        'max_observations_per_tick', 'rotation_degrees_clockwise',
                        'orientation_basis', 'source_kinds', 'presence_interval_seconds'}
            if (type(automatic) is not dict or set(automatic) != required
                    or any(type(automatic[k]) is not bool for k in ('enabled', 'photo', 'video'))
                    or type(automatic['revision']) is not str or not 1 <= len(automatic['revision'].encode()) <= 48
                    or type(automatic['orientation_basis']) is not str
                    or not 1 <= len(automatic['orientation_basis'].encode()) <= 512
                    or type(automatic['rotation_degrees_clockwise']) is not int
                    or automatic['rotation_degrees_clockwise'] not in {0, 180}
                    or type(automatic['source_kinds']) is not list
                    or not 1 <= len(automatic['source_kinds']) <= 3
                    or any(type(k) is not str or k not in {'customer_capture','fixture','derived_replay'} for k in automatic['source_kinds'])
                    or len(set(automatic['source_kinds'])) != len(automatic['source_kinds'])):
                raise ValueError('Explicit generic equipment intake policy required')
            for name, maximum in (('video_window_seconds', 10), ('max_packet_seconds', 900),
                    ('max_packet_tasks', 90), ('max_admissions_per_tick', 20),
                    ('max_observations_per_tick', 600), ('presence_interval_seconds', 300)):
                if type(automatic[name]) is not int or not 1 <= automatic[name] <= maximum:
                    raise ValueError('Automatic equipment intake exceeds bounded policy: ' + name)
            if automatic['video'] and automatic['max_packet_tasks'] * automatic['video_window_seconds'] < automatic['max_packet_seconds']:
                raise ValueError('Automatic QR task budget cannot cover declared packet duration')
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
