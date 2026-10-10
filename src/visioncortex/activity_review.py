"""Activity classification is separate from action evidence and completion."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ActivityAssessment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['experiment', 'equipment_organization', 'cleanup', 'preparation', 'uncertain']
    workflow_relation: Literal['standalone', 'part_of_experiment', 'unresolved']
    reason: str = Field(min_length=1, max_length=1200)


LABELS = {'experiment':'实验操作', 'equipment_organization':'器材整理', 'cleanup':'清洁收尾',
          'preparation':'准备活动', 'uncertain':'活动类型待确认'}
REVIEW_DIR = 'JSON-Config-Files/Activity-Reviews'


def assessment(group: dict) -> dict:
    raw = group.get('activity_assessment') or (group.get('model_understanding') or {}).get('activity_assessment')
    if not isinstance(raw, dict):
        raw = None
    try:
        result = ActivityAssessment.model_validate({k: raw[k] for k in ('kind', 'workflow_relation', 'reason')}).model_dump()
    except (ValueError, TypeError, KeyError):
        result = {'kind':'uncertain', 'workflow_relation':'unresolved', 'reason':'尚未区分实验操作与辅助活动'}
    auxiliary = result['kind'] in {'equipment_organization', 'cleanup', 'preparation'} and result['workflow_relation'] == 'standalone'
    return {**result, 'label': LABELS[result['kind']], 'is_auxiliary':auxiliary,
            'source': (raw or {}).get('source', 'model' if raw else 'not_reviewed'),
            'review_receipt': (raw or {}).get('review_receipt'),
            'scope':'activity_classification_only_not_action_ground_truth'}


def counts(groups: list[dict]) -> dict:
    rows = [assessment(g) for g in groups]
    auxiliary = sum(a['is_auxiliary'] for a in rows)
    return {'experiments':len(rows)-auxiliary, 'auxiliary_activities':auxiliary,
            'activity_records':len(rows), 'unclassified_activities':sum(a['kind']=='uncertain' for a in rows)}


def binding(root: Path, group: dict) -> str:
    identity = {k:group.get(k) for k in ('group_id', 'group_uid', 'global_start_ms', 'global_end_ms',
                                       'participating_views', 'archive_folder', 'key_event_ids')}
    source = root / 'JSON-Config-Files/input_manifest.yaml'
    identity['source_manifest_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def apply(root: Path, groups: list[dict]) -> list[dict]:
    output = copy.deepcopy(groups)
    by_binding = {binding(root, g):g for g in output}
    records = []
    for path in (root / REVIEW_DIR).glob('*.json'):
        try:
            records.append((json.loads(path.read_text(encoding='utf-8')), path))
        except (OSError, ValueError):
            continue
    for row, path in sorted(records, key=lambda pair: (pair[0].get('created_at',''), pair[1].name)):
        group = by_binding.get(row.get('binding'))
        if group is None:
            continue
        try:
            value = ActivityAssessment.model_validate(row['assessment']).model_dump()
        except (ValueError, KeyError, TypeError):
            continue
        group['activity_assessment'] = {**value, 'source':'user_review', 'review_receipt':path.relative_to(root).as_posix()}
    return output


def save(root: Path, group: dict, revision: str, value: dict) -> Path:
    if revision != binding(root, group):
        raise ValueError('片段或原输入已更新，请刷新后再修改分类')
    accepted = ActivityAssessment.model_validate(value).model_dump()
    path = root / REVIEW_DIR / (uuid4().hex[:16] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version':'visioncortex-activity-review/1', 'binding':revision,
               'group_id':group.get('group_id'), 'created_at':datetime.now(timezone.utc).isoformat(),
               'source':'user_review', 'assessment':accepted, 'previous_assessment':assessment(group),
               'scope':'activity_classification_only_not_action_ground_truth', 'source_media_modified':False}
    # Append-only review; the original group, events and media are unchanged.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return path
