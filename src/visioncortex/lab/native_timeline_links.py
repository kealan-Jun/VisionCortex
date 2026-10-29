"""Read-only native builder links to immutable Timeline evidence sidecars.

Associations never become native model steps/findings or generate clips. The
native publisher remains the only writer of its Understanding and daily JSON.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
import stat
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import UUID

from .components import ROOT, pipeline_module
from .domain import clean
from .media import safe_path
from .storage import is_nas

SCHEMA = 'vc-lab-native-timeline-associations/1'
REFERENCE_SCHEMA = 'vc-lab-native-timeline-reference/1'
MAX_BYTES = 2 * 1024**2
MAX_DIRECTORY_ENTRIES = 1024
_NAME = re.compile(r'r(\d{6,})-([0-9a-f]{16})-([0-9a-f]{12})\.json\Z')
_SHA = re.compile('[0-9a-f]{64}\\Z')
_COMMITTED = ContextVar('vc_lab_verified_committed_association', default=None)


class PendingCommittedAssociation(OSError):
    """The newest filesystem candidate is not the caller's committed receipt."""


def association_path(archive, revision, snapshot_fingerprint, rules_sha256):
    _archive(archive)
    if type(revision) is not int or revision < 1 or not _sha(snapshot_fingerprint) or not _sha(rules_sha256):
        raise ValueError('Invalid association revision or fingerprint')
    return f'{archive}/LaboratoryDailyReport/Timeline/Associations/r{revision:06d}-{snapshot_fingerprint[:16]}-{rules_sha256[:12]}.json'


def _sha(value):
    return isinstance(value, str) and bool(_SHA.fullmatch(value))


def _archive(value):
    if type(value) is not str:raise ValueError('An exact native camera/day archive is required')
    validated=pipeline_module('device_day_contract').validate_archive_name(value)
    return validated[:10],validated[11:]


def build_sidecar(*, archive, customer_id, revision, snapshot_fingerprint, rules, canonical_report, activities, person_projection, media_sources=None,
                  artifact_refs=None, operation_projection=None, bindings_latest_path=None):
    day, camera = _archive(archive)
    association_path(archive, revision, snapshot_fingerprint, rules['sha256'])
    UUID(str(customer_id))
    by_activity = {}
    for person in (person_projection or {}).get('persons', []):
        for event in person.get('events', []) + person.get('untimed_events', []):
            by_activity.setdefault(event['activity_id'], []).append(dict(person_id=person['person_id'],
                attribution=event.get('person_attribution', [])))
    observations = []
    for activity in activities:
        payload = activity.get('payload') or {}
        observations.append(clean(dict(activity_id=activity['id'],kind=activity['kind'],source_type=activity['source_type'],
            start_at=activity.get('start_at'),end_at=activity.get('end_at'),evidence=activity.get('evidence', []),
            run_ids=activity.get('run_ids', []),sample_id=activity.get('sample_id'),person_id=activity.get('person_id'),
            text=activity.get('text_content'),text_is_untrusted_evidence_data=True,model_revision=activity.get('model_revision'),
            operator_scan_event=payload.get('operator_scan_event'),operator_context=payload.get('operator_context'),
            operator_scan_snapshot=payload.get('operator_scan'),scan_intake=payload.get('scan_intake'),
            capture_equipment_intake=payload.get('capture_equipment_intake'),
            qr_observation=({k:payload['video_qr'].get(k) for k in (
                'schema_version','source_sha256','media_id','media_version','source_stream_key','source_session_key',
                'original_decoded_frame_count','qr_decode_calls','world_clock_status','coverage','identity_status','operation_status')}
                if payload.get('video_qr') else None),
            comment_transition=payload.get('comment_transition'),retained_clip=payload.get('retained_clip'),
            value_candidate=payload.get('value_candidate'),
            original_source_spans=payload.get('original_source_spans', []),fields=payload.get('fields', []),
            association_revisions=activity.get('association_revisions', []),person_roles=by_activity.get(activity['id'], []))))
    result = dict(schema_version=SCHEMA,archive=archive,day=day,camera_key=camera,customer_id=str(customer_id),
        snapshot_revision=revision,snapshot_fingerprint=snapshot_fingerprint,processing_rules=rules,
        report_scope='camera_evidence_and_explicit_operator_scope_only',observations=observations,media_sources=clean(media_sources or {}),
        native_entry_refresh_status='pending_native_publication',
        cross_person_projection=dict(reference=canonical_report,reference_base='archive_root',scope='customer_all_streams_not_camera_narrative',
                                     json_pointer='/document/person_timelines'),
        interpretation=dict(status='associations_only',native_model_steps_generated=False,native_findings_generated=False,
                            physical_clip_generation=False,speaker_attribution='not_established',
                            unknown_times_or_sources_not_assigned_to_segments=True))
    if artifact_refs is not None:result['comment_artifacts']=clean(artifact_refs)
    if operation_projection is not None:result['operation_segments']=clean(operation_projection)
    if bindings_latest_path is not None:result['bindings_latest_path']=bindings_latest_path
    if len((json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()) > MAX_BYTES:
        raise ValueError('Native Timeline associations exceed their 2MiB bound')
    return result


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:raise ValueError('Duplicate association JSON key')
        value[key] = item
    return value


@contextmanager
def _association_directory(root):
    """No directory creation: NAS traversal stays on the original pinned mount."""
    relative = 'LaboratoryDailyReport/Timeline/Associations'
    descriptor = None
    guard = None
    try:
        if is_nas(root):
            from .native_linked_layout import module
            guard = module('device_day_write_guard').CURRENT.get()
            if guard is None:raise OSError('Native association NAS reads require the current pinned guard')
            guard.validate()
            mount, canonical = guard.mount_for(root / relative)
            if mount is None:raise OSError('Native association NAS path has no pinned mount')
            descriptor = os.dup(mount.descriptor)
            for part in canonical.relative_to(mount.path).parts:
                guard.validate(volume=False)
                try:
                    child = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
                except FileNotFoundError:
                    guard.validate()
                    yield None,guard
                    return
                os.close(descriptor);descriptor=child
                if os.fstat(descriptor).st_dev!=mount.snapshot['device']:raise OSError('Association path crossed an unverified filesystem')
        else:
            folder = safe_path(root,relative,must_exist=False)
            try:descriptor = os.open(folder,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            except FileNotFoundError:
                yield None,None
                return
        yield descriptor,guard
        if guard is not None:guard.validate()
    finally:
        if descriptor is not None:os.close(descriptor)


def _selected(root, archive):
    """Current rule release only; ambiguity/corruption fails before a main write."""
    day,camera = _archive(archive)
    root=Path(root)
    if root.name!=archive:raise ValueError('Native association root/archive mismatch')
    rules_sha = hashlib.sha256((ROOT/'contracts/lab/rules.lock.json').read_bytes()).hexdigest()
    with _association_directory(root) as (directory,guard):
        if directory is None:return None
        candidates=[]
        with os.scandir(directory) as entries:
            for count,entry in enumerate(entries,1):
                if count>MAX_DIRECTORY_ENTRIES:raise ValueError('Association directory exceeds its entry bound')
                match=_NAME.fullmatch(entry.name)
                if match and match[3]==rules_sha[:12]:candidates.append((int(match[1]),entry.name,match))
        if not candidates:return None
        highest=max(row[0] for row in candidates)
        candidates=[row for row in candidates if row[0]==highest]
        if len(candidates)!=1:raise ValueError('Conflicting native association revision')
        revision,name,match=candidates[0]
        if guard is not None:guard.validate(volume=False)
        with os.fdopen(os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory),'rb') as handle:
            before=os.fstat(handle.fileno())
            if guard is not None and before.st_dev!=os.fstat(directory).st_dev:raise OSError('Association file crossed an unverified filesystem')
            if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_BYTES:raise ValueError('Association file exceeds regular-file bound')
            raw=handle.read(MAX_BYTES+1);after=os.fstat(handle.fileno())
        if (len(raw)>MAX_BYTES or len(raw)!=before.st_size or
                tuple(getattr(before,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')) !=
                tuple(getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))):
            raise ValueError('Association source changed during bounded read')
        if guard is not None:guard.validate()
    digest,size=hashlib.sha256(raw).hexdigest(),len(raw)
    value = json.loads(raw,object_pairs_hook=_unique,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite association JSON')))
    if (value.get('schema_version') != SCHEMA or value.get('archive') != archive or value.get('day') != day or
            value.get('camera_key') != camera or type(value.get('snapshot_revision')) is not int or
            value['snapshot_revision'] != revision or not _sha(value.get('snapshot_fingerprint')) or
            value['snapshot_fingerprint'][:16] != match[2] or value.get('processing_rules',{}).get('sha256') != rules_sha or
            value.get('report_scope') != 'camera_evidence_and_explicit_operator_scope_only' or
            not isinstance(value.get('observations'),list)):
        raise ValueError('Native association release/archive/snapshot pin mismatch')
    UUID(value['customer_id'])
    reference = dict(schema_version=REFERENCE_SCHEMA,sidecar='LaboratoryDailyReport/Timeline/Associations/'+name,sha256=digest,byte_size=size,
        snapshot_revision=revision,snapshot_fingerprint=value['snapshot_fingerprint'],processing_rules=value['processing_rules'],
        scope='camera_associations_not_native_model_synthesis',native_entry_refresh_status='pending_committed_verification',cross_person_projection=value['cross_person_projection'],
        model_steps_generated=False,findings_generated=False,clip_created=False)
    if value.get('bindings_latest_path'):
        reference['comment_bindings']=value['bindings_latest_path']
    return value, reference


def association_revision(root, archive):
    selected = _verified_selected(root,archive)
    if selected is None:return ()
    ref=selected[1]
    return (('vc-lab-native-associations/1',ref['sidecar'],ref['sha256'],ref['snapshot_revision'],ref['snapshot_fingerprint']),)


def attach_day(document, root, archive):
    selected = _verified_selected(root,archive)
    return document if selected is None else dict(document,**{'关联时间线':selected[1]})


def attach_segment(document, root, index, segment):
    selected = _verified_selected(root,index['archive'])
    if selected is None:return document
    value, reference = selected
    source_sha=(segment.get('source_ref') or {}).get('sha256')
    matched=[]
    if _sha(source_sha):
        lo, hi = Decimal(str(segment['start_ms']))/1000, Decimal(str(segment['end_ms']))/1000
        for activity in value['observations']:
            for evidence in activity.get('evidence', []):
                # Exact original source and offset, never filename or world-time proximity.
                source=(value.get('media_sources') or {}).get(str(evidence.get('media_id')), {})
                if (evidence.get('verified_sha256') != source_sha or source.get('source_sha256') != source_sha or
                        source.get('source_relative_path') != index['archive']+'/'+str((segment.get('source_ref') or {}).get('path'))):continue
                try:
                    left,right=Decimal(str(evidence['start_seconds'])),Decimal(str(evidence['end_seconds']))
                    if not left.is_finite() or not right.is_finite() or left<0 or right<left:continue
                except (KeyError,InvalidOperation,ValueError):continue
                if left<hi and (right>lo or left==right and lo<=left):
                    matched.append(dict(activity_id=activity['activity_id'],evidence_id=evidence['id'],
                        source_sha256=source_sha,start_seconds=str(left),end_seconds=str(right)))
    return dict(document,**{'关联时间线':dict(reference,segment_evidence_links=matched,
        segment_association_status='source_pinned_overlap_is_association_only' if matched else 'pending_no_exact_source_evidence',
        native_model_status=document.get('status','pending'))})


def _pin(root, archive, selected):
    value, ref = selected
    return (str(Path(root).absolute()), archive, value['customer_id'], ref['sidecar'],
            ref['sha256'], ref['byte_size'], ref['snapshot_revision'], ref['snapshot_fingerprint'])


@contextmanager
def verified_committed_association(root, archive, *, customer_id, receipt):
    """Caller supplies an already committed, customer-scoped DB receipt.

    No credentials or database are discovered here. Recheck exact file identity
    at entry and at every builder hook; a newer uncommitted candidate is pending.
    The task-local permission ends even when a builder or writer raises.
    """
    selected = _selected(root, archive)
    if selected is None:
        raise PendingCommittedAssociation('No committed association candidate')
    value, reference = selected
    if (value['customer_id'] != str(customer_id)
            or archive + '/' + reference['sidecar'] != receipt['relative_path']
            or reference['sha256'] != receipt['sha256']
            or reference['byte_size'] != receipt['byte_size']):
        raise PendingCommittedAssociation('Newest association is not the committed receipt')
    token = _COMMITTED.set(_pin(root, archive, selected))
    try:
        yield value, dict(reference,native_entry_refresh_status='included_by_native_builder')
    finally:
        _COMMITTED.reset(token)


def _verified_selected(root, archive):
    allowed = _COMMITTED.get()
    if allowed is None or allowed[:2] != (str(Path(root).absolute()), archive):
        return None
    selected = _selected(root, archive)
    if selected is None or _pin(root, archive, selected) != allowed:
        raise PendingCommittedAssociation('Association changed after committed verification')
    return selected[0], dict(selected[1],native_entry_refresh_status='included_by_native_builder')
