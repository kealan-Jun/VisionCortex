"""Opt-in research contracts for coarse indexing and evidence-backed fine review.

This module is deliberately not imported by TimelineRuntime. It never starts a
model, opens a customer source, resolves recycled labels or writes business facts.
All intervals are half-open integer microseconds on one explicit stream/session.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import math
from threading import RLock
import time
from typing import Callable

from .domain import fingerprint, instant

SCHEMA = 'vc-lab-coarse-fine-research/1'
TRIGGERS = frozenset({'action', 'audio', 'ocr', 'binding', 'uncertain',
                      'unfinished', 'conflict', 'manual', 'blind_audit'})
STAGES = ('load', 'read', 'decode', 'preprocess', 'encode', 'coarse_memory',
          'selection', 'fine_model', 'validation', 'database', 'publication')


def _us(value):
    if type(value) is not int or value < 0:
        raise ValueError('Nonnegative integer microseconds required')
    return value


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Pinned lowercase SHA256 required')
    return value


@dataclass(frozen=True)
class Packet:
    media_id: str
    source_sha256: str
    stream_id: str
    session_id: str
    captured_at: str
    duration_us: int
    clock_basis: str
    clock_uncertainty_us: int | None = None

    def __post_init__(self):
        _sha(self.source_sha256)
        if not all((self.media_id, self.stream_id, self.session_id, self.clock_basis)):
            raise ValueError('Explicit media, stream, session and clock qualification required')
        instant(self.captured_at)
        if _us(self.duration_us) == 0:
            raise ValueError('Packet duration must be positive; 15 minutes is not required')
        if self.clock_uncertainty_us is not None:
            _us(self.clock_uncertainty_us)

    @property
    def key(self):
        return self.stream_id, self.session_id

    @property
    def start_us(self):
        delta = instant(self.captured_at) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds

    @property
    def end_us(self):
        return self.start_us + self.duration_us


@dataclass(frozen=True)
class Span:
    media_id: str
    source_sha256: str
    start_us: int
    end_us: int

    def __post_init__(self):
        _sha(self.source_sha256)
        if _us(self.start_us) >= _us(self.end_us):
            raise ValueError('Evidence span must be nonempty')


@dataclass(frozen=True)
class Hint:
    hint_id: str
    span: Span
    kind: str
    text: str
    model_revision: str
    config_sha256: str
    uncertain: bool = False
    unfinished: bool = False
    conflicting: bool = False
    # Only externally attributable identities belong here, never latest-scan guesses.
    run_ids: tuple[str, ...] = ()
    binding_revision_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _sha(self.config_sha256)
        object.__setattr__(self, 'run_ids', tuple(self.run_ids))
        object.__setattr__(self, 'binding_revision_ids', tuple(self.binding_revision_ids))
        if not self.hint_id or not self.model_revision or not self.kind:
            raise ValueError('Versioned immutable coarse hint required')


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    span: Span
    kind: str
    evidence_id: str
    run_ids: tuple[str, ...] = ()
    binding_revision_ids: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'run_ids', tuple(self.run_ids))
        object.__setattr__(self, 'binding_revision_ids', tuple(self.binding_revision_ids))
        if self.kind not in TRIGGERS or not self.anchor_id or not self.evidence_id:
            raise ValueError('Known trigger with immutable evidence reference required')


@dataclass(frozen=True)
class Policy:
    before_us: int = 3000000
    after_us: int = 3000000
    max_fine_duration_us: int = 30000000
    audit_period_us: int | None = None
    audit_length_us: int = 2000000
    audit_seed: str = 'vc-lab-research-explicit-seed/1'

    def __post_init__(self):
        for value in (self.before_us, self.after_us, self.max_fine_duration_us, self.audit_length_us):
            _us(value)
        if not self.max_fine_duration_us or not self.audit_length_us:
            raise ValueError('Finite positive capacity and audit length required')
        if self.audit_period_us is not None and (not _us(self.audit_period_us) or self.audit_length_us > self.audit_period_us):
            raise ValueError('Audit length must fit its finite period')
        if not self.audit_seed:
            raise ValueError('Audit selection requires a reproducible seed')


@dataclass(frozen=True)
class FineTask:
    task_id: str
    stream_id: str
    session_id: str
    trigger_ids: tuple[str, ...]
    trigger_kinds: tuple[str, ...]
    source_spans: tuple[Span, ...]
    focal_span: Span
    hint_ids: tuple[str, ...]
    external_evidence_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    binding_revision_ids: tuple[str, ...]
    status: str
    attention_reasons: tuple[str, ...]
    context_gaps: tuple[tuple[int, int], ...]
    config_sha256: str
    source_packets_sha256: str

    @property
    def key(self):
        return self.stream_id, self.session_id

    def request(self):
        """Evidence references are mandatory; hints are explicitly unverified."""
        return dict(schema_version=SCHEMA, task=asdict(self),
                    input_policy=dict(temporal='every_decoded_source_frame', spatial='native_pixels',
                                      capacity_failure='needs_attention_no_automatic_identical_retry'),
                    coarse_hint_role='unverified_leads_not_observations_or_answers',
                    output_role='candidate_revision_only',
                    instructions='Read original visual evidence and available anchored audio/OCR; '
                    'correct or reject coarse leads. Cite source SHA and precise source offsets. '
                    'Do not infer unseen actions, exact readings, material use or physical completion.')


@dataclass(frozen=True)
class Plan:
    packets: tuple[Packet, ...]
    hints: tuple[Hint, ...]
    anchors: tuple[Anchor, ...]
    tasks: tuple[FineTask, ...]
    continuity: tuple[dict, ...]
    coverage: tuple[dict, ...]
    policy_sha256: str
    schema_version: str = SCHEMA


def _validate_span(span, by_id):
    packet = by_id.get(span.media_id)
    if packet is None or packet.source_sha256 != span.source_sha256 or span.end_us > packet.duration_us:
        raise ValueError('Evidence must match the exact pinned source and valid source offsets')
    return packet


def _missing(lo, hi, intervals):
    output, cursor = [], lo
    for start, end in sorted(intervals):
        if start > cursor:
            output.append((cursor, min(start, hi)))
        cursor = max(cursor, end)
    if cursor < hi:
        output.append((cursor, hi))
    return tuple((a, b) for a, b in output if a < b)


def build_plan(packets, hints=(), anchors=(), *, policy=None):
    """Plan workloads only. Packet boundaries and trigger windows are not experiments."""
    policy = policy or Policy()
    packets, hints, anchors = tuple(packets), tuple(hints), tuple(anchors)
    by_id = {p.media_id: p for p in packets}
    if len(by_id) != len(packets):
        raise ValueError('Media IDs are immutable and unique')
    if len({h.hint_id for h in hints}) != len(hints) or len({a.anchor_id for a in anchors}) != len(anchors):
        raise ValueError('Duplicate immutable evidence ID')
    if {h.hint_id for h in hints} & {a.anchor_id for a in anchors}:
        raise ValueError('Coarse and external evidence IDs require separate identities')
    packets = tuple(sorted(packets, key=lambda p: (p.key, p.start_us, p.media_id)))
    continuity, groups, seen, frontiers = [], {}, {}, {}
    for packet in packets:
        group = groups.setdefault(packet.key, [])
        if group:
            # A short nested/repeated packet must not hide the longer packet's
            # still-covered tail and create a fictitious recording gap.
            previous = frontiers[packet.key]
            delta = packet.start_us - previous.end_us
            if delta:
                continuity.append(dict(stream_id=packet.stream_id, session_id=packet.session_id,
                                       kind='gap' if delta > 0 else 'overlap', delta_us=delta,
                                       previous_media_id=previous.media_id, media_id=packet.media_id))
        if packet.source_sha256 in seen:
            continuity.append(dict(kind='repeated_source_bytes', source_sha256=packet.source_sha256,
                                   previous_media_id=seen[packet.source_sha256], media_id=packet.media_id))
        seen[packet.source_sha256] = packet.media_id
        group.append(packet)
        if packet.key not in frontiers or packet.end_us > frontiers[packet.key].end_us:
            frontiers[packet.key] = packet
    causes = []
    for hint in hints:
        _validate_span(hint.span, by_id)
        kinds = ('action',) if hint.kind not in {'waiting', 'rest', 'chat', 'unknown'} else ()
        kinds += tuple(name for name, active in [('uncertain', hint.uncertain), ('unfinished', hint.unfinished), ('conflict', hint.conflicting)] if active)
        if kinds:
            causes.append((hint.hint_id, hint.span, kinds, (hint.hint_id,), (), hint.run_ids, hint.binding_revision_ids))
    for anchor in anchors:
        _validate_span(anchor.span, by_id)
        causes.append((anchor.anchor_id, anchor.span, (anchor.kind,), (), (anchor.evidence_id,), anchor.run_ids, anchor.binding_revision_ids))
    # Deterministic blind audits are independent of model importance scores.
    if policy.audit_period_us is not None:
        for packet in packets:
            for lo in range(0, packet.duration_us, policy.audit_period_us):
                width = min(policy.audit_period_us, packet.duration_us - lo)
                length = min(width, policy.audit_length_us)
                seed = fingerprint([policy.audit_seed, packet.media_id, packet.source_sha256, lo])
                start = lo + int(seed[:16], 16) % (width - length + 1)
                span = Span(packet.media_id, packet.source_sha256, start, start + length)
                causes.append(('audit:' + seed, span, ('blind_audit',), (), (), (), ()))
    tasks, config_sha = [], fingerprint(asdict(policy))
    for cause_id, focal, kinds, hint_ids, external_ids, run_ids, binding_ids in causes:
        packet = _validate_span(focal, by_id)
        group = groups[packet.key]
        lo = max(min(p.start_us for p in group), packet.start_us + focal.start_us - policy.before_us)
        hi = min(max(p.end_us for p in group), packet.start_us + focal.end_us + policy.after_us)
        spans, intervals, reasons = [], [], []
        for other in group:
            a, b = max(lo, other.start_us), min(hi, other.end_us)
            if a < b:
                spans.append(Span(other.media_id, other.source_sha256, a - other.start_us, b - other.start_us))
                intervals.append((a, b))
                if other.clock_uncertainty_us is None or other.clock_uncertainty_us > 0:
                    reasons.append('uncertain_source_clock')
                if sum(p.source_sha256 == other.source_sha256 for p in packets) > 1:
                    reasons.append('repeated_source_bytes_require_review')
        gaps = _missing(lo, hi, intervals)
        if gaps:
            reasons.append('missing_source_context')
        if any(b > c for (_, b), (c, _) in zip(sorted(intervals), sorted(intervals)[1:])):
            reasons.append('overlapping_source_context')
        if sum(s.end_us - s.start_us for s in spans) > policy.max_fine_duration_us:
            reasons.append('fine_duration_capacity')
        payload = dict(stream_id=packet.stream_id, session_id=packet.session_id,
                       trigger_ids=(cause_id,), trigger_kinds=kinds, source_spans=tuple(spans),
                       focal_span=focal, hint_ids=hint_ids, external_evidence_ids=external_ids, run_ids=tuple(sorted(set(run_ids))),
                       binding_revision_ids=tuple(sorted(set(binding_ids))),
                       status='needs_attention' if reasons else 'pending',
                       attention_reasons=tuple(sorted(set(reasons))), context_gaps=gaps,
                       config_sha256=config_sha,
                       source_packets_sha256=fingerprint([asdict(by_id[s.media_id]) for s in spans]))
        task_id = fingerprint({**payload, 'source_spans': [asdict(s) for s in spans], 'focal_span': asdict(focal)})
        tasks.append(FineTask(task_id=task_id, **payload))
    coverage = []
    for packet in packets:
        parts = [(task, s) for task in tasks for s in task.source_spans if s.media_id == packet.media_id]
        boundaries = sorted({0, packet.duration_us, *(v for _, s in parts for v in (s.start_us, s.end_us))})
        for lo, hi in zip(boundaries, boundaries[1:]):
            covering = [task for task, span in parts if span.start_us <= lo and hi <= span.end_us]
            coverage.append(dict(media_id=packet.media_id, source_sha256=packet.source_sha256,
                                 start_us=lo, end_us=hi, captured_at=(instant(packet.captured_at) + timedelta(microseconds=lo)).isoformat(),
                                 clock_basis=packet.clock_basis, clock_uncertainty_us=packet.clock_uncertainty_us,
                                 fine_status='planned_not_yet_analysed' if covering else 'not_selected_for_fine_still_unanalysed',
                                 task_ids=sorted({task.task_id for task in covering}),
                                 semantic_completeness='NOT_PROVEN', raw_evidence_retained=True))
    tasks.sort(key=lambda t: (t.key, by_id[t.focal_span.media_id].start_us + t.focal_span.start_us, t.task_id))
    return Plan(packets, hints, anchors, tuple(tasks), tuple(continuity), tuple(coverage), config_sha)


class StreamScheduler:
    """Thread-safe in-memory lease interface; one active task per stream/session.

    A remote worker must explicitly lease/finish and persist its own state. This
    is not a distributed queue and never puts live locks or SQLite on SMB.
    """
    def __init__(self, plan, *, max_concurrent=1):
        if type(max_concurrent) is not int or not 1 <= max_concurrent <= 4:
            raise ValueError('Research concurrency must be explicitly 1..4')
        self.max_concurrent, self.tasks = max_concurrent, {t.task_id: t for t in plan.tasks}
        self.states = {t.task_id: t.status for t in plan.tasks}
        self.active, self.events, self._lock = {}, [], RLock()

    def claim(self, worker, *, now=None, lease_seconds=300):
        if not worker or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError('Named worker and bounded positive lease required')
        now = time.monotonic() if now is None else now
        with self._lock:
            for token, active in list(self.active.items()):
                if active['expires'] <= now:
                    self.states[active['task_id']] = 'needs_attention'
                    self.events.append(dict(kind='expired_no_identical_auto_retry', **active))
                    del self.active[token]
            if len(self.active) >= self.max_concurrent:
                return None
            busy = {self.tasks[a['task_id']].key for a in self.active.values()}
            task = next((t for t in self.tasks.values() if self.states[t.task_id] == 'pending' and t.key not in busy), None)
            if task is None:
                return None
            token = fingerprint([task.task_id, worker, now, len(self.events)])
            self.active[token] = dict(task_id=task.task_id, worker=worker, expires=now + lease_seconds)
            self.states[task.task_id] = 'running'
            self.events.append(dict(kind='claimed', token=token, task_id=task.task_id, worker=worker))
            return dict(token=token, task=task, stream_state_key=fingerprint(task.key))

    def finish(self, token, worker, *, outcome, result_sha256=None, now=None):
        if outcome not in {'candidate_returned', 'needs_attention'}:
            raise ValueError('Only research candidates or explicit attention allowed')
        if outcome == 'candidate_returned':
            _sha(result_sha256)
        now = time.monotonic() if now is None else now
        with self._lock:
            active = self.active.get(token)
            if not active or active['worker'] != worker or now >= active['expires']:
                raise ValueError('Stale or foreign stream lease')
            self.states[active['task_id']] = outcome
            self.events.append(dict(kind=outcome, result_sha256=result_sha256, **active))
            del self.active[token]

    def backlog(self):
        with self._lock:
            return {status: sum(v == status for v in self.states.values()) for status in
                    ('pending', 'running', 'candidate_returned', 'needs_attention')}


@dataclass(frozen=True)
class ConsumedFrame:
    span: Span
    source_frame_index: int
    source_timestamp_us: int

    def __post_init__(self):
        _us(self.source_frame_index)
        _us(self.source_timestamp_us)
        if not self.span.start_us <= self.source_timestamp_us < self.span.end_us:
            raise ValueError('Actual input frame must lie inside its declared evidence span')


class StreamMemoryProgress:
    """Commit after the actual model update returns; planned work is not counted.

    Model-specific state objects stay with their worker, indexed by this key.
    Queries use independent snapshots; this API has no Q/A-to-memory entry point.
    """
    def __init__(self, stream_id, session_id, model_revision, config_sha256):
        _sha(config_sha256)
        if not all((stream_id, session_id, model_revision)):
            raise ValueError('Independent stream/session and pinned model required')
        self.key = stream_id, session_id
        self.model_revision, self.config_sha256 = model_revision, config_sha256
        self.updates, self.total_input_frames, self._unique = [], 0, set()
        self._packets = {}
        self.latest_processed_us = None

    def commit(self, packet, frames, *, actual_update_count_after, state_sha256, successful):
        frames = tuple(frames)
        _sha(state_sha256)
        if packet.key != self.key or successful is not True or not frames:
            raise ValueError('Only successful actual model inputs for this stream can advance memory')
        if type(actual_update_count_after) is not int or actual_update_count_after != len(self.updates) + 1:
            raise ValueError('Actual update count must advance once; no static scheduled counters')
        for frame in frames:
            if frame.span.media_id != packet.media_id or frame.span.source_sha256 != packet.source_sha256 or frame.span.end_us > packet.duration_us:
                raise ValueError('Model consumed a changed or foreign source')
        if packet.media_id in self._packets and self._packets[packet.media_id] != packet:
            raise ValueError('Memory source capture identity or time mapping changed')
        positions = [(f.source_timestamp_us, f.source_frame_index) for f in frames]
        if any(b < a for a, b in zip(positions, positions[1:])):
            raise ValueError('Actual memory input frames must remain in chronological source order')
        newest = max(packet.start_us + f.source_timestamp_us for f in frames)
        if self.latest_processed_us is not None and newest < self.latest_processed_us:
            raise ValueError('Memory updates cannot silently rewind or mix source order')
        self.total_input_frames += len(frames)
        self._unique.update((packet.media_id, packet.source_sha256, f.source_frame_index) for f in frames)
        self.latest_processed_us = newest
        self._packets[packet.media_id] = packet
        row = dict(actual_update_count=actual_update_count_after, state_sha256=state_sha256,
                   state_changed=not self.updates or self.updates[-1]['state_sha256'] != state_sha256,
                   input_frames=[asdict(f) for f in frames], cumulative_input_frames=self.total_input_frames,
                   cumulative_unique_source_frames=len(self._unique), latest_processed_epoch_us=newest,
                   clock_basis=packet.clock_basis, clock_uncertainty_us=packet.clock_uncertainty_us)
        self.updates.append(row)
        return row


@dataclass(frozen=True)
class Claim:
    claim_id: str
    kind: str
    text: str
    evidence: tuple[Span, ...]
    replaces_claim_ids: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'evidence', tuple(self.evidence))
        object.__setattr__(self, 'replaces_claim_ids', tuple(self.replaces_claim_ids))


@dataclass(frozen=True)
class Revision:
    task_id: str
    revision_id: str
    parent_revision_id: str | None
    model_revision: str
    raw_answer_sha256: str
    claims: tuple[Claim, ...]
    status: str = field(default='candidate_not_confirmed', init=False)

    def __post_init__(self):
        object.__setattr__(self, 'claims', tuple(self.claims))


class RevisionLedger:
    """Append-only candidate revisions. Evidence presence is not truth verification."""
    def __init__(self):
        self._items = []

    @property
    def items(self):
        return tuple(self._items)

    def append(self, task, revision, *, native_receipt):
        if native_receipt is None:
            raise ValueError('Original-frame delivery receipt required; coarse text alone is insufficient')
        native_receipt.validate(task)
        _sha(revision.raw_answer_sha256)
        if revision.task_id != task.task_id or not revision.model_revision or not revision.revision_id:
            raise ValueError('Pinned task, revision and model required')
        if any(r.revision_id == revision.revision_id for r in self._items):
            raise ValueError('Revision IDs are immutable')
        history = [r for r in self._items if r.task_id == task.task_id]
        if revision.parent_revision_id != (history[-1].revision_id if history else None):
            raise ValueError('Revision must append to the current candidate history')
        old_claims = {c.claim_id for r in history for c in r.claims}
        if len({c.claim_id for c in revision.claims}) != len(revision.claims) or {c.claim_id for c in revision.claims} & old_claims:
            raise ValueError('Unique claim IDs required')
        for claim in revision.claims:
            if not claim.claim_id or not claim.text or not claim.evidence:
                raise ValueError('Unsupported claims without concrete evidence are rejected')
            if claim.kind not in {'visible_action', 'object', 'sequence', 'uncertainty', 'spoken', 'ocr_candidate'}:
                raise ValueError('Physical completion, precise confirmed readings and material accounting are prohibited')
            if set(claim.replaces_claim_ids) - old_claims:
                raise ValueError('A correction must identify actual earlier claims')
            for span in claim.evidence:
                if not any(span.media_id == s.media_id and span.source_sha256 == s.source_sha256 and
                           s.start_us <= span.start_us < span.end_us <= s.end_us for s in task.source_spans):
                    raise ValueError('Claim evidence is outside the actual fine-review input')
        self._items.append(revision)


class StageClocks:
    """Explicit synchronizer supports CUDA callers; this module never imports CUDA."""
    def __init__(self):
        self.records, self.resources = [], []

    def resource(self, *, stream_id, latest_processed_epoch_us=None, gpu_used_mib=None,
                 gpu_total_mib=None, process_rss_bytes=None, cache_bytes=None):
        if not stream_id:
            raise ValueError('Resource curve must identify the actual stream')
        values = dict(gpu_used_mib=gpu_used_mib, gpu_total_mib=gpu_total_mib,
                      process_rss_bytes=process_rss_bytes, cache_bytes=cache_bytes,
                      latest_processed_epoch_us=latest_processed_epoch_us)
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0) for v in values.values()):
            raise ValueError('Measured resource values must be nonnegative; unknown is null')
        if gpu_total_mib is not None and (gpu_total_mib == 0 or (gpu_used_mib is not None and gpu_used_mib > gpu_total_mib)):
            raise ValueError('Invalid measured GPU memory')
        row = dict(at=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(), stream_id=stream_id, **values)
        self.resources.append(row)
        return row

    @contextmanager
    def stage(self, name, *, sync: Callable[[], None] | None = None, mode='cold', stream_id=None):
        if name not in STAGES or mode not in {'cold', 'reused_memory', 'verification_recompute'}:
            raise ValueError('Known stage and separate cache/timing mode required')
        if sync:
            sync()
        started, tick, outcome = datetime.now(timezone.utc), time.monotonic(), 'returned'
        try:
            yield
        except Exception:
            outcome = 'failed'
            raise
        finally:
            if sync:
                sync()
            self.records.append(dict(stage=name, mode=mode, stream_id=stream_id, started_at=started.isoformat(),
                                     finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - tick,
                                     outcome=outcome, gpu_synchronizer_supplied=sync is not None))


def nightly_budget(clocks, backlog, *, whole_wall_seconds=None, source_seconds=None,
                   complete_input=None, accepted_quality=None, publication_verified=None):
    """Unknown stages stay null; rates/ETAs never extrapolate partial trials."""
    if whole_wall_seconds is not None and (isinstance(whole_wall_seconds, bool) or not math.isfinite(whole_wall_seconds) or whole_wall_seconds < 0):
        raise ValueError('Actual nonnegative whole wall time required')
    if source_seconds is not None and (isinstance(source_seconds, bool) or not math.isfinite(source_seconds) or source_seconds <= 0):
        raise ValueError('Actual positive source duration required')
    if any(v is not None and (type(v) is not int or v < 0) for v in backlog.values()):
        raise ValueError('Actual backlog counts must be nonnegative integers; unknown is null')
    grouped = {}
    for mode in ('cold', 'reused_memory', 'verification_recompute'):
        grouped[mode] = {stage: None for stage in STAGES}
        for stage in STAGES:
            rows = [r['elapsed_seconds'] for r in clocks.records if r['mode'] == mode and r['stage'] == stage]
            if rows:
                grouped[mode][stage] = sum(rows)
    required_backlog = {'pending', 'running', 'needs_attention'}
    known_backlog = (required_backlog <= backlog.keys() and
                     not (backlog.keys() - required_backlog - {'candidate_returned'}) and
                     all(v is not None for v in backlog.values()))
    completed = known_backlog and all(backlog[k] == 0 for k in required_backlog)
    proven = all(value is True for value in (complete_input, accepted_quality, publication_verified)) and completed
    returned_stages = {r['stage'] for r in clocks.records if r['mode'] == 'cold' and r['outcome'] == 'returned'}
    measured = (whole_wall_seconds is not None and whole_wall_seconds > 0 and source_seconds is not None and
                all(s in returned_stages for s in STAGES))
    return dict(schema_version='vc-lab-coarse-fine-nightly/1', acceptance='DEMONSTRATED' if proven and measured else 'NOT_DEMONSTRATED',
                stages_seconds=grouped, stage_totals_are_not_wall_time=True, whole_wall_seconds=whole_wall_seconds,
                source_seconds=source_seconds, observed_real_time_factor=whole_wall_seconds / source_seconds if whole_wall_seconds is not None and source_seconds else None,
                source_complete=complete_input, quality_accepted=accepted_quality, publication_verified=publication_verified,
                backlog_counts_known=known_backlog, successfully_returned_cold_stages=sorted(returned_stages),
                next_shift_backlog=dict(backlog), next_shift_eta_seconds=None,
                resource_curve=tuple(dict(r) for r in clocks.resources),
                budgets=[dict(hours=h, fits=(whole_wall_seconds <= h * 3600) if proven and measured else None)
                         for h in (8, 10, 12)])
