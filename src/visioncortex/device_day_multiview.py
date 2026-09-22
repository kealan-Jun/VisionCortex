"""Adapt sealed recorder evidence to the offline alignment/grouping functions.

No detector or cloud model is invoked here. Recorder boundaries stay source
boundaries; the shared grouping functions decide experiment boundaries.
"""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import time

from .device_day_contract import atomic_json, digest, file_hash, read_json, safe_child


def _place_recordings(infos, sources):
    """Supply recorder wall-clock placement to the shared segmented reader.

    A day contains real pauses and unavailable slices. The offline reader's
    provisional concatenated offsets must not compress those gaps before it
    checks the CSV clock. Physical file durations and frame offsets stay intact.
    """
    from .device_day_models import capture_us
    for camera, rows in sources.items():
        if camera not in infos:
            continue
        clocks = [s['record'].get('processing', {}).get('clock_mapping') or {} for s in rows]
        if not all(c.get('basis') == 'recorder_csv_interpolation' and c.get('points') for c in clocks):
            raise ValueError('Shared recorder alignment requires sealed capture-clock mappings')
        origins = [capture_us(clock, 0) for clock in clocks]
        if any(b < a for a, b in zip(origins, origins[1:], strict=False)):
            raise ValueError('Recorder capture clocks regress within one camera')
        for info, origin in zip(infos[camera].segments, origins, strict=True):
            info.virtual_start_ms = (origin - origins[0]) / 1000
            info.virtual_end_ms = info.virtual_start_ms + info.duration_ms
        infos[camera].duration_ms = max(s.virtual_end_ms for s in infos[camera].segments)


def _validate_capture_alignment(transforms, infos, sources, config):
    """Quarantine mismatched clock fits before they can pair action candidates."""
    from .device_day_models import capture_us
    reference = next(iter(transforms.values())).reference_view_id
    first = sources[reference][0]['record']['processing']['clock_mapping']
    origin = capture_us(first, 0) / 1000 - transforms[reference].to_global(infos[reference].segments[0].virtual_start_ms)
    tolerance = float(config['alignment'].get('maximum_alignment_uncertainty_ms', 5000))
    errors = []
    for camera, transform in transforms.items():
        for source, info, part in zip(sources[camera], infos[camera].segments, transform.segment_transforms, strict=True):
            clock = source['record']['processing']['clock_mapping']
            residual = max(abs(origin + part.to_global(info.virtual_start_ms+t) + transform.visual_correction_ms
                               - capture_us(clock, t)/1000) for t in (0, info.duration_ms/2, info.duration_ms))
            if residual > tolerance:
                part.state, part.confidence = 'failed', 0
                part.clock_anomalies.append('shared_fit_disagrees_with_retained_capture_clock')
                errors.append({'archive': source['archive'], 'recording_id': source['record']['recording_id'],
                               'residual_ms': residual, 'limit_ms': tolerance})
    return errors


def _mapped(value, offset, transform, frame_offset, namespace, key=""):
    """Translate one recorder's local ledger into its camera's virtual view."""
    if isinstance(value, dict):
        return {k: _mapped(v, offset, transform, frame_offset, namespace, k) for k, v in value.items()}
    if isinstance(value, list):
        item_key = "track_id" if key.endswith("track_ids") else key
        return [_mapped(v, offset, transform, frame_offset, namespace, item_key) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if key == "frame_index":
            return value + frame_offset
        if key.endswith("track_id"):
            return namespace * 1_000_000_000 + int(value)
        if key.endswith("global_ms") or key.startswith("global_") and key.endswith("_ms"):
            return transform.to_global(value + offset)
        if key.endswith("local_ms") or key.startswith("local_") and key.endswith("_ms"):
            return value + offset
    if isinstance(value, str) and key == "instance_keys":
        prefix, sep, token = value.rpartition(":")
        if sep and token.isdigit():
            return prefix + ":" + str(namespace * 1_000_000_000 + int(token))
    return value


class _FrameWindows:
    """Read only the existing fine-index frames needed by the shared refiner."""
    def __init__(self, sources, transforms):
        self.sources, self.transforms = sources, transforms

    def iter_global_frames(self, view_id, *, start_ms, end_ms):
        from .schemas import FrameEvidence
        transform = self.transforms[view_id]
        for source in self.sources.get(view_id, []):
            offset = source['info'].virtual_start_ms
            left = max(0, transform.to_local(start_ms) - offset)
            right = min(source['info'].duration_ms, transform.to_local(end_ms) - offset)
            if right < left:
                continue
            for path in source['fine_indexes']:
                with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
                    for (payload,) in db.execute(
                            'SELECT payload_json FROM fine_frames WHERE view_id=? '
                            'AND local_ms>=? AND local_ms<=? ORDER BY local_ms,frame_index',
                            (view_id, left, right)):
                        raw = _mapped(json.loads(payload), offset, transform,
                                      source['info'].frame_start_index, source['namespace'])
                        yield FrameEvidence.model_validate(raw)


def shared_analysis(views, infos, sources, config):
    """The same alignment, event audit, bounds, pairing and key curation as offline."""
    from .alignment import build_alignments, alignment_quality_report
    from .actions import audit_candidates, refine_liquid_events_with_context, build_experiment_segments
    from .action_state_machine import attach_continuous_action_states
    from .action_semantics import attach_action_observability, build_semantic_review_plan
    from .grouping import (normalize_experiment_segments, prepare_formal_experiment_segments,
                           build_experiment_groups, select_key_events)
    from .schemas import ActionCandidate
    from .device_day_models import capture_us
    settings = deepcopy(config)
    _place_recordings(infos, sources)
    def first_capture(view):
        record = sources[view.view_id][0]['record']
        clock = record.get('processing', {}).get('clock_mapping') or {'origin_us': record.get('start_us', 0)}
        return (capture_us(clock, 0), view.role.value != 'first_person', view.view_id)
    # Start the shared timeline at its earliest recording, not the first
    # alphabetically named camera (which may only start in the afternoon).
    settings['alignment']['reference_view'] = min(views, key=first_capture).view_id
    transforms, _ = build_alignments(views, infos, settings)
    clock_errors = _validate_capture_alignment(transforms, infos, sources, settings)
    quality = alignment_quality_report(views, infos, transforms, settings)
    candidates, coarse = [], []
    for view in views:
        for source, info in zip(sources[view.view_id], infos[view.view_id].segments, strict=True):
            source['info'] = info
            seen = set()
            for batch in source['record']['processing'].get('batches') or []:
                raw_candidates = [c for e in batch.get('activity_audit', {}).get('events', [])
                                  for c in e.get('candidates', [])]
                # Keep candidates rejected in a single view: the offline auditor
                # must see both views before deciding. Enriched audit rows win.
                raw_candidates.extend(batch.get('fine_candidates') or [])
                for target, rows in ((candidates, raw_candidates), (coarse, batch.get('coarse_candidates', []))):
                    for raw in rows:
                        identity = (target is coarse, raw['candidate_id'])
                        if identity in seen:
                            continue
                        seen.add(identity)
                        mapped = _mapped(raw, info.virtual_start_ms, transforms[view.view_id],
                                         info.frame_start_index, source['namespace'])
                        mapped.update(view_id=view.view_id, role=view.role.value,
                                      candidate_id=source['record']['recording_id'] + '-' + raw['candidate_id'])
                        mapped.setdefault('provenance', {})['recorder_source'] = {
                            'recording_id': source['record']['recording_id'], 'candidate_id': raw['candidate_id']}
                        target.append(ActionCandidate.model_validate(mapped))
    events, rejected = audit_candidates(candidates, transforms, settings)
    rejected.extend(refine_liquid_events_with_context(
        events, {}, frame_index=_FrameWindows(sources, transforms), config=settings))
    states = attach_continuous_action_states(events, settings)
    attach_action_observability(events)
    semantic = build_semantic_review_plan(events, settings)
    boundaries, normalizations, continuity, selection = [], [], [], []
    raw = build_experiment_segments(events, views, settings, coarse_windows=coarse, decision_receipts=boundaries)
    normalized = normalize_experiment_segments(raw, events, views, settings, decision_receipts=normalizations)
    segments, formal = prepare_formal_experiment_segments(normalized, events, views, coarse, settings)
    groups = build_experiment_groups(segments, events, views, settings,
                                     decision_receipts=continuity, coarse_windows=coarse)
    keys = select_key_events(groups, segments, events, settings, decision_receipts=selection)
    # The shared continuity auditor uses infinity for an absent context chain.
    # JSON encodes that absence as null, never as a zero-length observed gap.
    for decision in continuity:
        facts = decision.get('facts', {})
        if facts.get('maximum_context_step_ms') == float('inf'):
            facts['maximum_context_step_ms'] = None
            facts['context_step_status'] = 'no_bounded_context_chain'
    return {'alignment_quality': quality, 'capture_alignment_errors': clock_errors,
            'transforms': {k: v.model_dump(mode='json') for k, v in transforms.items()},
            'events': [e.model_dump(mode='json') for e in events], 'rejected': rejected,
            'segments': [s.model_dump(mode='json') for s in segments],
            'groups': [g.model_dump(mode='json') for g in groups],
            'selected_key_events': [e.model_dump(mode='json') for e in keys],
            'boundary_decisions': boundaries, 'normalization_decisions': normalizations,
            'formal_decisions': formal, 'continuity_decisions': continuity,
            'selection_decisions': selection, 'state_machine': states, 'semantic_review_plan': semantic}


def build_device_day_multiview(config, day, indexes):
    from .capture_layout import camera_group
    partitions = {}
    for name, index in indexes:
        group = camera_group(config.get('collection_ingest', {}), name[11:])
        partitions.setdefault(group, []).append((name, index))
    if len(partitions) <= 1:
        return _build_source_multiview(config, day, indexes)
    results = {}
    for group, rows in partitions.items():
        try:
            results[group] = _build_source_multiview(config, day, rows)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            results[group] = {'status': 'failed', 'error_type': type(exc).__name__}
    result = {'key': digest({group: row.get('key', row) for group, row in results.items()}),
              'date': day, 'status': 'completed' if all(row.get('status') == 'completed' for row in results.values()) else 'partial',
              'source_groups': {group: {key: row.get(key) for key in
                  ('key', 'status', 'receipt_path', 'alignment_quality', 'error_type')} for group, row in results.items()},
              'detector_invoked': False, 'cloud_model_invoked': False, 'evidence_status': 'PARTIAL_EVIDENCE'}
    for key in ('experiments', 'aligned_recordings', 'missing_sources', 'formal_decisions', 'capture_alignment_errors'):
        result[key] = [item for row in results.values() for item in row.get(key, [])]
    result['algorithms'] = sorted({item for row in results.values() for item in row.get('algorithms', [])})
    path = Path(config['storage']['local_runtime_root'])/'device-day/Multiview'/day/result['key']/'SourceGroups.json'
    atomic_json(path, result)
    return result | {'receipt_path': str(path)}


def _build_source_multiview(config, day, indexes):
    from .schemas import ViewInput, VideoSegmentInput
    root = Path(config['storage']['archive_root'])
    backend = Path(config['storage']['local_cache_root'])
    roles = config.get('collection_ingest', {}).get('camera_role_map', {})
    sources, missing = {}, []
    namespace = 0
    for name, index in sorted(indexes):
        camera = name[11:]
        if name[:10] != day or roles.get(camera) not in {'first_person', 'third_person'}:
            continue
        published = {s['recording_id'] for s in index.get('segments', [])}
        for record in sorted(index.get('recordings', []), key=lambda r: r.get('start_us') or 0):
            if record['recording_id'] not in published:
                continue
            refs = {s['kind']: s['retained'] for s in record.get('sources', [])}
            try:
                video, clock = [safe_child(root, name+'/'+refs[k]['path']) for k in ('video', 'clock')]
                for k, p in (('video', video), ('clock', clock)):
                    if p.stat().st_size != refs[k]['size_bytes']:
                        raise ValueError('Source size disagrees with sealed reference')
                fine = [safe_child(backend, a['path']) for a in record.get('processing', {}).get('audit_artifacts') or []
                        if a['path'].endswith('FineIndex.sqlite3') and a.get('storage_root') == 'local_cache_root']
                if not all(p.is_file() for p in fine):
                    raise ValueError('Fine evidence index missing')
                namespace += 1
                sources.setdefault(camera, []).append({'archive': name, 'record': record, 'refs': refs,
                    'video': video, 'clock': clock, 'fine_indexes': fine, 'namespace': namespace})
            except (OSError, ValueError, KeyError) as exc:
                missing.append({'archive': name, 'recording_id': record['recording_id'], 'reason': type(exc).__name__})
    identities = {v: [{'recording_id': s['record']['recording_id'], 'refs': s['refs'],
                       'file_identities': [list(_file_identity(p)) for p in [s['video'], s['clock'], *s['fine_indexes']]],
                       'processing': s['record']['processing']} for s in rows] for v, rows in sources.items()}
    modules = ['alignment', 'actions', 'grouping', 'action_state_machine', 'action_semantics', 'archive', 'video_io', 'workflow_video', 'device_day_multiview']
    key = digest([day, identities, missing, roles, {k: config.get(k) for k in ('alignment','segmentation','continuity','performance','key_materials')},
                  {m: file_hash(Path(__file__).with_name(m+'.py')) for m in modules}])
    path = Path(config['storage']['local_runtime_root'])/'device-day/Multiview'/day/key/'Result.json'
    if path.is_file():
        saved = read_json(path)
        if (saved.get('key') == key and saved.get('result_digest') == digest(saved['result'])
                and _cache_usable(saved, root)):
            return saved['result'] | {'reused': True, 'receipt_path': str(path)}
    base = {'key': key, 'date': day, 'missing_sources': missing,
            'algorithms': ['alignment.build_alignments', 'actions.audit_candidates', 'actions.refine_liquid_events_with_context',
                           'actions.build_experiment_segments', 'grouping.normalize_experiment_segments',
                           'grouping.prepare_formal_experiment_segments', 'grouping.build_experiment_groups', 'grouping.select_key_events'],
            'detector_invoked': False, 'cloud_model_invoked': False, 'evidence_status': 'PARTIAL_EVIDENCE'}
    views = [ViewInput(view_id=v, role=roles[v], segments=[VideoSegmentInput(video=s['video'], timestamps_csv=s['clock']) for s in rows])
             for v, rows in sources.items()]
    if {v.role.value for v in views} != {'first_person','third_person'}:
        result = base | {'status': 'waiting_for_other_role', 'groups': [], 'experiments': []}
    else:
        ready, infos = _probe_sources(sources, roles, config, missing)
        if {v.role.value for v in ready} != {'first_person', 'third_person'}:
            result = base | {'status': 'waiting_for_readable_views', 'groups': [], 'experiments': []}
        else:
            analysis = shared_analysis(ready, infos, sources, config)
            result = base | analysis | {'status': 'completed', 'experiments': _project_groups(analysis, sources, config),
                                        'aligned_recordings': _aligned_recordings(analysis, {v.view_id:sources[v.view_id] for v in ready})}
            if analysis['alignment_quality']['formal_evidence_ready']:
                for experiment in result['experiments']:
                    try:
                        _materialize([experiment], ready, infos, sources, config)
                    except (OSError, ValueError, RuntimeError, KeyError) as exc:
                        experiment['materialization_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    atomic_json(path, {'key': key, 'result': result, 'result_digest': digest(result), 'saved_at': time.time()})
    return result | {'reused': False, 'receipt_path': str(path)}


def _file_identity(path):
    from .device_day_verification import ArtifactVerifier
    return ArtifactVerifier._identity(path)


def _probe_sources(sources, roles, config, missing):
    """Cache the shared prober per immutable recording, not per expanding day."""
    from concurrent.futures import ThreadPoolExecutor
    from .schemas import ViewInput, VideoSegmentInput, VideoInfo
    from .video_io import probe_views, _build_segmented_view_info
    from .alignment import read_timestamp_csv_bounded, read_timestamp_csv_endpoints
    root = Path(config['storage']['local_runtime_root'])/'device-day/Multiview/MediaInfo'
    implementation = [file_hash(Path(__file__).with_name(m+'.py')) for m in ('video_io', 'alignment')]
    align = config['alignment']
    jobs = [(camera, source) for camera, rows in sources.items() for source in rows]
    def inspect(job):
        camera, source = job
        try:
            before = [_file_identity(p) for p in (source['video'], source['clock'])]
            key = digest([str(source['video']), before, source['refs'], implementation, align])
            path = root/(key+'.json')
            if path.is_file():
                saved = read_json(path)
                if saved.get('key') == key and saved.get('info_digest') == digest(saved['info']):
                    return camera, source, VideoInfo.model_validate(saved['info'])
            view = ViewInput(view_id=camera, role=roles[camera], video=source['video'], timestamps_csv=source['clock'])
            info = probe_views([view], workers=1)[camera]
            # The very same bounded CSV reader as build_alignments, isolated
            # per recording so malformed CSV cannot abort other cameras.
            if align.get('csv_sampling_mode') == 'legacy_endpoints':
                read_timestamp_csv_endpoints(source['clock'], info.fps)
            else:
                read_timestamp_csv_bounded(source['clock'], info.fps,
                    sample_count=max(2, int(align.get('csv_samples_per_segment', 5))),
                    full_read_limit_bytes=int(align.get('csv_bounded_full_read_limit_bytes', 2*1024*1024)))
            if before != [_file_identity(p) for p in (source['video'], source['clock'])]:
                raise ValueError('Source changed during media probe')
            data = info.model_dump(mode='json')
            atomic_json(path, {'key': key, 'info': data, 'info_digest': digest(data)})
            return camera, source, info
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            return camera, source, exc
    successful = {camera: [] for camera in sources}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='multiview-media') as pool:
        for camera, source, info in pool.map(inspect, jobs):
            if isinstance(info, Exception):
                missing.append({'camera': camera, 'archive': source['archive'],
                                'recording_id': source['record']['recording_id'],
                                'reason': type(info).__name__, 'stage': 'media_probe'})
            else:
                successful[camera].append((source, info))
    views, infos = [], {}
    for camera, rows in successful.items():
        if not rows:
            continue
        geometry = {}
        for _, info in rows:
            size = (info.width, info.height)
            geometry[size] = geometry.get(size, 0) + info.duration_ms
        size = max(geometry, key=geometry.get)
        compatible = []
        for source, info in rows:
            if (info.width, info.height) == size:
                compatible.append((source, info))
            else:
                missing.append({'camera': camera, 'archive': source['archive'],
                                'recording_id': source['record']['recording_id'], 'stage': 'media_geometry',
                                'reason': 'mixed_resolution_requires_separate_view',
                                'resolution': [info.width, info.height], 'shared_view_resolution': list(size)})
        rows = compatible
        # Keep valid recordings from a camera even when a sibling file fails.
        view = ViewInput(view_id=camera, role=roles[camera], segments=[
            VideoSegmentInput(video=s['video'], timestamps_csv=s['clock']) for s, _ in rows])
        try:
            infos[camera] = _build_segmented_view_info(view, [info for _, info in rows])
            views.append(view)
            sources[camera] = [source for source, _ in rows]
        except ValueError:
            missing.append({'camera': camera, 'reason': 'inconsistent_media_geometry', 'stage': 'media_probe'})
    return views, infos


def _outputs_valid(outputs, root):
    from .device_day_verification import verify_artifact_cached
    return bool(outputs) and all(
        verify_artifact_cached(root/o['archive'], o[k])
        for o in outputs for k in ('video', 'aligned_video') if k in o)


def _cache_usable(saved, root):
    result = saved['result']
    transient = any(s.get('stage') == 'media_probe' for s in result.get('missing_sources', []))
    transient |= any(e.get('materialization_error') for e in result.get('experiments', []))
    if transient and time.time()-saved.get('saved_at', 0) >= 300:
        return False
    return all(_outputs_valid(e['outputs'], root)
               for e in result.get('experiments', []) if e.get('outputs'))


def _aligned_recordings(analysis, sources):
    from .schemas import AlignmentTransform
    from .device_day_models import capture_us
    transforms = {k: AlignmentTransform.model_validate(v) for k, v in analysis['transforms'].items()}
    reference = analysis['alignment_quality']['reference_view_id']
    first = sources[reference][0]
    clock = first['record']['processing'].get('clock_mapping') or {'origin_us': first['record']['start_us']}
    origin = capture_us(clock, 0) - transforms[reference].to_global(first['info'].virtual_start_ms)*1000
    rows = []
    for camera, records in sources.items():
        tr = transforms[camera]
        for s in records:
            info = s['info']
            part = tr._segment_for_local((info.virtual_start_ms+info.virtual_end_ms)/2)
            if tr.state == 'failed' or part and part.state == 'failed':
                continue
            def global_ms(local):
                return part.to_global(local) + tr.visual_correction_ms if part else tr.to_global(local)
            knots = [[round(origin+global_ms(info.virtual_start_ms+t)*1000), t/1000]
                     for t in (0, info.duration_ms/2, info.duration_ms)]
            rows.append({'archive': s['archive'], 'recording_id': s['record']['recording_id'],
                         'source_sha256': s['refs']['video']['sha256'],
                         'start_us': knots[0][0], 'end_us': knots[-1][0], 'seek_points': knots,
                         'state': part.state if part else tr.state, 'basis': tr.alignment_basis,
                         'uncertainty_ms': part.uncertainty_ms if part else tr.uncertainty_ms,
                         'reference_view_id': reference, 'global_origin_us': origin})
    return rows


def _materialize(experiments, views, infos, sources, config):
    """Reuse the offline materializer, then project its outputs into the five directories."""
    from .archive import ArchiveLayout, materialize_experiment_clips
    from .schemas import AlignmentTransform, ExperimentGroup, ExperimentSegment, EvidenceEvent
    from .device_day_contract import artifact
    from .device_day import copy_verified
    import tempfile
    archive_root = Path(config['storage']['archive_root'])
    staging = Path(config['storage']['local_runtime_root'])/'device-day/Multiview/Staging'
    staging.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        group = ExperimentGroup.model_validate(experiment['group'])
        from .workflow_video import routed_intervals
        routes = routed_intervals(group)
        unverified_ms = sum(r['end_ms']-r['start_ms'] for r in routes if not r.get('third_person_view'))
        experiment['playback_coverage'] = {
            'duration_ms': group.global_end_ms-group.global_start_ms,
            'unverified_third_person_ms': unverified_ms,
            'complete_experiment_verified': False,
        }
        if unverified_ms:
            # Recorder publication already includes full per-camera activity
            # intervals. Keep the group's source references for comparison;
            # do not publish a mostly blank workflow as another activity MP4.
            experiment.update(outputs=[], status='partial_view_coverage',
                              publication_reason='unverified_third_person_intervals_use_source_comparison')
            continue
        if any(r['third_person_view'] != group.third_person_view for r in routes):
            # A routed multi-camera movie is not one camera's recording.
            experiment.update(outputs=[], status='routed_source_comparison',
                              publication_reason='multiple_third_person_cameras_use_source_comparison')
            continue
        transforms = {k: AlignmentTransform.model_validate(v) for k,v in experiment['transforms'].items()}
        segments = [ExperimentSegment.model_validate(s) for s in experiment['atomic_segments']]
        events = [EvidenceEvent.model_validate(e) for e in experiment['events']]
        outputs = []
        # New slices update the day, but unchanged completed experiments keep
        # their sealed outputs instead of being encoded again.
        first = next(s for s in experiment['sources'] if s['camera'] == group.first_person_view)
        metadata = safe_child(archive_root, first['archive']+'/ProcessedClips/Clips/'
                              +_experiment_folder(experiment, first['archive'])+'/ExperimentActivity.json')
        if metadata.is_file():
            previous = read_json(metadata)
            if (previous.get('experiment_id') == experiment['experiment_id']
                    and previous.get('materializer') == 'visioncortex.archive.materialize_experiment_clips'
                    and _outputs_valid(previous.get('outputs', []), archive_root)):
                experiment.update(outputs=previous['outputs'], materializer=previous['materializer'])
                continue
        from .device_day_verification import verify_artifact_cached
        for reference in experiment['sources']:
            if not verify_artifact_cached(archive_root/reference['archive'], reference['source_ref']):
                raise ValueError('Experiment source integrity verification failed')
        with tempfile.TemporaryDirectory(dir=staging) as temporary:
            layout = ArchiveLayout(Path(temporary))
            layout.create()
            materialize_experiment_clips(layout, [group], segments, events, views, infos, transforms, config)
            for role, camera in (('first_person', group.first_person_view), ('third_person', group.third_person_view)):
                source_refs = [s for s in experiment['sources'] if s['camera'] == camera]
                if not source_refs:
                    raise ValueError('Selected offline group has no corresponding source')
                name = source_refs[0]['archive']
                folder = safe_child(archive_root, name+'/ProcessedClips/Clips/'+_experiment_folder(experiment, name))
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder/'ExperimentActivity.mp4'
                copy_verified(layout.root/group.videos[role.replace('_', '-')], destination)
                output = {'archive': name, 'camera': camera, 'role': role,
                          'video': artifact(archive_root/name, destination),
                          'json_path': (folder/'ExperimentActivity.json').relative_to(archive_root/name).as_posix()}
                if role == 'first_person':
                    aligned = folder/'AlignedFirstThird.mp4'
                    copy_verified(layout.root/group.videos['aligned_first_third'], aligned)
                    output['aligned_video'] = artifact(archive_root/name, aligned)
                outputs.append(output)
        experiment['outputs'] = outputs
        experiment['materializer'] = 'visioncortex.archive.materialize_experiment_clips'
        for output in outputs:
            atomic_json(archive_root/output['archive']/output['json_path'],
                        {'scope': 'aligned_experiment', **experiment})


def _experiment_folder(experiment, archive):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from .device_day_contract import TIMEZONE
    refs = [s for s in experiment['sources'] if s['archive'] == archive]
    def label(value):
        return datetime.fromtimestamp(value/1e6, ZoneInfo(TIMEZONE)).strftime('%H-%M-%S')
    return (f"{label(min(s['start_us'] for s in refs))}_{label(max(s['end_us'] for s in refs))}"
            f"_ExperimentActivity_{digest(experiment['experiment_id'])[:8]}")


def retire_superseded_outputs(config, archive, experiments):
    """Retain obsolete generated groups in backend history, not the clip list.

    Called under the device/day index lock after the replacement index is
    published. Only adapter-owned group folders are eligible, never recorder
    segments, originals, audio, CSVs, or user-added files.
    """
    from .device_day_contract import validate_archive_name
    validate_archive_name(archive)
    root = safe_child(Path(config['storage']['archive_root']), archive)
    history = Path(config['storage']['local_cache_root'])/'device-day-multiview-history'/archive
    keep = {Path(o['json_path']).parent.as_posix() for e in experiments for o in e.get('outputs', [])
            if o['archive'] == archive}
    moved = []
    for folder in (root/'ProcessedClips/Clips').glob('*'):
        if folder.is_symlink() or not folder.is_dir() or folder.relative_to(root).as_posix() in keep:
            continue
        metadata = folder/'ExperimentActivity.json'
        if not metadata.is_file() or metadata.is_symlink():
            continue
        data = read_json(metadata)
        if (data.get('scope') != 'aligned_experiment'
                or data.get('materializer') != 'visioncortex.archive.materialize_experiment_clips'):
            continue
        children = list(folder.iterdir())
        if any(p.is_symlink() or not p.is_file() or p.name not in {
                'ExperimentActivity.json','ExperimentActivity.mp4','AlignedFirstThird.mp4'} for p in children):
            continue
        identity = digest(data)
        destination = history/identity/folder.name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        receipt = {'from': str(folder), 'to': str(destination), 'metadata_digest': identity,
                   'reason': 'superseded_multiview_projection', 'source_media_changed': False,
                   'historical_references': 'resolve_original_prefix_using_from_to_mapping'}
        atomic_json(destination.parent/'Relocation.json', receipt | {'status': 'prepared'})
        folder.rename(destination)
        atomic_json(destination.parent/'Relocation.json', receipt | {'status': 'completed'})
        moved.append(receipt)
    return moved


def _project_groups(analysis, sources, config):
    """Map shared experiment boundaries back to immutable device/day media."""
    from .schemas import AlignmentTransform
    from .device_day_models import capture_us
    transforms = {k: AlignmentTransform.model_validate(v) for k, v in analysis['transforms'].items()}
    experiments = []
    for group in analysis['groups']:
        refs = []
        for camera in group['participating_views']:
            tr = transforms[camera]
            left, right = tr.to_local(group['global_start_ms']), tr.to_local(group['global_end_ms'])
            for s in sources[camera]:
                info = s['info']
                a, b = max(left, info.virtual_start_ms), min(right, info.virtual_end_ms)
                if a >= b:
                    continue
                clock = s['record']['processing'].get('clock_mapping') or {'origin_us': s['record']['start_us']}
                refs.append({'archive': s['archive'], 'camera': camera, 'recording_id': s['record']['recording_id'],
                             'source_ref': s['refs']['video'], 'start_ms': a-info.virtual_start_ms, 'end_ms': b-info.virtual_start_ms,
                             'global_start_ms': tr.to_global(a), 'global_end_ms': tr.to_global(b),
                             'start_us': capture_us(clock, a-info.virtual_start_ms), 'end_us': capture_us(clock, b-info.virtual_start_ms)})
        renderer_identity = [config['performance'].get('ffmpeg_video_encoder'),
                             *[file_hash(Path(__file__).with_name(m+'.py'))
                               for m in ('archive', 'video_io', 'workflow_video')]]
        group_segments = [s for s in analysis['segments'] if s['segment_id'] in group['atomic_experiment_ids']]
        group_events = [e for e in analysis['events'] if any(e['event_id'] in s['event_ids'] for s in group_segments)]
        group_transforms = {v: analysis['transforms'][v] for v in group['participating_views']}
        experiments.append({'experiment_id': 'AlignedExperiment'+digest([group, refs, group_events, group_transforms, renderer_identity])[:24],
                            'group': group, 'sources': refs,
                            'atomic_segments': group_segments,
                            'events': group_events,
                            'alignment_quality': analysis['alignment_quality'],
                            'transforms': group_transforms,
                            'key_events': [e for e in analysis['selected_key_events'] if e['event_id'] in group['key_event_ids']],
                            'status': 'offline_group_selected', 'physical_action_confirmed': False,
                            'evidence_status': 'PARTIAL_EVIDENCE'})
    return experiments
