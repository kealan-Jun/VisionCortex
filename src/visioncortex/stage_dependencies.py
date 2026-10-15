"""Reviewable stage dependencies and selective rebuild planning.

This is a new identity recipe, never a compatibility whitelist. Historical
receipts keep their own recipe and are not rewritten by this planner.
"""
from pathlib import Path

from .source_identity import SOURCE_RECIPE
from .source_identity import source_manifest as source_manifest

DEVICE_SOURCES = {
    'retention': ['nas_recordings', 'device_day_audio'],
    'vision': ['shared_inference', 'scan_scheduler', 'device_day_models', 'device_day_audit', 'detection', 'detection_inference',
               'actions', 'candidate_index', 'video_io', 'cuda_decode_admission', 'alignment', 'movement_verification', 'coarse_recall',
               'grouping', 'action_state_machine', 'action_semantics', 'archive', 'open_vocabulary_runtime', 'source_frames', 'media_time'],
    'stt': ['device_day_stt', 'speech', 'speech_worker', 'speech_qwen'],
    'understanding': ['device_day_models', 'device_day_audit', 'device_day_semantic_cache', 'scene_requests', 'scene_transport',
                      'device_day_call_templates', 'device_day_steps',
                      'mllm', 'mllm_provider', 'video_io', 'media_time', 'device_day_coverage', 'device_day_window_inputs'],
    'report': ['device_day_reports', 'daily_reports', 'report_presentations', 'device_day_narrative',
               'device_day_synthesis', 'device_day_call_templates', 'device_day_content', 'device_day_file_index'],
}
CONTROL_MODULES = frozenset({
    'runtime_control', 'runtime_options', 'runtime_process', 'sqlite_store', 'observed_inventory',
    'receipt_projection', 'publication_journal', 'provider_control', 'build_identity', 'owned_subprocess',
    'runtime_services', 'task_events', 'input_availability', 'submission', 'knowledge', 'knowledge_api',
    'stage_execution', 'stage_dependencies', 'artifact_reader', 'timeline_invalidation', 'multimodal_usage',
    'device_day_cohort', 'pricing_table', 'usage_reconciliation',
    'device_day_activity', 'device_day_queue', 'device_day_retention_worker', 'device_day_admission',
    'device_day_progress', 'device_day_overview', 'device_day_publication_worker',
    'device_day_write_guard',
})


def _changed_stages(paths):
    python_paths = {Path(path).as_posix() for path in paths if Path(path).suffix == '.py'}
    names = {Path(path).stem for path in python_paths if '/' not in path}
    changed = {stage for stage, sources in DEVICE_SOURCES.items() if names.intersection(sources)}
    packages = {path.split('/', 1)[0] for path in python_paths if '/' in path}
    # Package responsibilities are explicit. Unknown packages invalidate all
    # stages; a nested module never inherits a flat module's name-only policy.
    package_stages = {'analysis': {'vision'}, 'media': {'vision', 'understanding'},
                      'evidence': {'vision', 'understanding', 'report'},
                      'training': set(), 'training_runtime': set(), 'web_api': set()}
    for package in packages:
        changed.update(package_stages.get(package, DEVICE_SOURCES))
    # Context delivery and deterministic person projections do not execute a
    # visual model. Explicit identities keep their late revisions report-only.
    context_files = {'evidence/field_events.py', 'evidence/person_day.py', 'evidence/device_events.py',
                     'application/field_event_consumer.py'}
    if python_paths and python_paths <= context_files:
        changed = {'report'}
    if python_paths == {'application/retained_video.py'}:
        changed = {'report'}
    if python_paths == {'media/retained_intervals.py'}:
        changed = {'report'}
    if 'analysis/windows.py' in python_paths:
        changed.add('understanding')
    known = CONTROL_MODULES | {name for items in DEVICE_SOURCES.values() for name in items} | {'api', 'cli'}
    if names - known:
        changed.update(DEVICE_SOURCES)  # Unknown execution edits fail closed.
    return changed, names, packages


def impact(paths):
    changed, names, packages = _changed_stages(paths)
    from .stage_execution import required_stages
    affected = [stage for stage in DEVICE_SOURCES if changed.intersection(required_stages(stage))]
    return {'changed_stages': sorted(changed), 'affected_stages': affected,
            'runtime_restart_required': bool(names & CONTROL_MODULES or packages & {'web_api', 'application'}),
            'historical_receipts_rewritten': False, 'automatic_cache_acceptance': False}


def compare(before, after):
    paths = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    return {'source_recipe': SOURCE_RECIPE, 'changed_files': paths, **impact(paths)}


def stage_source_manifest(stage, directory=None):
    """Bind actual implementations, including moved files and unknown modules."""
    return {name: checksum for name, checksum in source_manifest(directory).items()
            if stage in _changed_stages([name])[0]
            or name in {'source_identity.py', 'stage_dependencies.py', 'stage_execution.py'}}
