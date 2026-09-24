"""Reviewable stage dependencies and selective rebuild planning.

This is a new identity recipe, never a compatibility whitelist. Historical
receipts keep their own recipe and are not rewritten by this planner.
"""
from pathlib import Path
import hashlib

DEVICE_SOURCES = {
    'retention': ['nas_recordings', 'device_day_audio'],
    'vision': ['shared_inference', 'scan_scheduler', 'device_day_models', 'device_day_audit', 'detection', 'detection_inference',
               'actions', 'candidate_index', 'video_io', 'cuda_decode_admission', 'alignment', 'movement_verification', 'coarse_recall',
               'grouping', 'action_state_machine', 'action_semantics', 'archive', 'open_vocabulary_runtime', 'source_frames', 'media_time'],
    'stt': ['device_day_stt', 'speech', 'speech_worker', 'speech_qwen'],
    'understanding': ['device_day_models', 'device_day_audit', 'device_day_semantic_cache', 'scene_requests', 'scene_transport',
                      'mllm', 'mllm_provider', 'video_io', 'media_time'],
    'report': ['device_day_reports', 'daily_reports', 'report_presentations'],
}
CONTROL_MODULES = frozenset({
    'runtime_control', 'runtime_options', 'runtime_process', 'sqlite_store', 'observed_inventory',
    'receipt_projection', 'publication_journal', 'provider_control', 'build_identity', 'owned_subprocess',
    'runtime_services', 'task_events', 'input_availability', 'submission', 'knowledge', 'knowledge_api',
    'stage_execution', 'stage_dependencies', 'artifact_reader', 'timeline_invalidation', 'multimodal_usage',
})


def impact(paths):
    names = {Path(path).stem for path in paths if Path(path).suffix == '.py'}
    changed = {stage for stage, sources in DEVICE_SOURCES.items() if names.intersection(sources)}
    known = CONTROL_MODULES | {name for items in DEVICE_SOURCES.values() for name in items} | {'api', 'cli'}
    if names - known:
        changed.update(DEVICE_SOURCES)  # Unknown execution edits fail closed.
    from .stage_execution import required_stages
    affected = [stage for stage in DEVICE_SOURCES if changed.intersection(required_stages(stage))]
    return {'changed_stages': sorted(changed), 'affected_stages': affected,
            'runtime_restart_required': bool(names & CONTROL_MODULES),
            'historical_receipts_rewritten': False, 'automatic_cache_acceptance': False}


def source_manifest(directory=None):
    root = directory or Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def compare(before, after):
    paths = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    return {'changed_files': paths, **impact(paths)}
