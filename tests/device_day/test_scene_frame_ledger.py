"""Deterministic request-ledger and stage-continuity checks; no model startup."""
import ast
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from repo_paths import ROOT
from test_device_day import FakeModels, capture, item_and_layout

from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import read_json
from visioncortex.scene_transport import (
    FRAME_LEDGER_VERSION,
    compact,
    expand,
    merge_frame_ledger,
    prepare_frame_ledger,
    project,
)

BASELINE_SOURCES = {
    'scene_requests.py': '2efc95923cb363235a4e23b646d614671588c30768533286f38c6ee06c98d34e',
    'scene_transport.py': 'd9092d64080d98528263dd25264fe4f596bfbdab70af5d177cac02c5d95e0264',
}


def frame(identifier='f1', **fields):
    return {'frame_id': identifier, 'sha256': 'a' * 64, 'size_bytes': 100,
            'local_ms': 100, 'path': 'SceneFrames/F1.jpg',
            'source_path': 'MetaVideo/Source.mp4', 'frame_kind': 'scene_sample', **fields}


def test_duplicate_extra_references_form_one_image_with_all_selected_events():
    event_a = frame(frame_kind='action_keyframe', event_id='event-a', action_type='tip_change',
                    path='KeyFrames/A.jpg', selected_event_digest='b' * 64)
    event_b = frame(frame_kind='action_keyframe', event_id='event-b', action_type='hand_contact',
                    path='KeyFrames/B.jpg', selected_event_digest='c' * 64)
    samples, references = [frame('sample', sha256='d' * 64)], [event_a, event_b]
    original = deepcopy((samples, references))
    images = merge_frame_ledger([*samples, *references])
    assert [image['frame_id'] for image in images] == ['sample', 'f1']
    assert images[1]['path'] == event_a['path']
    assert [a['event_id'] for a in images[1]['frame_associations']] == ['event-a', 'event-b']
    assert (samples, references) == original
    wire, _, aliases = compact({'frames': images}, [(f['frame_id'], Path(f['path'])) for f in images])
    assert len(wire['frames']) == 2
    restored = expand({'frame_observations': [{'frame_id': identifier} for identifier in aliases['frame_aliases']],
                       'steps': []}, aliases)
    assert [f['frame_id'] for f in restored['frame_observations']] == ['sample', 'f1']


def test_sample_event_overlap_retains_sample_kind_and_action_association():
    image = frame()
    action = frame(frame_kind='action_keyframe', event_id='event-a', action_type='tip_change',
                   key_frame_selection={'ledger_row': 17}, path='KeyFrames/A.jpg')
    images = merge_frame_ledger([image, action])
    assert len(images) == 1 and images[0]['frame_kind'] == 'scene_sample'
    assert images[0]['frame_associations'][1]['event_id'] == 'event-a'
    assert images[0]['frame_associations'][1]['key_frame_selection'] == {'ledger_row': 17}
    assert len(merge_frame_ledger([*images, action])[0]['frame_associations']) == 2


@pytest.mark.parametrize('fields', [
    {'sha256': 'b' * 64}, {'size_bytes': 101}, {'local_ms': 101},
    {'source_path': 'MetaVideo/Other.mp4'}, {'capture_us': 201},
    {'source_native_index': 11}, {'source_frame_index_estimate': 11},
])
def test_same_identifier_cannot_hide_changed_pixels_source_or_time(fields):
    original = frame(capture_us=200, source_native_index=10, source_frame_index_estimate=10)
    with pytest.raises(ValueError, match='conflicting content or time'):
        merge_frame_ledger([original, original | fields])


@pytest.mark.parametrize('name', ['capture_us', 'source_native_index', 'source_frame_index_estimate'])
@pytest.mark.parametrize('value', [None, 10])
def test_optional_native_identity_presence_cannot_be_discarded(name, value):
    with pytest.raises(ValueError, match='conflicting content or time'):
        merge_frame_ledger([frame(), frame(**{name: value})])
    with pytest.raises(ValueError, match='conflicting content or time'):
        merge_frame_ledger([frame(**{name: value}), frame()])


@pytest.mark.parametrize('name', ['capture_us', 'source_native_index', 'source_frame_index_estimate'])
def test_native_identity_null_cannot_equal_a_verified_value(name):
    with pytest.raises(ValueError, match='conflicting content or time'):
        merge_frame_ledger([frame(**{name: None}), frame(**{name: 10})])


def test_nested_native_source_proof_cannot_be_discarded_or_conflict():
    source = {'source_frame': {'source_pts': 42, 'packet_position': 100}, 'status': 'verified'}
    verified = frame(source_frame_verification=source)
    for other in (frame(), frame(source_frame_verification=None),
                  frame(source_frame_verification={**source, 'source_frame': {'source_pts': 43, 'packet_position': 101}})):
        with pytest.raises(ValueError, match='conflicting content or time'):
            merge_frame_ledger([verified, other])
    assert merge_frame_ledger([verified, verified])[0]['source_frame_verification'] == source


@pytest.mark.parametrize('fields', [
    {'sha256': None}, {'size_bytes': 0}, {'local_ms': float('nan')}, {'source_path': ''},
])
def test_merge_requires_content_identity(fields):
    with pytest.raises(ValueError, match='identity is incomplete'):
        merge_frame_ledger([frame(), frame(**fields)])


def test_unique_ledger_retains_paid_request_semantics():
    metadata = {'frames': [frame('sample'), frame('left', local_ms=0)], 'physical_action_confirmed': False}
    original = deepcopy(metadata)
    paths = [(f['frame_id'], Path(f['path'])) for f in metadata['frames']]
    assert prepare_frame_ledger(metadata, paths) is paths
    assert metadata == original and project(metadata)[0] == project(original)[0]
    with pytest.raises(ValueError, match='Duplicate scene frame identifiers'):
        project({'frames': [frame(), frame()]})


@pytest.fixture
def device_config(default_config, tmp_path):
    from test_device_day import device_config as fixture
    config = fixture.__wrapped__(default_config, tmp_path)
    config['device_day']['understanding_coverage_since_us'] = None
    config['runtime']['local_only'] = True
    return config


def test_only_understanding_recipe_changes_and_completed_cv_stt_are_reused(device_config, monkeypatch):
    from visioncortex import stage_dependencies
    capture(device_config)
    record, layout = item_and_layout(device_config)
    before_backend = FakeModels()
    before = DeviceDayRunner(device_config, before_backend)
    original_hash = before._hash
    manifest = stage_dependencies.source_manifest() if hasattr(stage_dependencies, 'stage_source_manifest') else None
    def closures():
        if manifest is not None:
            return {stage: stage_dependencies.stage_source_manifest(stage) for stage in ('vision', 'stt')}
        return {stage: {name: hashlib.sha256((ROOT / 'src/visioncortex' / f'{name}.py').read_bytes()).hexdigest()
                        for name in stage_dependencies.DEVICE_SOURCES[stage]}
                for stage in ('vision', 'stt')}
    with monkeypatch.context() as previous:
        previous.setattr(before, '_hash', lambda p: BASELINE_SOURCES.get(Path(p).name, original_hash(p)))
        if manifest is not None:
            previous.setattr(stage_dependencies, 'source_manifest', lambda directory=None: manifest | BASELINE_SOURCES)
        previous_closures = closures()
        keys = {stage: before._key(stage, record, {}) for stage in ('vision', 'stt', 'understanding')}
        assert before.process(record, stage='retention')['status'] == 'completed'
        assert before.process(record, stage='vision')['status'] == 'completed'
        assert before.process(record, stage='stt')['status'] == 'completed'
    old_receipts = {stage: before._receipt(layout, record, stage).read_bytes() for stage in ('vision', 'stt')}
    after_backend = FakeModels()
    after = DeviceDayRunner(device_config, after_backend)
    assert closures() == previous_closures
    assert after._key('vision', record, {}) == keys['vision']
    assert after._key('stt', record, {}) == keys['stt']
    assert after._key('understanding', record, {}) != keys['understanding']
    for stage in ('vision', 'stt'):
        assert after.process(record, stage=stage)['status'] == 'completed'
        assert after._receipt(layout, record, stage).read_bytes() == old_receipts[stage]
    assert before_backend.vision_calls == 1 and after_backend.vision_calls == 0
    assert read_json(after._receipt(layout, record, 'stt'))['model_invocation'] == 'NOT_PROVEN'
    original_hash = after._hash
    monkeypatch.setattr(after, '_hash', lambda p: 'f' * 64 if Path(p).name == 'scene_transport.py' else original_hash(p))
    assert after._key('vision', record, {}) == keys['vision']
    assert after._key('stt', record, {}) == keys['stt']
    assert after._key('understanding', record, {}) != keys['understanding']
    assert FRAME_LEDGER_VERSION == 'visioncortex-scene-frame-ledger/2'


def test_cv_owner_and_decoder_are_byte_identical_to_reviewed_baseline():
    path = ROOT / 'src/visioncortex/device_day_models.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'b526a59180791dc2f95cad7b866662e98e96c35e2c6b50b172bd877cdd4a45b4'


def test_other_vision_dependency_change_invalidates_its_stage(device_config, monkeypatch):
    capture(device_config)
    record, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    keys = {stage: runner._key(stage, record, {}) for stage in ('vision', 'stt')}
    original = runner._hash
    monkeypatch.setattr(runner, '_hash', lambda p: 'e' * 64 if Path(p).name == 'detection.py' else original(p))
    assert runner._key('vision', record, {}) != keys['vision']
    assert runner._key('stt', record, {}) == keys['stt']


def test_conflict_rejection_does_not_mutate_the_caller_or_parent_ledger():
    parent = [frame(), frame(sha256='b' * 64)]
    images = list(parent)
    metadata = {'frames': images}
    old = deepcopy(metadata)
    with pytest.raises(ValueError, match='conflicting content or time'):
        prepare_frame_ledger(metadata, [('f1', Path('A.jpg')), ('f1', Path('B.jpg'))])
    assert metadata == old and parent == old['frames']


@pytest.fixture
def scene(default_config, tmp_path):
    from test_scene_requests import scene as fixture
    return fixture.__wrapped__(default_config, tmp_path)


def test_window_validation_receipt_and_model_share_the_unique_ledger(scene):
    from test_scene_requests import Analyzer, run
    _, metadata, paths, folder = scene
    first = metadata['frames'][0]
    parent = deepcopy(first)
    metadata['frames'].extend([first | {'event_id': 'event-a'}, first | {'event_id': 'event-b'}])
    paths.extend([paths[0], paths[0]])
    class InspectingAnalyzer(Analyzer):
        def _call(self, prompt, request, images, **kwargs):
            assert len(request['frames']) == len(images) == 2
            assert request['frame_ledger_version'] == FRAME_LEDGER_VERSION
            assert [a.get('event_id') for a in request['frames'][0]['frame_associations']] == [None, 'event-a', 'event-b']
            return super()._call(prompt, request, images, **kwargs)
    analyzer = InspectingAnalyzer()
    raw, parsed, reused = run(scene, analyzer)
    assert not reused and analyzer.calls == 1
    assert len(parsed['frame_observations']) == len(metadata['frames']) == 2
    assert read_json(folder / 'Input.json')['metadata']['frames'] == metadata['frames']
    assert first == parent
    assert raw['status'] == 'completed'


def test_unique_window_retains_existing_paid_cache_identity(scene, monkeypatch):
    from test_scene_requests import Analyzer, run

    from visioncortex import scene_requests
    from visioncortex.device_day_cache_identity import _normalized
    from visioncortex.device_day_models import SCENE_PROMPT
    from visioncortex.device_day_semantic_cache import identity
    config, metadata, paths, _ = scene
    old_metadata = deepcopy(metadata)
    old_identity = identity(config, SCENE_PROMPT, metadata)
    analyzer = Analyzer()
    # Reconstruct only the prior reader statement boundary. Its complete AST
    # must match the fixed pre-repair owner, rather than assume cache parity.
    source = Path(scene_requests.__file__).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'window')
    node.body = [n for n in node.body if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id == 'prepare_frame_ledger')]
    paid_guard = ast.parse("CURRENT.get().source == 'device_day_backfill' or CURRENT.get().yield_signal is not None").body[0].value
    guards = [n for n in ast.walk(node) if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(paid_guard)]
    assert len(guards) == 1
    guards[0].test = guards[0].test.values[0]
    assert hashlib.sha256(json.dumps(_normalized(node), sort_keys=True).encode()).hexdigest() == 'd6c88e0bdca370143a98bb9d9d6b5c1b8e35a891ec3466b7f9a986906972cf8e'
    namespace = dict(vars(scene_requests))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<prior-window-reader>', 'exec'), namespace)  # noqa: S102 - pinned, locally owned AST above
    with monkeypatch.context() as previous:
        previous.setattr(scene_requests, 'window', namespace['window'])
        assert run(scene, analyzer)[2] is False
    assert metadata == old_metadata and identity(config, SCENE_PROMPT, metadata) == old_identity
    assert prepare_frame_ledger(metadata, paths) is paths
    assert run(scene, analyzer)[2] is True
    assert analyzer.calls == 1


def test_changed_understanding_owner_keeps_full_cv_scan_identity(default_config, tmp_path, monkeypatch):
    from test_scan_dependencies import setup as fixture

    from visioncortex import stage_dependencies
    from visioncortex.device_day_models import DeviceDayModels
    config, retention, info = fixture.__wrapped__(default_config, tmp_path)
    model = DeviceDayModels(config)
    manifest = stage_dependencies.source_manifest() if hasattr(stage_dependencies, 'stage_source_manifest') else None
    with monkeypatch.context() as previous:
        if manifest is not None:
            previous.setattr(stage_dependencies, 'source_manifest', lambda directory=None: manifest | BASELINE_SOURCES)
        before = {phase: model.scan_identity(retention, info, phase, [(0, 1000)], 2, 640)
                  for phase in ('coarse', 'fine')}
    assert {phase: model.scan_identity(retention, info, phase, [(0, 1000)], 2, 640)
            for phase in before} == before
    original = model._identity_hash
    monkeypatch.setattr(model, '_identity_hash', lambda p: 'e' * 64 if p.name == 'source_frames.py' else original(p))
    assert all(model.scan_identity(retention, info, phase, [(0, 1000)], 2, 640) != identity
               for phase, identity in before.items())
