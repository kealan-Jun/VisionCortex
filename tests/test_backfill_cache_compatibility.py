"""Read old receipts without presenting the recursive recipe as old execution."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from visioncortex import device_day, device_day_cache_identity, device_day_inplace
from visioncortex import device_day_runtime_identity, stage_dependencies
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import STAGES, VERSION, digest, read_json
from visioncortex.ast_identity import dump_python312
from device_day_identity_baseline_v1 import (
    DEVICE_SOURCES_V1, EXECUTION_IDENTITY_V1, INPLACE_EXECUTION_IDENTITY_V1,
    SOURCE_SHA256_V1, bind_stage_key_v1, runtime_recipes_v1,
)



def function_namespace(source, name, filename, *, class_name=None, authored_python312=False):
    """Execute only the requested identity recipe, never old runtime classes."""
    tree = ast.parse(source)
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef)
                     and node.name == class_name).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    function = deepcopy(function)
    namespace = {'__file__': str(filename), '__package__': 'visioncortex',
                 '__name__': 'visioncortex._baseline_cache_test', 'Path': Path,
                 'Any': Any, 'deepcopy': deepcopy, 'VERSION': VERSION, 'digest': digest}
    if authored_python312:
        # The fixed receipt was authored with Python 3.12. Adapt only its AST
        # representation, leaving the historical recipe/source untouched.
        def receipt_dump(node, *, include_attributes=False):
            assert not include_attributes
            return dump_python312(node)
        namespace['ast'] = SimpleNamespace(parse=ast.parse, FunctionDef=ast.FunctionDef,
                                           dump=receipt_dump)
        function.body = [node for node in function.body
                         if not (isinstance(node, ast.Import)
                                 and [alias.name for alias in node.names] == ['ast'])]
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(filename), 'exec'), namespace)
    return namespace[name]


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    # Virtual local source paths select immutable identity witnesses; no old
    # operational module is copied, imported or required from Git history.
    root = tmp_path_factory.mktemp('identity-v1')
    return SimpleNamespace(root=root, key=bind_stage_key_v1(root),
                           dependencies=deepcopy(DEVICE_SOURCES_V1),
                           runtime=vars(runtime_recipes_v1(root)),
                           execution=EXECUTION_IDENTITY_V1,
                           inplace_execution=INPLACE_EXECUTION_IDENTITY_V1)


@contextmanager
def baseline_recipes(baseline):
    # _key imports these recipes at invocation time. Replace them only while
    # calculating the fixed old descriptor; the new descriptor runs normally.
    with pytest.MonkeyPatch.context() as patch:
        for name in ('compatible_runtime_hash', 'compatible_performance', 'independent_vision_backend_hash'):
            patch.setattr(device_day_runtime_identity, name, baseline.runtime[name])
        patch.setattr(stage_dependencies, 'DEVICE_SOURCES', baseline.dependencies)
        yield


def old_runner(config, baseline, backend=None):
    runner = DeviceDayRunner(config, backend or FakeModels())
    runner._key = MethodType(baseline.key, runner)
    runner._execution_identity = baseline.execution
    runner._inplace_execution_identity = baseline.inplace_execution
    actual_hash = runner._hash

    def source_witness(path):
        path = Path(path)
        if path.parent.resolve() == baseline.root.resolve():
            return SOURCE_SHA256_V1[path.name]
        return actual_hash(path)

    runner._hash = source_witness
    return runner


def mode_config(config, mode, inplace):
    value = deepcopy(config)
    value['device_day']['inplace_preprocessing'] = inplace
    # Tests never read model binaries; the supplied backend is deterministic.
    value['models'] = {}
    value['device_day'].pop('understanding_coverage_since_us', None)
    value['device_day'].pop('experiment_steps_since_us', None)
    if mode in {'full_coverage', 'structured_steps'}:
        value['device_day']['understanding_coverage_since_us'] = 1
    if mode == 'structured_steps':
        value['device_day']['experiment_steps_since_us'] = 1
    return value


def enable_backfill(config):
    value = deepcopy(config)
    value['device_day']['backfill'] = {
        'enabled': True, 'stages': ['vision', 'understanding', 'report'],
        'idle_seconds': 15, 'quantum_seconds': 30, 'failure_cooldown_seconds': 60,
    }
    return value


def key_inputs(record):
    return {
        'retention': deepcopy(record),
        'vision': {'recording': {'recording_id': record['recording_id']},
                   'sources': [{'sha256': 'synthetic-source-A', 'size_bytes': 31}]},
        'stt': {'recording': deepcopy(record), 'audio': {'status': 'no_input'}},
        'understanding': {'vision': {'key': 'synthetic-vision'}, 'stt': {'key': 'synthetic-stt'},
                          'context': {'comments': [], 'protocol': None}},
        'report': {'key': 'synthetic-understanding'},
    }


@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('mode', ['legacy', 'full_coverage', 'structured_steps'])
def test_recursive_recipe_invalidates_old_keys_but_backfill_scheduling_does_not(device_config, baseline, inplace, mode):  # noqa: F811
    config = mode_config(device_config, mode, inplace)
    capture(config)
    record, _ = item_and_layout(config)
    old = old_runner(config, baseline)
    if inplace:
        assert device_day_inplace.active(old, record, initialize=True)
    inputs = key_inputs(record)
    with baseline_recipes(baseline):
        historical = {stage: old._key(stage, record, inputs[stage]) for stage in STAGES}
    current_keys = None
    for current_config in (config, enable_backfill(config)):
        current = DeviceDayRunner(current_config, FakeModels())
        actual = {stage: current._key(stage, record, inputs[stage]) for stage in STAGES}
        assert all(actual[stage] != historical[stage] for stage in STAGES)
        if current_keys is None:
            current_keys = actual
        else:
            assert actual == current_keys


class CountingModels(FakeModels):
    def __init__(self):
        super().__init__()
        self.stt_calls = 0

    def transcribe(self, *args):
        self.stt_calls += 1
        return super().transcribe(*args)


@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('mode', ['legacy', 'full_coverage', 'structured_steps'])
def test_historical_receipts_remain_readable_but_new_recipe_executes_and_keeps_history(device_config, baseline, inplace, mode):  # noqa: F811
    config = mode_config(device_config, mode, inplace)
    capture(config)
    record, layout = item_and_layout(config)
    backend = CountingModels()
    old = old_runner(config, baseline, backend)
    with baseline_recipes(baseline):
        assert old.process(record)['status'] == 'completed'
    assert backend.vision_calls == backend.semantic_calls == backend.stt_calls == 1
    receipts = {stage: read_json(old._receipt(layout, record, stage)) for stage in STAGES}
    saved_bytes = {stage: old._receipt(layout, record, stage).read_bytes() for stage in STAGES}
    current_backend = CountingModels()
    current = DeviceDayRunner(enable_backfill(config), current_backend)
    inputs = {
        'retention': record,
        'vision': device_day.visual_input(receipts['retention']),
        'stt': receipts['retention'],
        'understanding': {'vision': receipts['vision'], 'stt': receipts['stt'],
                          'context': device_day.load_context(layout, record)},
        'report': receipts['understanding'],
    }
    for stage in STAGES:
        path = old._receipt(layout, record, stage)
        # Reading a verified historical result is separate from accepting it
        # under a new execution identity. Neither read rewrites the receipt.
        assert current._load(path, receipts[stage]['key'], layout) == receipts[stage]
        new_key = current._key(stage, record, inputs[stage])
        assert new_key != receipts[stage]['key']
        assert current._load(path, new_key, layout) is None
        assert path.read_bytes() == saved_bytes[stage]
    assert current_backend.vision_calls == current_backend.semantic_calls == current_backend.stt_calls == 0
    assert current.process(record)['status'] == 'completed'
    assert current_backend.vision_calls == current_backend.semantic_calls == current_backend.stt_calls == 1
    for stage in STAGES:
        path = current._receipt(layout, record, stage)
        assert read_json(path)['key'] != receipts[stage]['key']
        history = path.parent / 'history' / f'{stage}-{digest(receipts[stage])}.json'
        assert history.read_bytes() == saved_bytes[stage]
    assert current.process(record)['status'] == 'completed'
    assert current_backend.vision_calls == current_backend.semantic_calls == current_backend.stt_calls == 1


def test_new_retention_key_cannot_overwrite_history_when_history_write_fails(device_config, baseline, monkeypatch):  # noqa: F811
    config = mode_config(device_config, 'legacy', True)
    capture(config)
    record, layout = item_and_layout(config)
    old = old_runner(config, baseline)
    with baseline_recipes(baseline):
        assert old.process(record)['status'] == 'completed'
    path = old._receipt(layout, record, 'retention')
    saved_bytes = path.read_bytes()
    current = DeviceDayRunner(config, CountingModels())
    write = device_day_inplace.durable_json
    def history_unavailable(destination, value):
        if destination.parent.name == 'history' and destination.name.startswith('retention-'):
            raise OSError('retention history unavailable')
        return write(destination, value)
    monkeypatch.setattr(device_day_inplace, 'durable_json', history_unavailable)
    with pytest.raises(OSError, match='retention history unavailable'):
        current.process(record, stage='retention')
    assert path.read_bytes() == saved_bytes


@pytest.mark.parametrize('legacy_understanding', [False, True])
def test_changed_recursive_scan_backend_keeps_its_own_identity(baseline, legacy_understanding):
    old_path = baseline.root / 'device_day_models.py'
    old_checksum = SOURCE_SHA256_V1['device_day_models.py']
    old_canonical = baseline.runtime['compatible_runtime_hash'](old_path, old_checksum)
    expected = baseline.runtime['independent_vision_backend_hash'](
        old_path, old_canonical, legacy_understanding=legacy_understanding)
    current_path = Path(device_day.__file__).with_name('device_day_models.py')
    current_checksum = hashlib.sha256(current_path.read_bytes()).hexdigest()
    current_canonical = device_day_runtime_identity.compatible_runtime_hash(current_path, current_checksum)
    assert current_canonical == current_checksum
    assert device_day_runtime_identity.independent_vision_backend_hash(
        current_path, current_canonical, legacy_understanding=legacy_understanding) == current_checksum
    assert current_checksum != expected


@pytest.mark.parametrize('filename,stage', [
    ('detection.py', 'vision'), ('shared_inference.py', 'vision'),
    ('source_frames.py', 'vision'), ('device_day_stt.py', 'stt'),
    ('scene_requests.py', 'understanding'), ('device_day_reports.py', 'report'),
    ('device_day_contract.py', 'retention'), ('video_io.py', 'vision'),
    ('device_day_models.py', 'vision'), ('device_day_inputs.py', 'stt'),
])
def test_unknown_source_hash_is_never_accepted_by_reviewed_aliases(device_config, monkeypatch, filename, stage):  # noqa: F811
    config = mode_config(device_config, 'full_coverage', filename == 'device_day_inputs.py')
    capture(config)
    record, _ = item_and_layout(config)
    current = DeviceDayRunner(config, FakeModels())
    if filename == 'device_day_inputs.py':
        assert device_day_inplace.active(current, record, initialize=True)
    inputs = key_inputs(record)[stage]
    expected = current._key(stage, record, inputs)
    original_hash = current._hash
    unknown = hashlib.sha256(('unreviewed semantic edit:' + filename).encode()).hexdigest()
    monkeypatch.setattr(current, '_hash', lambda path:
                        unknown if Path(path).name == filename else original_hash(path))
    assert current._key(stage, record, inputs) != expected


@pytest.mark.parametrize('legacy_understanding', [False, True])
def test_unknown_backend_semantic_ast_edit_remains_unaliased(tmp_path, legacy_understanding):
    path = Path(device_day.__file__).with_name('device_day_models.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DeviceDayModels')
    vision = next(node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name == 'vision')
    vision.body.insert(0, ast.Return(value=ast.Constant(value='unreviewed_semantic_change')))
    modified = tmp_path / path.name
    modified.write_text(ast.unparse(ast.fix_missing_locations(tree)), encoding='utf-8')
    checksum = hashlib.sha256(modified.read_bytes()).hexdigest()
    assert device_day_runtime_identity.compatible_runtime_hash(modified, checksum) == checksum
    assert device_day_runtime_identity.independent_vision_backend_hash(
        modified, checksum, legacy_understanding=legacy_understanding) == checksum
    if legacy_understanding:
        # Exercise the legacy AST branch itself, rather than only its early
        # checksum guard: an altered CV AST cannot gain the historical alias.
        legacy_checksum = 'fd755dac9528bbc51d868de8ce8b08ed29cf43da4f1724c411af9927d8d2c4e4'
        assert device_day_runtime_identity.independent_vision_backend_hash(
            modified, legacy_checksum, legacy_understanding=True) == legacy_checksum


def test_unknown_inplace_execution_semantic_edit_remains_unaliased(tmp_path, baseline):
    path = Path(device_day_inplace.__file__)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    execute.body.insert(0, ast.Return(value=ast.Constant(value='unreviewed_execution_change')))
    modified = tmp_path / path.name
    modified.write_text(ast.unparse(ast.fix_missing_locations(tree)), encoding='utf-8')
    identity = function_namespace(modified.read_text(encoding='utf-8'), 'execution_identity', modified)
    assert identity() != baseline.inplace_execution


def test_unknown_execution_semantics_and_real_input_changes_still_invalidate(device_config, baseline):  # noqa: F811
    source = Path(device_day.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DeviceDayRunner')
    execute = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == '_process')
    execute.body.insert(0, ast.Return(value=ast.Constant(value='unreviewed_execution_change')))
    assert device_day_cache_identity.execution_identity(ast.unparse(ast.fix_missing_locations(tree))) != baseline.execution
    config = mode_config(device_config, 'full_coverage', False)
    capture(config)
    record, _ = item_and_layout(config)
    current = DeviceDayRunner(config, FakeModels())
    inputs = key_inputs(record)
    initial = {stage: current._key(stage, record, inputs[stage]) for stage in STAGES}
    revised = record | {'source_signature': 'synthetic-source-changed'}
    revised_inputs = deepcopy(inputs)
    revised_inputs['retention'] = revised
    revised_inputs['vision']['sources'][0]['sha256'] = 'synthetic-source-changed'
    assert all(current._key(stage, revised, revised_inputs[stage]) != initial[stage] for stage in STAGES)
    changed = deepcopy(config)
    changed['device_day']['inactive_frames'] += 1
    different = DeviceDayRunner(changed, FakeModels())
    assert different._key('vision', record, inputs['vision']) != initial['vision']
    assert different._key('understanding', record, inputs['understanding']) != initial['understanding']
