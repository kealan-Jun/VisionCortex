import ast
import os
from pathlib import Path

import pytest

from visioncortex.source_identity import SOURCE_RECIPE, source_digest, source_manifest
from visioncortex.stage_dependencies import compare, impact, stage_source_manifest
from visioncortex.ast_identity import dump_python312
from visioncortex import source_identity


def test_recursive_manifest_binds_nested_edits_additions_removals_and_names(tmp_path):
    package = tmp_path / "analysis"
    package.mkdir()
    flat = tmp_path / "events.py"
    nested = package / "events.py"
    flat.write_text("flat = 1\n")
    nested.write_text("nested = 1\n")
    first = source_manifest(tmp_path)
    assert set(first) == {"events.py", "analysis/events.py"}
    nested.write_text("nested = 2\n")
    second = source_manifest(tmp_path)
    assert source_digest(first) != source_digest(second)
    assert compare(first, second)["changed_files"] == ["analysis/events.py"]
    nested.rename(package / "renamed.py")
    third = source_manifest(tmp_path)
    assert source_digest(second) != source_digest(third)
    (package / "renamed.py").unlink()
    assert source_digest(third) != source_digest(source_manifest(tmp_path))


def test_exclusions_use_relative_paths_not_colliding_basenames(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "api.py").write_text("control = 1\n")
    (tmp_path / "nested/api.py").write_text("execution = 1\n")
    assert set(source_manifest(tmp_path, exclude={"api.py"})) == {"nested/api.py"}


def test_source_links_fail_closed_without_reading_external_content(tmp_path):
    external = tmp_path / "outside.txt"
    external.write_text("unreviewed")
    (tmp_path / "linked.py").symlink_to(external)
    with pytest.raises(ValueError, match="linked.py"):
        source_manifest(tmp_path)


def test_missing_or_non_directory_source_root_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        source_manifest(tmp_path / "missing")
    source_file = tmp_path / "file.py"
    source_file.write_text("value = 1\n")
    with pytest.raises(ValueError, match="existing directory"):
        source_manifest(source_file)


def test_source_scan_errors_propagate_without_partial_manifest(tmp_path, monkeypatch):
    def inaccessible_walk(root, *, followlinks, onerror):
        yield str(root), ["private"], ["visible.py"]
        onerror(PermissionError("unreadable source directory"))
    (tmp_path / "visible.py").write_text("value = 1\n")
    monkeypatch.setattr(source_identity.os, "walk", inaccessible_walk)
    with pytest.raises(PermissionError, match="unreadable source directory"):
        source_manifest(tmp_path)


def test_directory_source_links_also_fail_closed(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "linked_package").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="linked_package"):
        source_manifest(tmp_path)


def test_stage_manifest_covers_actual_nested_code_without_training_coupling(tmp_path):
    (tmp_path / 'analysis').mkdir()
    (tmp_path / 'training').mkdir()
    (tmp_path / 'analysis/audit.py').write_text('verdict = 1\n')
    (tmp_path / 'training/run.py').write_text('train = 1\n')
    manifest = stage_source_manifest('vision', tmp_path)
    assert set(manifest) == {'analysis/audit.py'}
    (tmp_path / 'analysis/audit.py').write_text('verdict = 2\n')
    assert stage_source_manifest('vision', tmp_path) != manifest


def test_equal_size_metadata_collisions_still_bind_changed_content(tmp_path, monkeypatch):
    path = tmp_path / 'run.py'
    path.write_text('value = 1\n')
    monkeypatch.setattr(source_identity, '_snapshot', lambda _: (1, 2, 10, 3, 3))
    before = source_manifest(tmp_path)
    path.write_text('value = 2\n')
    assert source_manifest(tmp_path) != before


def test_mid_inventory_content_change_fails_even_with_identical_metadata(tmp_path, monkeypatch):
    path = tmp_path / 'run.py'
    path.write_text('value = 1\n')
    original = Path.read_bytes
    def changing_read(current):
        content = original(current)
        if current == path:
            path.write_text('value = 2\n')
        return content
    monkeypatch.setattr(source_identity, '_snapshot', lambda _: (1, 2, 10, 3, 3))
    monkeypatch.setattr(Path, 'read_bytes', changing_read)
    with pytest.raises(ValueError, match='Source changed'):
        source_manifest(tmp_path)


def test_device_dependency_replacement_with_same_size_and_mtime_gets_new_hash(tmp_path):
    from visioncortex.device_day import DeviceDayRunner
    runner = DeviceDayRunner.__new__(DeviceDayRunner)
    runner._hash_cache = {}
    path = tmp_path / 'current-weight.pt'
    path.write_bytes(b'old')
    metadata = path.stat()
    before = runner._hash(path)
    replacement = tmp_path / 'replacement.pt'
    replacement.write_bytes(b'new')
    os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    replacement.replace(path)
    assert runner._hash(path) != before
    assert len(runner._hash_cache) == 1


def test_nested_unknown_modules_do_not_inherit_flat_exclusions():
    result = impact(["new_package/api.py"])
    assert result["changed_stages"] == ["report", "retention", "stt", "understanding", "vision"]
    assert impact(["analysis/candidates.py"])["changed_stages"] == ["vision"]
    assert impact(["api.py"])["changed_stages"] == []
    assert compare({}, {"new.py": "hash"})["source_recipe"] == SOURCE_RECIPE


def test_manifest_is_stable_and_ignores_non_sources(tmp_path):
    (tmp_path / "b.py").write_text("b = 1\n")
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "cache.json").write_text("{}")
    assert list(source_manifest(Path(tmp_path))) == ["a.py", "b.py"]
    assert source_digest(source_manifest(tmp_path)) == source_digest(source_manifest(tmp_path))


def test_ast_receipt_serialization_preserves_execution_edits_and_type_parameters():
    original = ast.parse("def run():\n    return 1\n").body[0]
    representation = dump_python312(original)
    assert "type_params=[]" in representation
    assert dump_python312(ast.parse("def run():\n    return 2\n").body[0]) != representation
    # A genuine future/generic field cannot collapse onto a non-generic receipt.
    generic = ast.parse("def run():\n    return 1\n").body[0]
    generic._fields = tuple(field for field in generic._fields if field != "type_params") + ("type_params",)
    generic.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert dump_python312(generic) != representation
