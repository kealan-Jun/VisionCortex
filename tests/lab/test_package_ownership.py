"""Migration boundaries: one package owner and isolated historical namespaces."""
import hashlib
import json
import sys

from field_recognition.runtime.paths import PROJECT_ROOT, source_path
from visioncortex.lab.components import enable_components, legacy_module, pipeline_module

from ._support import ROOT


def test_field_source_lineage_and_main_repository_resource_paths():
    manifest = json.loads((ROOT / "src/field_recognition/SOURCE_LINEAGE.json").read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src/field_recognition").rglob("*.py")}
    assert {item["path"] for item in manifest["files"]} == actual
    assert PROJECT_ROOT == ROOT
    assert source_path("measurement_records.py") == ROOT / "src/field_recognition/measurements/measurement_records.py"
    for item in manifest["files"]:
        digest = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
        assert digest == item["sha256"]
        if not item["changes"]:
            assert digest == item["upstream_sha256"]
    assert {item["path"] for item in manifest["files"] if item["changes"]} == {
        "src/field_recognition/runtime/database.py", "src/field_recognition/runtime/paths.py",
    }


def test_historical_components_do_not_shadow_current_package_or_modify_search_path():
    import visioncortex.config as current_config
    before = list(sys.path)
    enable_components()
    assert sys.path == before
    assert sys.modules["visioncortex.config"] is current_config
    assert legacy_module("analysis.activity_recall").__name__ == "visioncortex_lab_legacy.analysis.activity_recall"
    assert pipeline_module("device_day_contract").__name__ == "visioncortex_lab_pipeline.device_day_contract"
