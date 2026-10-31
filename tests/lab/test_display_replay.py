"""An evaluation manifest may locate fields, never prescribe their readings."""

import copy
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "tools/evaluation/replay_display_ocr.py"
spec = importlib.util.spec_from_file_location("display_replay_fixture", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def manifest():
    return dict(
        schema_version="vc-display-ocr-replay/1",
        cases=[
            dict(
                id="arbitrary-source",
                split="holdout",
                path="/local/example.mp4",
                registration=dict(fields=[dict(layout=dict(color="red", digits=[]))]),
            )
        ],
    )


@pytest.mark.parametrize(
    "key",
    [
        "glyph_adapter",
        "expected_value",
        "expected_reading",
        "forced_value",
        "correction_map",
    ],
)
def test_engineering_replay_rejects_reading_specific_rules(key):
    document = manifest()
    document["cases"][0]["registration"]["fields"][0]["layout"][key] = "fixture"
    with pytest.raises(ValueError, match="reading/font-specific"):
        module.validate_manifest(document)


def test_manifest_is_not_limited_to_selected_cameras_or_devices():
    document = manifest()
    other = copy.deepcopy(document["cases"][0])
    other["id"] = "new-unknown-camera"
    document["cases"].append(other)
    assert len(module.validate_manifest(document)) == 2
    document["cases"][1]["split"] = ""
    with pytest.raises(ValueError, match="holdout"):
        module.validate_manifest(document)
