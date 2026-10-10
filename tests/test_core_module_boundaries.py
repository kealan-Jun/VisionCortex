import json
import os
import subprocess
import sys
from pathlib import Path

from visioncortex.evidence.layout import ArchiveLayout
from visioncortex.evidence.serialization import write_json
from visioncortex.schemas import ViewRole


def test_layout_and_serialization_do_not_load_analysis_or_model_execution():
    # A status reader or uploader should not need cv2, detector/model modules,
    # archive composition or the pipeline merely to resolve/write its layout.
    script = """
import sys
from visioncortex.evidence.layout import ArchiveLayout
from visioncortex.evidence.serialization import write_json
for name in ('cv2', 'torch', 'visioncortex.archive', 'visioncortex.pipeline',
             'visioncortex.detection', 'visioncortex.mllm'):
    assert name not in sys.modules, name
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_native_layout_keeps_historical_paths_and_json_contract(tmp_path):
    layout = ArchiveLayout(tmp_path / "experiment")
    layout.create()
    assert layout.key_frames == layout.root / "Key-Materials/Key-Frames"
    assert layout.experiment_clips == layout.root / "Experiment-Clips"
    destination = layout.json_config / "receipt.json"
    destination.write_text("previous content", encoding="utf-8")
    write_json(
        destination, {"source": Path("original.mp4"), "role": ViewRole.FIRST_PERSON}
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "source": "original.mp4",
        "role": "first_person",
    }
    assert not destination.with_suffix(".json.tmp").exists()
    # Recorder-native atomic_json keeps its independent durable-write policy;
    # the historical serializer remains available as the same public callable.
    from visioncortex.archive import write_json as legacy_write_json

    assert legacy_write_json is write_json


def test_core_owners_do_not_import_their_legacy_composition_facades():
    import ast
    import visioncortex

    package = Path(visioncortex.__file__).parent
    forbidden = {"archive", "actions", "pipeline", "api", "cli"}
    for directory in ("analysis", "evidence", "media"):
        for source in (package / directory).glob("*.py"):
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level == 2:
                    assert (node.module or "").split(".")[0] not in forbidden, source
                elif isinstance(node, ast.Import):
                    assert not any(
                        alias.name.startswith("visioncortex.")
                        and alias.name.split(".")[1] in forbidden
                        for alias in node.names
                    ), source
