import json

from labvision_evidence.config import load_config
from labvision_evidence.pipeline import create_dry_run


def test_dry_run_builds_contract_without_video_or_ffmpeg(tmp_path):
    output = create_dry_run(tmp_path / "dry", load_config())
    assert (output / "JSON-Config-Files" / "Experiment-Groups-Step-Level-Analysis.json").is_file()
    assert (output / "Key-Materials" / "Key-Materials-Model-Understanding.json").is_file()
    evaluation = json.loads((output / "JSON-Config-Files" / "evidence_package_eval.json").read_text(encoding="utf-8"))
    assert evaluation["passed"] is True
    assert evaluation["dry_run"] is True
