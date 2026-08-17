import json

from labvision_evidence.config import load_config
from labvision_evidence.pipeline import create_dry_run


def test_dry_run_builds_contract_without_video_or_ffmpeg(tmp_path):
    output = create_dry_run(tmp_path / "dry", load_config())
    assert (output / "JSON-Config-Files" / "Experiment-Groups-Step-Level-Analysis.json").is_file()
    assert (output / "Key-Materials" / "Key-Materials-Model-Understanding.json").is_file()
    assert (output / "JSON-Config-Files" / "evidence_index.sqlite").is_file()
    index_manifest = json.loads(
        (output / "JSON-Config-Files" / "evidence_index_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert index_manifest["validation"]["passed"] is True
    assert index_manifest["token_usage"]["total_tokens"] == 0
    assert list((output / "Lab-Daily-Reports").glob("*/Lab-Daily-Report-*.json"))
    assert list((output / "Professional-PDFs").glob("Lab-Daily-Report-*.pdf"))
    daily_eval_path = next((output / "Lab-Daily-Reports").glob("*/Daily-Report-Eval.json"))
    daily_evaluation = json.loads(daily_eval_path.read_text(encoding="utf-8"))
    assert daily_evaluation["passed"] is True
    evaluation = json.loads((output / "JSON-Config-Files" / "evidence_package_eval.json").read_text(encoding="utf-8"))
    assert evaluation["passed"] is True
    assert evaluation["dry_run"] is True
