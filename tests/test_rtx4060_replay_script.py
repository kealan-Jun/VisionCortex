from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "rtx4060"
    / "05-Replay-Quality-Ledgers.ps1"
)


def test_replay_script_uses_repository_virtual_environment_only():
    content = SCRIPT.read_text(encoding="utf-8")

    assert ".venv\\Scripts\\python.exe" in content
    assert "& $visionCortexPython -B" in content
    assert "pip install" not in content
    assert "--output" not in content
    assert "labvision_evidence.cli replay-quality-ledger" in content


def test_replay_script_preflights_runtime_and_restores_process_environment():
    content = SCRIPT.read_text(encoding="utf-8")

    assert "labvision_evidence.runtime_preflight" in content
    assert "--expected-source $visionCortexSourceRoot" in content
    assert "$runtimeProbe" not in content
    assert " -c " not in content
    assert "finally" in content
    assert "Remove-Item Env:PYTHONPATH" in content


def test_replay_script_preflights_both_ledgers_before_running_either_replay():
    content = SCRIPT.read_text(encoding="utf-8")

    dev041_inspection = content.index(
        "inspect-quality-ledger-inputs `\n        --archive $Dev041Archive"
    )
    dev042_inspection = content.index(
        "inspect-quality-ledger-inputs `\n        --archive $Dev042Archive"
    )
    dev041_replay = content.index(
        "replay-quality-ledger `\n        --archive $Dev041Archive"
    )
    dev042_replay = content.index(
        "replay-quality-ledger `\n        --archive $Dev042Archive"
    )

    assert dev041_inspection < dev042_inspection < dev041_replay < dev042_replay
