import json
import shutil
import subprocess
from pathlib import Path

import pytest


requires_powershell = pytest.mark.skipif(
    shutil.which("powershell.exe") is None,
    reason="RTX 4060 replay scripts require Windows PowerShell",
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "rtx4060"
    / "05-Replay-Quality-Ledgers.ps1"
)
PATH_MODULE = SCRIPT.with_name("Replay-Path-Resolution.psm1")
NAS_DIAGNOSTIC_SCRIPT = SCRIPT.with_name("06-Diagnose-Nas-Connectivity.ps1")
NAS_DIAGNOSTIC_MODULE = SCRIPT.with_name("Nas-Diagnostic-Classification.psm1")


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


def test_replay_script_resolves_exact_run_ids_via_unc_before_mapped_drive():
    content = SCRIPT.read_text(encoding="utf-8")
    module_content = PATH_MODULE.read_text(encoding="utf-8")

    assert "\\\\192.168.66.149\\video_database\\VisionCortexExperimentArchive" in content
    assert "Y:\\VisionCortexExperimentArchive" in content
    assert "Import-Module -Name $replayPathModule -Force" in content
    assert "collection-20260818-154357-29d1" in content
    assert "collection-20260818-173813-fc76" in content
    assert "$archiveRoots = @($CanonicalNasArchiveRoot, $MappedNasArchiveRoot)" in content
    assert "Get-ChildItem" not in content + module_content
    assert "-Recurse" not in content + module_content
    assert "bounded path diagnosis" in module_content


def test_replay_script_checks_runtime_before_nas_path_resolution():
    content = SCRIPT.read_text(encoding="utf-8")

    runtime_preflight = content.index("labvision_evidence.runtime_preflight")
    nas_resolution = content.index("$dev041Resolution = Resolve-ExactReplayArchive")

    assert runtime_preflight < nas_resolution


@requires_powershell
def test_path_resolver_prefers_first_visible_exact_root_and_returns_one_receipt(
    tmp_path: Path,
):
    missing_root = tmp_path / "missing"
    visible_root = tmp_path / "visible"
    relative = Path(".VisionCortex-Run-Staging") / "dataset" / "run-123"
    expected = visible_root / relative
    expected.mkdir(parents=True)
    command = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"Import-Module -Name '{PATH_MODULE}' -Force",
            (
                "$receipt = Resolve-ExactReplayArchive "
                "-Label 'fixture' "
                f"-RelativePath '{relative}' "
                f"-ArchiveRoots @('{missing_root}', '{visible_root}')"
            ),
            "$receipt | ConvertTo-Json -Depth 5 -Compress",
        ]
    )

    completed = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout.strip())

    assert receipt["run_id"] == "run-123"
    assert Path(receipt["resolved_archive"]) == expected.resolve()
    assert receipt["selected_archive_root"] == str(visible_root)
    assert len(receipt["path_resolution_attempts"]) == 2
    assert receipt["path_resolution_attempts"][0]["archive_root_visible"] is False
    assert receipt["path_resolution_attempts"][1]["exact_run_root_visible"] is True


def test_nas_diagnostic_is_read_only_and_bounded_to_exact_runs():
    content = NAS_DIAGNOSTIC_SCRIPT.read_text(encoding="utf-8")

    assert "Test-NetConnection" in content
    assert "Get-SmbMapping" in content
    assert "Get-SmbConnection" in content
    assert "collection-20260818-154357-29d1" in content
    assert "collection-20260818-173813-fc76" in content
    assert "Get-ChildItem" not in content
    assert "-Recurse" not in content
    assert "New-SmbMapping" not in content
    assert "Remove-SmbMapping" not in content
    assert "New-PSDrive" not in content
    assert "cmd.exe" not in content
    assert "net use" not in content.casefold()
    assert "Credential" not in content


@requires_powershell
def test_nas_diagnostic_classifier_distinguishes_external_failure_layers():
    command = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"Import-Module -Name '{NAS_DIAGNOSTIC_MODULE}' -Force",
            "$cases = @(",
            "  (Get-VisionCortexNasDiagnosis -ActiveAdapterCount 1 -Tcp445Succeeded $false -CanonicalShareVisible $false -CanonicalArchiveVisible $false -MappedArchiveVisible $false -Dev041Visible $false -Dev042Visible $false),",
            "  (Get-VisionCortexNasDiagnosis -ActiveAdapterCount 1 -Tcp445Succeeded $true -CanonicalShareVisible $false -CanonicalArchiveVisible $false -MappedArchiveVisible $false -Dev041Visible $false -Dev042Visible $false),",
            "  (Get-VisionCortexNasDiagnosis -ActiveAdapterCount 1 -Tcp445Succeeded $true -CanonicalShareVisible $true -CanonicalArchiveVisible $true -MappedArchiveVisible $false -Dev041Visible $true -Dev042Visible $false),",
            "  (Get-VisionCortexNasDiagnosis -ActiveAdapterCount 1 -Tcp445Succeeded $true -CanonicalShareVisible $true -CanonicalArchiveVisible $true -MappedArchiveVisible $false -Dev041Visible $true -Dev042Visible $true)",
            ")",
            "$cases | ConvertTo-Json -Depth 4 -Compress",
        ]
    )

    completed = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )
    cases = json.loads(completed.stdout.strip())

    assert [case["code"] for case in cases] == [
        "nas_host_or_smb_port_unreachable",
        "smb_share_session_or_authorization_unavailable",
        "exact_staging_unavailable",
        "retained_staging_reachable",
    ]
    assert cases[-1]["replay_permitted_by_connectivity"] is True


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
