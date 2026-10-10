"""Customer source/asset gates use tiny synthetic fixtures, never host assets."""
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from tools.packaging import build_rtx3050_offline_package as ubuntu
from tools.packaging import build_rtx4050_offline_package as desktop
from tools.packaging import build_rtx4090_complete_package as windows
from tools.packaging.documentation import (
    DOCUMENTATION_ENTRYPOINTS, documentation_resources, validate_source_snapshot,
)


def commit_fixture(repo, files):
    repo.mkdir()
    for name, data in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "--", *files], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "synthetic customer source"], check=True, capture_output=True)
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def test_desktop_source_snapshot_closes_relocated_readmes_without_private_guidance(tmp_path, monkeypatch):
    repo, output = tmp_path / "repo", tmp_path / "package"
    files = {
        "README.md": b"[Guide](docs/current.md)",
        "docs/current.md": b"[Home](../README.md) [Desktop](../deployment/rtx4050-windows/README.md)",
        "deployment/rtx4050-windows/README.md": b"[Guide](../../docs/current.md)",
        "src/visioncortex/example.py": b"VALUE = 1\n",
        "AGENTS.md": b"synthetic private guidance",
    }
    commit = commit_fixture(repo, files)
    monkeypatch.setattr(desktop, "ROOT", repo)
    monkeypatch.setitem(DOCUMENTATION_ENTRYPOINTS, "portable_desktop", ("README.md", "deployment/rtx4050-windows/README.md"))
    snapshot = desktop.copy_source(output, allow_working_tree=False)
    assert snapshot["base_commit"] == commit and not snapshot["working_tree_snapshot"]
    assert not (output / "AGENTS.md").exists()
    assert not (output / "deployment/rtx4050-windows/README.md").exists()
    assert (output / "README.md").read_bytes() == b"[Guide](docs/current.md)"
    assert b"../SOURCE-README.md" in (output / "docs/current.md").read_bytes()
    assert b"../README.md" in (output / "docs/current.md").read_bytes()
    assert set(documentation_resources(output, "customer_package")) == {
        "README.md", "SOURCE-README.md", "docs/current.md",
    }
    validate_source_snapshot(output, snapshot)
    for record in snapshot["source_files"]:
        assert record["sha256"] == hashlib.sha256((output / record["package_path"]).read_bytes()).hexdigest()
        if "documentation_projection" in record:
            assert record["source_sha256"] == hashlib.sha256(files[record["path"]]).hexdigest()


@pytest.mark.parametrize("missing", [False, True])
def test_windows_fixed_commit_export_requires_shared_lifecycle_helpers(tmp_path, monkeypatch, missing):
    repo, output = tmp_path / "repo", tmp_path / "package"
    files = {"README.md": b"Customer usage", "deployment/rtx4090/02-Start-Web.ps1": b"# shared wrapper"}
    for name in ("Start-VisionCortex.bat", "Start-VisionCortex.ps1", "Start-VisionCortex.command", "start-visioncortex.sh"):
        files[name] = b"# customer first-clone launcher"
    for name in ("02-启动Web.ps1", "03-停止Web.ps1", "Runtime-Config.psm1"):
        files["deployment/rtx4060/" + name] = b"# shared lifecycle dependency"
    if missing:
        files.pop("deployment/rtx4060/Runtime-Config.psm1")
    commit = commit_fixture(repo, files)
    monkeypatch.setitem(DOCUMENTATION_ENTRYPOINTS, "windows_offline", ("README.md",))
    if missing:
        with pytest.raises(ValueError, match="shared lifecycle dependency"):
            windows._git_archive(repo, commit, output)
    else:
        windows._git_archive(repo, commit, output)
        assert (output / "deployment/rtx4060/Runtime-Config.psm1").read_bytes() == b"# shared lifecycle dependency"
        for name in ("Start-VisionCortex.bat", "Start-VisionCortex.ps1", "Start-VisionCortex.command", "start-visioncortex.sh"):
            assert (output / name).read_bytes() == files[name]


def test_model_copy_uses_explicit_asset_root_and_preserves_checksum_gates(tmp_path, monkeypatch):
    runtime, package = tmp_path / "assets", tmp_path / "package"
    source_names = {
        "models/ClosedSetYOLO/first_person/best.pt": "Models/ClosedSetYOLO/first_person/best.pt",
        "models/ClosedSetYOLO/third_person/best.pt": "Models/ClosedSetYOLO/third_person/best.pt",
        "models/Public/yolov8s-worldv2.pt": "Engines/yolov8s-worldv2.pt",
        "models/Public/ViT-B-32.pt": "Engines/ViT-B-32.pt",
        "models/Public/grounding-dino-base/model.safetensors": "Engines/grounding-dino-base/model.safetensors",
        "models/Public/sam2.1_hiera_base_plus.pt": "Engines/sam2.1_hiera_base_plus.pt",
        "models/Public/labpics-semantic-materials.torch": "Engines/labpics-semantic-materials.torch",
    }
    hashes = {}
    for destination, name in source_names.items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic " + destination).encode())
        hashes[destination] = ubuntu._sha256(path)
    for name in ("config.json", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        (runtime / "Engines/grounding-dino-base" / name).write_bytes(b"{}")
    monkeypatch.setattr(ubuntu, "MODEL_HASHES", hashes)
    ubuntu._copy_runtime_models(package, runtime)
    for name, expected in hashes.items():
        assert ubuntu._sha256(package / name) == expected
    (runtime / "Engines/ViT-B-32.pt").write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ubuntu._copy_runtime_models(tmp_path / "rejected", runtime)


def test_explicit_research_assets_export_only_verified_weights_and_public_metadata(tmp_path):
    runtime = tmp_path / "assets"
    runtime.mkdir()
    (runtime / "candidate.pt").write_bytes(b"synthetic research weight")
    digest = ubuntu._sha256(runtime / "candidate.pt")
    private = "synthetic private training receipt value"
    registry = tmp_path / "public-candidates.json"
    registry.write_text(json.dumps({"candidates": {"fixture": {
        "artifact": {"path": "candidate.pt", "sha256": digest, "training_receipt": private},
        "policy": {"production_enabled": False}, "private_history": private,
    }}}))
    package = tmp_path / "package"
    records = ubuntu._copy_research_candidates(package, runtime, registry)
    assert records == [{"candidate_id": "fixture", "sha256": digest, "production_enabled": False,
                        "evidence": "research_only_not_ground_truth"}]
    outputs = [p for p in package.rglob("*") if p.is_file()]
    assert len(outputs) == 2 and all(private.encode() not in p.read_bytes() for p in outputs)
    assert json.loads(registry.read_text())["candidates"]["fixture"]["private_history"] == private
    value = json.loads(registry.read_text())
    value["candidates"]["fixture"]["artifact"]["sha256"] = "a" * 64
    registry.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ubuntu._copy_research_candidates(tmp_path / "rejected", runtime, registry)


def test_ubuntu_builder_requires_explicit_assets_before_any_material_action(monkeypatch):
    monkeypatch.setattr("sys.argv", ["builder", "--output", "out", "--wheelhouse", "wheels"])
    monkeypatch.setattr(ubuntu, "build_package", lambda *args, **kwargs: pytest.fail("Asset defaults must not invoke a build"))
    with pytest.raises(SystemExit) as caught:
        ubuntu.main()
    assert caught.value.code == 2
