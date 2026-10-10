from __future__ import annotations

import hashlib
import subprocess

import pytest

from repo_paths import ROOT
from tools.packaging.documentation import (
    DOCUMENTATION_ENTRYPOINTS, documentation_resources, export_customer_source,
    project_documentation, validate_customer_documents, validate_source_snapshot,
)
from tools.packaging.primitives import export_git_commit, safe_path, sha256_file


def test_streaming_hash_accounts_for_every_byte(tmp_path):
    path = tmp_path / "artifact"
    data = b"synthetic structural fixture" * 23
    path.write_bytes(data)
    chunks = []
    assert sha256_file(path, on_chunk=chunks.append, chunk_size=31) == hashlib.sha256(data).hexdigest()
    assert sum(chunks) == len(data)
    assert max(chunks) <= 31


@pytest.mark.parametrize("name", ["../outside", "/outside", "D:/outside", "bad\\path", ".", "a//b", "bad\npath"])
def test_package_paths_reject_escape_and_ambiguous_spellings(tmp_path, name):
    with pytest.raises(ValueError):
        safe_path(tmp_path, name)


def test_package_paths_reject_linked_parents(tmp_path):
    (tmp_path / "inside").mkdir()
    (tmp_path / "redirected").symlink_to(tmp_path / "inside", target_is_directory=True)
    with pytest.raises(ValueError, match="links"):
        safe_path(tmp_path, "redirected/file", reject_links=True)


def test_delivery_document_selection_closes_links_without_sweeping_history(tmp_path, monkeypatch):
    monkeypatch.setitem(DOCUMENTATION_ENTRYPOINTS, "test", ("README.md",))
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text('[Guide](docs/guide.md) <img src="docs/banner.svg">')
    (tmp_path / "docs/guide.md").write_text('[Home](../README.md)')
    (tmp_path / "docs/banner.svg").write_text('<svg/>')
    (tmp_path / "docs/history.md").write_text('Historical evidence, not linked by the delivery entry.')
    assert documentation_resources(tmp_path, "test") == ("README.md", "docs/banner.svg", "docs/guide.md")
    (tmp_path / "docs/banner.svg").unlink()
    with pytest.raises(ValueError, match="Missing"):
        documentation_resources(tmp_path, "test")


def test_portable_configuration_navigation_keeps_only_referenced_test_resources():
    resources = documentation_resources(ROOT, "portable_desktop")
    assert "configs/README.md" in resources
    assert {name for name in resources if name.startswith("tests/")} == {
        "tests/README.md",
    }


def test_fixed_commit_export_ignores_working_tree_changes(tmp_path):
    repo, destination = tmp_path / "repo", tmp_path / "export"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "source.py").write_text('version = "committed"\n')
    subprocess.run(["git", "-C", str(repo), "add", "--", "source.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "fixture"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (repo / "source.py").write_text('version = "dirty"\n')
    export_git_commit(repo, commit, destination)
    assert (destination / "source.py").read_text() == 'version = "committed"\n'
    with pytest.raises(FileExistsError, match="empty owned"):
        export_git_commit(repo, commit, destination)
    with pytest.raises(ValueError, match="immutable"):
        export_git_commit(repo, "HEAD", tmp_path / "mutable")


def test_customer_navigation_rejects_private_history_instead_of_publishing_remote_links(tmp_path, monkeypatch):
    monkeypatch.setitem(DOCUMENTATION_ENTRYPOINTS, "test", ("README.md",))
    (tmp_path / "docs/history").mkdir(parents=True)
    (tmp_path / "README.md").write_text('[History](docs/history/README.md)')
    (tmp_path / "docs/history/README.md").write_text('Private evidence is retained separately.')
    with pytest.raises(ValueError, match="private_entry.*docs/history/README.md"):
        documentation_resources(tmp_path, "test")


def test_documentation_compatibility_port_preserves_bytes_without_reading_git_remotes(tmp_path, monkeypatch):
    def unexpected_git(*args, **kwargs):
        raise AssertionError("Customer documentation must not query or project a remote")
    monkeypatch.setattr(subprocess, "check_output", unexpected_git)
    original = {"README.md": b"[Current guide](docs/current.md)"}
    projected, receipts = project_documentation(tmp_path, "a" * 40, original)
    assert projected == original and receipts == {}
    with pytest.raises(ValueError, match="immutable"):
        project_documentation(tmp_path, "HEAD", original)


@pytest.mark.parametrize("category,text", [
    ("private_home", "Local interpreter: /home/fixture-account/env/bin/python"),
    ("device_address", "Recorder: " + "192.168." + "12.34"),
    ("internal_run", "Recorded run: exp_20300102_030405"),
    ("repository_roles", "development-only repository"),
    ("credential_value", "Credential: " + "sk-" + "synthetic_fixture_value_never_a_real_key"),
])
def test_customer_document_hygiene_reports_category_without_private_value(category, text):
    with pytest.raises(ValueError) as caught:
        validate_customer_documents({"docs/current.md": text.encode()})
    assert category in str(caught.value)
    assert "docs/current.md" in str(caught.value)
    assert text not in str(caught.value)


def test_public_symbols_and_official_links_are_valid_customer_documentation():
    validate_customer_documents({"README.md": b"""Local URL: http://127.0.0.1:8002
Use $VISIONCORTEX_SITE_CONFIG and <PRIVATE_CONFIG>.
[Documentation](https://example.invalid/docs)
"""})


def test_customer_source_export_filters_private_files_and_preserves_runtime_helpers(tmp_path, monkeypatch):
    monkeypatch.setitem(DOCUMENTATION_ENTRYPOINTS, "test", ("README.md",))
    repo, destination = tmp_path / "repo", tmp_path / "export"
    repo.mkdir()
    files = {
        "README.md": "[Schema](docs/contracts/example.json)",
        "docs/contracts/example.json": "{}",
        "src/visioncortex/example.py": "VERSION = 'committed'",
        "deployment/rtx3090ti-ubuntu/_common.sh": "# shared lifecycle helper",
        "deployment/rtx3090ti-ubuntu/render_service.py": "# service renderer",
        "deployment/rtx4060/Runtime-Config.psm1": "# private config loader interface",
        "AGENTS.md": "Internal guidance",
        "CLAUDE.md": "Internal guidance",
        "docs/history/README.md": "Private evidence index",
        "docs/2030-01-01_DEV-001-evidence.md": "Internal run journal",
        "release/freeze/private.json": "{}",
        "tests/unrelated.py": "# test only",
    }
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "--", *files], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "fixture"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (repo / "src/visioncortex/example.py").write_text("VERSION = 'dirty'")
    export_customer_source(repo, commit, destination, "test")
    exported = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    assert exported == {name for name in files if name.startswith(("src/", "deployment/", "docs/contracts/")) or name == "README.md"}
    for name in exported:
        assert (destination / name).read_text() == files[name]
    assert (repo / "AGENTS.md").read_text() == "Internal guidance"


def test_old_source_snapshot_with_historical_projection_cannot_be_resealed(tmp_path):
    with pytest.raises(ValueError, match="historical_projection"):
        validate_source_snapshot(tmp_path, {"documentation_projections": {"docs/current.md": {"targets": []}}})


def test_public_document_relocation_keeps_local_links_and_source_identity(tmp_path):
    raw = b'[Guide](../../docs/current.md#usage) [Official](https://example.invalid/docs)\n```sh\n[example](../../docs/current.md)\n```'
    source = "deployment/desktop/README.md"
    projected, receipts = project_documentation(tmp_path, "a" * 40, {source: raw}, {source: "README.md"})
    assert b"[Guide](docs/current.md#usage)" in projected[source]
    assert b"https://example.invalid/docs" in projected[source]
    assert b"[example](../../docs/current.md)" in projected[source]
    receipt = receipts[source]
    assert receipt["recipe"] == "public-document-relocation/1"
    assert receipt["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["projection_sha256"] == hashlib.sha256(projected[source]).hexdigest()
    assert receipt["base_commit_context"] == "a" * 40
    assert "targets" not in receipt
