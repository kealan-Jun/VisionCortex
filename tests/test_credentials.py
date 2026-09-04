from __future__ import annotations

import os
from pathlib import Path

import pytest

from visioncortex.credentials import ensure_ark_api_key


def _config(name: str = "TEST_ARK_API_KEY") -> dict:
    return {"mllm": {"enabled": True, "api_key_env": name}}


@pytest.mark.skipif(os.name != "posix", reason="0600 credential files are POSIX-only")
def test_secure_credential_file_is_loaded_without_returning_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "ark-test-secret-value"
    credential = tmp_path / "ark_api_key"
    credential.write_text(secret + "\n", encoding="utf-8")
    credential.chmod(0o600)
    monkeypatch.setenv("VISIONCORTEX_ARK_API_KEY_FILE", str(credential))
    monkeypatch.delenv("TEST_ARK_API_KEY", raising=False)

    report = ensure_ark_api_key(_config())

    assert os.environ["TEST_ARK_API_KEY"] == secret
    assert report["configured"] is True
    assert report["source"] == "credential_file"
    assert report["secret_recorded"] is False
    assert secret not in str(report)


@pytest.mark.skipif(os.name != "posix", reason="0600 credential files are POSIX-only")
def test_credential_file_must_be_mode_600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = tmp_path / "ark_api_key"
    credential.write_text("ark-test-secret-value\n", encoding="utf-8")
    credential.chmod(0o640)
    monkeypatch.setenv("VISIONCORTEX_ARK_API_KEY_FILE", str(credential))
    monkeypatch.delenv("TEST_ARK_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="permissions must be 600"):
        ensure_ark_api_key(_config())


def test_credential_file_must_not_be_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.write_text("ark-test-secret-value\n", encoding="utf-8")
    target.chmod(0o600)
    credential = tmp_path / "ark_api_key"
    credential.symlink_to(target)
    monkeypatch.setenv("VISIONCORTEX_ARK_API_KEY_FILE", str(credential))
    monkeypatch.delenv("TEST_ARK_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="non-symbolic-link"):
        ensure_ark_api_key(_config())


def test_invalid_environment_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_ARK_API_KEY", "unexpected")
    with pytest.raises(RuntimeError, match="unexpected format"):
        ensure_ark_api_key(_config())
