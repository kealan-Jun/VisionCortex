import os

import pytest

from visioncortex import cli


def _settings():
    return {
        "project": {},
        "mllm": {"enabled": True, "api_key_env": "TEST_CLI_ARK_API_KEY"},
    }


@pytest.mark.parametrize("source", ["environment", "credential_file"])
def test_run_loads_credential_before_pipeline_without_recording_secret(
    monkeypatch, tmp_path, capsys, source
):
    if source == "credential_file" and os.name != "posix":
        pytest.skip("0600 credential files require POSIX")
    settings = _settings()
    secret = "ark-test-cli-credential"
    credential = tmp_path / "ark_api_key"
    monkeypatch.setenv("VISIONCORTEX_ARK_API_KEY_FILE", str(credential))
    monkeypatch.delenv("TEST_CLI_ARK_API_KEY", raising=False)
    if source == "credential_file":
        credential.write_text(secret + "\n", encoding="utf-8")
        credential.chmod(0o600)
    else:
        monkeypatch.setenv("TEST_CLI_ARK_API_KEY", secret)
    manifest = object()
    output = tmp_path / "output"
    monkeypatch.setattr(cli, "load_config", lambda *_: settings)
    monkeypatch.setattr(cli, "load_manifest", lambda *_: manifest)

    class Pipeline:
        def __init__(self, config, _progress):
            assert os.environ["TEST_CLI_ARK_API_KEY"] == secret
            assert secret not in str(config)
            assert config["project"]["output_root"] == str(output)

        def run(self, pipeline_manifest):
            assert pipeline_manifest is manifest
            return output

    monkeypatch.setattr(cli, "EvidencePipeline", Pipeline)
    cli.run_command(tmp_path / "manifest.yaml", None, output)
    captured = capsys.readouterr()
    assert str(output) in captured.out
    assert secret not in captured.out + captured.err


@pytest.mark.parametrize("invalid_environment", [False, True])
def test_run_rejects_missing_or_invalid_credential_before_material_work(
    monkeypatch, tmp_path, invalid_environment
):
    monkeypatch.setattr(cli, "load_config", lambda *_: _settings())
    monkeypatch.setenv("VISIONCORTEX_ARK_API_KEY_FILE", str(tmp_path / "missing"))
    monkeypatch.delenv("TEST_CLI_ARK_API_KEY", raising=False)
    if invalid_environment:
        monkeypatch.setenv("TEST_CLI_ARK_API_KEY", "invalid")

    def forbidden(*_args, **_kwargs):
        pytest.fail("credential validation must precede manifest/GPU work")

    monkeypatch.setattr(cli, "load_manifest", forbidden)
    monkeypatch.setattr(cli, "EvidencePipeline", forbidden)
    expected = "unexpected format" if invalid_environment else "unavailable"
    with pytest.raises(RuntimeError, match=expected):
        cli.run_command(tmp_path / "manifest.yaml", None, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("cv_only", ["disabled_mllm", "preprocessing_only"])
def test_cv_only_run_does_not_access_ark_credentials(monkeypatch, tmp_path, cv_only):
    settings = _settings()
    if cv_only == "disabled_mllm":
        settings["mllm"]["enabled"] = False
    else:
        settings["project"]["preprocessing_acceptance_only"] = True
    monkeypatch.setattr(cli, "load_config", lambda *_: settings)
    monkeypatch.setattr(cli, "load_manifest", lambda *_: object())

    def forbidden(*_args, **_kwargs):
        pytest.fail("CV-only runs must not read credentials or require Ark")

    monkeypatch.setattr(cli, "ensure_ark_api_key", forbidden)

    class Pipeline:
        def __init__(self, config, _progress):
            assert config is settings

        def run(self, _manifest):
            return tmp_path

    monkeypatch.setattr(cli, "EvidencePipeline", Pipeline)
    cli.run_command(tmp_path / "manifest.yaml", None, None)
