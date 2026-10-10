"""Package a sealed source snapshot for an already prepared Ubuntu host.

This builds a candidate, never a formally accepted release. It does not inspect
NAS, install dependencies, copy model binaries or start services.
"""
from __future__ import annotations

from tools.packaging.primitives import relative_path

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "PackageManifest.json"
SCHEMA = "visioncortex-prepared-automation-candidate/1"
TEXT_SUFFIXES = {".py", ".js", ".html", ".css", ".json", ".yaml", ".toml", ".sh", ".service"}
SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|ark-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
SENSITIVE_KEYS = {"apikey", "password", "secret", "token", "secretkey", "accesstoken", "privatekey", "authorization", "accesskey"}
RUNTIME_DIRECTORIES = {"runtime", "cache", "staging", "media", "outputs", "input", "inputs"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()




def plain_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink source file: {path.name}")
    if path.stat().st_size > 4 * 1024**2 or path.suffix not in TEXT_SUFFIXES:
        raise ValueError(f"Runtime, media, model or unsupported payload: {path.name}")
    data = path.read_bytes()
    text = data.decode("utf-8")
    if SECRET.search(text):
        raise ValueError(f"Secret-shaped literal in payload: {path.name}")
    return data


def audit_config(value, ancestors=None):
    ancestors = set(ancestors or ())
    if isinstance(value, (dict, list)):
        if id(value) in ancestors:
            raise ValueError("Recursive configuration is unsupported")
        ancestors.add(id(value))
        values = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in values:
            if isinstance(key, str):
                normalized = re.sub("[^a-z0-9]", "", key.lower())
                if normalized in SENSITIVE_KEYS and child not in (None, ""):
                    if not isinstance(child, str) or not re.fullmatch(r"\$\{[A-Z][A-Z0-9_]*\}", child):
                        raise ValueError("Customer configuration contains a literal credential field")
            audit_config(child, ancestors)
    elif isinstance(value, str):
        if SECRET.search(value) or "\x00" in value:
            raise ValueError("Customer configuration contains secret-shaped content")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite configuration value")
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise ValueError("Unsupported configuration value")


def profile(path: Path, seen=None, source_check=None) -> dict:
    seen = set(seen or ())
    if path.resolve() in seen:
        raise ValueError("Configuration inheritance cycle")
    seen.add(path.resolve())
    if source_check is not None:
        source_check(path)
    data = yaml.safe_load(plain_file(path)) or {}
    if not isinstance(data, dict):
        raise ValueError("Customer configuration must be a mapping")
    audit_config(data)
    extends = data.pop("extends", None)
    if extends is None:
        return data
    if not isinstance(extends, str) or Path(extends).is_absolute() or ".." in Path(extends).parts:
        raise ValueError("Customer configuration inheritance escapes its directory")
    base = path.parent / extends
    if base.resolve().parent != path.resolve().parent:
        raise ValueError("Customer configuration inheritance escapes its directory")
    return merge(profile(base, seen, source_check), data)


def merge(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
    return result


def validate_host(python: str, service_name: str, port: int, web_ai_settings: int):
    if not Path(python).is_absolute() or any(ord(c) < 32 or ord(c) == 127 for c in python):
        raise ValueError("Prepared Python must be an absolute path without control characters")
    if len(service_name) > 64 or not re.fullmatch(r"visioncortex-[a-z0-9][a-z0-9-]*\.service", service_name, re.ASCII):
        raise ValueError("Invalid VisionCortex service name")
    if type(port) is not int or not 1024 <= port <= 65535 or type(web_ai_settings) is not int or web_ai_settings not in {0, 1}:
        raise ValueError("Invalid prepared port or AI settings flag")


def build_package(source_root: Path, config: Path, python: str, output: Path, *,
                  service_name="visioncortex-analysis.service", port=8001, web_ai_settings=1,
                  source_manifest: Path | None = None) -> dict:
    validate_host(python, service_name, port, web_ai_settings)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("Source snapshot must be a regular directory")
    source_root = source_root.resolve()
    output = output.absolute()
    archive = output.with_name(output.name + ".tar.gz")
    if output.exists() or output.is_symlink() or archive.exists() or archive.is_symlink():
        raise FileExistsError("Candidate destination already exists")
    if output.resolve().is_relative_to(source_root):
        raise ValueError("Candidate destination must be outside the sealed source snapshot")
    source_manifest = source_manifest or source_root.parent / "SelfTestManifest.json"
    sealed = json.loads(plain_file(source_manifest))
    if sealed.get("schema_version") != "visioncortex-automation-selftest/1" or not isinstance(sealed.get("files"), dict):
        raise ValueError("A supported sealed source snapshot manifest is required")
    for name, expected in sealed["files"].items():
        relative_path(name)
        if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
            raise ValueError("Invalid sealed source digest")
    def source_check(path):
        if path.is_relative_to(source_root):
            for child in (path, *path.parents):
                if child == source_root.parent:
                    break
                if child.is_symlink():
                    raise ValueError("Source snapshot contains a symlink")
            name = path.relative_to(source_root).as_posix()
            if sealed["files"].get(name) != digest(plain_file(path)):
                raise ValueError(f"Source file does not match its seal: {name}")
        elif path.resolve().is_relative_to(source_root):
            raise ValueError("Source snapshot path is redirected")
    selected = []
    for subtree in (source_root / "src", source_root / "configs/models"):
        if subtree.is_symlink() or not subtree.is_dir():
            raise ValueError("Required source subtree is missing or redirected")
        for path in sorted(subtree.rglob("*")):
            if path.is_symlink():
                raise ValueError("Source snapshot contains a symlink")
            if subtree.name == "src":
                top = path.relative_to(subtree).parts[0]
                if top.endswith(".egg-info"):
                    continue
                if path.is_file() and not (path.parent.name == "__pycache__" and path.suffix == ".pyc") and top != "visioncortex":
                    raise ValueError("Unsupported source payload outside the explicit package allowlist")
            if path.is_file() and not (path.parent.name == "__pycache__" and path.suffix == ".pyc"):
                if path != source_root / "src/visioncortex/BuildManifest.json":
                    selected.append(path)
    selected.extend(source_root / name for name in (
        "configs/default.yaml", "configs/mllm-providers.json", "pyproject.toml",
        "deployment/rtx3090ti-ubuntu/_common.sh",
        "deployment/rtx3090ti-ubuntu/render_service.py",
        "deployment/rtx3090ti-ubuntu/06-Run-LAN-Server.sh",
        "deployment/rtx3090ti-ubuntu/09-Install-Analysis-Service.sh",
        "deployment/rtx3090ti-ubuntu/visioncortex-analysis.service",
    ))
    payload = {}
    for path in selected:
        name = path.relative_to(source_root).as_posix()
        if any(part.lower() in RUNTIME_DIRECTORIES for part in Path(name).parts[1:-1]):
            raise ValueError("Runtime payload cannot be included in a service candidate")
        source_check(path)
        raw = plain_file(path)
        if path.suffix in {".yaml", ".json"}:
            audit_config(yaml.safe_load(raw))
        payload["App/" + name] = raw
    customer = profile(config.absolute(), source_check=source_check)
    payload["App/configs/PreparedCustomer.yaml"] = yaml.safe_dump(customer, allow_unicode=True, sort_keys=False).encode()
    source_files = {name.removeprefix("App/"): digest(raw) for name, raw in sorted(payload.items())}
    source_digest = digest(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode())
    identity = {"version": "prepared-automation-candidate", "commit": None, "source_digest": source_digest,
                "lock_digest": digest(payload["App/pyproject.toml"]), "lock_digest_scope": "pyproject_metadata_not_dependency_lock",
                "candidate_kind": "sealed_source_snapshot", "release_ready": False,
                "base_commit_context": sealed.get("base_commit")}
    payload["App/src/visioncortex/BuildManifest.json"] = json.dumps(identity, indent=2).encode()
    prepared = {"python": python, "config": "App/configs/PreparedCustomer.yaml", "service_name": service_name,
                "port": port, "web_ai_settings": web_ai_settings}
    payload["PreparedHost.json"] = json.dumps(prepared, ensure_ascii=False, indent=2).encode()
    payload["install.sh"] = plain_file(ROOT / "deployment/rtx3090ti-ubuntu/install.sh")
    files = {name: digest(raw) for name, raw in sorted(payload.items())}
    manifest = {"schema_version": SCHEMA, "candidate_kind": "prepared_host_service", "release_ready": False,
                "commit": None, "base_commit_context": sealed.get("base_commit"), "source_digest": source_digest,
                "source_files": source_files, "files": files, "prepared_environment_reused": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".visioncortex-candidate-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        for name, raw in payload.items():
            destination = staged / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            if destination.suffix == ".sh":
                destination.chmod(0o755)
        (staged / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        tar_path = Path(temporary) / "candidate.tar.gz"
        with tarfile.open(tar_path, "w:gz") as bundle:
            bundle.add(staged, arcname=output.name)
        staged.rename(output)
        tar_path.rename(archive)
    return {"output": str(output), "archive": str(archive), "source_digest": source_digest,
            "release_ready": False, "commit": None, "file_count": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--service-name", default="visioncortex-analysis.service")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--web-ai-settings", default=1, type=int, choices=(0, 1))
    args = parser.parse_args()
    result = build_package(args.source_root, args.config, args.python, args.output,
                           service_name=args.service_name, port=args.port, web_ai_settings=args.web_ai_settings,
                           source_manifest=args.source_manifest)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
