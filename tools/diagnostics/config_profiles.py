"""Inspect profile inheritance and effective roles without touching runtime storage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from visioncortex.config import load_config


def profile_chain(path: Path, seen: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    path = path.resolve()
    if path in seen:
        raise ValueError("Configuration inheritance cycle")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    extends = payload.get("extends")
    if extends is None:
        return (path,)
    if not isinstance(extends, str) or not extends.strip() or Path(extends).is_absolute():
        raise ValueError("Profile inheritance must be a relative same-directory path")
    parent = (path.parent / extends).resolve()
    if parent.parent != path.parent:
        raise ValueError("Profile inheritance must stay in its directory")
    return (path, *profile_chain(parent, (*seen, path)))


def inspect_profiles(directory: Path) -> dict:
    records = []
    for path in sorted(directory.glob("*.yaml")):
        chain = profile_chain(path)
        config = load_config(path)
        local = path.name.endswith("-local.yaml")
        storage, ingest = config["storage"], config["collection_ingest"]
        if local and (any("production" in item.name for item in chain)
                      or storage["sync_to_nas"] or ingest["enabled"]
                      or storage["run_output_mode"] != "local"
                      or storage["manifest_storage"] != "local"
                      or storage["require_nas_source_paths"]):
            raise ValueError(f"Local profile inherits a production integration: {path.name}")
        records.append({
            "profile": path.name, "chain": [item.name for item in chain],
            "kind": "base_not_deployment" if path.stem.endswith("-base") else "local" if local else "deployment",
            "runtime_role": config["runtime"]["role"],
            "collection_enabled": ingest["enabled"], "device_day_enabled": config["device_day"]["enabled"],
            "nas_sync_enabled": storage["sync_to_nas"], "output_mode": storage["run_output_mode"],
            "mllm_enabled": config["mllm"]["enabled"], "speech_enabled": config["speech_recognition"]["enabled"],
        })
    return {"profiles": records, "storage_or_model_access": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[2] / "configs")
    args = parser.parse_args()
    print(json.dumps(inspect_profiles(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
