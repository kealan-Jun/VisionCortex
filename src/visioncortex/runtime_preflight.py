from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Sequence


RUNTIME_PREFLIGHT_SCHEMA_VERSION = "visioncortex-runtime-preflight/1.0.0"
REQUIRED_RUNTIME_MODULES = ("openpyxl", "cv2", "numpy", "pydantic", "yaml")


def inspect_project_runtime(expected_source: Path) -> dict[str, object]:
    """Import the complete replay runtime and prove its frozen source root."""

    dependency_versions: dict[str, str] = {}
    for module_name in REQUIRED_RUNTIME_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"required project dependency is unavailable: {module_name}"
            ) from exc
        dependency_versions[module_name] = str(
            getattr(module, "__version__", "available")
        )

    package = importlib.import_module("visioncortex")
    importlib.import_module("visioncortex.cli")
    importlib.import_module("visioncortex.replay_acceptance")

    expected = expected_source.resolve()
    package_file = getattr(package, "__file__", None)
    if not package_file:
        raise RuntimeError("visioncortex package has no resolvable source file")
    actual = Path(package_file).resolve()
    if expected not in actual.parents:
        raise RuntimeError(
            f"unexpected package source: expected beneath {expected}, observed {actual}"
        )

    return {
        "schema_version": RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "expected_source_root": str(expected),
        "package_source": str(actual),
        "dependency_versions": dependency_versions,
        "cli_imported": True,
        "quality_replay_imported": True,
        "video_operations": 0,
        "model_calls": 0,
        "token_usage": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen VisionCortex project runtime without media I/O."
    )
    parser.add_argument(
        "--expected-source",
        type=Path,
        required=True,
        help="Frozen checkout src directory that must own visioncortex.",
    )
    args = parser.parse_args(argv)
    result = inspect_project_runtime(args.expected_source)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
