"""Portable ASGI launcher with explicit configuration and no import-time runtime."""

import argparse
import os

import uvicorn
from pydantic import ValidationError

from field_recognition.runtime.environment import load_environment_file
from field_recognition.runtime.production_config import ProductionConfig


def app_factory():
    from field_recognition.api.application import create_app
    from field_recognition.runtime.factory import create_runtime

    return create_app(create_runtime())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file", help="Explicit KEY=value file; process variables take priority"
    )
    parser.add_argument("--host", help="Override FIELD_HOST (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Override FIELD_PORT (default: 8188)")
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            load_environment_file(args.env_file)
        for name, value in (("FIELD_HOST", args.host), ("FIELD_PORT", args.port)):
            if value is not None:
                os.environ[name] = str(value)
        config = ProductionConfig.from_environment()
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        parser.error("; ".join(".".join(map(str, e["loc"])) + ": " + e["msg"] for e in errors))
    except ValueError:
        parser.error(
            "Invalid configuration or missing environment file; check the selected profile"
        )
    uvicorn.run(
        "field_recognition.runtime.server:app_factory",
        factory=True,
        host=config.host,
        port=config.port,
    )
