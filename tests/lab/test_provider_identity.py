"""Serving configuration swaps at a real localhost transport; no model/DB."""

import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

from visioncortex.lab.domain import Conflict
from visioncortex.lab.local_provider import (
    LocalHTTP,
    LocalProviders,
    ProviderUnavailable,
)


@contextmanager
def server():
    state = dict(health="a" * 64, result="a" * 64, posts=0)
    model = dict(
        protocol="vc-lab-media-infer/1",
        model_revision="synthetic-model/1",
        runtime_revision="f" * 64,
        token_env="VC_LAB_SYNTHETIC_SERVICE_TOKEN",
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body):
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.respond(
                dict(
                    status="ready",
                    kind="asr",
                    model_revision=model["model_revision"],
                    execution_identity=state["health"],
                )
            )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            raw = base64.b64decode(body["media_base64"])
            state["posts"] += 1
            self.respond(
                dict(
                    source_sha256=hashlib.sha256(raw).hexdigest(),
                    model_revision=model["model_revision"],
                    runtime_revision=model["runtime_revision"],
                    execution_identity=state["result"],
                    actual_model_invocation=True,
                    cache_hit=False,
                )
            )

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    model["base_url"] = f"http://127.0.0.1:{http.server_port}"
    try:
        yield state, model
    finally:
        http.shutdown()
        http.server_close()
        worker.join(timeout=2)


def test_same_weights_and_script_cannot_hide_a_serving_configuration_swap(monkeypatch):
    monkeypatch.setenv(
        "VC_LAB_SYNTHETIC_SERVICE_TOKEN", "synthetic-token-no-live-secret"
    )
    with server() as (state, model):
        model["execution_identity"] = "a" * 64
        client = LocalHTTP(model)
        state["health"] = "b" * 64
        with pytest.raises(ProviderUnavailable):
            client.preflight("transcription")
        assert state["posts"] == 0
        state["health"] = "a" * 64
        client.preflight("transcription")
        state["result"] = "b" * 64
        with pytest.raises(Conflict):
            client.infer(b"synthetic-audio", "fixture")
        state["result"] = "a" * 64
        _, receipt = client.infer(b"synthetic-audio", "fixture")
        assert (
            receipt["execution_identity"] == "a" * 64
            and receipt["serving_configuration_pinned"]
        )


def test_legacy_manifest_is_compatible_without_claiming_a_configuration_pin(
    monkeypatch,
):
    monkeypatch.setenv(
        "VC_LAB_SYNTHETIC_SERVICE_TOKEN", "synthetic-token-no-live-secret"
    )
    with server() as (_, model):
        client = LocalHTTP(model)
        client.preflight("transcription")
        _, receipt = client.infer(b"synthetic-audio", "fixture")
        assert not receipt["serving_configuration_pinned"]
        assert receipt["execution_identity"] == "a" * 64


@pytest.mark.parametrize("pin", [None, True, "short", "g" * 64])
def test_manifest_rejects_an_invalid_explicit_execution_pin(tmp_path, pin):
    path = tmp_path / "providers.json"
    model = dict(
        base_url="http://127.0.0.1:1",
        protocol="vc-lab-media-infer/1",
        model_revision="synthetic-model/1",
        runtime_revision="f" * 64,
        token_env="VC_LAB_SYNTHETIC_SERVICE_TOKEN",
        execution_identity=pin,
    )
    path.write_text(
        json.dumps(
            dict(
                schema_version="vc-lab-local-providers/1",
                customer_code="synthetic-fixture",
                enabled=True,
                data_scope="fixture",
                models={"transcription": model},
            )
        )
    )
    settings = SimpleNamespace(
        timeline_providers_path=path, customer_code="synthetic-fixture"
    )
    with pytest.raises(ProviderUnavailable):
        LocalProviders(settings).load()
