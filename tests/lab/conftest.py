"""Laboratory fixtures are local to this test domain and opt-in for PostgreSQL."""
import hashlib
import json
from uuid import uuid4

import pytest
import qrcode

from visioncortex.lab.domain import Core

from ._fixtures import db as database_fixture, settings as settings_fixture


@pytest.fixture(scope="session", name="vc_lab_db")
def vc_lab_database():
    return database_fixture.__wrapped__()


@pytest.fixture(scope="session", name="db")
def lab_db(vc_lab_db):
    return vc_lab_db


@pytest.fixture(name="vc_lab_core")
def vc_lab_core_instance(vc_lab_db):
    customer = Core.customer_instance(vc_lab_db, "fixture-" + str(uuid4()), scope="fixture")
    return Core(vc_lab_db, customer["id"])


@pytest.fixture(name="core")
def lab_core(vc_lab_core):
    return vc_lab_core


@pytest.fixture(name="vc_lab_settings")
def vc_lab_configuration(core, tmp_path):
    return settings_fixture.__wrapped__(core, tmp_path)


@pytest.fixture(name="settings")
def lab_settings(vc_lab_settings):
    return vc_lab_settings


@pytest.fixture
def photo(settings):
    folder = settings.incoming / "capture-one"
    folder.mkdir()
    path = folder / "image.png"
    qrcode.make("Q-M001").save(path)
    receipt = dict(capture_id="physical-capture-1", source_instance="camera-one",
                   source_revision="1", kind="photo", relative_path="capture-one/image.png",
                   captured_at="2026-10-23T10:58:00+08:00",
                   time_basis={"basis": "fixture_receipt", "uncertainty_us": 0},
                   source_kind="fixture", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (folder / "CaptureReceipt.json").write_text(json.dumps(receipt))
    return receipt
