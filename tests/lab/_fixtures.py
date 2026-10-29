"""Opt-in PostgreSQL fixtures. No default database or runtime.env is read."""
import os

import pytest

from visioncortex.lab.config import Settings
from visioncortex.lab.db import Database

from ._support import ROOT


@pytest.fixture(scope="session", name="vc_lab_db")
def db():
    url = os.environ.get("VC_LAB_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Explicit isolated VC_LAB_TEST_DATABASE_URL required for PostgreSQL tests")
    database = Database(url)
    database.migrate(ROOT / "migrations/lab")
    return database


@pytest.fixture(name="vc_lab_settings")
def settings(core, tmp_path):
    result = Settings(database_url=core.db.url, runtime_root=tmp_path / "working",
                      customer_code="fixture-" + str(core.customer),
                      token="fixture-only-not-secret", source_kind="fixture",
                      source_root=None, unified_archive_required=False,
                      live_sources_manifest=None, model_manifest=None,
                      allow_external_inference=False)
    result.prepare()
    return result
