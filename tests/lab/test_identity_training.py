from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

import pytest
import qrcode

from visioncortex.lab.domain import Conflict
from visioncortex.lab.training import validate_dataset, prepare_dataset


def test_identity_allocation_is_atomic_and_codes_decode_to_permanent_objects(core):
    prefix = "T" + uuid4().hex[:9].upper()
    def issue(n):
        return core.issue_identity(f"issue-{n}", "instrument", prefix, metadata={"fixture": True})
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(issue, range(16)))
    assert len({r["code"] for r in rows}) == 16
    assert sorted(r["code"] for r in rows) == [f"{prefix}-EQ-{n:06d}" for n in range(1,17)]
    assert issue(0) == rows[0]
    with core.db.transaction(readonly=True) as conn:
        tag = conn.execute("SELECT id FROM entity WHERE customer_id=%s AND kind='tag' AND code=%s", (core.customer, rows[0]["code"])).fetchone()
    assert core.resolve_tag(tag["id"], datetime.now(timezone.utc))["target_id"] == rows[0]["id"]
    with pytest.raises(Conflict):
        core.issue_identity("wrong-prefix", "instrument", prefix + "X")


def test_new_material_identity_has_unknown_opening_and_reusable_label_is_separate(core):
    prefix = "M" + uuid4().hex[:9].upper()
    with pytest.raises(ValueError, match="dimension"):
        core.issue_identity("no-unit", "material_unit", prefix)
    material = core.issue_identity("material", "material_unit", prefix, unit="g", metadata={"physical_form": "powder"})
    label = core.issue_identity("label", "tag", prefix)
    assert material["code"] == f"{prefix}-MA-000001"
    assert label["code"] == f"{prefix}-TG-000001"
    assert material["balance"] is None and label["label_mode"] == "reusable"


def test_dataset_capture_group_and_duplicate_content_cannot_leak_between_splits(settings):
    folder = settings.runtime_root / "training-subset"
    folder.mkdir()
    images = []
    for n, split in enumerate(("train", "val")):
        path = folder / f"{n}.png"
        qrcode.make("fixture-" + str(n)).save(path)
        from PIL import Image
        with Image.open(path) as im:width, height = im.size
        images.append(dict(image_id=str(n), image_path=path.name, split=split, capture_group_id="group-" + str(n), width=width, height=height, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    data = dict(schema_version="visioncortex-yolo-reviewed-ground-truth/1", truth_status="reviewed_ground_truth", customer_code=settings.customer_code, reviewed_by="fixture-only-not-real-training-truth", classes=["balance"], images=images, annotations=[])
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(data))
    assert validate_dataset(manifest, folder, customer_code=settings.customer_code)["image_count"] == 2
    result = prepare_dataset(manifest, folder, settings.runtime_root / "training-view", customer_code=settings.customer_code)
    assert result["existing_dataset_receipt"]["source_copy_bytes"] == 0
    images[1]["capture_group_id"] = "group-0"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="leak"):
        validate_dataset(manifest, folder, customer_code=settings.customer_code)


def test_identity_catalogue_is_explicitly_unbound_and_has_no_invented_material_stock():
    from ._support import private_fixture
    data = json.loads(private_fixture("identity-catalogue-draft.json").read_text())
    assert len(data["equipment"]) == len({r["code"] for r in data["equipment"]}) == 32
    assert all(r["status"] == "reserved_unbound" and r["physical_instance_id"] is None for r in data["equipment"])
    assert data["material_instances"] == []
