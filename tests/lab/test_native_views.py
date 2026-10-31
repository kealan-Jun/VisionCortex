"""Model-view pixel/transform contracts, not real OCR quality gold."""

import hashlib

import numpy as np

from visioncortex.lab.native_views import support_crop


def test_support_view_crops_original_colors_and_keeps_baseline_room(monkeypatch):
    image = np.zeros((60, 120, 3), np.uint8)
    image[:, :, 0] = 90
    image[15:45, 22:28, 2] = 240
    image[15:45, 72:88, 2] = 240
    quads = [[[10, 8], [40, 8], [40, 50], [10, 50]],
             [[65, 8], [95, 8], [95, 50], [65, 50]]]
    native = dict(image=image, layout=dict(color="red"), quads=quads)
    monkeypatch.setattr("visioncortex.lab.native_segments.observe", lambda native: dict(
        slots=[dict(native_cell_polygon=q) for q in quads]))
    before = image.copy()
    view, metadata = support_crop(native)
    x, y, xx, yy = metadata["native_crop_box"]
    assert np.array_equal(image, before) and np.array_equal(view, image[y:yy, x:xx])
    assert (x, y, xx, yy) == (20, 13, 90, 60)
    assert metadata["source_bgr_sha256"] == hashlib.sha256(image.tobytes()).hexdigest()
    assert metadata["native_to_prepared"] == [[1, 0, -x], [0, 1, -y], [0, 0, 1]]
    assert not metadata["resampled"] and not metadata["strokes_completed"]


def test_dark_slot_cannot_be_completed_to_make_a_support_view(monkeypatch):
    image = np.zeros((60, 120, 3), np.uint8)
    quads = [[[10, 8], [40, 8], [40, 50], [10, 50]]]
    monkeypatch.setattr("visioncortex.lab.native_segments.observe", lambda native: dict(
        slots=[dict(native_cell_polygon=q) for q in quads]))
    view, metadata = support_crop(dict(image=image, layout=dict(color="red"), quads=quads))
    assert view is None and metadata["status"] == "pending"
