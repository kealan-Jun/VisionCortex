"""Pixel/ambiguity contracts only; these fixtures are not field-quality gold."""

import copy

import numpy as np
import pytest

from visioncortex.lab.native_points import observe
from visioncortex.lab.native_views import color_residual
from visioncortex.lab.video_numeric import resolve_observed_points


QUADS = [[[10, 10], [35, 10], [35, 75], [10, 75]],
         [[70, 10], [95, 10], [95, 75], [70, 75]]]


def scene():
    return np.full((100, 120, 3), [100, 80, 100], np.uint8)


def test_source_point_persistence_and_no_pixel_mutation():
    im = scene()
    im[72:75, 48:51] = [160, 160, 230]
    before = im.copy()
    result = observe(im, QUADS, "red")
    assert np.array_equal(im, before)
    assert len(result["points"]) == 1
    point = result["points"][0]
    assert point["box"] == [48, 72, 51, 75]
    assert max(point["threshold_values"]) - min(point["threshold_values"]) >= 4
    assert not result["confirmed"] and not result["OFF_inferred"]
    assert resolve_observed_points(["12"], 2, result["points"])["numeric_candidate"] == "1.2"


@pytest.mark.parametrize("kind", ["clipped_stroke", "white_glare", "singleton", "inside_glyph", "absent"])
def test_partial_strokes_glare_noise_and_missing_dots_stay_unknown(kind):
    im = scene()
    if kind == "clipped_stroke":
        im[72:75, 30:51] = [160, 160, 230]
    elif kind == "white_glare":
        im[72:75, 48:51] = [230, 230, 230]
    elif kind == "singleton":
        im[73, 49] = [160, 160, 230]
    elif kind == "inside_glyph":
        im[72:75, 30:33] = [160, 160, 230]
    result = observe(im, QUADS, "red")
    assert not result["points"]
    assert resolve_observed_points(["1.2"], 2, result["points"])["numeric_candidate"] is None


def test_multiple_source_points_and_conflicting_model_dot_are_not_resolved():
    im = scene()
    im[72:75, 48:51] = [160, 160, 230]
    im[72:75, 58:61] = [160, 160, 230]
    points = observe(im, QUADS, "red")["points"]
    assert len(points) == 2
    assert resolve_observed_points(["1.2"], 2, points)["numeric_candidate"] is None
    assert resolve_observed_points(["12.3"], 3, [points[0]])["numeric_candidate"] is None


@pytest.mark.parametrize("kind", ["nonfinite", "reversed", "clipped"])
def test_source_quadrilateral_validation(kind):
    q = copy.deepcopy(QUADS)
    if kind == "nonfinite":
        q[0][0][0] = float("nan")
    elif kind == "reversed":
        q[0].reverse()
    else:
        q[0][0][0] = -1
    with pytest.raises(ValueError):
        observe(scene(), q, "red")


def test_photometric_view_retains_source_shape_but_cannot_supply_evidence():
    im = scene()
    im[72:75, 48:51] = [160, 160, 230]
    before = im.copy()
    view, metadata = color_residual(im)
    assert view.shape == im.shape and view.dtype == im.dtype
    assert np.array_equal(im, before)
    assert metadata["source_bgr_sha256"] != metadata["prepared_bgr_sha256"]
    assert not metadata["supplies_stroke_or_point_evidence"]
    assert not metadata["source_pixels_replaced"]
