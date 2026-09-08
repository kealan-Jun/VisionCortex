"""Loss and geometry contracts; fixtures are synthetic, not quality evidence."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")
from visioncortex.project_ignore_training import (  # noqa: E402
    IgnoreDataset,
    IgnoreNegativeBCE,
    ignored_anchor_mask,
    loss_usage,
    make_ignore_criterion,
)


def test_negative_gradient_mask_preserves_positive_terms_and_other_images():
    logits = torch.zeros((2, 3, 2), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 1] = 0.8
    loss = IgnoreNegativeBCE()
    points = torch.tensor([[4., 4.], [8., 8.], [12., 12.]])
    loss.mask = ignored_anchor_mask(
        points, [torch.tensor([[0., 0., 8., 8.]]), torch.empty((0, 4))], (16, 16)
    )
    loss(logits, targets).sum().backward()
    assert logits.grad[0, 0, 0] == 0
    assert logits.grad[0, 1].eq(0).all()  # Closed boundary.
    assert logits.grad[0, 0, 1].item() == pytest.approx(-0.3)
    assert logits.grad[0, 2].eq(0.5).all()
    assert logits.grad[1].eq(0.5).all()
    assert loss.retained_positive_terms == 1


@pytest.mark.parametrize("regions", [
    None, [], [None], [torch.zeros((1, 3))],
    [torch.tensor([[0., 0., float("nan"), 4.]])],
    [torch.tensor([[0., 0., 0., 4.]])],
    [torch.tensor([[-1., 0., 4., 4.]])],
    [torch.tensor([[0., 0., 17., 4.]])],
])
def test_missing_and_invalid_ignore_metadata_fails_closed(regions):
    with pytest.raises(ValueError):
        ignored_anchor_mask(torch.tensor([[4., 4.]]), regions, (16, 16))


def test_stale_or_wrong_batch_mask_is_rejected():
    loss = IgnoreNegativeBCE()
    logits = torch.zeros((2, 4, 3))
    with pytest.raises(ValueError, match="missing"):
        loss(logits, logits)
    loss.mask = torch.zeros((1, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="match"):
        loss(logits, logits)


@pytest.mark.parametrize("end2end", [False, True])
def test_native_assignment_and_regression_retained_in_all_loss_branches(end2end):
    from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=2)
            self.model = [SimpleNamespace(stride=torch.tensor([8., 16., 32.]), nc=2, reg_max=16)]
            self.end2end = end2end

    model = Model()
    criterion = make_ignore_criterion(model)
    native = E2ELoss(model) if end2end else v8DetectionLoss(model)
    def prediction():
        return dict(
            boxes=torch.zeros((1, 64, 84), requires_grad=True),
            scores=torch.zeros((1, 2, 84), requires_grad=True),
            feats=[torch.zeros((1, 1, n, n)) for n in (8, 4, 2)],
        )
    preds = dict(one2many=prediction(), one2one=prediction()) if end2end else prediction()
    batch = dict(
        img=torch.zeros((1, 3, 64, 64)), batch_idx=torch.tensor([0]),
        cls=torch.tensor([[0.]]), bboxes=torch.tensor([[0.5, 0.5, 0.8, 0.8]]),
        ignore_xyxy_px=(torch.tensor([[0., 0., 64., 64.]]),),
    )
    total, items = criterion(preds, batch)
    _, reference = native(preds, batch)
    torch.testing.assert_close(items[[0, 2]], reference[[0, 2]])
    assert items[1] < reference[1]
    total.sum().backward()
    for counts in loss_usage(criterion).values():
        assert counts["calls"] == 1
        assert counts["ignored_anchor_visits"] == 84
        assert counts["retained_positive_terms"] > 0
    batch.pop("ignore_xyxy_px")
    with pytest.raises(ValueError, match="explicit"):
        criterion(preds, batch)


@pytest.mark.parametrize("size, augment, rect", [
    ((101, 73), True, False), ((721, 1281), True, False),
    ((101, 73), False, True), ((1281, 721), False, True),
])
def test_loader_and_letterbox_transform_ignores_exactly_like_known_boxes(
    tmp_path, size, augment, rect
):
    from PIL import Image
    from ultralytics.cfg import get_cfg

    width, height = size
    path = tmp_path / "images" / "sample.png"
    path.parent.mkdir()
    Image.new("RGB", size, (60, 80, 100)).save(path)
    label = tmp_path / "labels" / "sample.txt"
    label.parent.mkdir()
    label.write_text("0 0.5 0.5 0.5 0.5\n")
    metadata = {str(path): dict(source_size=list(size), xyxy_px=[
        [width / 4, height / 4, width * 3 / 4, height * 3 / 4]
    ])}
    dataset = IgnoreDataset(
        img_path=str(path.parent), imgsz=96, batch_size=1, augment=augment,
        rect=rect, hyp=get_cfg(), data={"names": {0: "tube"}, "nc": 1},
        ignore_metadata=metadata,
    )
    item = dataset[0]
    h, w = item["img"].shape[-2:]
    xywh = item["bboxes"][0] * torch.tensor([w, h, w, h])
    expected = torch.cat((xywh[:2] - xywh[2:] / 2, xywh[:2] + xywh[2:] / 2))
    torch.testing.assert_close(item["ignore_xyxy_px"][0], expected)
    batch = dataset.collate_fn([item, dataset[0]])
    assert len(batch["ignore_xyxy_px"]) == 2
    metadata[str(path)]["source_size"] = [1, 1]
    with pytest.raises(ValueError, match="dimensions"):
        dataset[0]
    metadata.clear()
    with pytest.raises(ValueError, match="Missing"):
        dataset[0]
