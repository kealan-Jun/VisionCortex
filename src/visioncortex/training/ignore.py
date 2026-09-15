"""Explicit partial-supervision adapter for the pinned local detector runtime.

Unresolved regions are never labels or ground truth. They only suppress negative
classification terms at anchors inside those regions. Known positive terms and
the native assignment/box/DFL losses remain intact. Ordinary validation images
must still be completely reviewed. Online augmentation is disabled; verified
offline derivatives supply already transformed regions through the same loader.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import torch
from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from ..project_sampling import SourceSamplingMixin


def ignored_anchor_mask(anchors, regions, image_shape):
    """Return B×A boolean coverage; missing or malformed metadata is an error."""
    if not isinstance(regions, (tuple, list)) or not regions:
        raise ValueError("Every batch image requires explicit ignore metadata")
    height, width = image_shape
    masks = []
    for boxes in regions:
        if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Ignore regions must be explicit N-by-4 pixel tensors")
        boxes = boxes.to(device=anchors.device, dtype=anchors.dtype)
        if (
            not torch.isfinite(boxes).all()
            or (boxes[:, :2] < 0).any()
            or (boxes[:, 2:] > boxes.new_tensor([width, height])).any()
            or (boxes[:, 2:] <= boxes[:, :2]).any()
        ):
            raise ValueError("Invalid ignore rectangle in transformed image")
        # Closed edges are conservative: boundary anchors must not learn an
        # unresolved object as background because of floating-point rounding.
        masks.append(
            ((anchors[:, None] >= boxes[None, :, :2]).all(-1)
             & (anchors[:, None] <= boxes[None, :, 2:]).all(-1)).any(-1)
        )
    return torch.stack(masks)


class IgnoreNegativeBCE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None
        self.calls = 0
        self.ignored_anchor_visits = 0
        self.retained_positive_terms = 0

    def forward(self, logits, targets):
        if self.mask is None or self.mask.shape != logits.shape[:2]:
            raise ValueError("Ignore mask is missing or does not match this batch")
        ignored = self.mask[..., None]
        keep = ~ignored | (targets > 0)
        self.calls += 1
        self.ignored_anchor_visits += int(self.mask.sum())
        self.retained_positive_terms += int((ignored & (targets > 0)).sum())
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ) * keep


class IgnoreDetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bce = IgnoreNegativeBCE()

    def get_assigned_targets_and_loss(self, preds, batch):
        points, strides = make_anchors(preds["feats"], self.stride, 0.5)
        self.bce.mask = ignored_anchor_mask(
            points * strides, batch.get("ignore_xyxy_px"), batch["img"].shape[-2:]
        )
        try:
            return super().get_assigned_targets_and_loss(preds, batch)
        finally:
            self.bce.mask = None


class ObservedE2ELoss(E2ELoss):
    """Observe both branches; use native or explicitly selected one2many loss."""

    def __init__(self, model, trace):
        super().__init__(model, loss_fn=IgnoreDetectionLoss)
        self.branch_trace = trace
        self.objective_policy = trace.policy
        if self.objective_policy == "one2many":
            self.o2m, self.o2o = 1.0, 0.0

    def update(self):
        super().update()
        if self.objective_policy == "one2many":
            self.o2m, self.o2o = 1.0, 0.0

    def __call__(self, preds, batch):
        if not torch.is_grad_enabled():
            raise ValueError("Training branch trace must not include validation calls")
        preds = self.one2many.parse_output(preds)
        many = self.one2many.loss(preds["one2many"], batch)
        one = self.one2one.loss(preds["one2one"], batch)
        # Returning many directly leaves the unused head's gradients absent;
        # multiplying its loss by zero would still allow AdamW weight decay.
        weighted = many[0] if self.objective_policy == "one2many" else (
            many[0] * self.o2m + one[0] * self.o2o
        )
        self.branch_trace.record(
            batch, dict(one2many=self.o2m, one2one=self.o2o),
            many[0], one[0], weighted, one[1],
        )
        return weighted, one[1]


def make_ignore_criterion(model, branch_trace=None):
    if getattr(model, "end2end", False):
        return (
            ObservedE2ELoss(model, branch_trace) if branch_trace is not None
            else E2ELoss(model, loss_fn=IgnoreDetectionLoss)
        )
    if branch_trace is not None:
        raise ValueError("Branch trace requires a native two-branch end2end detector")
    return IgnoreDetectionLoss(model)


def loss_usage(criterion):
    branches = (
        {"one2many": criterion.one2many, "one2one": criterion.one2one}
        if isinstance(criterion, E2ELoss)
        else {"detection": criterion}
    )
    return {
        name: dict(
            calls=loss.bce.calls,
            ignored_anchor_visits=loss.bce.ignored_anchor_visits,
            retained_positive_terms=loss.bce.retained_positive_terms,
        )
        for name, loss in branches.items()
    }


class IgnoreLetterBox(LetterBox):
    @staticmethod
    def _update_labels(labels, ratio, padw, padh):
        # Reuse exactly the same Instances resize/pad operation as known boxes.
        # Both begin normalized to their original source dimensions; the loader
        # resize and LetterBox therefore cannot give them different geometries.
        ignored = dict(img=labels["img"], instances=labels.pop("ignore_instances"))
        LetterBox._update_labels(ignored, ratio, padw, padh)
        labels["ignore_xyxy_px"] = torch.from_numpy(ignored["instances"].bboxes.copy())
        return LetterBox._update_labels(labels, ratio, padw, padh)


class IgnoreDataset(YOLODataset):
    def __init__(self, *args, ignore_metadata, **kwargs):
        self.ignore_metadata = ignore_metadata
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index):
        labels = super().get_image_and_label(index)
        key = str(Path(labels["im_file"]).absolute())
        if key not in self.ignore_metadata:
            raise ValueError(f"Missing explicit ignore metadata: {key}")
        record = self.ignore_metadata[key]
        height, width = labels["ori_shape"]
        if [width, height] != record["source_size"]:
            raise ValueError("Ignore metadata dimensions differ from decoded source")
        boxes = np.asarray(record["xyxy_px"], dtype=np.float32).reshape(-1, 4)
        boxes = boxes / np.array([width, height, width, height], dtype=np.float32)
        labels["ignore_instances"] = Instances(
            boxes, np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xyxy", normalized=True,
        )
        return labels

    def build_transforms(self, hyp=None):
        return Compose([
            IgnoreLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
            Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
        ])


class IgnoreTrainer(SourceSamplingMixin, DetectionTrainer):
    def __init__(self, *args, ignore_metadata, branch_loss_trace=None, **kwargs):
        if importlib.metadata.version("ultralytics") != "8.4.28":
            raise ValueError("Ignore adapter is verified only with Ultralytics 8.4.28")
        self.ignore_metadata = ignore_metadata
        self.branch_loss_trace = branch_loss_trace
        self.branch_loss_trace_path = branch_loss_trace.path if branch_loss_trace else None
        super().__init__(*args, **kwargs)
        unsupported = [
            key for key in (
                "multi_scale", "hsv_h", "hsv_s", "hsv_v", "bgr", "degrees",
                "translate", "scale", "shear", "perspective", "flipud", "fliplr",
                "mosaic", "mixup", "cutmix", "copy_paste", "erasing",
            ) if getattr(self.args, key)
        ]
        if unsupported or self.args.auto_augment is not None:
            raise ValueError(f"Ignore adapter forbids augmentation: {unsupported}")
        if (
            self.args.task != "detect" or self.args.single_cls
            or self.args.classes is not None or self.args.fraction != 1.0
            or self.args.compile
        ):
            raise ValueError("Ignore adapter requires complete-class eager detection")
        self.add_callback("on_pretrain_routine_end", self.install_ignore_loss)
        if branch_loss_trace is not None:
            if self.sampling_sources is None:
                raise ValueError("Branch trace requires actual training-source sampling evidence")
            self.add_callback(
                "on_train_epoch_start",
                lambda trainer: trainer.branch_loss_trace.start_epoch(int(trainer.epoch)),
            )

    @staticmethod
    def install_ignore_loss(trainer):
        trainer.model.criterion = make_ignore_criterion(trainer.model, trainer.branch_loss_trace)
        if trainer.ema is not None:
            trainer.ema.ema.criterion = make_ignore_criterion(trainer.ema.ema)

    def build_dataset(self, img_path, mode="train", batch=None):
        cfg = self.args
        return IgnoreDataset(
            img_path=img_path, imgsz=cfg.imgsz, batch_size=batch,
            augment=mode == "train", hyp=cfg, rect=cfg.rect or mode == "val",
            cache=False, single_cls=False,
            stride=max(int(self.model.stride.max()), 32),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ", task="detect", classes=None, data=self.data,
            fraction=1.0, ignore_metadata=self.ignore_metadata,
        )
