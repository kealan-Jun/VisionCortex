import json
from pathlib import Path

import pytest

from visioncortex.project_annotation_training import (
    canonical_sha,
    freeze_normalization_statistics,
    frozen_state_sha256,
    project_evaluation,
    run_experiment,
    restrict_to_class_outputs,
    sha256,
    synchronize_frozen_ema,
    transfer_class_outputs,
    validate_export,
    recover_evaluated_experiment,
    OptimizationTrace,
    optimization_trace_usage,
    training_sampling_sources,
)


def export_fixture(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    rows, records, files = [], [], {}
    for i, split in enumerate(["train", "val"]):
        path = root / "first_person/images" / split / f"F{i}.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(bytes([i, 50, 90]))
        source = dict(
            path=str(path), sha256=sha256(path), source_group=f"recording-{i}"
        )
        annotation = dict(
            status="reviewed",
            completeness="all_visible_instances",
            ignore_regions=[],
            truth_status="project_annotations",
            independent_ground_truth=False,
            role="first_person",
            split=split,
            boxes=[dict(id="box", class_id=0, xyxy_px=[1, 2, 20, 30])],
        )
        rows.append(dict(id=f"F{i}", revision=1, source=source, annotation=annotation))
        records.append(
            dict(
                image_id=f"F{i}",
                revision=1,
                source_sha256=source["sha256"],
                annotation_sha256=canonical_sha(annotation),
                role="first_person",
                split=split,
            )
        )
        files[str(path.relative_to(root))] = sha256(path)
        label = root / "first_person/labels" / split / f"F{i}.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 0.105 0.2 0.19 0.35\n")
        files[str(label.relative_to(root))] = sha256(label)
    (root / "annotations.json").write_text(json.dumps(dict(images=rows)))
    files["annotations.json"] = sha256(root / "annotations.json")
    receipt = dict(
        schema_version="annotation-workbench-yolo-export/1",
        truth_status="project_annotations",
        independent_ground_truth=False,
        check=dict(export_ready=True),
        files=files,
        records=records,
    )
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root, rows


def test_project_export_keeps_truth_status_and_rejects_file_drift(tmp_path):
    root, _ = export_fixture(tmp_path)
    pin = sha256(root / "receipt.json")
    receipt, rows = validate_export(root, pin, "first_person")
    assert len(rows) == 2 and receipt["independent_ground_truth"] is False
    Path(rows[1]["source"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_export(root, pin, "first_person")


def partial_export_fixture(tmp_path):
    root, rows = export_fixture(tmp_path)
    rows[0]["annotation"].update(
        status="reviewed_partial", completeness="all_visible_outside_ignore",
        review_notes="All visible instances outside the retained region checked",
        ignore_regions=[dict(id="I0", xyxy_px=[30, 30, 70, 60], reason="unresolved")],
    )
    metadata = {}
    for r in rows:
        r["source"].update(width=100, height=80)
        a = r["annotation"]
        metadata[f"first_person/images/{a['split']}/{r['id']}.png"] = dict(
            image_id=r["id"], source_size=[100, 80], annotation_sha256=canonical_sha(a),
            xyxy_px=[i["xyxy_px"] for i in a["ignore_regions"]],
        )
    (root / "annotations.json").write_text(json.dumps(dict(images=rows)))
    (root / "ignore-regions.json").write_text(json.dumps(metadata))
    receipt = json.loads((root / "receipt.json").read_text())
    receipt.update(schema_version="annotation-workbench-partial-training-export/1")
    receipt["check"]["supervision"] = "outside_ignore"
    receipt["supervision"] = dict(
        mode="outside_ignore", complete_annotations=False,
        required_trainer_contract="ignore_negative_classification_at_anchor_centers/1",
        known_positive_terms="retained", validation_and_test="complete_annotations_only",
        augmentation="disabled", production_ready=False,
    )
    for pin, row in zip(receipt["records"], rows, strict=True):
        pin["annotation_sha256"] = canonical_sha(row["annotation"])
    for name in ["annotations.json", "ignore-regions.json"]:
        receipt["files"][name] = sha256(root / name)
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root, receipt


def augmented_partial_fixture(tmp_path):
    import copy
    from PIL import Image
    from visioncortex.project_augmentation import yolo_labels

    root, receipt = partial_export_fixture(tmp_path)
    rows = json.loads((root / "annotations.json").read_text())["images"]
    for pin, row in zip(receipt["records"], rows, strict=True):
        s = row["source"]
        Image.new("RGB", (100, 80), "white").save(s["path"])
        s.update(sha256=sha256(Path(s["path"])), camera_id="test-camera",
                 baseline_exposure="unknown", license="test fixture")
        pin["source_sha256"] = s["sha256"]
        (root / "first_person/labels" / row["annotation"]["split"] / f"{row['id']}.txt").write_text(
            yolo_labels(row["annotation"]["boxes"], 100, 80))
    row = rows[0]
    a, s = row["annotation"], row["source"]
    policy = dict(schema_version="annotation-workbench-augmentation/1", variants_per_image=1)
    identity = dict(source_id=row["id"], source_sha256=s["sha256"], annotation_sha256=canonical_sha(a),
                    policy=policy, index=0, supervision="outside_ignore")
    iid = "awaug-" + canonical_sha(identity)[:24]
    boxes, regions = copy.deepcopy(a["boxes"]), copy.deepcopy(a["ignore_regions"])
    boxes[0]["xyxy_px"] = [1 * .8 + 10, 2 * .8 + 8, 20 * .8 + 10, 30 * .8 + 8]
    regions[0]["xyxy_px"] = [34.0, 32.0, 66.0, 56.0]
    d = dict(id=iid, source_image_id=row["id"], source_revision=1, source_path=s["path"],
             source_sha256=s["sha256"], source_group=s["source_group"],
             camera_id=s["camera_id"], license=s["license"], baseline_exposure=s["baseline_exposure"],
             annotation_sha256=canonical_sha(a), role="first_person", split="train",
             truth_status="project_annotations", independent_ground_truth=False,
             derivation="deterministic_training_augmentation", counts_as_new_source=False,
             variant_index=0, policy_sha256=canonical_sha(policy), supervision="outside_ignore",
             source_to_output_matrix=[[.8, 0, 10], [0, .8, 8], [0, 0, 1]], output_size=[100, 80],
             boxes=boxes, ignore_regions=regions,
             derived_annotation_sha256=canonical_sha(dict(boxes=boxes, ignore_regions=regions)),
             image_file=f"first_person/images/train/{iid}.png", label_file=f"first_person/labels/train/{iid}.txt",
             operation_order=["brightness", "contrast", "gamma_power", "gaussian_blur", "resize_and_pad", "jpeg_444_then_png"])
    im = Image.new("RGB", (100, 80), (114, 114, 114))
    with Image.open(s["path"]) as original:
        im.paste(original.resize((80, 64)), (10, 8))
    im.save(root / d["image_file"])
    (root / d["label_file"]).write_text(yolo_labels(boxes, 100, 80))
    for kind in ["image", "label"]:
        d[f"{kind}_sha256"] = sha256(root / d[f"{kind}_file"])
        receipt["files"][d[f"{kind}_file"]] = d[f"{kind}_sha256"]
    metadata = json.loads((root / "ignore-regions.json").read_text())
    metadata[d["image_file"]] = dict(image_id=iid, source_size=[100, 80],
                                     annotation_sha256=d["derived_annotation_sha256"], xyxy_px=[[34, 32, 66, 56]])
    (root / "ignore-regions.json").write_text(json.dumps(metadata))
    (root / "annotations.json").write_text(json.dumps(dict(images=rows)))
    receipt.update(schema_version="annotation-workbench-partial-training-export/2",
                   augmentation=dict(policy=policy, records=[d], counts_as_new_sources=False,
                                     validation_and_test="originals_only_unchanged", derived_roles={"first_person": 1}))
    receipt["supervision"]["augmentation"] = "offline_affine_ignore_regions/1"
    for name in receipt["files"]:
        receipt["files"][name] = sha256(root / name)
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root, receipt


def test_verified_offline_partial_derivatives_keep_parent_sampling_identity(tmp_path):
    root, receipt = augmented_partial_fixture(tmp_path)
    _, rows = validate_export(root, sha256(root / "receipt.json"), "first_person", supervision="outside_ignore")
    sources = training_sampling_sources(root, "first_person", receipt, rows)
    assert len(sources) == 2 and len(rows) == 2  # one parent + derivative, one untouched validation
    derived = next(s for s in sources.values() if "parent_image_id" in s)
    assert derived["parent_image_id"] == "F0" and derived["counts_as_new_source"] is False
    assert {s["source_group"] for s in sources.values()} == {"recording-0"}
    with pytest.raises(ValueError, match="passing"):
        validate_export(root, sha256(root / "receipt.json"), "first_person")
    receipt["schema_version"] = "annotation-workbench-partial-training-export/1"
    (root / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="exact ignore contract"):
        validate_export(root, sha256(root / "receipt.json"), "first_person", supervision="outside_ignore")


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(ignore_regions=[]),
    lambda d: d["ignore_regions"][0].update(xyxy_px=[30, 30, 70, 60]),  # stale original coordinates
    lambda d: d.update(boxes=[]),
    lambda d: d.update(source_image_id="F1"),  # validation cannot become a training parent
    lambda d: d.update(role="third_person"),
    lambda d: d.update(source_group="wrong-group"),
    lambda d: d.update(source_revision=2),
    lambda d: d.update(source_to_output_matrix=[[.8, 0, -1], [0, .8, 8], [0, 0, 1]]),
    lambda d: d.update(source_to_output_matrix=[[float("nan"), 0, 10], [0, .8, 8], [0, 0, 1]]),
])
def test_partial_derivative_tampering_fails_even_after_repinning_receipt(tmp_path, mutation):
    root, receipt = augmented_partial_fixture(tmp_path)
    mutation(receipt["augmentation"]["records"][0])
    (root / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        validate_export(root, sha256(root / "receipt.json"), "first_person", supervision="outside_ignore")


def test_derived_label_drift_cannot_be_hidden_by_updating_file_hash(tmp_path):
    root, receipt = augmented_partial_fixture(tmp_path)
    d = receipt["augmentation"]["records"][0]
    (root / d["label_file"]).write_text("")
    d["label_sha256"] = receipt["files"][d["label_file"]] = sha256(root / d["label_file"])
    (root / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="YOLO labels differ"):
        validate_export(root, sha256(root / "receipt.json"), "first_person", supervision="outside_ignore")


def test_partial_training_requires_explicit_mode_and_exact_frozen_ignore_metadata(tmp_path):
    root, receipt = partial_export_fixture(tmp_path)
    with pytest.raises(ValueError, match="passing"):
        validate_export(root, sha256(root / "receipt.json"), "first_person")
    _, rows = validate_export(root, sha256(root / "receipt.json"), "first_person", supervision="outside_ignore")
    assert rows[0]["annotation"]["status"] == "reviewed_partial"
    metadata = json.loads((root / "ignore-regions.json").read_text())
    metadata["first_person/images/train/F0.png"]["xyxy_px"] = []
    (root / "ignore-regions.json").write_text(json.dumps(metadata))
    receipt["files"]["ignore-regions.json"] = sha256(root / "ignore-regions.json")
    (root / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Ignore metadata differs"):
        validate_export(root, sha256(root / "receipt.json"), "first_person", supervision="outside_ignore")


def test_partial_recovery_cannot_promote_predictions_without_loss_consumption(tmp_path):
    root, receipt = partial_export_fixture(tmp_path)
    receipt["project"] = dict(classes=[dict(name="tube")])
    (root / "receipt.json").write_text(json.dumps(receipt))
    out = tmp_path / "failed-run"
    out.mkdir()
    (out / "experiment.json").write_text(json.dumps(dict(
        status="failed", supervision="outside_ignore", role="first_person",
        dataset_receipt_sha256=sha256(root / "receipt.json"),
        configuration=dict(data=str(root / "first_person/partial-training.yaml")),
    )))
    with pytest.raises(ValueError, match="recorded ignore loss usage"):
        recover_evaluated_experiment(out)


def test_balanced_recovery_requires_recorded_sampling_trace(tmp_path):
    root, receipt = partial_export_fixture(tmp_path)
    receipt["project"] = dict(classes=[dict(name="tube")])
    (root / "receipt.json").write_text(json.dumps(receipt))
    out = tmp_path / "failed-balanced-run"
    out.mkdir()
    (out / "experiment.json").write_text(json.dumps(dict(
        status="failed", supervision="outside_ignore", sampling_policy="source_balanced",
        role="first_person", dataset_receipt_sha256=sha256(root / "receipt.json"),
        configuration=dict(data=str(root / "first_person/partial-training.yaml")),
    )))
    with pytest.raises(ValueError, match="recorded source sampling usage"):
        recover_evaluated_experiment(out)


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(status="needs_review"),
        lambda a: a.update(completeness="partial"),
        lambda a: a.update(ignore_regions=[{"reason": "dense"}]),
        lambda a: a.update(truth_status="pseudo_labels_not_ground_truth"),
        lambda a: a.update(independent_ground_truth=True),
    ],
)
def test_project_training_rechecks_annotation_not_just_receipt_flag(tmp_path, change):
    root, rows = export_fixture(tmp_path)
    change(rows[0]["annotation"])
    (root / "annotations.json").write_text(json.dumps(dict(images=rows)))
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["files"]["annotations.json"] = sha256(root / "annotations.json")
    receipt["records"][0]["annotation_sha256"] = canonical_sha(rows[0]["annotation"])
    (root / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Unreviewed"):
        validate_export(root, sha256(root / "receipt.json"), "first_person")


def test_bad_identity_aborts_before_gpu_startup(tmp_path):
    root, _ = export_fixture(tmp_path)
    with pytest.raises(ValueError, match="receipt SHA256"):
        run_experiment(
            root,
            "0" * 64,
            "first_person",
            Path("missing.pt"),
            "0" * 64,
            tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("setting", [
    {"nominal_batch": 0}, {"nominal_batch": 4.5}, {"nominal_batch": True},
    {"warmup_epochs": float("nan")}, {"warmup_epochs": -1},
    {"warmup_bias_lr": float("inf")}, {"warmup_bias_lr": -0.1},
    {"patience": -1}, {"patience": 1.5},
    {"freeze_layers": -1}, {"freeze_layers": 11},
    {"freeze_layers": True}, {"freeze_layers": 0.5},
])
def test_invalid_optimization_schedule_stops_before_data_or_gpu(tmp_path, setting):
    with pytest.raises(ValueError, match="bounded experiment settings"):
        run_experiment(tmp_path / "missing", "0" * 64, "third_person",
                       tmp_path / "missing.pt", "0" * 64, tmp_path / "run", **setting)
    assert not (tmp_path / "run").exists()


def test_backbone_override_cannot_misrepresent_class_output_probe(tmp_path):
    with pytest.raises(ValueError, match="own fixed training scope"):
        run_experiment(tmp_path / "missing", "0" * 64, "third_person",
                       tmp_path / "missing.pt", "0" * 64, tmp_path / "run",
                       training_scope="class_outputs", freeze_layers=0)
    assert not (tmp_path / "run").exists()


def test_trace_counts_optimizer_updates_not_accumulated_batches(tmp_path):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    trainer = SimpleNamespace(optimizer=optimizer, epoch=0, accumulate=3)
    trace = OptimizationTrace(tmp_path / "updates.jsonl")
    trace.install(trainer)
    for batch in range(6):
        trainer.epoch = batch // 2
        parameter.square().sum().backward()
        if (batch + 1) % 3 == 0:
            optimizer.step()
            optimizer.zero_grad()
    usage = trace.finish()
    assert usage["optimizer_steps"] == 2
    assert int(optimizer.state[parameter]["step"]) == 2
    assert [r["epoch"] for r in map(json.loads, trace.path.read_text().splitlines())] == [1, 2]
    optimizer.step()  # Detached hook cannot contaminate the closed receipt.
    assert optimization_trace_usage(trace.path) == usage
    trace.path.write_text(trace.path.read_text() + trace.path.read_text().splitlines()[-1] + "\n")
    with pytest.raises(ValueError, match="Invalid optimization trace"):
        optimization_trace_usage(trace.path)


def test_new_run_recovery_rejects_missing_or_changed_optimizer_evidence(tmp_path):
    root, _ = export_fixture(tmp_path)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["project"] = dict(classes=[dict(name="tube")])
    (root / "receipt.json").write_text(json.dumps(receipt))
    out = tmp_path / "failed-new-run"
    out.mkdir()
    previous = dict(
        status="failed", role="first_person",
        dataset_receipt_sha256=sha256(root / "receipt.json"),
        configuration=dict(data=str(root / "first_person/data.yaml"), nbs=4),
    )
    (out / "experiment.json").write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="recorded optimizer steps"):
        recover_evaluated_experiment(out)
    trace = out / "optimization-trace.jsonl"
    trace.write_text(json.dumps(dict(step=1, epoch=0, accumulation=1, learning_rates=[.001])) + "\n")
    previous["optimization_usage"] = optimization_trace_usage(trace)
    (out / "experiment.json").write_text(json.dumps(previous))
    trace.write_text(json.dumps(dict(step=1, epoch=0, accumulation=1, learning_rates=[.1])) + "\n")
    with pytest.raises(ValueError, match="recorded optimizer steps"):
        recover_evaluated_experiment(out)


def test_unregistered_training_image_cannot_enter_as_background(tmp_path):
    root, _ = export_fixture(tmp_path)
    extra = root / "first_person/images/train/unreviewed.png"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"not registered")
    with pytest.raises(ValueError, match="Unregistered"):
        validate_export(root, sha256(root / "receipt.json"), "first_person")


def test_internal_project_metrics_count_duplicate_and_missing_instances(tmp_path):
    _, rows = export_fixture(tmp_path)
    val = [rows[1]]
    predictions = [
        dict(
            event_id="recording-1",
            role="first_person",
            frame_index=0,
            detections=[
                dict(class_name="tube", xyxy=[1, 2, 20, 30], confidence=0.9),
                dict(class_name="tube", xyxy=[1, 2, 20, 30], confidence=0.8),
            ],
        )
    ]
    report = project_evaluation(predictions, val, ["tube", "pipette"])
    assert report["micro"]["true_positive"] == 1
    assert report["micro"]["false_positive"] == 1
    assert report["independent_ground_truth"] is False
    assert not report["production_ready"]
    assert report["insufficient_positive_classes"] == ["tube", "pipette"]


def test_class_extension_preserves_shared_rows_and_leaves_new_outputs_initialized():
    torch = pytest.importorskip("torch")
    source = {
        "model.23.cv3.0.2.weight": torch.arange(42.0).reshape(21, 2, 1, 1),
        "model.23.cv3.0.2.bias": torch.arange(21.0),
        "model.23.cv2.0.weight": torch.ones(21, 2, 1, 1),
    }
    target = {
        "model.23.cv3.0.2.weight": torch.full((23, 2, 1, 1), -9.0),
        "model.23.cv3.0.2.bias": torch.full((23,), -8.0),
        "model.23.cv2.0.weight": torch.zeros(23, 2, 1, 1),
    }
    copied = transfer_class_outputs(source, target, 21, 23)
    assert len(copied) == 2
    assert torch.equal(target[copied[0]][:21], source[copied[0]])
    assert torch.equal(target[copied[1]][:21], source[copied[1]])
    assert torch.all(target[copied[0]][21:] == -9)
    assert torch.all(target[copied[1]][21:] == -8)
    assert torch.all(target["model.23.cv2.0.weight"] == 0)
    with pytest.raises(ValueError, match="No compatible"):
        transfer_class_outputs({}, target, 21, 23)


def test_legacy_box_semantics_migrate_without_promoting_single_tip_labels():
    torch = pytest.importorskip("torch")
    source = {
        "model.23.cv3.0.2.weight": torch.arange(42.0).reshape(21, 2, 1, 1),
        "model.23.cv3.0.2.bias": torch.arange(21.0),
    }
    target = {
        "model.23.cv3.0.2.weight": torch.full((23, 2, 1, 1), -9.0),
        "model.23.cv3.0.2.bias": torch.full((23,), -8.0),
    }
    transfer_class_outputs(source, target, 21, 23, migrate_legacy_tip_box=True)
    for key in source:
        assert torch.equal(target[key][22], source[key][11])
        assert torch.equal(target[key][10], source[key][10])
        assert torch.equal(target[key][12], source[key][12])
    assert torch.all(target["model.23.cv3.0.2.weight"][11] == -9)
    assert torch.all(target["model.23.cv3.0.2.bias"][11] == -8)


def test_classification_probe_learns_without_changing_features_boxes_or_bn():
    import copy

    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
            self.cv3 = nn.ModuleList([
                nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.Conv2d(4, 23, 1))
            ])
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.cv2 = nn.Conv2d(4, 4, 1)

        def forward(self, image):
            features = self.features(image)
            return self.cv3[0](features), self.one2one_cv3[0](features), self.cv2(features)

    model = Detector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    selected = restrict_to_class_outputs(model, 23)
    assert len(selected) == 4
    before = frozen_state_sha256(model, selected)
    outputs_before = {n: p.detach().clone() for n, p in model.named_parameters() if n in selected}
    ema = copy.deepcopy(model)
    for offset in [0., 10.]:
        model.train()  # Match the real trainer's epoch reset.
        freeze_normalization_statistics(model)
        assert model.training and model.cv3.training
        predictions = model(torch.randn(2, 3, 5, 5) + offset)
        loss = sum(p.square().mean() for p in predictions[:2])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert frozen_state_sha256(model, selected) == before
    assert all(not torch.equal(outputs_before[n], p) for n, p in model.named_parameters() if n in selected)
    with torch.no_grad():
        ema.features[1].running_mean.add_(3.)
        ema.cv2.weight.add_(2.)
    ema_outputs = {n: p.detach().clone() for n, p in ema.named_parameters() if n in selected}
    synchronize_frozen_ema(model, ema, selected)
    assert frozen_state_sha256(ema, selected) == before
    assert all(torch.equal(ema_outputs[n], p) for n, p in ema.named_parameters() if n in selected)
    with pytest.raises(ValueError, match="No verified"):
        restrict_to_class_outputs(nn.Conv2d(3, 23, 1), 23)
