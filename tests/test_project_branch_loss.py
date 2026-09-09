"""Synthetic trace integrity checks; not model or image-quality evidence."""

import copy
import json

import pytest

from visioncortex.project_branch_loss import branch_loss_trace_usage, validate_branch_loss_row


def trace_row(epoch=0, *, epochs=2, batch_size=1):
    gain = max(1 - epoch / max(epochs - 1, 1), 0) * (0.8 - 0.1) + 0.1
    many, one = [2., 4., 6.], [1., 3., 5.]
    return dict(
        schema_version="visioncortex-branch-loss/1", phase="train", policy="native",
        call=epoch + 1, epoch=epoch, epochs=epochs, batch_size=batch_size,
        files=[f"/train/image-{i}.png" for i in range(batch_size)],
        gains=dict(one2many=gain, one2one=1-gain),
        branch_vectors=dict(one2many=many, one2one=one),
        weighted_vector=[a * gain + b * (1-gain) for a, b in zip(many, one, strict=True)],
        logged_one2one_items=[x / batch_size for x in one],
    )


def write_trace(tmp_path, rows):
    path, samples = tmp_path / "branch-loss-trace.jsonl", tmp_path / "source-sampling-trace.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    samples.write_text("".join(json.dumps(dict(epoch=r["epoch"], files=r["files"])) + "\n" for r in rows))
    return path, samples


def test_recompute_observed_native_losses_with_short_final_batch(tmp_path):
    rows = [trace_row(0, batch_size=2), trace_row(1)]
    path, samples = write_trace(tmp_path, rows)
    report = branch_loss_trace_usage(path, samples, expected_epochs=2)
    assert report["calls"] == 2 and report["image_visits"] == 3
    assert report["per_epoch"][0]["mean_components_per_image"]["one2many"] == [1., 2., 3.]
    assert report["per_epoch"][1]["gains"]["one2many"] == pytest.approx(.1)
    assert report["native_csv_branch"] == "one2one"
    assert not report["validation_calls_included"]


@pytest.mark.parametrize("change", [
    lambda r: r.update(phase="val"),
    lambda r: r.update(epoch=True),
    lambda r: r.update(files=[]),
    lambda r: r.update(batch_size=2),
    lambda r: r["gains"].update(one2many=.3, one2one=.7),
    lambda r: r["gains"].update(one2many=float("nan")),
    lambda r: r["branch_vectors"].pop("one2one"),
    lambda r: r["branch_vectors"]["one2many"].__setitem__(0, -1.),
    lambda r: r["weighted_vector"].__setitem__(0, 123.),
    lambda r: r["logged_one2one_items"].__setitem__(0, 123.),
])
def test_trace_rejects_invalid_or_misattributed_losses(change):
    row = trace_row()
    change(row)
    with pytest.raises(ValueError):
        validate_branch_loss_row(row)


def test_trace_rejects_missing_calls_wrong_inputs_and_incomplete_epochs(tmp_path):
    rows = [trace_row(0), trace_row(1)]
    path, samples = write_trace(tmp_path, rows)
    with pytest.raises(ValueError, match="completed epochs"):
        branch_loss_trace_usage(path, samples, expected_epochs=3)
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="batches"):
        branch_loss_trace_usage(path, samples)
    changed = copy.deepcopy(rows)
    changed[1]["files"] = ["/validation/not-a-training-input.png"]
    path.write_text("".join(json.dumps(r) + "\n" for r in changed))
    with pytest.raises(ValueError, match="inputs differ"):
        branch_loss_trace_usage(path, samples)
    changed = copy.deepcopy(rows)
    changed[1]["call"] = 4
    path.write_text("".join(json.dumps(r) + "\n" for r in changed))
    with pytest.raises(ValueError, match="order"):
        branch_loss_trace_usage(path, samples)
    path.write_text('{"partial":')
    with pytest.raises(json.JSONDecodeError):
        branch_loss_trace_usage(path, samples)


def test_one2many_trace_rejects_wrong_and_mixed_objectives(tmp_path):
    rows = [trace_row(0), trace_row(1)]
    for row in rows:
        row.update(policy="one2many", gains=dict(one2many=1., one2one=0.),
                   weighted_vector=row["branch_vectors"]["one2many"])
    path, samples = write_trace(tmp_path, rows)
    usage = branch_loss_trace_usage(path, samples, expected_epochs=2, expected_policy="one2many")
    assert all(r["gains"] == dict(one2many=1., one2one=0.) for r in usage["per_epoch"])
    with pytest.raises(ValueError, match="objective policy"):
        branch_loss_trace_usage(path, samples, expected_policy="native")
    rows[0] = trace_row(0)
    path, samples = write_trace(tmp_path, rows)
    with pytest.raises(ValueError, match="order"):
        branch_loss_trace_usage(path, samples)
    wrong = trace_row()
    wrong["policy"] = "one2many"
    with pytest.raises(ValueError, match="declared branch objective"):
        validate_branch_loss_row(wrong)
