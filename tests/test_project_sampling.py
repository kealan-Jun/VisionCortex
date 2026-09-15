"""Synthetic sampling/integrity contracts; not model-quality evidence."""

from collections import Counter
import json
from types import SimpleNamespace

import pytest

from visioncortex.project_annotation_training import run_experiment, training_sampling_sources
from visioncortex.project_sampling import SourceSamplingMixin, sampling_plan, sampling_usage


def sources_fixture(tmp_path):
    return {
        str(tmp_path / f"{i}.png"): dict(image_id=f"F{i}", source_group="long" if i < 3 else "short")
        for i in range(4)
    }


def test_no_sampling_metadata_preserves_native_preprocessing():
    class Native:
        def preprocess_batch(self, batch):
            return batch

    class Trainer(SourceSamplingMixin, Native):
        pass

    batch = {"native_only_field": 1}
    assert Trainer().preprocess_batch(batch) is batch


def test_source_probability_not_proportional_to_near_duplicate_frame_count(tmp_path):
    sources = sources_fixture(tmp_path)
    plan = sampling_plan(list(sources)[::-1], sources, "source_balanced", 42)
    mass = Counter()
    for path, weight in zip(plan["files"], plan["weights"], strict=True):
        mass[sources[path]["source_group"]] += weight
    assert mass == {"long": 1., "short": 1.}
    assert plan["samples_per_epoch"] == 4
    assert plan["replacement"]


@pytest.mark.parametrize("alter", [
    lambda p: p[:-1], lambda p: p + [p[0]], lambda p: p[:-1] + ["/test/held-out.png"],
])
def test_loader_membership_must_match_training_export_exactly(tmp_path, alter):
    sources = sources_fixture(tmp_path)
    with pytest.raises(ValueError, match="membership"):
        sampling_plan(alter(list(sources)), sources, "source_balanced", 42)


def test_sampling_metadata_filters_role_and_validation_before_runtime(tmp_path):
    rows = [dict(id="a", revision=1, source=dict(source_group="train", sha256="1"),
                 annotation=dict(split="train")),
            dict(id="b", revision=1, source=dict(source_group="validation", sha256="2"),
                 annotation=dict(split="val"))]
    receipt = dict(files={"first_person/images/train/a.png": "1",
                          "first_person/images/val/b.png": "2",
                          "third_person/images/train/third.png": "3"})
    result = training_sampling_sources(tmp_path, "first_person", receipt, rows)
    assert {v["image_id"] for v in result.values()} == {"a"}
    receipt["files"]["first_person/images/train/b.png"] = "2"
    with pytest.raises(ValueError, match="unregistered"):
        training_sampling_sources(tmp_path, "first_person", receipt, rows)


@pytest.mark.parametrize("kwargs", [dict(sampling_policy="unknown"),
                                    dict(sampling_policy="source_balanced")])
def test_unsupported_sampling_rejected_before_model_or_output_creation(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_experiment(tmp_path / "missing", "0" * 64, "first_person", tmp_path / "missing.pt",
                       "0" * 64, tmp_path / "run", **kwargs)
    assert not (tmp_path / "run").exists()


def trainer_fixture(tmp_path, monkeypatch, policy):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    sources = sources_fixture(tmp_path)

    class Dataset(torch.utils.data.Dataset):
        im_files = list(sources)
        collate_fn = staticmethod(lambda batch: dict(im_file=[x["im_file"] for x in batch]))

        def __len__(self):
            return len(self.im_files)

        def __getitem__(self, i):
            return dict(im_file=self.im_files[i])

    class Native:
        def __init__(self):
            self.args = SimpleNamespace(seed=42, workers=0, rect=False, resume=False, compile=False)
            self.save_dir = tmp_path
            self.epoch = 0
            self.dataset = Dataset()

        def build_dataset(self, *args):
            return self.dataset

        def get_dataloader(self, path, batch_size, rank, mode):
            if mode == "val":
                return ("native-validation", path, batch_size, rank)
            return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=self.dataset.collate_fn)

        def preprocess_batch(self, batch):
            return dict(batch, native_preprocessed=True)

    class Trainer(SourceSamplingMixin, Native):
        pass

    return Trainer(sampling_sources=sources, sampling_policy=policy), sources


@pytest.mark.parametrize("policy", ["image_uniform", "source_balanced"])
def test_trace_counts_consumed_batches_not_loader_prefetch_and_val_stays_native(tmp_path, monkeypatch, policy):
    trainer, sources = trainer_fixture(tmp_path, monkeypatch, policy)
    assert trainer.get_dataloader("test", 2, -1, "val") == ("native-validation", "test", 2, -1)
    assert trainer.sampling_trace_path is None
    loader = trainer.get_dataloader("train", 2, -1, "train")
    fetched = list(loader)
    assert trainer.sampling_trace_path.read_text() == ""
    for batch in fetched:
        assert trainer.preprocess_batch(batch)["native_preprocessed"]
    usage = sampling_usage(trainer.sampling_plan_path, trainer.sampling_trace_path, sources, policy)
    assert usage["preprocessed_samples"] == 4
    assert usage["preprocessed_batches"] == 2
    assert usage["completed_epochs"] == 1
    if policy == "image_uniform":
        assert usage["source_visits"] == {"long": 3, "short": 1}
    with pytest.raises(ValueError, match="Non-training"):
        trainer.preprocess_batch(dict(im_file=[str(tmp_path / "val.png")]))
    assert sampling_usage(trainer.sampling_plan_path, trainer.sampling_trace_path, sources, policy) == usage


@pytest.mark.parametrize("records", [[], [dict(epoch=0, files=[0])],
    [dict(epoch=1, files=[0, 1, 2, 3])], [dict(epoch=0, files=[0, 0, 1, 2])],
    [dict(epoch=0, files=[0, 1, 2, 99])], [dict(epoch=True, files=[0, 1, 2, 3])],
])
def test_incomplete_or_leaking_uniform_trace_cannot_be_claimed_complete(tmp_path, records):
    sources = sources_fixture(tmp_path)
    paths = list(sources)
    plan = tmp_path / "plan.json"
    trace = tmp_path / "trace.jsonl"
    plan.write_text(json.dumps(sampling_plan(paths, sources, "image_uniform", 42)))
    trace.write_text("".join(json.dumps(dict(r, files=[paths[i] if i < 4 else "/val.png"
                                                     for i in r["files"]])) + "\n" for r in records))
    with pytest.raises(ValueError):
        sampling_usage(plan, trace, sources, "image_uniform")
