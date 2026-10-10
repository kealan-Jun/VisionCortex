"""Source-group sampling for frozen local training experiments.

The trace records batches entering preprocessing, not DataLoader prefetches or
new independent samples. Verified offline derivatives retain their parent group;
unique_images counts file identities, not new sources. Validation keeps the native loader.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def sampling_plan(files, sources, policy, seed):
    if policy not in {"image_uniform", "source_balanced"}:
        raise ValueError("Unknown project sampling policy")
    paths = [str(Path(p).absolute()) for p in files]
    if not paths or len(set(paths)) != len(paths) or set(paths) != set(sources):
        raise ValueError("Sampling files differ from the frozen training membership")
    if any(not r.get("image_id") or not r.get("source_group") for r in sources.values()):
        raise ValueError("Sampling requires image and source-group identities")
    if len({r["image_id"] for r in sources.values()}) != len(sources):
        raise ValueError("Sampling aliases cannot duplicate a frozen image identity")
    counts = Counter(sources[p]["source_group"] for p in paths)
    weights = [1.0 / counts[sources[p]["source_group"]] for p in paths]
    return dict(
        schema_version="visioncortex-project-source-sampling/1", policy=policy,
        seed=seed, files=paths, sources=sources, source_counts=dict(counts),
        weights=weights if policy == "source_balanced" else None,
        replacement=policy == "source_balanced", samples_per_epoch=len(paths),
        validation="native_loader_unchanged", truth_status="project_annotations",
        independent_ground_truth=False,
    )


def sampling_usage(plan_path, trace_path, sources, policy):
    plan_path, trace_path = Path(plan_path), Path(trace_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan != sampling_plan(plan["files"], sources, policy, plan["seed"]):
        raise ValueError("Sampling plan differs from the frozen training sources")
    epochs, batches = {}, 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        epoch, files = record["epoch"], record["files"]
        if (
            type(epoch) is not int or epoch < 0 or not isinstance(files, list)
            or not files or any(p not in sources for p in files)
        ):
            raise ValueError("Sampling trace contains invalid or non-training samples")
        if epoch not in epochs:
            if epoch != len(epochs):
                raise ValueError("Sampling epochs are not consecutive")
            epochs[epoch] = []
        if epoch != len(epochs) - 1:
            raise ValueError("Sampling trace returned to an earlier epoch")
        epochs[epoch].extend(files)
        batches += 1
    if not epochs or any(len(files) != len(sources) for files in epochs.values()):
        raise ValueError("Sampling trace does not contain completed training epochs")
    if policy == "image_uniform" and any(set(files) != set(sources) for files in epochs.values()):
        raise ValueError("Uniform sampling did not visit every training image once")
    all_files = [p for files in epochs.values() for p in files]
    return dict(
        plan_path=str(plan_path), plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        trace_path=str(trace_path), trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        policy=policy, completed_epochs=len(epochs), preprocessed_batches=batches,
        preprocessed_samples=len(all_files), unique_images=len(set(all_files)),
        source_visits=dict(Counter(sources[p]["source_group"] for p in all_files)),
        image_visits=dict(Counter(sources[p]["image_id"] for p in all_files)),
        epoch_source_visits=[dict(Counter(sources[p]["source_group"] for p in files))
                             for files in epochs.values()],
        measurement="actual_training_preprocessing_not_prefetch_or_independent_samples",
    )


class SourceSamplingMixin:
    def __init__(self, *args, sampling_sources=None, sampling_policy="image_uniform", **kwargs):
        if sampling_policy not in {"image_uniform", "source_balanced"}:
            raise ValueError("Unknown project sampling policy")
        if sampling_policy == "source_balanced" and not sampling_sources:
            raise ValueError("Balanced sampling requires frozen training sources")
        self.sampling_sources = sampling_sources
        self.sampling_policy = sampling_policy
        self.sampling_plan_path = self.sampling_trace_path = None
        super().__init__(*args, **kwargs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train" or self.sampling_sources is None:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1 or self.args.rect or self.args.resume or self.args.compile:
            raise ValueError("Source sampling supports fresh single-GPU non-rectangular runs only")
        if self.sampling_plan_path is not None:
            raise ValueError("Preserve the existing sampling trace; do not rebuild this loader")
        if self.sampling_policy == "image_uniform":
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = loader.dataset
        else:
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        plan = sampling_plan(dataset.im_files, self.sampling_sources, self.sampling_policy, self.args.seed)
        self.sampling_plan_path = Path(self.save_dir) / "source-sampling-plan.json"
        self.sampling_trace_path = Path(self.save_dir) / "source-sampling-trace.jsonl"
        with self.sampling_plan_path.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
        self.sampling_trace_path.touch(exist_ok=False)
        if self.sampling_policy == "source_balanced":
            import torch
            from ultralytics.data.build import InfiniteDataLoader, seed_worker
            from ultralytics.utils import RANK

            # A separate sampler generator avoids consuming the model/augmentation RNG.
            sampler = torch.utils.data.WeightedRandomSampler(
                plan["weights"], len(dataset), replacement=True,
                generator=torch.Generator().manual_seed(self.args.seed),
            )
            devices = torch.cuda.device_count()
            workers = min((os.cpu_count() or 1) // max(devices, 1), self.args.workers)
            loader = InfiniteDataLoader(
                dataset=dataset, batch_size=min(batch_size, len(dataset)), shuffle=False,
                sampler=sampler, num_workers=workers, prefetch_factor=4 if workers else None,
                pin_memory=devices > 0, collate_fn=dataset.collate_fn,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(6148914691236517205 + RANK),
                drop_last=False,
            )
        return loader

    def preprocess_batch(self, batch):
        if self.sampling_sources is None:
            return super().preprocess_batch(batch)
        files = [str(Path(p).absolute()) for p in batch["im_file"]]
        if self.sampling_sources is not None and any(p not in self.sampling_sources for p in files):
            raise ValueError("Non-training image entered training preprocessing")
        batch = super().preprocess_batch(batch)
        if self.sampling_sources is not None:
            if self.sampling_trace_path is None:
                raise ValueError("Training sampling trace was not initialized")
            with self.sampling_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(epoch=self.epoch, files=files)) + "\n")
        return batch
