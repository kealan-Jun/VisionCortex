"""Shared role, decoder-lane and inference scheduling for both entry points."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextvars import copy_context
from .schemas import ViewInput, ViewRole

_COARSE_DECODE_LOCK = threading.Lock()
_COARSE_CUDA_USERS = 0


def scan_views_concurrently(config, views, infos, transforms, work_dir, **kwargs):
    from .runtime_control import resource_slot
    with resource_slot(config, 'vision', units=max(1, len(views))):
        return _admitted_scan(config, views, infos, transforms, work_dir, **kwargs)


def _admitted_scan(config, views, infos, transforms, work_dir, **kwargs):
    """Share optional NVDEC capacity across independent recorder jobs.

    Excess ready sources use CPU decoding immediately; they do not wait behind
    another camera's decoder. Explicit caller assignments retain precedence.
    """
    global _COARSE_CUDA_USERS
    if kwargs.get("phase", "fine") == "coarse":
        # Both entry points use the same detector and action algorithm. An
        # independently accepted coarse engine need not replace fine evidence
        # inference, where small numeric shifts can change threshold decisions.
        config = deepcopy(config)
        for role in ("first_person", "third_person"):
            engine = config.get("models", {}).get(f"{role}_coarse_engine")
            if engine:
                config["models"][f"{role}_engine"] = engine
        if "coarse_inference_batch_wait_ms" in config["performance"]:
            config["performance"]["inference_batch_wait_ms"] = config["performance"]["coarse_inference_batch_wait_ms"]
    perf = config["performance"]
    limit = perf.get("coarse_cuda_max_concurrent_sources", 0)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 32:
        raise ValueError("coarse_cuda_max_concurrent_sources must be an integer in [0,32]")
    if (not limit or kwargs.get("phase", "fine") != "coarse"
            or kwargs.get("decode_backends") is not None or not perf.get("ffmpeg_hwaccel", True)):
        return _scan_views_concurrently(config, views, infos, transforms, work_dir, **kwargs)
    backends = {}
    reserved = 0
    with _COARSE_DECODE_LOCK:
        for view in views:
            gpu = _COARSE_CUDA_USERS < limit
            backends[view.view_id] = "cuda" if gpu else "cpu"
            if gpu:
                _COARSE_CUDA_USERS += 1
                reserved += 1
    try:
        return _scan_views_concurrently(config, views, infos, transforms, work_dir,
                                       **(kwargs | {"decode_backends": backends}))
    finally:
        with _COARSE_DECODE_LOCK:
            _COARSE_CUDA_USERS -= reserved


def _scan_views_concurrently(config, views, infos, transforms, work_dir, *,
                            windows=None, sample_fps=None, image_size=None,
                            keyframes_only=False, phase="fine", decode_backends=None,
                            progress_callback=None, view_runtime=None, scanner=None):
    if scanner is None:
        from .detection import scan_videos as scanner
    scan_videos = scanner
    if progress_callback is None:
        def progress_callback(*args):
            pass
    if view_runtime is None:
        view_runtime = {}
    role_groups = [
        [view for view in views if view.role == ViewRole.FIRST_PERSON],
        [view for view in views if view.role == ViewRole.THIRD_PERSON],
    ]
    role_groups = [group for group in role_groups if group]
    kwargs = {
        "windows": windows,
        "sample_fps": sample_fps,
        "image_size": image_size,
        "keyframes_only": keyframes_only,
        "phase": phase,
        "progress_callback": lambda view_id, completed, total: progress_callback(
            phase, view_id, completed, total
        ),
    }
    perf = config["performance"]
    concurrent_roles = (
        bool(perf.get("concurrent_role_scanners", True))
        and len(role_groups) > 1
    )
    workers_per_role = max(
        1, int(perf.get("yolo_inference_workers", 1))
    )
    same_role_workers = max(1, min(4, int(perf.get("fine_inference_workers_per_role", 1))))
    same_role_workers = min(
        same_role_workers, max(1, int(perf.get("fine_active_decode_slots", 1)))
    )
    parallel_same_role = (
        phase == "fine" and len(role_groups) == 1
        and same_role_workers > 1 and len(role_groups[0]) > 1
    )
    parallel_scanners = concurrent_roles or parallel_same_role
    if parallel_same_role:
        workers_per_role = same_role_workers
    scanner_groups: list[tuple[list[ViewInput], str | None]] = []
    for role_group in role_groups:
        worker_count = (
            min(workers_per_role, len(role_group))
            if parallel_scanners
            else 1
        )
        partitions = [role_group[index::worker_count] for index in range(worker_count)]
        partitions = [partition for partition in partitions if partition]
        for index, partition in enumerate(partitions, start=1):
            scanner_groups.append(
                (
                    partition,
                    (
                        f"worker_{index:02d}"
                        if len(partitions) > 1
                        else None
                    ),
                )
            )
    requested_sources = int(perf.get("source_workers", len(views)))
    if requested_sources < len(views):
        raise ValueError(
            f"source_workers={requested_sources} cannot keep {len(views)} views active"
        )
    lanes = list(
        perf.get(
            "coarse_decode_lanes"
            if phase in {"motion_probe", "coarse"}
            else "fine_decode_lanes",
            [],
        )
    )
    if not lanes:
        lanes = ["cuda" if perf.get("ffmpeg_hwaccel") else "cpu"] * len(views)
    if len(lanes) < len(views):
        lanes.extend([lanes[-1]] * (len(views) - len(lanes)))
    default_decode_backends = {
        view.view_id: lanes[index] for index, view in enumerate(views)
    }
    kwargs["decode_backends"] = {
        view.view_id: (decode_backends or default_decode_backends).get(
            view.view_id, default_decode_backends[view.view_id]
        )
        for view in views
    }
    if (
        phase == "coarse"
        and windows is None
        and perf.get("synchronized_segment_waves")
        and concurrent_roles
        and all(view.segments for view in views)
    ):
        segment_counts = {view.view_id: len(view.segments) for view in views}
        if len(set(segment_counts.values())) != 1:
            raise ValueError(
                f"synchronized segment waves require equal segment counts: {segment_counts}"
            )
        kwargs["wave_barrier"] = threading.Barrier(len(views))
    for view in views:
        runtime = view_runtime.setdefault(view.view_id, {})
        runtime.update(
            {
                "role": view.role.value,
                "decode_backend": kwargs["decode_backends"][view.view_id],
                "state": f"{phase}_running",
            }
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / f"scheduler_{phase}.json").write_text(
        json.dumps(
            {
                "schema_version": "visioncortex-role-scheduler/1",
                "phase": phase,
                "mode": (
                    "concurrent_same_role_workers"
                    if parallel_same_role
                    else
                    "concurrent_role_workers"
                    if concurrent_roles and len(scanner_groups) > len(role_groups)
                    else "concurrent_roles"
                    if concurrent_roles
                    else "sequential_role_residency"
                ),
                "role_order": [group[0].role.value for group in role_groups],
                "yolo_inference_workers_per_role": workers_per_role,
                "active_scanner_count": len(scanner_groups),
                "scanner_groups": [
                    {
                        "scanner_id": scanner_id,
                        "role": group[0].role.value,
                        "view_ids": [view.view_id for view in group],
                    }
                    for group, scanner_id in scanner_groups
                ],
                "configured_decode_lanes": lanes,
                "active_view_ids": [view.view_id for view in views],
                "decode_backends": kwargs["decode_backends"],
                "reason": (
                    "bounded same-role contexts sharing the configured decode-slot budget"
                    if parallel_same_role
                    else
                    "configured concurrent role scanners"
                    if concurrent_roles
                    else "one TensorRT role model resident at a time to preserve batch capacity"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not parallel_scanners:
        result = {}
        for group in role_groups:
            group_kwargs = dict(kwargs)
            group_kwargs["decode_backends"] = {
                view.view_id: kwargs["decode_backends"][view.view_id]
                for view in group
            }
            if (
                phase == "coarse"
                and windows is None
                and perf.get("synchronized_segment_waves")
                and all(view.segments for view in group)
                and len(group) > 1
            ):
                group_kwargs["wave_barrier"] = threading.Barrier(len(group))
            for view in group:
                view_runtime[view.view_id]["decode_backend"] = group_kwargs[
                    "decode_backends"
                ][view.view_id]
            result.update(
                scan_videos(
                    group,
                    infos,
                    transforms,
                    work_dir,
                    config,
                    **group_kwargs,
                )
            )
        for view in views:
            view_runtime[view.view_id]["state"] = f"{phase}_completed"
        return result
    result = {}
    worker_config = config
    if parallel_same_role:
        worker_config = deepcopy(config)
        decode_slots = max(1, int(perf.get("fine_active_decode_slots", 1)))
        worker_config["performance"]["fine_active_decode_slots"] = max(
            1, decode_slots // len(scanner_groups)
        )
    with ThreadPoolExecutor(
        max_workers=len(scanner_groups),
        thread_name_prefix="role-scanner",
    ) as executor:
        futures = [
            executor.submit(
                copy_context().run, scan_videos,
                group,
                infos,
                transforms,
                work_dir,
                worker_config,
                scanner_id=scanner_id,
                **kwargs,
            )
            for group, scanner_id in scanner_groups
        ]
        for future in futures:
            result.update(future.result())
    for view in views:
        view_runtime[view.view_id]["state"] = f"{phase}_completed"
    return result
