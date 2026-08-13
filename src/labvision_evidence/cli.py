from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .config import load_config, load_manifest
from .detection import validate_models
from .daily_reports import generate_daily_report_from_archive
from .pipeline import EvidencePipeline, create_dry_run
from .archive import ArchiveLayout, refresh_key_material_metadata, write_json, _artifact_json
from .schemas import RunSummary
from .storage import fixed_archive_staging_paths, prepare_from_nas_index, promote_fixed_archive


app = typer.Typer(no_args_is_help=True, help="多视角化学实验视频证据流水线")


def _progress(stage: str, progress: float, message: str) -> None:
    typer.echo(f"[{progress:6.1%}] {stage}: {message}")


@app.command("generate-daily-report")
def generate_daily_report_command(
    archive: Annotated[Path, typer.Option("--archive", "-a", exists=True, file_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
) -> None:
    """Generate or refresh a zero-additional-token report from an accepted archive."""

    result = generate_daily_report_from_archive(archive, load_config(config))
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("run")
def run_command(
    manifest: Annotated[Path, typer.Option("--manifest", "-m", exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    settings = load_config(config)
    if output:
        settings["project"]["output_root"] = str(output)
    result = EvidencePipeline(settings, _progress).run(load_manifest(manifest))
    typer.echo(str(result))


@app.command("dry-run")
def dry_run_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("outputs/dry-run"),
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
) -> None:
    result = create_dry_run(output, load_config(config))
    typer.echo(str(result))


@app.command("validate-models")
def validate_models_command(
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
) -> None:
    typer.echo(json.dumps(validate_models(load_config(config)), ensure_ascii=False, indent=2))


@app.command("prepare-engine")
def prepare_engine_command(
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
) -> None:
    settings = load_config(config)
    perf = settings["performance"]
    from ultralytics import YOLO

    for role in ("first_person", "third_person"):
        source = Path(settings["models"][role])
        destination = Path(settings["models"][f"{role}_engine"])
        if destination.is_file():
            typer.echo(f"{role}: 已存在 {destination}")
            continue
        typer.echo(f"{role}: 从 {source} 导出 TensorRT，首次导出可能较久")
        temporary_root = Path(tempfile.mkdtemp(prefix=f"labvision-{role}-"))
        try:
            temporary_source = temporary_root / "model.pt"
            shutil.copy2(source, temporary_source)
            model = YOLO(str(temporary_source))
            exported = Path(
                model.export(
                    format="engine",
                    imgsz=int(perf["image_size"]),
                    half=bool(perf["half"]),
                    dynamic=True,
                    batch=int(perf["batch_size"]),
                    workspace=3,
                    device=perf["device"],
                )
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(exported), str(destination))
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
        typer.echo(f"{role}: {destination}")


@app.command("run-fixed-benchmark")
def run_fixed_benchmark_command(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)] = Path(
        "configs/rtx4060-laptop-production.yaml"
    ),
) -> None:
    """Run the registered six-view benchmark and reuse its fixed NAS archive."""

    experiment_id = "exp_20260810_144014_e918b762"
    archive_name = "Six-View-Three-Hour-Experiment-2026-08-13"
    settings = load_config(config)
    settings["storage"]["sync_to_nas"] = True
    run_id = f"cli-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    fixed_root, nas_root, history_root = fixed_archive_staging_paths(
        settings, archive_name, run_id
    )
    local_runtime_root = Path(settings["storage"]["local_runtime_root"]).resolve()
    settings["project"]["output_root"] = str(local_runtime_root / "runs" / run_id)
    settings["storage"]["active_archive_path"] = str(nas_root)
    started = time.perf_counter()
    manifest, manifest_path, ingest = prepare_from_nas_index(
        settings, experiment_id, lambda message: typer.echo(f"[NAS] {message}")
    )
    manifest.experiment_id = archive_name
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    typer.echo(
        f"input_views={len(manifest.views)} input_mode={ingest['input_mode']} "
        f"source_copy_bytes={ingest['copied_source_bytes']} manifest={ingest['manifest']} "
        f"nas_staging={nas_root} fixed_output={fixed_root}"
    )
    result = EvidencePipeline(settings, _progress).run(manifest)
    receipt = promote_fixed_archive(nas_root, fixed_root, history_root)
    typer.echo(
        f"{fixed_root} total_seconds={time.perf_counter() - started:.6f} "
        f"previous_package_retained={receipt['previous_package_retained']}"
    )


@app.command("serve")
def serve_command(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
) -> None:
    import os

    import uvicorn

    if config:
        os.environ["LABVISION_CONFIG"] = str(config.resolve())
    uvicorn.run("labvision_evidence.api:app", host=host, port=port, reload=False)


@app.command("refresh-key-json")
def refresh_key_json_command(
    archive: Annotated[Path, typer.Option("--archive", "-a", exists=True, file_okay=False)],
) -> None:
    """Rewrite key-material JSON sidecars using the normalized event contract."""
    layout = ArchiveLayout(archive.resolve())
    package = RunSummary.model_validate_json(
        (layout.json_config / "evidence_package.json").read_text(encoding="utf-8")
    )
    transforms = {item.view_id: item for item in package.alignments}
    key_events = [event for event in package.events if event.key_frames or event.key_clips]
    refresh_key_material_metadata(layout, key_events, package.experiment_groups, transforms)
    normalized = [
        _artifact_json(
            next(
                group
                for group in package.experiment_groups
                if event.event_id in group.key_event_ids
            ),
            event,
            "key_material_event_index",
            "",
            None,
            transforms,
        )
        for event in key_events
    ]
    write_json(layout.key_materials / "Key-Materials-Model-Understanding.json", normalized)
    typer.echo(f"rewritten_events={len(normalized)}")


if __name__ == "__main__":
    app()
