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

from .archive import ArchiveLayout, _artifact_json, refresh_key_material_metadata, write_json
from .config import load_config, load_manifest
from .collection_state import record_collection_state
from .daily_reports import generate_daily_report_from_archive
from .detection import validate_models
from .indexing import build_archive_index
from .pipeline import EvidencePipeline, create_dry_run
from .replay_acceptance import replay_quality_decisions_from_ledgers
from .schemas import RunSummary, VideoInfo
from .storage import (
    fixed_archive_staging_paths,
    initialize_nas_archive,
    prepare_from_nas_index,
    promote_fixed_archive,
    safe_archive_name,
)
from .validation import validate_experiment_and_material_quality


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


@app.command("validate-archive-quality")
def validate_archive_quality_command(
    archive: Annotated[Path, typer.Option("--archive", "-a", exists=True, file_okay=False)],
    baseline: Annotated[
        Path | None, typer.Option("--baseline", "-b", exists=True, dir_okay=False)
    ] = None,
    write: Annotated[bool, typer.Option("--write/--no-write")] = True,
) -> None:
    """Evaluate an existing archive without re-decoding video or calling a model."""

    package_path = archive / "JSON-Config-Files" / "evidence_package.json"
    package = RunSummary.model_validate_json(package_path.read_text(encoding="utf-8-sig"))
    baseline_payload = (
        json.loads(baseline.read_text(encoding="utf-8-sig")) if baseline is not None else None
    )
    key_events = [event for event in package.events if event.key_frames or event.key_clips]
    report = validate_experiment_and_material_quality(
        package.experiment_groups, key_events, baseline_payload
    )
    if write:
        write_json(archive / "JSON-Config-Files" / "quality_acceptance.json", report)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("replay-quality-ledger")
def replay_quality_ledger_command(
    archive: Annotated[Path, typer.Option("--archive", "-a", exists=True, file_okay=False)],
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Replay CV quality decisions from JSON only; never open video or call MLLM."""

    result = replay_quality_decisions_from_ledgers(archive, load_config(config))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        typer.echo(str(output.resolve()))
        return
    typer.echo(rendered)


@app.command("register-archived-collection")
def register_archived_collection_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    archive: Annotated[Path, typer.Option("--archive", "-a", exists=True, file_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)] = None,
) -> None:
    """Register an existing accepted archive in the central collection ledger."""

    accepted_files = (
        archive / "JSON-Config-Files" / "evidence_package_eval.json",
        archive / "JSON-Config-Files" / "quality_acceptance.json",
    )
    for path in accepted_files:
        if not path.is_file():
            raise typer.BadParameter(f"Acceptance receipt is missing: {path}")
        if not json.loads(path.read_text(encoding="utf-8-sig")).get("passed"):
            raise typer.BadParameter(f"Acceptance receipt did not pass: {path}")
    report_evaluations = list(
        (archive / "Lab-Daily-Reports").glob("*/Daily-Report-Eval.json")
    )
    if not report_evaluations or not all(
        json.loads(path.read_text(encoding="utf-8-sig")).get("passed")
        for path in report_evaluations
    ):
        raise typer.BadParameter("A passing daily-report evaluation is required")
    professional_pdfs = list(
        (archive / "Professional-PDFs").glob(
            "VisionCortex-Professional-Evidence-Report-*.pdf"
        )
    )
    if not professional_pdfs:
        raise typer.BadParameter("A professional evidence PDF is required")
    settings = load_config(config)
    run_id = f"archive-registration-{datetime.now():%Y%m%d-%H%M%S}"
    ledger = record_collection_state(
        settings,
        experiment_id,
        archive_name=archive.name,
        run_id=run_id,
        state="archived",
        details={
            "formal_archive": str(archive.resolve()),
            "registration_only": True,
            "daily_report_evaluations": len(report_evaluations),
            "professional_pdf_count": len(professional_pdfs),
        },
    )
    typer.echo(
        json.dumps(
            {
                "state": "archived",
                "source_experiment_id": experiment_id,
                "archive": str(archive.resolve()),
                "ledger": str(ledger),
                "model_calls": 0,
                "token_usage": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
    preprocessing_only: Annotated[
        bool,
        typer.Option(
            "--preprocessing-only/--full-pipeline",
            help="Stop after bounded dual-view CV acceptance; do not call MLLM or promote the fixed archive.",
        ),
    ] = False,
) -> None:
    """Run the registered six-view benchmark and reuse its fixed NAS archive."""

    experiment_id = "exp_20260810_144014_e918b762"
    archive_name = "Six-View-Three-Hour-Experiment-2026-08-13"
    settings = load_config(config)
    settings["storage"]["sync_to_nas"] = True
    settings["project"]["preprocessing_acceptance_only"] = preprocessing_only
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
    if preprocessing_only:
        typer.echo(
            f"{result} total_seconds={time.perf_counter() - started:.6f} "
            "run_mode=preprocessing_acceptance_only token_calls=0 "
            "fixed_archive_promotion=skipped"
        )
        return
    receipt = promote_fixed_archive(nas_root, fixed_root, history_root)
    typer.echo(
        f"{fixed_root} total_seconds={time.perf_counter() - started:.6f} "
        f"previous_package_retained={receipt['previous_package_retained']}"
    )


@app.command("run-index-collection")
def run_index_collection_command(
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    archive_name: Annotated[str, typer.Option("--archive-name")],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)] = Path(
        "configs/rtx4060-laptop-production.yaml"
    ),
) -> None:
    """Run one indexed NAS collection through staging and verified promotion."""

    settings = load_config(config)
    safe_name = safe_archive_name(archive_name)
    run_id = f"collection-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    fixed_root, staging_root, history_root = fixed_archive_staging_paths(
        settings, safe_name, run_id
    )
    if fixed_root.exists() and not fixed_root.is_dir():
        raise typer.BadParameter(
            f"Formal archive path exists but is not a directory: {fixed_root}"
        )
    formal_archive_preexisting = fixed_root.is_dir()
    settings["storage"]["sync_to_nas"] = True
    settings["storage"]["active_archive_path"] = str(staging_root)
    settings["project"]["output_root"] = str(
        Path(settings["storage"]["local_runtime_root"]).resolve() / "runs" / run_id
    )
    initialize_nas_archive(settings, safe_name)
    started = time.perf_counter()
    record_collection_state(
        settings,
        experiment_id,
        archive_name=safe_name,
        run_id=run_id,
        state="queued",
        details={
            "staging": str(staging_root),
            "source_copy_bytes": 0,
            "formal_archive": str(fixed_root),
            "formal_archive_preexisting": formal_archive_preexisting,
            "promotion_mode": "verified_atomic_replace",
        },
    )
    try:
        record_collection_state(
            settings,
            experiment_id,
            archive_name=safe_name,
            run_id=run_id,
            state="processing",
            details={
                "staging": str(staging_root),
                "formal_archive": str(fixed_root),
                "formal_archive_preexisting": formal_archive_preexisting,
                "promotion_mode": "verified_atomic_replace",
            },
        )
        manifest, manifest_path, ingest = prepare_from_nas_index(
            settings, experiment_id, lambda message: typer.echo(f"[NAS] {message}")
        )
        manifest_path.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        typer.echo(
            f"run_id={run_id} input_views={len(manifest.views)} "
            f"input_mode={ingest['input_mode']} source_copy_bytes="
            f"{ingest['copied_source_bytes']} staging={staging_root}"
        )
        EvidencePipeline(settings, _progress).run(manifest)
        receipt = promote_fixed_archive(staging_root, fixed_root, history_root)
        record_collection_state(
            settings,
            experiment_id,
            archive_name=safe_name,
            run_id=run_id,
            state="archived",
            details={
                "formal_archive": str(fixed_root),
                "promotion_verification": receipt.get("verification"),
                "previous_package_retained": receipt.get(
                    "previous_package_retained", False
                ),
            },
        )
        typer.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "state": "archived",
                    "source_experiment_id": experiment_id,
                    "archive": str(fixed_root),
                    "staging": str(staging_root),
                    "source_copy_bytes": ingest["copied_source_bytes"],
                    "total_seconds": round(time.perf_counter() - started, 6),
                    "promotion": receipt,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        record_collection_state(
            settings,
            experiment_id,
            archive_name=safe_name,
            run_id=run_id,
            state="failed",
            details={
                "staging": str(staging_root),
                "error": f"{type(exc).__name__}: {exc}",
                "total_seconds": round(time.perf_counter() - started, 6),
                "formal_archive": str(fixed_root),
                "formal_archive_preexisting": formal_archive_preexisting,
                "formal_archive_preserved": formal_archive_preexisting,
            },
        )
        raise


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
    refresh_key_material_metadata(
        layout,
        key_events,
        package.experiment_groups,
        transforms,
        archive_id=package.experiment_id,
    )
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
            package.experiment_id,
        )
        for event in key_events
    ]
    write_json(layout.key_materials / "Key-Materials-Model-Understanding.json", normalized)
    probe_path = layout.json_config / "video_probe.json"
    infos = (
        {
            view_id: VideoInfo.model_validate(payload)
            for view_id, payload in json.loads(
                probe_path.read_text(encoding="utf-8-sig")
            ).items()
        }
        if probe_path.is_file()
        else {}
    )
    index_manifest = build_archive_index(
        layout.root,
        package.experiment_id,
        normalized,
        package.events,
        package.experiment_groups,
        infos,
    )
    typer.echo(
        f"rewritten_events={len(normalized)} indexed_artifacts="
        f"{index_manifest['counts']['artifacts']}"
    )


if __name__ == "__main__":
    app()
