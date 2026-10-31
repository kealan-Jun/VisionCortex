"""Repeatable offline replay through the existing TemporalVideo runner.

Site paths/ROIs belong in a private manifest. There are no stored readings,
camera lists, font templates, business writes or service switches in this tool.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import platform

from visioncortex.lab.config import Settings
from visioncortex.lab.rules import Rules
from visioncortex.lab.storage import is_nas


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read(path, limit):
    if path.stat().st_size > limit:
        raise ValueError("Replay document exceeds its byte bound")
    return json.loads(path.read_text())


def validate_manifest(document):
    if document.get("schema_version") != "vc-display-ocr-replay/1":
        raise ValueError("Versioned display replay manifest required")
    cases = document.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError("One to eight bounded replay cases required")
    names = set()
    for case in cases:
        name = case.get("id")
        if not isinstance(name, str) or not name or len(name) > 80 or name in names:
            raise ValueError("Distinct bounded case IDs required")
        names.add(name)
        if case.get("split") not in {"development", "holdout"}:
            raise ValueError("Declare development or holdout before execution")
        if not Path(case["path"]).is_absolute():
            raise ValueError("Absolute original video path required")
        for field in case["registration"]["fields"]:
            layout = field["layout"]
            prohibited = {
                "glyph_adapter",
                "expected_value",
                "expected_reading",
                "forced_value",
                "correction_map",
            }
            if prohibited.intersection(layout) or layout.get("glyph_ocr_reconcile"):
                raise ValueError(
                    "Engineering replay excludes reading/font-specific corrections"
                )
    return cases


def worker(configuration_path):
    """One bounded CPU inference in an explicitly selected existing environment."""
    import cv2
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
    from lab_model_service import checkpoint, load_configuration
    from ppocr_onnx import PPOCRONNX, settings as model_settings

    config = load_configuration(Path(configuration_path))
    raw = sys.stdin.buffer.read(12 * 1024**2 + 1)
    if len(raw) > 12 * 1024**2:
        raise ValueError("Bounded native PNG input required")
    pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or pixels.shape[0] * pixels.shape[1] > 262144:
        raise ValueError("Bounded decoded numeric field required")
    verified = {}
    det, dr = checkpoint(
        Path(configuration_path).parent,
        "PP-OCRv6_medium_det",
        config,
        verified=verified,
    )
    rec, rr = checkpoint(
        Path(configuration_path).parent,
        "PP-OCRv6_medium_rec",
        config,
        verified=verified,
    )
    configuration = dict(
        config["services"]["ocr"], input_mode="field_roi", cpu_threads=1
    )
    signature = digest(dict(revisions=[dr, rr], settings=model_settings(configuration)))
    cv2.setNumThreads(1)
    result = PPOCRONNX(det, rec, configuration).predict(pixels)
    print(
        json.dumps(
            dict(
                result=result,
                model_signature=signature,
                python=platform.python_version(),
                opencv=cv2.__version__,
                numpy=np.__version__,
            )
        )
    )


def main(argv=None):
    from visioncortex.lab.temporal_video import TemporalVideo
    import cv2
    import numpy as np

    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-manifest", type=Path)
    parser.add_argument("--algorithm", choices=["baseline", "native"], default="native")
    parser.add_argument(
        "--mode", choices=["geometry", "replay", "infer"], default="replay"
    )
    parser.add_argument("--model-services", type=Path)
    parser.add_argument("--ocr-python", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--max-new-calls", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.max_new_calls <= 32 or args.mode != "infer" and args.max_new_calls:
        parser.error("Only explicit infer mode accepts a bounded new-call budget")
    if args.storage_manifest:
        os.environ["VC_LAB_STORAGE_MANIFEST"] = str(args.storage_manifest.absolute())
    document = read(args.manifest, 256 * 1024)
    cases = validate_manifest(document)
    output = args.output.absolute()
    if (
        is_nas(output.parent)
        or output.parent.resolve() != output.parent
        or not output.parent.is_dir()
    ):
        raise ValueError("Existing canonical local output parent required")
    output.mkdir(exist_ok=False)
    rules = Rules().identity()
    plan = dict(
        schema_version="vc-display-ocr-replay/1",
        manifest_sha256=digest(document),
        rules=rules,
        algorithm=args.algorithm,
        mode=args.mode,
        max_new_calls=args.max_new_calls,
        cases=[dict(id=c["id"], split=c["split"]) for c in cases],
        cache_file_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest()
        if args.cache
        else None,
        model_configuration_sha256=hashlib.sha256(
            args.model_services.read_bytes()
        ).hexdigest()
        if args.model_services
        else None,
        source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(
            python=platform.python_version(),
            opencv=cv2.__version__,
            numpy=np.__version__,
            opencv_threads=1,
            case_RNG_seed=0,
        ),
        roi_basis="explicit_registered_geometry_not_automatic_panel_discovery",
        confirmed_values_are_not_acceptance_truth=True,
    )
    (output / "RunPlan.json").write_text(json.dumps(plan, indent=2) + "\n")
    cache = {}
    model = None
    config = None
    model_signature = None
    new_calls = hits = attempts = unknown_calls = 0
    actual = []
    verified = {}
    if args.mode != "geometry":
        if args.model_services is None:
            parser.error("Explicit existing local model configuration required")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
        from lab_model_service import checkpoint, load_configuration
        from ppocr_onnx import settings as model_settings

        config = load_configuration(args.model_services)
        det, det_revision = checkpoint(
            output, "PP-OCRv6_medium_det", config, verified=verified
        )
        rec, rec_revision = checkpoint(
            output, "PP-OCRv6_medium_rec", config, verified=verified
        )
        configuration = dict(
            config["services"]["ocr"], input_mode="field_roi", cpu_threads=1
        )
        model = dict(
            revisions=[det_revision, rec_revision],
            settings=model_settings(configuration),
        )
        model_signature = digest(model)
        if args.cache:
            existing = read(args.cache, 16 * 1024**2)
            if existing.get("model_signature") != model_signature:
                raise ValueError("Recorded OCR model/settings differ from this replay")
            cache = existing["entries"]
            if not isinstance(cache, dict) or len(cache) > 512:
                raise ValueError("Bounded OCR receipt cache required")

    def infer(raw):
        nonlocal new_calls, hits, attempts, unknown_calls
        import cv2
        import numpy as np

        pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        key = digest(
            dict(
                model=model_signature,
                shape=list(pixels.shape),
                bgr=hashlib.sha256(pixels.tobytes()).hexdigest(),
            )
        )
        if key in cache:
            hits += 1
            return cache[key]["result"], dict(
                mode="exact_pixels_recorded_result",
                actual_neural_calls=0,
                original_execution=cache[key]["execution"],
                cache_key=key,
            )
        if args.mode == "replay":
            raise RuntimeError("Recorded OCR unavailable for exact input: " + key)
        if attempts >= args.max_new_calls:
            raise RuntimeError("Explicit replay inference budget exhausted")
        executable = args.ocr_python
        if (
            executable is None
            or not executable.is_file()
            or str(executable.resolve()).startswith(("/mnt/", "/media/"))
        ):
            raise ValueError("Explicit existing local OCR Python required")
        attempts += 1
        attempt_path = output / f"InferenceAttempt-{attempts:02d}.json"
        attempt = dict(input_key=key, status="started", actual_neural_calls=None)
        attempt_path.write_text(json.dumps(attempt) + "\n")
        try:
            child = subprocess.run(
                [
                    str(executable),
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(args.model_services),
                ],
                input=raw,
                capture_output=True,
                timeout=90,
            )
            if child.returncode or len(child.stdout) > 1024**2:
                raise RuntimeError(
                    "Local OCR worker failed with exit " + str(child.returncode)
                )
            response = json.loads(child.stdout)
            if response["model_signature"] != model_signature:
                raise ValueError("OCR worker model/settings changed")
            result = response["result"]
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            unknown_calls += 1
            attempt.update(status="failed_unknown_call_count", reason=str(exc))
            attempt_path.write_text(json.dumps(attempt) + "\n")
            raise RuntimeError(
                "Bounded OCR attempt failed; preserved count uncertainty"
            ) from exc
        calls = result["onnx_execution"]["calls"]
        if len(calls) != 1 or calls[0]["kind"] != "rec":
            raise ValueError("Replay requires exactly one local ROI recognition")
        new_calls += 1
        attempt.update(status="completed", actual_neural_calls=1)
        attempt_path.write_text(json.dumps(attempt) + "\n")
        execution = dict(
            mode="local_PP_OCRv6_medium_CPU",
            actual_neural_calls=1,
            model=model,
            input_bgr_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
            shape=list(pixels.shape),
            cache_key=key,
            worker_runtime={k: response[k] for k in ["python", "opencv", "numpy"]},
        )
        cache[key] = dict(result=result, execution=execution)
        actual.append(execution)
        # Persist after each successful invocation; interruption cannot hide cost.
        (output / "OcrCache.json").write_text(
            json.dumps(dict(model_signature=model_signature, entries=cache)) + "\n"
        )
        return result, execution

    completed = []
    started = time.monotonic()
    settings = Settings(
        database_url="", runtime_root=output, customer_code="offline-replay"
    )
    for index, case in enumerate(cases):
        cv2.setRNGSeed(0)
        registration = copy.deepcopy(case["registration"])
        for field in registration["fields"]:
            field["layout"]["numeric_anchor"] = True
            field["layout"]["numeric_native_ocr"] = args.algorithm == "native"
        try:
            events, receipts = TemporalVideo(settings).run(
                case["path"],
                case["ref"],
                registration,
                None if args.mode == "geometry" else infer,
                case["provider"],
            )
            result = dict(
                id=case["id"],
                split=case["split"],
                status="processed",
                events=events,
                receipts=receipts,
                independent_accuracy=None,
                production_changes=0,
            )
            fields = [f for event in events for f in event.get("fields", [])]
            executions = [
                r["temporal_execution"] for r in receipts if "temporal_execution" in r
            ]
            result["metrics"] = dict(
                input_frames=sum(r["input_frame_count"] for r in executions),
                recorded_field_events=len(fields),
                candidate_events=sum(
                    f.get("reading_status") in {"recognized_candidate", "glyphs_only"}
                    for f in fields
                ),
                quality_accepted=False,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            result = dict(
                id=case["id"],
                split=case["split"],
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                independent_accuracy=None,
                production_changes=0,
            )
        raw = json.dumps(result, ensure_ascii=False, indent=2).encode()
        if (
            len(raw) > 8 * 1024**2
            or sum(p.stat().st_size for p in output.iterdir() if p.is_file()) + len(raw)
            > 32 * 1024**2
        ):
            raise ValueError("Local replay output budget exceeded")
        (output / f"Case-{index:02d}.json").write_bytes(raw)
        completed.append(
            {k: v for k, v in result.items() if k not in {"events", "receipts"}}
        )
    if Rules().identity() != rules:
        raise ValueError("Processing source/rules changed during replay")
    summary = dict(
        plan=plan,
        cases=completed,
        actual_neural_calls=new_calls,
        inference_attempts=attempts,
        failed_attempts_with_unknown_call_count=unknown_calls,
        exact_input_cache_hits=hits,
        models=model,
        actual_executions=actual,
        wall_seconds=time.monotonic() - started,
        independent_accuracy=None,
        database_writes=0,
        production_changes=0,
        glyph_templates_used=False,
        expected_readings_in_rules=False,
    )
    (output / "Summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "OcrCache.json").write_text(
        json.dumps(dict(model_signature=model_signature, entries=cache)) + "\n"
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                cases=completed,
                actual_neural_calls=new_calls,
                cache_hits=hits,
            ),
            ensure_ascii=False,
        )
    )
    return 0 if all(c["status"] == "processed" for c in completed) else 2


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        raise SystemExit(main())
